#!/usr/bin/env python3
"""Prepare real/spatial datasets for training.

This script downloads bAbI tasks (17 and 19), parses them into the expected JSON
format, and augments with trajectory-style synthetic spatial samples so the model
can train with spatiotemporal fields.
"""

import argparse
import json
import os
import random
import tarfile
import urllib.request
from pathlib import Path


BABI_URL = "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz"


def download_babi(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    tar_path = raw_dir / "babi_tasks_1-20_v1-2.tar.gz"
    extracted = raw_dir / "tasks_1-20_v1-2"

    if not tar_path.exists():
        print(f"Downloading bAbI from {BABI_URL} ...")
        urllib.request.urlretrieve(BABI_URL, tar_path)

    if not extracted.exists():
        print("Extracting bAbI archive ...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(raw_dir)

    return extracted


def parse_babi_file(path: Path, task_type: str):
    samples = []
    context_lines = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            line_no = int(parts[0])
            text = parts[1]

            if line_no == 1:
                context_lines = []

            if "\t" in text:
                question, answer_part = text.split("\t", 1)
                answer = answer_part.split("\t")[0]
                trajectory_len = max(4, min(32, len(context_lines)))
                velocity = [[random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)] for _ in range(trajectory_len)]
                coords = []
                x, y = 0.0, 0.0
                for vx, vy in velocity:
                    x += vx
                    y += vy
                    coords.append([round(x, 3), round(y, 3)])

                samples.append(
                    {
                        "context": " ".join(context_lines),
                        "question": question.strip(),
                        "answer": answer.strip(),
                        "task_type": task_type,
                        "difficulty": min(3, max(1, len(context_lines) // 3)),
                        "source": "babi",
                        "velocity": velocity,
                        "coordinates": coords,
                        "temporal_distances": [1.0] * trajectory_len,
                        "spatial_distances": [round((c[0] ** 2 + c[1] ** 2) ** 0.5, 3) for c in coords],
                    }
                )
            else:
                context_lines.append(text)
    return samples


def gen_spatial_augmented_sample(task_type="object_location", difficulty=1):
    objs = ["table", "chair", "lamp", "cup", "box", "shelf", "book", "plant"]
    a, b = random.sample(objs, 2)
    rel = random.choice(["left of", "right of", "above", "below", "near", "behind"])

    steps = random.randint(8, 24)
    velocity = [[random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)] for _ in range(steps)]
    coords = []
    x, y = 0.0, 0.0
    for vx, vy in velocity:
        x += vx
        y += vy
        coords.append([round(x, 3), round(y, 3)])

    context = f"In a room, the {a} is {rel} the {b}. The agent explores the scene over time."
    question = f"Where is the {a} relative to the {b}?"
    answer = rel.replace(" of", "")

    return {
        "context": context,
        "question": question,
        "answer": answer,
        "task_type": task_type,
        "difficulty": difficulty,
        "source": "spatial_augmented",
        "velocity": velocity,
        "coordinates": coords,
        "temporal_distances": [1.0] * steps,
        "spatial_distances": [round((c[0] ** 2 + c[1] ** 2) ** 0.5, 3) for c in coords],
    }


def write_splits(samples, out_dir: Path):
    random.shuffle(samples)
    n = len(samples)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    splits = {
        "real_train": samples[:train_end],
        "real_val": samples[train_end:val_end],
        "real_test": samples[val_end:],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        out = out_dir / f"{name}.json"
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote {name}: {len(data)} -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw", type=str)
    parser.add_argument("--out-dir", default="data/processed", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--synthetic-aug", default=3000, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    extracted = download_babi(raw_dir)
    base = extracted / "en-valid"

    all_samples = []
    for task_num, task_type in [(17, "spatial_relation"), (19, "navigation")]:
        for split in ["train", "valid", "test"]:
            p = base / f"qa{task_num}_{split}.txt"
            if p.exists():
                parsed = parse_babi_file(p, task_type)
                all_samples.extend(parsed)
                print(f"Parsed {p}: {len(parsed)}")

    for _ in range(args.synthetic_aug):
        difficulty = random.choices([1, 2, 3], weights=[0.4, 0.35, 0.25])[0]
        all_samples.append(gen_spatial_augmented_sample(difficulty=difficulty))

    print(f"Total prepared samples: {len(all_samples)}")
    write_splits(all_samples, out_dir)


if __name__ == "__main__":
    main()
