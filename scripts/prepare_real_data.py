#!/usr/bin/env python3
"""Prepare multi-source real-ish datasets for training.

Sources:
- bAbI tasks 17/19 (downloaded)
- optional local SpartQA JSON files
- optional local CLEVR scene/question JSON files

All samples are normalized into a common schema with optional image_vector fields.
"""

import argparse
import json
import random
import tarfile
import urllib.request
from pathlib import Path

BABI_URL = "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz"


def add_traj(sample, steps=16, visual_dim=512):
    velocity = [[random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)] for _ in range(steps)]
    coords, x, y = [], 0.0, 0.0
    for vx, vy in velocity:
        x += vx
        y += vy
        coords.append([round(x, 3), round(y, 3)])
    sample["velocity"] = velocity
    sample["coordinates"] = coords
    sample["temporal_distances"] = [1.0] * steps
    sample["spatial_distances"] = [round((a * a + b * b) ** 0.5, 3) for a, b in coords]
    sample["image_vector"] = [0.0] * visual_dim
    return sample


def parse_babi(raw_dir):
    tar_path = raw_dir / "babi_tasks_1-20_v1-2.tar.gz"
    extracted = raw_dir / "tasks_1-20_v1-2"
    if not tar_path.exists():
        urllib.request.urlretrieve(BABI_URL, tar_path)
    if not extracted.exists():
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(raw_dir)

    samples = []
    base = extracted / "en-valid"
    for task_num, task_type in [(17, "spatial_relation"), (19, "navigation")]:
        for split in ["train", "valid", "test"]:
            p = base / f"qa{task_num}_{split}.txt"
            if not p.exists():
                continue
            ctx = []
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    n, txt = line.split(" ", 1)
                    if int(n) == 1:
                        ctx = []
                    if "\t" in txt:
                        q, rest = txt.split("\t", 1)
                        a = rest.split("\t")[0]
                        s = {
                            "context": " ".join(ctx), "question": q.strip(), "answer": a.strip(),
                            "task_type": task_type, "difficulty": min(3, max(1, len(ctx)//3)), "source": "babi"
                        }
                        samples.append(add_traj(s, steps=max(4, min(32, len(ctx)))))
                    else:
                        ctx.append(txt)
    return samples


def parse_spartqa(path):
    if path is None or not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text())
    out = []
    for item in data:
        out.append(add_traj({
            "context": item.get("context", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "task_type": item.get("task_type", "spatial_relation"),
            "difficulty": item.get("difficulty", 2),
            "source": "spartqa",
        }))
    return out


def parse_clevr(path):
    if path is None or not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text())
    items = data.get("questions", data if isinstance(data, list) else [])
    out = []
    for item in items:
        out.append(add_traj({
            "context": item.get("scene_description", "synthetic CLEVR scene"),
            "question": item.get("question", ""),
            "answer": str(item.get("answer", "unknown")),
            "task_type": "object_location",
            "difficulty": 2,
            "source": "clevr",
        }))
    return out


def gen_aug(n):
    objs = ["table", "chair", "lamp", "cup", "box", "shelf", "book", "plant"]
    rels = ["left", "right", "above", "below", "near", "behind"]
    out = []
    for _ in range(n):
        a, b = random.sample(objs, 2)
        rel = random.choice(rels)
        out.append(add_traj({
            "context": f"In a room, {a} is {rel} of {b}.",
            "question": f"Where is {a} relative to {b}?",
            "answer": rel,
            "task_type": "spatial_relation",
            "difficulty": random.choice([1,2,3]),
            "source": "augmented",
        }, steps=random.randint(8, 24)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--spartqa-path", default=None)
    ap.add_argument("--clevr-path", default=None)
    ap.add_argument("--synthetic-aug", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    raw_dir = Path(args.raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    samples.extend(parse_babi(raw_dir))
    samples.extend(parse_spartqa(args.spartqa_path))
    samples.extend(parse_clevr(args.clevr_path))
    samples.extend(gen_aug(args.synthetic_aug))

    random.shuffle(samples)
    n = len(samples)
    train_end, val_end = int(0.7*n), int(0.85*n)
    splits = {
        "real_train": samples[:train_end],
        "real_val": samples[train_end:val_end],
        "real_test": samples[val_end:],
    }
    for name, data in splits.items():
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps(data, indent=2))
        print(f"Wrote {name}: {len(data)} -> {p}")


if __name__ == "__main__":
    main()
