#!/usr/bin/env python3
"""Prepare multi-source real spatial QA datasets for training.

Supported sources:
- bAbI (auto download, tasks 17/19)
- SpartQA (local parsed JSON)
- CLEVR (local question JSON)
- GQA (local question JSON)
- NLVR2 (local JSON/JSONL)
- ScanQA (local JSON)

All sources are normalized to the same schema expected by the training pipeline.
"""

import argparse
import json
import random
import tarfile
import urllib.request
from pathlib import Path

BABI_URL = "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz"


def read_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    return data


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
    sample.setdefault("image_vector", [0.0] * visual_dim)
    return sample


def parse_babi(raw_dir, visual_dim=512):
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
            with open(p, "r", encoding="utf-8") as f:
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
                            "context": " ".join(ctx),
                            "question": q.strip(),
                            "answer": a.strip(),
                            "task_type": task_type,
                            "difficulty": min(3, max(1, len(ctx) // 3)),
                            "source": "babi",
                        }
                        samples.append(add_traj(s, steps=max(4, min(32, len(ctx))), visual_dim=visual_dim))
                    else:
                        ctx.append(txt)
    return samples


def parse_spartqa(path, visual_dim=512):
    if path is None or not Path(path).exists():
        return []
    data = read_json_or_jsonl(Path(path))
    items = data if isinstance(data, list) else data.get("data", [])
    out = []
    for item in items:
        out.append(add_traj({
            "context": item.get("context", item.get("story", "")),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "task_type": item.get("task_type", "spatial_relation"),
            "difficulty": item.get("difficulty", 2),
            "source": "spartqa",
        }, visual_dim=visual_dim))
    return out


def parse_clevr(path, visual_dim=512):
    if path is None or not Path(path).exists():
        return []
    data = read_json_or_jsonl(Path(path))
    items = data.get("questions", data if isinstance(data, list) else [])
    out = []
    for item in items:
        out.append(add_traj({
            "context": item.get("scene_description", "CLEVR scene"),
            "question": item.get("question", ""),
            "answer": str(item.get("answer", "unknown")),
            "task_type": "object_location",
            "difficulty": 2,
            "source": "clevr",
        }, visual_dim=visual_dim))
    return out


def parse_gqa(path, visual_dim=512):
    if path is None or not Path(path).exists():
        return []
    data = read_json_or_jsonl(Path(path))
    if isinstance(data, dict):
        items = list(data.values())
    else:
        items = data
    out = []
    for item in items:
        out.append(add_traj({
            "context": item.get("scene_graph", "GQA image-grounded scene") if isinstance(item.get("scene_graph"), str) else "GQA image-grounded scene",
            "question": item.get("question", ""),
            "answer": str(item.get("answer", "unknown")),
            "task_type": "object_location",
            "difficulty": 2,
            "source": "gqa",
        }, visual_dim=visual_dim))
    return out


def parse_nlvr2(path, visual_dim=512):
    if path is None or not Path(path).exists():
        return []
    data = read_json_or_jsonl(Path(path))
    items = data if isinstance(data, list) else data.get("data", [])
    out = []
    for item in items:
        out.append(add_traj({
            "context": item.get("sentence", item.get("context", "NLVR2 paired-image statement")),
            "question": item.get("question", "Is the statement true?"),
            "answer": str(item.get("label", item.get("answer", "unknown"))),
            "task_type": "spatial_relation",
            "difficulty": 2,
            "source": "nlvr2",
        }, visual_dim=visual_dim))
    return out


def parse_scanqa(path, visual_dim=512):
    if path is None or not Path(path).exists():
        return []
    data = read_json_or_jsonl(Path(path))
    items = data if isinstance(data, list) else data.get("data", [])
    out = []
    for item in items:
        answers = item.get("answers", [])
        answer = answers[0] if isinstance(answers, list) and answers else item.get("answer", "unknown")
        out.append(add_traj({
            "context": item.get("scene_id", "ScanQA 3D scene") + ": " + item.get("situation", "3D indoor environment"),
            "question": item.get("question", ""),
            "answer": str(answer),
            "task_type": "object_location",
            "difficulty": 3,
            "source": "scanqa",
        }, visual_dim=visual_dim))
    return out


def gen_aug(n, visual_dim=512):
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
            "difficulty": random.choice([1, 2, 3]),
            "source": "augmented",
        }, steps=random.randint(8, 24), visual_dim=visual_dim))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--spartqa-path", default=None)
    ap.add_argument("--clevr-path", default=None)
    ap.add_argument("--gqa-path", default=None)
    ap.add_argument("--nlvr2-path", default=None)
    ap.add_argument("--scanqa-path", default=None)
    ap.add_argument("--synthetic-aug", type=int, default=3000)
    ap.add_argument("--visual-dim", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    samples.extend(parse_babi(raw_dir, visual_dim=args.visual_dim))
    samples.extend(parse_spartqa(args.spartqa_path, visual_dim=args.visual_dim))
    samples.extend(parse_clevr(args.clevr_path, visual_dim=args.visual_dim))
    samples.extend(parse_gqa(args.gqa_path, visual_dim=args.visual_dim))
    samples.extend(parse_nlvr2(args.nlvr2_path, visual_dim=args.visual_dim))
    samples.extend(parse_scanqa(args.scanqa_path, visual_dim=args.visual_dim))
    samples.extend(gen_aug(args.synthetic_aug, visual_dim=args.visual_dim))

    random.shuffle(samples)
    n = len(samples)
    train_end, val_end = int(0.7 * n), int(0.85 * n)
    splits = {
        "real_train": samples[:train_end],
        "real_val": samples[train_end:val_end],
        "real_test": samples[val_end:],
    }

    for name, data in splits.items():
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Wrote {name}: {len(data)} -> {p}")


if __name__ == "__main__":
    main()
