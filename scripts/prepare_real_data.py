#!/usr/bin/env python3
"""Download and prepare real spatial reasoning datasets.

Supported datasets (auto-download):
  - bAbI Tasks 17 & 19  -- clean symbolic spatial reasoning (Facebook/Meta)
  - SpartQA             -- controlled synthetic spatial QA (Allen AI)
  - CLEVR (no images)   -- object-centric spatial logic (Meta)
  - GQA questions        -- real human spatial language (Stanford)

Manual download required (provide local path):
  - NLVR2               -- compositional spatial verification (Cornell)
  - ScanQA              -- 3D scene spatial QA (ATR-DBI)

Usage:
  # Auto-download everything + 1000 synthetic augmentation:
  python scripts/prepare_real_data.py --synthetic-aug 1000

  # With local dataset paths:
  python scripts/prepare_real_data.py \\
      --spartqa-path data/raw/spartqa \\
      --clevr-path data/raw/clevr \\
      --gqa-path data/raw/gqa \\
      --nlvr2-path data/raw/nlvr2 \\
      --scanqa-path data/raw/scanqa
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tarfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

# === Download URLs ===
BABI_URL = "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz"
CLEVR_NO_IMAGES_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip"
GQA_QUESTIONS_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"

# SpartQA: try multiple possible GitHub locations
SPARTQA_URLS = [
    "https://raw.githubusercontent.com/HLR/SpartQA_baselines/main/SpartQA_data/{split}.json",
    "https://raw.githubusercontent.com/HLR/SpartQA/main/data/{split}.json",
    "https://raw.githubusercontent.com/HLR/SpartQA_baselines/master/SpartQA_data/{split}.json",
]


# ============================================================
# Download helper
# ============================================================
def download(url, dest, desc=""):
    """Download a file. Returns True on success."""
    dest = Path(dest)
    if dest.exists():
        print(f"  [skip] {desc or dest.name} already exists")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or url}...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / 1e6
        print(f"  -> {dest} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {desc or url}: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ============================================================
# 1. bAbI Tasks (positional reasoning & path finding)
# ============================================================
def download_and_parse_babi():
    """Download bAbI tasks and extract spatial reasoning (tasks 17, 19)."""
    print("\n=== 1. bAbI Tasks ===")
    samples = []
    tar_path = RAW_DIR / "babi.tar.gz"

    if not download(BABI_URL, tar_path, "bAbI tasks (~11 MB)"):
        print("  [WARN] bAbI download failed, skipping.")
        return samples

    extract_dir = RAW_DIR / "tasks_1-20_v1-2"
    if not extract_dir.exists():
        print("  Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(RAW_DIR)

    babi_base = RAW_DIR / "tasks_1-20_v1-2" / "en-valid"
    task_map = {
        17: "spatial_relation",  # Positional reasoning
        19: "navigation",       # Path finding
    }

    for task_num, task_type in task_map.items():
        for split in ["train", "valid", "test"]:
            path = babi_base / f"qa{task_num}_{split}.txt"
            if path.exists():
                parsed = _parse_babi_file(path, task_type)
                samples.extend(parsed)
                print(f"  bAbI task {task_num} ({task_type}) {split}: {len(parsed)} samples")

    print(f"  Total bAbI: {len(samples)}")
    return samples


def _parse_babi_file(filepath, task_type):
    """Parse a single bAbI task file into structured samples."""
    samples = []
    context_lines = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            line_num = int(parts[0])
            text = parts[1]

            if line_num == 1:
                context_lines = []

            if "\t" in text:
                question, answer_part = text.split("\t", 1)
                answer = answer_part.split("\t")[0]
                samples.append({
                    "context": " ".join(context_lines),
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "task_type": task_type,
                    "difficulty": min(3, max(1, len(context_lines) // 3)),
                    "source": "babi",
                })
            else:
                context_lines.append(text)
    return samples


# ============================================================
# 2. SpartQA (controlled synthetic spatial QA)
# ============================================================
def download_and_parse_spartqa(local_path=None):
    """Download SpartQA from GitHub or use local path."""
    print("\n=== 2. SpartQA ===")
    samples = []

    if local_path and Path(local_path).exists():
        print(f"  Using local path: {local_path}")
        samples = _parse_spartqa_dir(Path(local_path))
        if samples:
            print(f"  Total SpartQA (local): {len(samples)}")
            return samples

    # Try downloading from GitHub
    spartqa_dir = RAW_DIR / "spartqa"
    spartqa_dir.mkdir(parents=True, exist_ok=True)

    downloaded_any = False
    for split in ["train", "dev", "test"]:
        dest = spartqa_dir / f"{split}.json"
        if dest.exists():
            downloaded_any = True
            continue
        for url_template in SPARTQA_URLS:
            url = url_template.format(split=split)
            if download(url, dest, f"SpartQA {split}"):
                downloaded_any = True
                break

    if downloaded_any:
        samples = _parse_spartqa_dir(spartqa_dir)

    if not samples:
        print("  [WARN] SpartQA download failed. Generating SpartQA-style synthetic data.")
        samples = _generate_spartqa_synthetic(2000)

    print(f"  Total SpartQA: {len(samples)}")
    return samples


def _parse_spartqa_dir(data_dir):
    """Parse SpartQA JSON files from a directory."""
    samples = []
    for json_file in sorted(Path(data_dir).rglob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    s = _convert_spartqa_item(item)
                    if s:
                        samples.append(s)
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        s = _convert_spartqa_item(value)
                        if s:
                            samples.append(s)
                    elif isinstance(value, list):
                        for item in value:
                            s = _convert_spartqa_item(item)
                            if s:
                                samples.append(s)
        except Exception as e:
            print(f"  [WARN] Failed to parse {json_file.name}: {e}")
    return samples


def _convert_spartqa_item(item):
    """Convert a single SpartQA item to our common format."""
    if not isinstance(item, dict):
        return None

    # SpartQA fields vary: story/context/passage, question/query, answer/label
    context = item.get("story", item.get("context", item.get("passage", "")))
    question = item.get("question", item.get("query", ""))
    answer = item.get("answer", item.get("label", ""))

    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    answer = str(answer)

    if not (context and question and answer):
        return None

    q_type = item.get("q_type", item.get("type", ""))
    task_type = "spatial_relation"
    if q_type in ("FB", "find_block"):
        task_type = "object_location"
    elif q_type in ("CO", "count"):
        task_type = "object_location"

    return {
        "context": str(context).strip(),
        "question": str(question).strip(),
        "answer": answer.strip(),
        "task_type": task_type,
        "difficulty": int(item.get("difficulty", 2)),
        "source": "spartqa",
    }


def _generate_spartqa_synthetic(n):
    """Generate SpartQA-style data as fallback."""
    ROOMS = ["kitchen", "bedroom", "bathroom", "living room", "garage",
             "garden", "office", "hallway", "basement", "attic"]
    OBJECTS = ["table", "chair", "lamp", "book", "cup", "box", "shelf",
               "plant", "clock", "mirror", "bed", "desk", "sofa", "vase"]
    COLORS = ["red", "blue", "green", "yellow", "white", "black", "brown"]
    SIZES = ["small", "large", "medium", "tiny"]
    RELATIONS = [
        ("left of", "right of"), ("right of", "left of"),
        ("above", "below"), ("below", "above"),
        ("in front of", "behind"), ("behind", "in front of"),
        ("inside", "outside"), ("on top of", "under"),
        ("near", "far from"), ("next to", "away from"),
    ]

    samples = []
    for _ in range(n):
        objs = random.sample(OBJECTS, min(4, len(OBJECTS)))
        rel, inv_rel = random.choice(RELATIONS)
        a = f"{random.choice(COLORS)} {random.choice(SIZES)} {objs[0]}"
        b = f"{random.choice(COLORS)} {random.choice(SIZES)} {objs[1]}"
        room = random.choice(ROOMS)

        ctx = f"In the {room}, the {a} is {rel} the {b}."
        difficulty = random.choices([1, 2, 3], weights=[0.4, 0.35, 0.25])[0]

        if difficulty >= 2 and len(objs) > 2:
            c = f"{random.choice(COLORS)} {random.choice(SIZES)} {objs[2]}"
            r2, _ = random.choice(RELATIONS)
            ctx += f" The {c} is {r2} the {b}."
        if difficulty >= 3 and len(objs) > 3:
            d = f"{random.choice(COLORS)} {random.choice(SIZES)} {objs[3]}"
            r3, _ = random.choice(RELATIONS)
            ctx += f" The {d} is {r3} the {a}."

        q_type = random.choice(["relation", "yes_no", "find"])
        if q_type == "relation":
            question = f"What is the spatial relation of the {a} to the {b}?"
            answer = rel
        elif q_type == "yes_no":
            if random.random() > 0.5:
                question = f"Is the {a} {rel} the {b}?"
                answer = "yes"
            else:
                question = f"Is the {a} {inv_rel} the {b}?"
                answer = "no"
        else:
            question = f"Which object is {inv_rel} the {b}?"
            answer = a

        samples.append({
            "context": ctx,
            "question": question,
            "answer": answer,
            "task_type": "spatial_relation",
            "difficulty": difficulty,
            "source": "spartqa_synthetic",
        })
    return samples


# ============================================================
# 3. CLEVR (object-centric spatial logic)
# ============================================================
def download_and_parse_clevr(local_path=None):
    """Download CLEVR questions/scenes (no images) for spatial reasoning."""
    print("\n=== 3. CLEVR ===")
    samples = []

    if local_path and Path(local_path).exists():
        print(f"  Using local path: {local_path}")
        samples = _parse_clevr_path(Path(local_path))
        if samples:
            print(f"  Total CLEVR (local): {len(samples)}")
            return samples

    clevr_dir = RAW_DIR / "clevr"
    clevr_dir.mkdir(parents=True, exist_ok=True)

    zip_path = clevr_dir / "CLEVR_v1.0_no_images.zip"
    if download(CLEVR_NO_IMAGES_URL, zip_path, "CLEVR no-images (~200 MB)"):
        try:
            print("  Extracting CLEVR...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(clevr_dir)
            samples = _parse_clevr_path(clevr_dir)
        except Exception as e:
            print(f"  [WARN] CLEVR extraction failed: {e}")

    if not samples:
        print("  [WARN] CLEVR download unavailable. Generating CLEVR-style synthetic data.")
        samples = _generate_clevr_synthetic(2000)

    print(f"  Total CLEVR: {len(samples)}")
    return samples


def _parse_clevr_path(clevr_dir):
    """Parse CLEVR scenes + questions into spatial reasoning samples."""
    samples = []
    clevr_dir = Path(clevr_dir)

    # Find scene and question files
    scene_files = sorted(clevr_dir.rglob("*scenes*.json"))
    question_files = sorted(clevr_dir.rglob("*questions*.json"))

    if not question_files:
        return samples

    # Load scenes indexed by image_index
    scenes_by_idx = {}
    for sf in scene_files:
        try:
            with open(sf) as f:
                data = json.load(f)
            for scene in data.get("scenes", []):
                idx = scene.get("image_index", scene.get("index"))
                if idx is not None:
                    scenes_by_idx[idx] = scene
        except Exception as e:
            print(f"  [WARN] Scene parse error {sf.name}: {e}")

    print(f"  Loaded {len(scenes_by_idx)} CLEVR scenes")

    # Spatial keywords for filtering questions
    spatial_kw = {"left", "right", "behind", "front", "above", "below",
                  "near", "far", "between", "beside", "same", "closer",
                  "farther", "next"}

    for qf in question_files:
        try:
            with open(qf) as f:
                data = json.load(f)

            questions = data.get("questions", [])
            count = 0
            for q in questions:
                if count >= 5000:
                    break

                scene_idx = q.get("image_index", -1)
                scene = scenes_by_idx.get(scene_idx)
                if not scene:
                    continue

                question_text = q.get("question", "")
                answer = str(q.get("answer", ""))

                # Filter for spatial questions
                q_words = set(question_text.lower().split())
                is_spatial = bool(q_words & spatial_kw)

                if not is_spatial:
                    continue

                context = _clevr_scene_to_text(scene)
                n_objs = len(scene.get("objects", []))

                samples.append({
                    "context": context,
                    "question": question_text,
                    "answer": answer,
                    "task_type": "object_location",
                    "difficulty": min(3, max(1, n_objs - 2)),
                    "source": "clevr",
                })
                count += 1

            print(f"  Parsed {count} spatial questions from {qf.name}")
        except Exception as e:
            print(f"  [WARN] Question parse error {qf.name}: {e}")

    return samples


def _clevr_scene_to_text(scene):
    """Convert a CLEVR scene to a text description."""
    parts = []
    objects = scene.get("objects", [])
    relationships = scene.get("relationships", {})

    for obj in objects:
        name = (f"{obj.get('size', '')} {obj.get('color', '')} "
                f"{obj.get('material', '')} {obj.get('shape', '')}").strip()
        coords = obj.get("3d_coords", [])
        if coords and len(coords) >= 2:
            parts.append(f"There is a {name} at ({coords[0]:.1f}, {coords[1]:.1f}).")
        else:
            parts.append(f"There is a {name}.")

    # Add explicit spatial relations from scene graph
    rel_names = {"behind": "behind", "front": "in front of",
                 "left": "to the left of", "right": "to the right of"}
    for rel_type, rel_lists in relationships.items():
        readable = rel_names.get(rel_type, rel_type)
        for i, related_indices in enumerate(rel_lists):
            if i >= len(objects):
                break
            for j in related_indices[:1]:  # Limit to avoid huge contexts
                if j < len(objects):
                    a_name = f"{objects[i].get('color', '')} {objects[i].get('shape', '')}".strip()
                    b_name = f"{objects[j].get('color', '')} {objects[j].get('shape', '')}".strip()
                    parts.append(f"The {a_name} is {readable} the {b_name}.")

    return " ".join(parts[:25])


def _generate_clevr_synthetic(n):
    """Generate CLEVR-style spatial reasoning data as fallback."""
    SHAPES = ["cube", "sphere", "cylinder", "cone", "pyramid"]
    COLORS = ["red", "blue", "green", "yellow", "gray", "brown", "cyan", "purple"]
    SIZES = ["small", "large"]
    MATERIALS = ["metal", "rubber", "glass"]

    samples = []
    for _ in range(n):
        n_objects = random.randint(3, 6)
        scene = []
        for _ in range(n_objects):
            scene.append({
                "shape": random.choice(SHAPES),
                "color": random.choice(COLORS),
                "size": random.choice(SIZES),
                "material": random.choice(MATERIALS),
                "x": round(random.uniform(-3, 3), 1),
                "y": round(random.uniform(-3, 3), 1),
            })

        def obj_name(o):
            return f"{o['size']} {o['color']} {o['material']} {o['shape']}"

        ctx_parts = []
        for o in scene:
            ctx_parts.append(f"There is a {obj_name(o)} at ({o['x']}, {o['y']}).")

        a, b = scene[0], scene[1]
        if a["x"] < b["x"]:
            rel = "left of"
        elif a["x"] > b["x"]:
            rel = "right of"
        elif a["y"] > b["y"]:
            rel = "above"
        else:
            rel = "below"
        ctx_parts.append(f"The {obj_name(a)} is {rel} the {obj_name(b)}.")

        q_type = random.choice(["relation", "count", "exist"])
        if q_type == "relation":
            question = (f"What is the spatial relation between the "
                        f"{obj_name(a)} and the {obj_name(b)}?")
            answer = rel
        elif q_type == "count":
            tc = random.choice(COLORS)
            count = sum(1 for o in scene if o["color"] == tc)
            question = f"How many {tc} objects are there?"
            answer = str(count)
        else:
            ts = random.choice(SHAPES)
            exists = any(o["shape"] == ts for o in scene)
            question = f"Is there a {ts} in the scene?"
            answer = "yes" if exists else "no"

        samples.append({
            "context": " ".join(ctx_parts),
            "question": question,
            "answer": answer,
            "task_type": "object_location",
            "difficulty": min(3, n_objects - 2),
            "source": "clevr_synthetic",
        })
    return samples


# ============================================================
# 4. GQA (real human spatial language)
# ============================================================
def download_and_parse_gqa(local_path=None):
    """Download GQA questions and filter for spatial reasoning subset."""
    print("\n=== 4. GQA ===")
    samples = []

    if local_path and Path(local_path).exists():
        print(f"  Using local path: {local_path}")
        samples = _parse_gqa_path(Path(local_path))
        if samples:
            print(f"  Total GQA (local): {len(samples)}")
            return samples

    gqa_dir = RAW_DIR / "gqa"
    gqa_dir.mkdir(parents=True, exist_ok=True)
    zip_path = gqa_dir / "questions1.2.zip"

    if download(GQA_QUESTIONS_URL, zip_path, "GQA questions (~1.4 GB)"):
        try:
            print("  Extracting GQA...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(gqa_dir)
            samples = _parse_gqa_path(gqa_dir)
        except Exception as e:
            print(f"  [WARN] GQA extraction failed: {e}")

    if not samples:
        print("  [WARN] GQA unavailable. Generating GQA-style synthetic data.")
        samples = _generate_gqa_synthetic(1500)

    print(f"  Total GQA: {len(samples)}")
    return samples


def _parse_gqa_path(gqa_dir):
    """Parse GQA JSON files, keeping only spatial questions."""
    samples = []
    gqa_dir = Path(gqa_dir)

    spatial_kw = {"left", "right", "above", "below", "behind", "front",
                  "near", "far", "between", "next", "on", "under", "inside",
                  "outside", "beside", "around", "across", "along", "through",
                  "top", "bottom", "side", "corner", "edge", "middle", "center"}

    for json_file in sorted(gqa_dir.rglob("*.json")):
        # Skip files > 2GB to avoid memory issues on Kaggle
        if json_file.stat().st_size > 2e9:
            print(f"  [skip] {json_file.name} too large ({json_file.stat().st_size / 1e9:.1f} GB)")
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)

            if not isinstance(data, dict):
                continue

            count = 0
            for qid, item in data.items():
                if not isinstance(item, dict):
                    continue
                if count >= 3000:
                    break

                question = item.get("question", "")
                answer = str(item.get("answer", ""))
                full_answer = item.get("fullAnswer", "")

                # Filter for spatial questions
                q_words = set(question.lower().split())
                if not q_words & spatial_kw:
                    continue

                context = full_answer if full_answer else "Question about a visual scene."

                samples.append({
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "task_type": "spatial_relation",
                    "difficulty": 2,
                    "source": "gqa",
                })
                count += 1

            if count > 0:
                print(f"  Parsed {count} spatial questions from {json_file.name}")
        except Exception as e:
            print(f"  [WARN] Failed to parse {json_file.name}: {e}")

    return samples


def _generate_gqa_synthetic(n):
    """Generate GQA-style spatial QA as fallback."""
    OBJECTS = ["man", "woman", "child", "dog", "cat", "car", "tree",
               "building", "table", "chair", "bench", "sign", "pole",
               "fence", "sidewalk", "window", "door", "wall", "floor"]
    RELATIONS = ["left of", "right of", "behind", "in front of", "on",
                 "under", "near", "next to", "above", "below"]

    samples = []
    for _ in range(n):
        a, b = random.sample(OBJECTS, 2)
        rel = random.choice(RELATIONS)
        context = f"In the image, a {a} is {rel} a {b}."

        q_type = random.choice(["where", "what", "yesno"])
        if q_type == "where":
            question = f"Where is the {a} relative to the {b}?"
            answer = rel
        elif q_type == "what":
            question = f"What is {rel} the {b}?"
            answer = a
        else:
            if random.random() > 0.5:
                question = f"Is the {a} {rel} the {b}?"
                answer = "yes"
            else:
                wrong_rel = random.choice([r for r in RELATIONS if r != rel])
                question = f"Is the {a} {wrong_rel} the {b}?"
                answer = "no"

        samples.append({
            "context": context,
            "question": question,
            "answer": answer,
            "task_type": "spatial_relation",
            "difficulty": random.choice([1, 2]),
            "source": "gqa_synthetic",
        })
    return samples


# ============================================================
# 5. NLVR2 (compositional spatial verification)
# ============================================================
def parse_nlvr2(local_path):
    """Parse NLVR2 from user-provided path.

    NLVR2 requires manual download and agreement:
      https://lil.nlp.cornell.edu/nlvr/
    """
    print("\n=== 5. NLVR2 ===")
    samples = []
    path = Path(local_path)

    if not path.exists():
        print(f"  [WARN] NLVR2 path not found: {path}")
        print("  NLVR2 requires manual download from: https://lil.nlp.cornell.edu/nlvr/")
        return samples

    spatial_kw = {"left", "right", "above", "below", "behind", "front",
                  "near", "far", "between", "top", "bottom", "side",
                  "touching", "edge", "corner", "middle", "center"}

    for json_file in sorted(path.rglob("*.json*")):
        try:
            with open(json_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    sentence = item.get("sentence", "")
                    label = item.get("label", "")

                    # Filter for spatial content
                    if not any(kw in sentence.lower() for kw in spatial_kw):
                        continue

                    samples.append({
                        "context": sentence,
                        "question": "Is this statement true or false?",
                        "answer": str(label).lower(),
                        "task_type": "spatial_relation",
                        "difficulty": 2,
                        "source": "nlvr2",
                    })
        except Exception as e:
            print(f"  [WARN] Failed to parse {json_file.name}: {e}")

    print(f"  Total NLVR2: {len(samples)}")
    return samples


# ============================================================
# 6. ScanQA (3D scene spatial QA)
# ============================================================
def parse_scanqa(local_path):
    """Parse ScanQA from user-provided path.

    ScanQA requires manual download:
      https://github.com/ATR-DBI/ScanQA
    """
    print("\n=== 6. ScanQA ===")
    samples = []
    path = Path(local_path)

    if not path.exists():
        print(f"  [WARN] ScanQA path not found: {path}")
        print("  ScanQA requires manual download from: https://github.com/ATR-DBI/ScanQA")
        return samples

    for json_file in sorted(path.rglob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)

            items = data if isinstance(data, list) else data.get("questions", data.get("data", []))
            for item in items:
                if not isinstance(item, dict):
                    continue
                question = item.get("question", "")
                answers = item.get("answers", [])
                answer = answers[0] if answers else item.get("answer", "")
                situation = item.get("situation", item.get("context", ""))

                if question and answer:
                    samples.append({
                        "context": str(situation),
                        "question": str(question),
                        "answer": str(answer),
                        "task_type": "navigation",
                        "difficulty": 3,
                        "source": "scanqa",
                    })
        except Exception as e:
            print(f"  [WARN] Failed to parse {json_file.name}: {e}")

    print(f"  Total ScanQA: {len(samples)}")
    return samples


# ============================================================
# Synthetic augmentation (fallback)
# ============================================================
def generate_augmented(n):
    """Generate diverse synthetic spatial reasoning samples."""
    OBJECTS = ["table", "chair", "lamp", "book", "cup", "box", "shelf",
               "plant", "clock", "mirror", "bed", "desk", "sofa", "vase",
               "rug", "door", "fridge", "stove", "sink", "counter"]
    RELATIONS = {
        "spatial_relation": [
            ("left of", "right of"), ("right of", "left of"),
            ("above", "below"), ("below", "above"),
            ("in front of", "behind"), ("behind", "in front of"),
            ("next to", "away from"), ("on top of", "under"),
        ],
        "object_location": [
            ("inside", "outside"), ("near", "far from"),
        ],
        "navigation": [
            ("north", "south"), ("south", "north"),
            ("east", "west"), ("west", "east"),
        ],
    }

    samples = []
    task_types = list(RELATIONS.keys())

    for _ in range(n):
        task = random.choice(task_types)
        objs = random.sample(OBJECTS, min(4, len(OBJECTS)))
        a, b = objs[0], objs[1]

        if task == "navigation":
            rel, inv = random.choice(RELATIONS[task])
            context = f"Go {rel} from the {a} to reach the {b}."
            question = f"What direction from the {a} to the {b}?"
            answer = rel
        else:
            rel, inv = random.choice(RELATIONS[task])
            context = f"The {a} is {rel} the {b}."
            question = f"Where is the {a} relative to the {b}?"
            answer = rel

        difficulty = random.choices([1, 2, 3], weights=[0.4, 0.35, 0.25])[0]
        if difficulty >= 2 and len(objs) > 2:
            r2, _ = random.choice(RELATIONS["spatial_relation"])
            context += f" The {objs[2]} is {r2} the {b}."
        if difficulty >= 3 and len(objs) > 3:
            r3, _ = random.choice(RELATIONS["spatial_relation"])
            context += f" The {objs[3]} is {r3} the {a}."

        samples.append({
            "context": context,
            "question": question,
            "answer": answer,
            "task_type": task,
            "difficulty": difficulty,
            "source": "augmented",
        })
    return samples


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare real spatial reasoning datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Datasets downloaded automatically:
  bAbI      -- clean symbolic reasoning (tasks 17 & 19)
  SpartQA   -- controlled synthetic spatial QA
  CLEVR     -- object-centric spatial logic (no images)
  GQA       -- real human spatial language

Datasets requiring manual download:
  NLVR2     -- https://lil.nlp.cornell.edu/nlvr/
  ScanQA    -- https://github.com/ATR-DBI/ScanQA
        """,
    )
    parser.add_argument("--spartqa-path", type=str, default=None,
                        help="Local path to SpartQA data directory")
    parser.add_argument("--clevr-path", type=str, default=None,
                        help="Local path to CLEVR data directory")
    parser.add_argument("--gqa-path", type=str, default=None,
                        help="Local path to GQA data directory")
    parser.add_argument("--nlvr2-path", type=str, default=None,
                        help="Local path to NLVR2 data (requires manual download)")
    parser.add_argument("--scanqa-path", type=str, default=None,
                        help="Local path to ScanQA data (requires manual download)")
    parser.add_argument("--synthetic-aug", type=int, default=0,
                        help="Number of synthetic augmentation samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # Auto-download datasets
    all_samples.extend(download_and_parse_babi())
    all_samples.extend(download_and_parse_spartqa(args.spartqa_path))
    all_samples.extend(download_and_parse_clevr(args.clevr_path))
    all_samples.extend(download_and_parse_gqa(args.gqa_path))

    # Manual-download datasets
    if args.nlvr2_path:
        all_samples.extend(parse_nlvr2(args.nlvr2_path))
    else:
        print("\n=== 5. NLVR2 (skipped -- provide --nlvr2-path) ===")

    if args.scanqa_path:
        all_samples.extend(parse_scanqa(args.scanqa_path))
    else:
        print("\n=== 6. ScanQA (skipped -- provide --scanqa-path) ===")

    # Synthetic augmentation
    if args.synthetic_aug > 0:
        print(f"\n=== Synthetic augmentation: {args.synthetic_aug} samples ===")
        all_samples.extend(generate_augmented(args.synthetic_aug))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Total samples collected: {len(all_samples)}")
    sources = Counter(s["source"] for s in all_samples)
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        pct = count / len(all_samples) * 100 if all_samples else 0
        print(f"  {source:<25} {count:>6} ({pct:>5.1f}%)")

    if not all_samples:
        print("[ERROR] No samples collected! Check network and paths.")
        sys.exit(1)

    # Split: 70% train, 15% val, 15% test
    random.shuffle(all_samples)
    n = len(all_samples)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    splits = {
        "real_train": all_samples[:train_end],
        "real_val": all_samples[train_end:val_end],
        "real_test": all_samples[val_end:],
    }

    print(f"\n{'=' * 60}")
    for name, data in splits.items():
        path = PROC_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        src_counts = Counter(s["source"] for s in data)
        print(f"Wrote {name}: {len(data)} samples -> {path}")
        for src, cnt in sorted(src_counts.items()):
            print(f"    {src}: {cnt}")

    print(f"\n{'=' * 60}")
    print("Done! Real datasets prepared in data/processed/")


if __name__ == "__main__":
    main()
