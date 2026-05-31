"""
src/eval/accuracy.py

Generate the model's actual yes/no answer for each held-out city and compute
accuracy — the metric that reflects whether the model uses elevation, unlike
eval_loss which is dominated by the repetitive question boilerplate.

Reports overall accuracy, per-class accuracy (so majority-class guessing can't
hide), and the class balance for context.

Usage:
    python -m src.eval.accuracy --config configs/coord_3d.yaml \
        --checkpoint outputs/coord_3d --val data/processed/val.jsonl
"""
import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _norm_yesno(text: str):
    """Map free text to 'yes'/'no'/None by first occurrence."""
    t = text.strip().lower()
    yi = t.find("yes")
    ni = t.find("no")
    if yi == -1 and ni == -1:
        return None
    if yi == -1:
        return "no"
    if ni == -1:
        return "yes"
    return "yes" if yi < ni else "no"


@torch.no_grad()
def evaluate_accuracy(config_path: str, checkpoint: str, val_path: str,
                      max_examples: int = None):
    from ..models.llm_wrapper import SpatialLLM
    from ..data.tokenizer import SpatialTokenizer, SPATIAL_PROMPT_TEMPLATE
    from ..utils.checkpoint import CheckpointManager

    cfg = yaml.safe_load(open(config_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SpatialLLM(
        base_llm=cfg["model"]["base_llm"],
        vit_model_name=cfg["model"]["vit_backbone"],
        coord_embed_dim=cfg["model"]["coord_embed_dim"],
        coord_num_freqs=cfg["model"]["coord_num_freqs"],
        coord_input_dim=cfg["model"].get("coord_input_dim", 2),
        fusion_num_heads=cfg["model"]["fusion_num_heads"],
        lora_r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        load_in_4bit=cfg["model"].get("load_in_4bit", False),
        use_place_memory=cfg["model"].get("use_place_memory", True),
        use_predictive_coding=cfg["model"].get("use_predictive_coding", True),
        use_neuromodulation=cfg["model"].get("use_neuromodulation", True),
    )
    CheckpointManager(cfg["training"]["output_dir"]).load(model, checkpoint, device=device)
    model.to(device).eval()
    tok = SpatialTokenizer(
        model_name=cfg["model"]["base_llm"],
        max_length=cfg["data"]["max_text_length"],
    )

    records = [json.loads(l) for l in open(val_path, encoding="utf-8")]
    if max_examples:
        records = records[:max_examples]

    correct = 0
    per_class = defaultdict(lambda: [0, 0])  # truth -> [correct, total]
    truth_balance = Counter()
    skipped = 0
    unparseable = 0
    sample_gens = []

    for rec in records:
        truth = _norm_yesno(rec["answer"])
        if truth is None:
            skipped += 1
            continue
        truth_balance[truth] += 1

        # Tokenize the prompt WITHOUT padding (right-padding to 512 would make the
        # model generate from a pad token). Elevation is NOT in the text — it only
        # reaches the model via the coordinate channel, so this fairly tests whether
        # the 3D coord pathway conveys elevation.
        prompt = SPATIAL_PROMPT_TEMPLATE.format(
            lat=rec["lat"], lon=rec["lon"], question=rec["question"]
        )
        enc = tok.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        coords = torch.tensor(
            [[rec["lat"], rec["lon"], rec.get("elevation", 0.0)]],
            dtype=torch.float32, device=device,
        )

        out_ids = model.generate_answer(
            input_ids=input_ids, attention_mask=attention_mask,
            coords=coords, max_new_tokens=5,
        )
        gen = tok.decode(out_ids[0])
        pred = _norm_yesno(gen)

        # capture the first few raw generations so a parsing/generation failure
        # surfaces as visible output instead of a fake 0.000 accuracy
        if len(sample_gens) < 8:
            sample_gens.append((truth, repr(gen)[:60], pred))

        per_class[truth][1] += 1
        if pred is None:
            unparseable += 1
        elif pred == truth:
            correct += 1
            per_class[truth][0] += 1

    total = sum(t for _, t in per_class.values())
    overall = correct / total if total else 0.0

    print(f"\n=== Accuracy: {Path(checkpoint).name} ===")
    print(f"  sample generations (truth | raw output | parsed):")
    for tr, raw, pr in sample_gens:
        print(f"    {tr:3s} | {raw:60s} | {pr}")
    print(f"  evaluated:        {total} (truth unparseable {skipped})")
    print(f"  model outputs that were NOT yes/no: {unparseable}/{total}")
    if unparseable == total and total:
        print("  ⚠️  EVERY generation failed to parse as yes/no — this is a")
        print("      generation problem, not a real 0.0 accuracy. See samples above.")
    print(f"  class balance:    {dict(truth_balance)}")
    print(f"  OVERALL accuracy: {overall:.3f}")
    for cls in ("yes", "no"):
        c, t = per_class[cls]
        if t:
            print(f"    {cls:3s}: {c}/{t} = {c/t:.3f}")
    # balanced accuracy = mean of per-class recalls (majority-guessing -> 0.5)
    recalls = [c / t for c, t in per_class.values() if t]
    if recalls:
        print(f"  BALANCED accuracy: {sum(recalls)/len(recalls):.3f}  "
              f"(0.5 = chance / majority-guessing)")
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val", default="data/processed/val.jsonl")
    ap.add_argument("--max_examples", type=int, default=None)
    args = ap.parse_args()
    evaluate_accuracy(args.config, args.checkpoint, args.val, args.max_examples)


if __name__ == "__main__":
    main()
