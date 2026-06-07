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


def read_fusion_gates(model):
    """Per spatial-module fusion gate strengths (tanh of the learned gate), one
    entry per fusion layer. For a shared-gate model each entry is a single value;
    with per_module_gates it's {coord/elev, grid, memory, tile} (trimmed to the
    gates that exist) — a direct read-out of which module each task leaned on."""
    fusion = getattr(model, "fusion", None)
    if fusion is None or not hasattr(fusion, "layers"):
        return None
    names = ["coord/elev", "grid", "memory", "tile"]
    attn_by_layer = []
    for layer in fusion.layers:
        g = layer.attn_gate.detach().float().tanh().tolist()
        if len(g) > 1:
            attn_by_layer.append({names[i]: round(v, 4) for i, v in enumerate(g)})
        else:
            attn_by_layer.append({"shared": round(g[0], 4)})
    return {
        "per_module": bool(getattr(model, "per_module_gates", False)),
        "attn_gate_by_layer": attn_by_layer,
        "ffn_gate_by_layer": [
            round(l.ffn_gate.detach().float().tanh().item(), 4) for l in fusion.layers
        ],
    }


@torch.no_grad()
def evaluate_accuracy(config_path: str, checkpoint: str, val_path: str,
                      max_examples: int = None, dump_gates: bool = False,
                      results_json: str = None, seed: int = None,
                      label: str = None):
    from ..models.llm_wrapper import SpatialLLM
    from ..data.tokenizer import (
        SpatialTokenizer, SPATIAL_PROMPT_TEMPLATE, SPATIAL_PROMPT_TEMPLATE_NOCOORDS,
    )
    from ..utils.checkpoint import CheckpointManager

    cfg = yaml.safe_load(open(config_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    coords_in_text = cfg.get("data", {}).get("coords_in_text", True)

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
        per_module_gates=cfg["model"].get("per_module_gates", False),
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
        if coords_in_text:
            prompt = SPATIAL_PROMPT_TEMPLATE.format(
                lat=rec["lat"], lon=rec["lon"], question=rec["question"]
            )
        else:
            prompt = SPATIAL_PROMPT_TEMPLATE_NOCOORDS.format(question=rec["question"])
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
    # balanced accuracy = mean of per-class recalls (majority-guessing -> 0.5)
    recalls = [c / t for c, t in per_class.values() if t]
    balanced = sum(recalls) / len(recalls) if recalls else None
    gates = read_fusion_gates(model)

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
    if balanced is not None:
        print(f"  BALANCED accuracy: {balanced:.3f}  (0.5 = chance / majority-guessing)")

    if dump_gates and gates:
        print(f"  fusion gates (tanh, per layer | which module the task leaned on):")
        for i, lg in enumerate(gates["attn_gate_by_layer"]):
            print(f"    layer {i}: {lg}")

    result = {
        "config": config_path,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "label": label,
        "evaluated": total,
        "class_balance": dict(truth_balance),
        "overall_accuracy": round(overall, 4),
        "balanced_accuracy": round(balanced, 4) if balanced is not None else None,
        "per_class_recall": {
            cls: round(c / t, 4) for cls, (c, t) in per_class.items() if t
        },
        "unparseable": unparseable,
        "gates": gates,
    }

    if results_json:
        Path(results_json).parent.mkdir(parents=True, exist_ok=True)
        Path(results_json).write_text(json.dumps(result, indent=2))
        print(f"  wrote results → {results_json}")

    # Clearly delimited block so the JSON can be copy-pasted straight back to the
    # repo (the eval output otherwise dies in the ephemeral Kaggle container).
    print("\n===RESULT-JSON-START===")
    print(json.dumps(result))
    print("===RESULT-JSON-END===")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val", default="data/processed/val.jsonl")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--dump-gates", action="store_true",
                    help="print per-module fusion gate strengths (which module the task used)")
    ap.add_argument("--results-json", default=None,
                    help="also write the structured result dict to this path (tracked under results/)")
    ap.add_argument("--seed", type=int, default=None, help="record the training seed in the result")
    ap.add_argument("--label", default=None,
                    help="record a label (e.g. 'permod_seed42') in the result")
    args = ap.parse_args()
    evaluate_accuracy(args.config, args.checkpoint, args.val, args.max_examples,
                      dump_gates=args.dump_gates, results_json=args.results_json,
                      seed=args.seed, label=args.label)


if __name__ == "__main__":
    main()
