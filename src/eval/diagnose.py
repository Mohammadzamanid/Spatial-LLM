"""
src/eval/diagnose.py — Pinpoint why a checkpoint generates garbage.

For one config+checkpoint, prints diagnostics on the spatial pathway:
- did the coord embedder weights actually load (vs stay random)?
- are spatial tokens / fused embeddings finite and reasonably scaled?
- what magnitude does the spatial soft-prompt add relative to text embeddings?

Usage:
    python -m src.eval.diagnose --config configs/coord_3d.yaml --checkpoint outputs/coord_3d
"""
import argparse, json
import torch, yaml


def diagnose(config_path, checkpoint, val_path="data/processed/val.jsonl"):
    from ..models.llm_wrapper import SpatialLLM
    from ..data.tokenizer import SpatialTokenizer, SPATIAL_PROMPT_TEMPLATE
    from ..utils.checkpoint import CheckpointManager

    cfg = yaml.safe_load(open(config_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def build():
        return SpatialLLM(
            base_llm=cfg["model"]["base_llm"], vit_model_name=cfg["model"]["vit_backbone"],
            coord_embed_dim=cfg["model"]["coord_embed_dim"],
            coord_num_freqs=cfg["model"]["coord_num_freqs"],
            coord_input_dim=cfg["model"].get("coord_input_dim", 2),
            fusion_num_heads=cfg["model"]["fusion_num_heads"],
            lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["lora_alpha"],
            lora_target_modules=cfg["lora"]["target_modules"], lora_dropout=cfg["lora"]["lora_dropout"],
            load_in_4bit=cfg["model"].get("load_in_4bit", False),
            use_place_memory=cfg["model"].get("use_place_memory", True),
            use_predictive_coding=cfg["model"].get("use_predictive_coding", True),
            use_neuromodulation=cfg["model"].get("use_neuromodulation", True),
            per_module_gates=cfg["model"].get("per_module_gates", False),
        )

    # Build twice: once random (pre-load), once loaded, to see if coord weights changed
    model = build().to(device).eval()
    coord_w_before = model.coord_embedder.proj[0].weight.detach().clone()
    CheckpointManager(cfg["training"]["output_dir"]).load(model, checkpoint, device=device)
    coord_w_after = model.coord_embedder.proj[0].weight.detach()
    changed = (coord_w_before - coord_w_after).abs().mean().item()
    print(f"\n[1] coord_embedder weights changed on load by mean|Δ| = {changed:.6f}")
    print(f"    -> if ~0, the coord embedder did NOT load (stayed random) = the bug")

    tok = SpatialTokenizer(model_name=cfg["model"]["base_llm"],
                           max_length=cfg["data"]["max_text_length"])
    rec = json.loads(open(val_path).readline())
    coords = torch.tensor([[rec["lat"], rec["lon"], rec.get("elevation", 0.0)]],
                          dtype=torch.float32, device=device)

    with torch.no_grad():
        spatial_tokens, _, group_sizes = model._encode_spatial(coords, None)
        prompt = SPATIAL_PROMPT_TEMPLATE.format(lat=rec["lat"], lon=rec["lon"], question=rec["question"])
        enc = tok.tokenizer(prompt, return_tensors="pt").to(device)
        text_embeds = model._get_embed()(enc["input_ids"])
        fused = model.fusion(
            text_embeds, spatial_tokens.to(text_embeds.dtype),
            group_sizes=group_sizes if model.per_module_gates else None,
        )

    def stats(name, t):
        print(f"    {name:16s} shape={tuple(t.shape)} "
              f"mean|x|={t.abs().mean():.4f} max|x|={t.abs().max():.4f} "
              f"NaN={torch.isnan(t).any().item()} Inf={torch.isinf(t).any().item()}")

    print(f"\n[2] magnitudes (text vs spatial soft-prompt vs fused):")
    stats("text_embeds", text_embeds)
    stats("spatial_tokens", spatial_tokens)
    stats("fused", fused)
    ratio = spatial_tokens.abs().mean().item() / max(text_embeds.abs().mean().item(), 1e-9)
    print(f"    spatial/text magnitude ratio = {ratio:.2f}")
    print(f"    -> if >> 1, the spatial soft-prompt overpowers the text = garbage generation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val", default="data/processed/val.jsonl")
    args = ap.parse_args()
    diagnose(args.config, args.checkpoint, args.val)


if __name__ == "__main__":
    main()
