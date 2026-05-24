"""
src/eval/benchmark.py
Runs the full evaluation pipeline against a held-out spatial QA set.
Usage: python -m src.eval.benchmark --config configs/train_config.yaml --checkpoint outputs/run_001
"""

import argparse
import json
import logging
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..data.loader import SpatialQADataset
from ..data.tokenizer import SpatialTokenizer
from ..models.llm_wrapper import SpatialLLM
from .metrics import mean_haversine_error, exact_match

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_coords_from_answer(answer: str) -> tuple[float, float] | None:
    """
    Attempt to parse (lat, lon) from a model answer string.
    Expects format like '37.52, 45.07' or '37.52N 45.07E'.
    Returns None if parsing fails.
    """
    import re
    pattern = r"(-?\d+\.?\d*)[°NS,\s]+(-?\d+\.?\d*)"
    match = re.search(pattern, answer)
    if match:
        try:
            return float(match.group(1)), float(match.group(2))
        except ValueError:
            pass
    return None


def run_benchmark(config_path: str, checkpoint_path: str | None = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on {device}")

    spatial_tok = SpatialTokenizer(
        model_name=cfg["model"]["base_llm"],
        max_length=cfg["data"]["max_text_length"],
    )

    val_ds = SpatialQADataset(
        jsonl_path=cfg["data"]["val_path"],
        tile_dir=cfg["data"]["tile_dir"],
    )

    model = SpatialLLM(
        base_llm=cfg["model"]["base_llm"],
        vit_model_name=cfg["model"]["vit_backbone"],
    )

    if checkpoint_path:
        state = torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location=device)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    model = model.to(device).eval()

    predictions, references, pred_coords_list, true_coords_list = [], [], [], []

    with torch.no_grad():
        for item in tqdm(val_ds, desc="Evaluating"):
            encoded = spatial_tok.encode_spatial(
                question=item["question"],
                lat=item["coords"][0].item(),
                lon=item["coords"][1].item(),
            )
            input_ids = encoded["input_ids"].unsqueeze(0).to(device)
            attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)
            coords = item["coords"].unsqueeze(0).to(device)
            pixel_values = item.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.unsqueeze(0).to(device)

            gen_ids = model.generate_answer(input_ids, attention_mask, coords, pixel_values)
            pred_text = spatial_tok.decode(gen_ids[0])

            predictions.append(pred_text)
            references.append(item["answer"])

            parsed = parse_coords_from_answer(pred_text)
            if parsed:
                pred_coords_list.append(parsed)
                true_coords_list.append(
                    (item["coords"][0].item(), item["coords"][1].item())
                )

    em = exact_match(predictions, references)
    logger.info(f"Exact Match: {em:.4f}")

    if pred_coords_list:
        geo_metrics = mean_haversine_error(pred_coords_list, true_coords_list)
        logger.info(f"Geo metrics: {json.dumps(geo_metrics, indent=2)}")
    else:
        logger.warning("No coordinate predictions parsed from output.")

    return {"exact_match": em, "geo_metrics": geo_metrics if pred_coords_list else {}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    run_benchmark(args.config, args.checkpoint)
