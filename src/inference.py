"""
src/inference.py
Production inference script for Spatial-LLM.
Usage:
    python -m src.inference \
        --checkpoint outputs/best \
        --config configs/train_config.yaml \
        --lat 35.6895 --lon 139.6917 \
        --question "What type of urban area is this?"
"""
import argparse
import logging

import torch
import yaml
from PIL import Image
from torchvision import transforms

from .data.tile_fetcher import fetch_tile
from .data.tokenizer import SpatialTokenizer
from .models.llm_wrapper import SpatialLLM
from .utils.checkpoint import CheckpointManager
from .utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

TILE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SpatialLLMInference:
    """
    Production inference wrapper.
    Handles: device placement, tile fetching, tokenisation, generation, cleanup.
    """

    def __init__(self, config_path: str, checkpoint_path: str, device: str = "auto"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading model on {self.device}")
        self.model = SpatialLLM(
            base_llm=self.cfg["model"]["base_llm"],
            vit_model_name=self.cfg["model"]["vit_backbone"],
            coord_embed_dim=self.cfg["model"]["coord_embed_dim"],
            coord_num_freqs=self.cfg["model"]["coord_num_freqs"],
            fusion_num_heads=self.cfg["model"]["fusion_num_heads"],
        )

        ckpt_mgr = CheckpointManager(checkpoint_path)
        self.model = ckpt_mgr.load_best(self.model, device=self.device)
        self.model = self.model.to(self.device).eval()

        self.tokenizer = SpatialTokenizer(
            model_name=self.cfg["model"]["base_llm"],
            max_length=self.cfg["data"]["max_text_length"],
        )

    @torch.no_grad()
    def predict(
        self,
        question: str,
        lat: float,
        lon: float,
        tile_path: str | None = None,
        fetch_tile_auto: bool = True,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Run inference for a single spatial question.
        Args:
            question: natural language question
            lat, lon: location in degrees
            tile_path: optional pre-fetched tile image path
            fetch_tile_auto: if True, auto-fetch tile from OSM
        Returns:
            answer string
        """
        # Optionally fetch tile
        pixel_values = None
        if tile_path is None and fetch_tile_auto:
            tile_path = fetch_tile(lat, lon, zoom=self.cfg["data"]["zoom_level"])

        if tile_path:
            try:
                img = Image.open(tile_path).convert("RGB")
                pixel_values = TILE_TRANSFORM(img).unsqueeze(0).to(self.device)
            except Exception as e:
                logger.warning(f"Could not load tile {tile_path}: {e}")

        # Tokenize
        encoded = self.tokenizer.encode_spatial(question=question, lat=lat, lon=lon)
        input_ids = encoded["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(self.device)
        coords = torch.tensor([[lat, lon]], dtype=torch.float32).to(self.device)

        # Generate
        gen_ids = self.model.generate_answer(
            input_ids, attention_mask, coords, pixel_values, max_new_tokens
        )
        return self.tokenizer.decode(gen_ids[0])


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Spatial-LLM Inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    engine = SpatialLLMInference(args.config, args.checkpoint, args.device)
    answer = engine.predict(args.question, args.lat, args.lon,
                            max_new_tokens=args.max_new_tokens)
    print(f"\n📍 ({args.lat}, {args.lon})")
    print(f"❓ {args.question}")
    print(f"💬 {answer}")


if __name__ == "__main__":
    main()
