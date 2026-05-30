"""src/utils/checkpoint.py — Model checkpoint manager."""
import json
import logging
import shutil
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Saves and loads model checkpoints, tracking best metric."""

    def __init__(self, output_dir: str, metric: str = "haversine_km", mode: str = "min"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.history: list[dict] = []

    def is_better(self, value: float) -> bool:
        return value < self.best_value if self.mode == "min" else value > self.best_value

    def save(self, model: torch.nn.Module, step: int, metrics: dict) -> Path:
        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(exist_ok=True)

        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        with open(ckpt_dir / "metrics.json", "w") as f:
            json.dump({"step": step, **metrics}, f, indent=2)

        self.history.append({"step": step, "dir": str(ckpt_dir), **metrics})

        if self.metric in metrics and self.is_better(metrics[self.metric]):
            self.best_value = metrics[self.metric]
            best_dir = self.output_dir / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(ckpt_dir, best_dir)
            logger.info(f"New best {self.metric}: {self.best_value:.4f} → saved to {best_dir}")

        logger.info(f"Checkpoint saved: {ckpt_dir}")
        return ckpt_dir

    @staticmethod
    def _load_state_any_format(directory: Path, device: str) -> dict:
        """Load a state dict from a checkpoint dir regardless of save format.
        Handles HF Trainer output (pytorch_model.bin / model.safetensors) and
        the legacy CheckpointManager format (model.pt)."""
        candidates = ["model.pt", "pytorch_model.bin", "model.safetensors"]
        for name in candidates:
            path = directory / name
            if path.exists():
                if name.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    return load_file(str(path), device=device)
                return torch.load(path, map_location=device)
        raise FileNotFoundError(
            f"No checkpoint file ({' / '.join(candidates)}) found in {directory}"
        )

    def load_best(self, model: torch.nn.Module, device: str = "cpu") -> torch.nn.Module:
        best_dir = self.output_dir / "best"
        # Fall back to the output dir itself (HF save_model writes here directly)
        target = best_dir if best_dir.exists() else self.output_dir
        state = self._load_state_any_format(target, device)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded checkpoint from {target}")
        return model

    def load(self, model: torch.nn.Module, checkpoint_path: str, device: str = "cpu"):
        state = self._load_state_any_format(Path(checkpoint_path), device)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return model
