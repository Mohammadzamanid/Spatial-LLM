"""
src/models/spatial_encoder.py
ViT-based encoder for map/satellite tile images.
Projects patch embeddings into the LLM's hidden dimension space.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class SpatialTileEncoder(nn.Module):
    """
    Encodes a map tile image (224x224) using a pretrained ViT.
    Output is projected to match the LLM's hidden dimension so it can
    be fed into the cross-attention fusion layer.

    Returns all patch tokens (including CLS) for cross-attention.
    """

    def __init__(
        self,
        vit_model_name: str = "google/vit-base-patch16-224",
        llm_hidden_dim: int = 4096,
        freeze_vit: bool = False,
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name)
        vit_hidden = self.vit.config.hidden_size  # 768 for ViT-base

        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(vit_hidden, llm_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(llm_hidden_dim),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224) normalized image tensor
        Returns:
            patch_tokens: (B, num_patches + 1, llm_hidden_dim)
                          num_patches = (224/16)^2 = 196, +1 for CLS
        """
        outputs = self.vit(pixel_values=pixel_values)
        patch_tokens = outputs.last_hidden_state  # (B, 197, 768)
        return self.proj(patch_tokens)            # (B, 197, llm_hidden_dim)

    @property
    def num_patch_tokens(self) -> int:
        img_size = self.vit.config.image_size
        patch_size = self.vit.config.patch_size
        return (img_size // patch_size) ** 2 + 1  # +1 for CLS
