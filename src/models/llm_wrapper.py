"""
src/models/llm_wrapper.py
Full Spatial-LLM model: LLM backbone + LoRA + spatial fusion.
Combines CoordinateEmbedder, SpatialTileEncoder, and SpatialFusionLayer.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from .coord_embedder import CoordinateEmbedderWithTokens
from .spatial_encoder import SpatialTileEncoder
from .fusion import MultiScaleSpatialFusion


class SpatialLLM(nn.Module):
    """
    End-to-end Spatial-LLM.

    Inputs:
        input_ids:      (B, T)       — tokenized text
        attention_mask: (B, T)       — text attention mask
        coords:         (B, 2)       — [lat, lon] in degrees
        pixel_values:   (B, 3, H, W) — map tile image (optional)
        labels:         (B, T)       — for causal LM loss (optional)

    Output:
        CausalLMOutputWithPast (includes loss if labels provided)
    """

    def __init__(
        self,
        base_llm: str = "mistralai/Mistral-7B-v0.1",
        vit_model_name: str = "google/vit-base-patch16-224",
        coord_embed_dim: int = 256,
        coord_num_freqs: int = 64,
        fusion_num_heads: int = 8,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: list[str] | None = None,
        lora_dropout: float = 0.05,
        freeze_vit: bool = False,
    ):
        super().__init__()

        # ── LLM backbone with LoRA ──────────────────────────────────────
        llm = AutoModelForCausalLM.from_pretrained(base_llm, torch_dtype=torch.float16)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules or ["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.llm = get_peft_model(llm, lora_cfg)
        llm_hidden_dim = llm.config.hidden_size

        # ── Spatial components ──────────────────────────────────────────
        self.coord_embedder = CoordinateEmbedderWithTokens(
            embed_dim=llm_hidden_dim,
            num_freqs=coord_num_freqs,
            num_tokens=4,
        )
        self.tile_encoder = SpatialTileEncoder(
            vit_model_name=vit_model_name,
            llm_hidden_dim=llm_hidden_dim,
            freeze_vit=freeze_vit,
        )
        self.fusion = MultiScaleSpatialFusion(
            hidden_dim=llm_hidden_dim,
            num_heads=fusion_num_heads,
            num_layers=2,
        )

        # Project coord embedding dim to match llm_hidden_dim if needed
        if coord_embed_dim != llm_hidden_dim:
            self.coord_proj = nn.Linear(coord_embed_dim, llm_hidden_dim)
        else:
            self.coord_proj = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        coords: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        # ── 1. Encode spatial context ───────────────────────────────────
        coord_tokens = self.coord_embedder(coords)           # (B, 4, D)

        if pixel_values is not None:
            tile_tokens = self.tile_encoder(pixel_values)    # (B, 197, D)
            spatial_tokens = torch.cat([coord_tokens, tile_tokens], dim=1)
        else:
            spatial_tokens = coord_tokens                    # (B, 4, D)

        # ── 2. LLM embedding layer ──────────────────────────────────────
        # Access the base model's embedding layer
        embed_layer = self.llm.base_model.model.model.embed_tokens
        text_embeds = embed_layer(input_ids)                 # (B, T, D)

        # ── 3. Fuse spatial into text embeddings ────────────────────────
        fused_embeds = self.fusion(text_embeds, spatial_tokens)  # (B, T, D)

        # ── 4. Forward through LLM with fused embeddings ────────────────
        outputs = self.llm(
            inputs_embeds=fused_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def generate_answer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        coords: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        max_new_tokens: int = 128,
    ) -> torch.Tensor:
        """Autoregressive generation with spatial context."""
        embed_layer = self.llm.base_model.model.model.embed_tokens
        text_embeds = embed_layer(input_ids)
        coord_tokens = self.coord_embedder(coords)

        if pixel_values is not None:
            tile_tokens = self.tile_encoder(pixel_values)
            spatial_tokens = torch.cat([coord_tokens, tile_tokens], dim=1)
        else:
            spatial_tokens = coord_tokens

        fused_embeds = self.fusion(text_embeds, spatial_tokens)

        return self.llm.generate(
            inputs_embeds=fused_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
