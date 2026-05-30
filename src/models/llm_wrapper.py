"""
src/models/llm_wrapper.py

Full Spatial-LLM — integrates all neuroscience-inspired components:
  1. GridCellEncoder        — entorhinal cortex multi-scale hex encoding
  2. HippocampalMemory      — place cell + episodic spatial memory
  3. SpatialTileEncoder     — ViT visual encoder for map tiles
  4. SpatialPredictiveCoding— hierarchical prediction error
  5. SpatialNeuromodulator  — context-conditioned gain control
  6. PredictionErrorGate    — dopamine-style novelty gating
  7. MultiScaleSpatialFusion— cross-attention fusion into LLM
  8. LoRA fine-tuned LLM    — Mistral / LLaMA backbone
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from .coord_embedder import CoordinateEmbedderWithTokens
from .fusion import MultiScaleSpatialFusion
from .grid_cell_encoder import GridCellEncoderWithTokens
from .neuromodulation import AdaptiveGain, PredictionErrorGate, SpatialNeuromodulator
from .place_cell_memory import HippocampalMemory
from .predictive_coding import SpatialPredictiveCoding
from .spatial_encoder import SpatialTileEncoder

logger = logging.getLogger(__name__)


def _get_embed_layer(model: nn.Module) -> nn.Embedding:
    """
    Robustly retrieve the token embedding layer across model families.
    Supports Mistral, LLaMA, Falcon, GPT-NeoX, and generic fallback.
    """
    # Canonical HF method first — works for every architecture including Qwen
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb
    except (AttributeError, NotImplementedError):
        pass

    # Try common attribute paths
    for attr_path in [
        "model.embed_tokens",           # Mistral, LLaMA, Qwen2.5
        "transformer.wte",              # GPT-2, Falcon
        "gpt_neox.embed_in",            # GPT-NeoX
        "model.decoder.embed_tokens",   # OPT
    ]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            logger.debug(f"Found embed layer at base_model.{attr_path}")
            return obj
        except AttributeError:
            continue

    # Fallback: search recursively
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and "embed" in name.lower():
            logger.debug(f"Found embed layer via search: {name}")
            return module

    raise RuntimeError(
        "Could not find token embedding layer. "
        "Please set model.embed_layer_path in configs."
    )


class SpatialLLM(nn.Module):
    """
    Neuroscience-inspired Spatial LLM.

    Spatial processing pipeline per forward pass:
      coords → [GridCells + Fourier] → PredictiveCoding
           → [HippocampalMemory retrieval]
           → AdaptiveGain
           → PredictionErrorGate
           → [+ ViT tile tokens if image provided]
           → SpatialNeuromodulator
           → CrossAttentionFusion with LLM hidden states
           → LoRA LLM → output
    """

    def __init__(
        self,
        base_llm: str = "mistralai/Mistral-7B-v0.1",
        vit_model_name: str = "google/vit-base-patch16-224",
        coord_embed_dim: int = 256,
        coord_num_freqs: int = 64,
        grid_num_modules: int = 6,
        num_place_cells: int = 512,
        fusion_num_heads: int = 8,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: Optional[list[str]] = None,
        lora_dropout: float = 0.05,
        freeze_vit: bool = False,
        use_place_memory: bool = True,
        use_predictive_coding: bool = True,
        use_neuromodulation: bool = True,
        load_in_4bit: bool = False,
    ):
        super().__init__()
        self.use_place_memory = use_place_memory
        self.use_predictive_coding = use_predictive_coding
        self.use_neuromodulation = use_neuromodulation

        # ── LLM backbone + LoRA ────────────────────────────────────────
        logger.info(f"Loading base LLM: {base_llm}")
        if load_in_4bit:
            logger.info("Using 4-bit quantization (QLoRA) — fits a T4 GPU")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            llm_base = AutoModelForCausalLM.from_pretrained(
                base_llm,
                quantization_config=bnb_config,
                device_map="auto",
            )
            llm_base = prepare_model_for_kbit_training(llm_base)
        else:
            llm_base = AutoModelForCausalLM.from_pretrained(
                base_llm, torch_dtype=torch.float16
            )
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules or ["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.llm = get_peft_model(llm_base, lora_cfg)
        self.llm.print_trainable_parameters()
        llm_dim = llm_base.config.hidden_size

        # ── Spatial encoders ───────────────────────────────────────────
        # 1. Fourier coordinate embedder (baseline)
        self.coord_embedder = CoordinateEmbedderWithTokens(
            embed_dim=llm_dim, num_freqs=coord_num_freqs, num_tokens=4
        )

        # 2. Grid cell encoder (entorhinal cortex)
        self.grid_encoder = GridCellEncoderWithTokens(
            embed_dim=llm_dim, num_modules=grid_num_modules
        )

        # 3. ViT tile encoder
        self.tile_encoder = SpatialTileEncoder(
            vit_model_name=vit_model_name,
            llm_hidden_dim=llm_dim,
            freeze_vit=freeze_vit,
        )

        # ── Neuroscience modules ───────────────────────────────────────
        # 4. Hippocampal place cell memory
        if use_place_memory:
            self.hippocampus = HippocampalMemory(
                embed_dim=llm_dim, num_cells=num_place_cells
            )

        # 5. Predictive coding hierarchy
        if use_predictive_coding:
            self.pred_coding = SpatialPredictiveCoding(
                spatial_dim=llm_dim, llm_dim=llm_dim
            )

        # 6. Adaptive gain (norepinephrine)
        if use_neuromodulation:
            self.adaptive_gain = AdaptiveGain(llm_dim)
            self.neuromod = SpatialNeuromodulator(llm_dim)
            self.pred_error_gate = PredictionErrorGate(llm_dim)

        # ── Fusion ────────────────────────────────────────────────────
        self.fusion = MultiScaleSpatialFusion(
            hidden_dim=llm_dim, num_heads=fusion_num_heads, num_layers=2
        )

        # Cache embed layer reference
        self._embed_layer = None

    def _get_embed(self) -> nn.Module:
        if self._embed_layer is None:
            self._embed_layer = _get_embed_layer(self.llm.base_model)
        return self._embed_layer

    def _encode_spatial(
        self,
        coords: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build spatial token sequence + compute PC loss.
        Returns:
            spatial_tokens: (B, S, D)
            pc_loss: scalar
        """
        pc_loss = torch.tensor(0.0, device=coords.device)

        # Fourier + grid cell tokens
        fourier_tokens = self.coord_embedder(coords)   # (B, 4, D)
        grid_tokens = self.grid_encoder(coords)         # (B, num_modules, D)

        # Pool to single vector for scalar modules
        spatial_vec = (fourier_tokens.mean(1) + grid_tokens.mean(1)) / 2  # (B, D)

        # Predictive coding
        if self.use_predictive_coding:
            spatial_vec, pc_loss = self.pred_coding(spatial_vec)

        # Hippocampal memory
        if self.use_place_memory:
            mem_contrib = self.hippocampus(coords, context=spatial_vec, store=True)
            spatial_vec = spatial_vec + mem_contrib

        # Neuromodulation
        if self.use_neuromodulation:
            spatial_vec, uncertainty = self.adaptive_gain(spatial_vec)
            spatial_vec = self.pred_error_gate(spatial_vec, pc_loss.expand(coords.shape[0]))
            spatial_vec = self.neuromod(spatial_vec, spatial_vec)

        # Assemble token sequence
        modulated_tokens = spatial_vec.unsqueeze(1)            # (B, 1, D)
        token_seq = torch.cat([fourier_tokens, grid_tokens, modulated_tokens], dim=1)

        if pixel_values is not None:
            tile_tokens = self.tile_encoder(pixel_values)      # (B, 197, D)
            token_seq = torch.cat([token_seq, tile_tokens], dim=1)

        return token_seq, pc_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        coords: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        spatial_tokens, pc_loss = self._encode_spatial(coords, pixel_values)

        embed_layer = self._get_embed()
        text_embeds = embed_layer(input_ids)                    # (B, T, D)
        fused = self.fusion(text_embeds, spatial_tokens)        # (B, T, D)

        outputs = self.llm(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Inject PC loss into total loss
        if labels is not None and self.use_predictive_coding:
            outputs.loss = outputs.loss + 0.1 * pc_loss

        return outputs

    @torch.no_grad()
    def generate_answer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        coords: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
    ) -> torch.Tensor:
        spatial_tokens, _ = self._encode_spatial(coords, pixel_values)
        embed_layer = self._get_embed()
        text_embeds = embed_layer(input_ids)
        fused = self.fusion(text_embeds, spatial_tokens)

        return self.llm.generate(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
