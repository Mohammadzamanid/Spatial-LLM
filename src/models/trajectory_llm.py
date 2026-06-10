"""
src/models/trajectory_llm.py

Milestone 2 — answer trajectory questions in NATURAL LANGUAGE.

The agent's path (a sequence of moves: heading, speed, vertical velocity) is encoded
by the recurrent TrajectoryCortex into a compact spatial summary, projected to a few
"spatial tokens", and fused (gated cross-attention) into a LoRA-adapted LLM. The text
prompt contains ONLY the question ("Are you back where you started?") — the moves reach
the model solely through the cortex channel, so the LLM must use the cortex's path
integration to answer.

Mirrors the proven SpatialLLM fusion path; the only change is the spatial-token source
(TrajectoryCortex over a move sequence instead of a single coordinate).
"""
import logging

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

from .fusion import MultiScaleSpatialFusion
from .llm_wrapper import _get_embed_layer
from .neuro.trajectory_cortex import TrajectoryCortex

logger = logging.getLogger(__name__)


class TrajectoryLLM(nn.Module):
    def __init__(
        self,
        base_llm: str = "Qwen/Qwen2.5-1.5B",
        cortex_dim: int = 128,
        cortex_task: str = "pathint",
        cortex_length_norm: bool = True,
        n_spatial_tokens: int = 8,
        fusion_num_heads: int = 8,
        gate_init: float = 2.0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: list[str] | None = None,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        logger.info(f"Loading base LLM: {base_llm}")
        try:                                                  # newer transformers
            llm_base = AutoModelForCausalLM.from_pretrained(base_llm, dtype=torch.float32)
        except TypeError:                                     # older transformers
            llm_base = AutoModelForCausalLM.from_pretrained(base_llm, torch_dtype=torch.float32)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha,
            target_modules=lora_target_modules or
            ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=lora_dropout, bias="none",
        )
        self.llm = get_peft_model(llm_base, lora_cfg)
        self.llm.print_trainable_parameters()
        llm_dim = llm_base.config.hidden_size

        # Spatial pathway: recurrent cortex over the move sequence -> spatial tokens.
        # cortex_length_norm=False (scale-free readout) + mixed-length pre-training is
        # what lets the cortex generalize to path lengths it never trained on
        # (see src/eval/generalize_trajectory.py and the FINDINGS stress-test).
        self.cortex = TrajectoryCortex(embed_dim=cortex_dim, task=cortex_task,
                                       length_norm=cortex_length_norm)
        self.n_tokens = n_spatial_tokens
        self.to_tokens = nn.Linear(cortex_dim, llm_dim * n_spatial_tokens)
        # gate_init>0 opens the fusion gates from step 0 — the answer depends entirely
        # on the spatial channel, so a zero gate (SpatialLLM's anti-garbage default)
        # would give the LLM no spatial signal and no gradient to ever open the gates.
        self.fusion = MultiScaleSpatialFusion(
            hidden_dim=llm_dim, num_heads=fusion_num_heads, num_layers=2, gate_init=gate_init
        )
        self._embed_ref = []   # cache embed layer without registering it as a submodule

    def _embed(self) -> nn.Module:
        if not self._embed_ref:
            self._embed_ref.append(_get_embed_layer(self.llm.base_model))
        return self._embed_ref[0]

    def _spatial_tokens(self, heading, speed, vz, k=None, ablate=False):
        """Encode the move sequence into (B, n_tokens, llm_dim) spatial tokens.
        ablate=True zeros the cortex output — the control showing the LLM cannot
        answer the spatial question from the (question-only) text alone."""
        h = self.cortex.encode(heading, speed, vz, k=k)          # (B, cortex_dim)
        tok = self.to_tokens(h).view(h.shape[0], self.n_tokens, -1)
        if ablate:
            tok = torch.zeros_like(tok)
        return tok

    def forward(self, input_ids, attention_mask, heading, speed, vz,
                labels=None, k=None, ablate_cortex=False):
        text = self._embed()(input_ids)                          # (B, T, D)
        spatial = self._spatial_tokens(heading, speed, vz, k, ablate=ablate_cortex)
        if spatial.dtype != text.dtype:
            spatial = spatial.to(text.dtype)
        fused = self.fusion(text, spatial)
        return self.llm(inputs_embeds=fused, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate_answer(self, input_ids, attention_mask, heading, speed, vz,
                        k=None, ablate_cortex=False, max_new_tokens: int = 5):
        text = self._embed()(input_ids)
        spatial = self._spatial_tokens(heading, speed, vz, k, ablate=ablate_cortex)
        fused = self.fusion(text, spatial.to(text.dtype))
        return self.llm.generate(
            inputs_embeds=fused, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
        )
