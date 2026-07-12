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
import math

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

from .fusion import MultiScaleSpatialFusion
from .llm_wrapper import _get_embed_layer
from .neuro.trajectory_cortex import TrajectoryCortex
from .neuro.theta_sweep import ThetaSweepSampler

logger = logging.getLogger(__name__)


class TrajectoryLLM(nn.Module):
    def __init__(
        self,
        base_llm: str = "Qwen/Qwen2.5-1.5B",
        cortex_dim: int = 128,
        cortex_task: str = "pathint",
        cortex_length_norm: bool = False,   # default: scale-free readout (generalizes across path lengths)
        cortex_constrained_velocity: bool = False,  # velocity-driven hex grid modules (faithful, metric)
        n_spatial_tokens: int = 8,
        fusion_num_heads: int = 8,
        gate_init: float = 2.0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: list[str] | None = None,
        lora_dropout: float = 0.05,
        use_theta_sweep: bool = False,       # add online theta-cycle look-ahead tokens (Vollan 2025)
        sweep_frac: float = 0.197,
        sweep_angle_deg: float = 25.0,
        sweep_steps: int = 8,
        n_sweep_cycles: int = 2,             # left + right theta cycles
        rsc_split: bool = False,             # RSC action/memory output pathways (Molecular Psychiatry 2024)
        perforant: bool = False,             # perforant semantic input pathway (Boccara 2019)
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
        # cortex_constrained_velocity=True swaps in velocity-driven hexagonal grid modules
        # (metrically accurate, length-invariant; the emergent grid-cell construction).
        self.cortex = TrajectoryCortex(embed_dim=cortex_dim, task=cortex_task,
                                       length_norm=cortex_length_norm,
                                       constrained_velocity=cortex_constrained_velocity)
        self.n_tokens = n_spatial_tokens
        self.to_tokens = nn.Linear(cortex_dim, llm_dim * n_spatial_tokens)
        # gate_init>0 opens the fusion gates from step 0 — the answer depends entirely
        # on the spatial channel, so a zero gate (SpatialLLM's anti-garbage default)
        # would give the LLM no spatial signal and no gradient to ever open the gates.
        self.fusion = MultiScaleSpatialFusion(
            hidden_dim=llm_dim, num_heads=fusion_num_heads, num_layers=2, gate_init=gate_init,
            rsc_split=rsc_split, perforant=perforant,
        )
        # Theta-cycle look-around (Vollan, Gardner, Moser & Moser, Nature 2025): each theta cycle the grid map
        # sweeps OUTWARD from the agent (alternating left/right, ~20% of module spacing). We turn those swept
        # grid codes into extra spatial tokens the LLM attends to — an active look-ahead interface, so the LLM
        # can reason about space AHEAD, not just where it stands. Requires the constrained-velocity hex grid
        # cortex (the integrator that exposes grid_code_at + module spacings).
        self.use_theta_sweep = bool(use_theta_sweep)
        if self.use_theta_sweep:
            if not cortex_constrained_velocity:
                raise ValueError("use_theta_sweep requires cortex_constrained_velocity=True (hex grid cortex)")
            self.theta_sweep = ThetaSweepSampler(sweep_frac=sweep_frac, angle_deg=sweep_angle_deg, steps=sweep_steps)
            self.n_sweep_cycles = n_sweep_cycles
            gm = self.cortex.integrator
            self.sweep_to_tokens = nn.Linear(gm.K * gm.M, llm_dim)   # one token per swept grid code
        self._embed_ref = []   # cache embed layer without registering it as a submodule

    def _embed(self) -> nn.Module:
        if not self._embed_ref:
            self._embed_ref.append(_get_embed_layer(self.llm.base_model))
        return self._embed_ref[0]

    @staticmethod
    def _current_pos_heading(heading, speed, vz):
        """Agent's current state from the move sequence: final position (path-integrated displacement from the
        origin) and final heading. (B,T) -> pos (B,2), heading (B,)."""
        pos = torch.stack([(speed * heading.cos()).sum(dim=1), (speed * heading.sin()).sum(dim=1)], dim=-1)
        return pos, heading[:, -1]

    def _sweep_tokens(self, heading, speed, vz, mode="real"):
        """Theta look-around tokens: from the agent's current position/heading, sweep the grid map ahead
        (alternating L/R cycles, ~sweep_frac of module spacing) and project each swept grid code to a token.
        mode 'real' = along the heading; 'shuffled' = along a wrong heading (control); 'ablated' = zeros."""
        gm = self.cortex.integrator
        pos, head = self._current_pos_heading(heading, speed, vz)
        B = pos.shape[0]; steps = self.theta_sweep.steps
        llm_dim = self.sweep_to_tokens.out_features
        if mode == "ablated":
            return torch.zeros(B, self.n_sweep_cycles * steps, llm_dim, device=pos.device, dtype=pos.dtype)
        use_head = head + math.pi if mode == "shuffled" else head          # wrong heading for the control
        length = self.theta_sweep.sweep_frac * self.theta_sweep.spacings(gm).mean()
        ks = torch.arange(1, steps + 1, device=pos.device, dtype=pos.dtype) / steps
        toks = []
        for cyc in range(self.n_sweep_cycles):
            side = -1.0 if cyc % 2 == 0 else 1.0
            direction = use_head + side * self.theta_sweep.angle           # (B,)
            d = torch.stack([direction.cos(), direction.sin()], -1)        # (B,2)
            swept = pos.unsqueeze(1) + ks.view(1, -1, 1) * length * d.unsqueeze(1)   # (B,steps,2)
            codes = gm.grid_code_at(swept.reshape(-1, 2)).view(B, steps, -1)         # (B,steps,K*M)
            toks.append(self.sweep_to_tokens(codes))                       # (B,steps,llm_dim)
        return torch.cat(toks, dim=1)                                      # (B, n_cycles*steps, llm_dim)

    def _spatial_tokens(self, heading, speed, vz, k=None, ablate=False, sweep_mode="real"):
        """Encode the move sequence into (B, n_tokens[, +sweep], llm_dim) spatial tokens.
        ablate=True zeros ALL spatial channels (cortex + sweep) — the control showing the LLM cannot answer
        the spatial question from the (question-only) text alone. With use_theta_sweep, theta look-ahead tokens
        are concatenated; sweep_mode in {real, shuffled, ablated} drives the sweep-specific ablation."""
        h = self.cortex.encode(heading, speed, vz, k=k)          # (B, cortex_dim)
        tok = self.to_tokens(h).view(h.shape[0], self.n_tokens, -1)
        if ablate:
            tok = torch.zeros_like(tok)
        if self.use_theta_sweep:
            sweep = self._sweep_tokens(heading, speed, vz, mode=("ablated" if ablate else sweep_mode))
            tok = torch.cat([tok, sweep], dim=1)
        return tok

    def forward(self, input_ids, attention_mask, heading, speed, vz,
                labels=None, k=None, ablate_cortex=False, sweep_mode="real", semantic_tokens=None):
        text = self._embed()(input_ids)                          # (B, T, D)
        spatial = self._spatial_tokens(heading, speed, vz, k, ablate=ablate_cortex, sweep_mode=sweep_mode)
        if spatial.dtype != text.dtype:
            spatial = spatial.to(text.dtype)
        if semantic_tokens is not None and semantic_tokens.dtype != text.dtype:
            semantic_tokens = semantic_tokens.to(text.dtype)
        fused = self.fusion(text, spatial, semantic_tokens=semantic_tokens)
        return self.llm(inputs_embeds=fused, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate_answer(self, input_ids, attention_mask, heading, speed, vz,
                        k=None, ablate_cortex=False, sweep_mode="real", max_new_tokens: int = 5):
        text = self._embed()(input_ids)
        spatial = self._spatial_tokens(heading, speed, vz, k, ablate=ablate_cortex, sweep_mode=sweep_mode)
        fused = self.fusion(text, spatial.to(text.dtype))
        return self.llm.generate(
            inputs_embeds=fused, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
        )
