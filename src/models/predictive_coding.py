"""
src/models/predictive_coding.py

Predictive Coding — inspired by the neocortex.

Biological basis:
  The brain doesn't passively receive input — it constantly predicts it.
  Higher cortical areas send top-down predictions; lower areas compute
  prediction errors and send those upward. Learning minimises prediction
  error at every level (Rao & Ballard, 1999; Karl Friston's Free Energy).

Implementation:
  A hierarchical stack where each level:
    1. Receives bottom-up input (sensory/spatial features)
    2. Generates a top-down prediction of the level below
    3. Computes a prediction error
    4. Passes the error upward

  The prediction error is an auxiliary training signal that forces the
  model to build internally consistent spatial representations.
  During inference, it acts as a self-consistency regulariser.

  Applied to Spatial-LLM:
    Level 0: raw coordinate / tile features
    Level 1: fused spatial representation
    Level 2: LLM hidden states
    Error flows: L2→L1→L0 (top-down predictions)
    Signals flow: L0→L1→L2 (bottom-up errors)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PredictiveCodingLevel(nn.Module):
    """Single level in the predictive coding hierarchy."""

    def __init__(self, dim: int, pred_dim: Optional[int] = None):
        super().__init__()
        pred_dim = pred_dim or dim

        # Bottom-up: encodes incoming signal
        self.bu_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        # Top-down: predicts the level below from this level's state
        self.td_predictor = nn.Sequential(
            nn.Linear(dim, pred_dim),
            nn.GELU(),
            nn.LayerNorm(pred_dim),
        )

        # Error unit: computes mismatch between prediction and actual
        self.error_gate = nn.Sequential(
            nn.Linear(pred_dim * 2, pred_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        top_down_pred: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, dim) bottom-up input
            top_down_pred: (B, dim) prediction from level above (or None for top level)
        Returns:
            state: (B, dim) this level's representation
            prediction_error: (B, dim) error signal passed upward
        """
        state = self.bu_encoder(x)

        if top_down_pred is not None:
            # Prediction error = gated difference
            concat = torch.cat([state, top_down_pred], dim=-1)
            gate = self.error_gate(concat)
            error = gate * (state - top_down_pred)
        else:
            error = state  # Top level: no prediction to compare against

        # Generate prediction for the level below
        td_pred = self.td_predictor(state)

        return state, error, td_pred


class SpatialPredictiveCoding(nn.Module):
    """
    3-level predictive coding hierarchy for spatial features.

    Levels:
      0 — raw spatial features (coord + tile)
      1 — fused representation
      2 — LLM-aligned abstract representation

    The total prediction error is added as an auxiliary loss during training.
    At inference, the final level's state is used for fusion with the LLM.
    """

    def __init__(self, spatial_dim: int, llm_dim: int, num_levels: int = 3):
        super().__init__()
        dims = [spatial_dim] + [
            int(spatial_dim * (llm_dim / spatial_dim) ** (i / (num_levels - 1)))
            for i in range(1, num_levels)
        ]
        # Ensure last dim matches llm_dim
        dims[-1] = llm_dim

        self.levels = nn.ModuleList([
            PredictiveCodingLevel(
                dim=dims[i],
                pred_dim=dims[i - 1] if i > 0 else dims[0],
            )
            for i in range(num_levels)
        ])

        # Projections between levels
        self.level_projs = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1])
            for i in range(num_levels - 1)
        ])

    def forward(
        self,
        spatial_features: torch.Tensor,  # (B, spatial_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            top_state: (B, llm_dim) — feed this to the LLM fusion layer
            total_pc_loss: scalar — auxiliary predictive coding loss
        """
        # Bottom-up pass
        states, errors, td_preds = [], [], []
        x = spatial_features
        for i, level in enumerate(self.levels):
            state, error, td_pred = level(x, top_down_pred=None)
            states.append(state)
            errors.append(error)
            td_preds.append(td_pred)
            if i < len(self.level_projs):
                x = self.level_projs[i](state)

        # Top-down pass: recompute errors with actual top-down predictions
        total_error = torch.tensor(0.0, device=spatial_features.device)
        for i in range(len(self.levels) - 2, -1, -1):
            # td_preds[i+1] is level i+1's prediction of level i
            pred = td_preds[i + 1]
            if pred.shape[-1] != states[i].shape[-1]:
                # Align dims if needed
                pred = F.adaptive_avg_pool1d(
                    pred.unsqueeze(1), states[i].shape[-1]
                ).squeeze(1)
            pc_error = F.mse_loss(pred, states[i].detach())
            total_error = total_error + pc_error

        return states[-1], total_error
