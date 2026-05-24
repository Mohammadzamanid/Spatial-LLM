"""
src/models/neuromodulation.py

Neuromodulation — inspired by dopamine, acetylcholine, and norepinephrine.

Biological basis:
  The brain doesn't just pass signals forward — neuromodulators globally
  regulate how those signals are processed:
  - Dopamine: gates learning based on prediction error / reward signal
  - Acetylcholine: controls the balance between top-down vs bottom-up
  - Norepinephrine: modulates signal gain (contrast / uncertainty)

Implementation:
  A lightweight context-conditioned gating network that:
  1. Takes a "context signal" (e.g. prediction error magnitude, task embedding)
  2. Outputs gain + bias modulation vectors
  3. Applies them to any hidden representation

  This allows the model to dynamically amplify or suppress spatial features
  based on how surprising/relevant the current location is — exactly what
  the biological neuromodulatory system does.

  SpatialNeuromodulator: wraps any layer with modulated gain
  AdaptiveGain: norepinephrine-style contrast control
  PredictionErrorGate: dopamine-style learning gate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpatialNeuromodulator(nn.Module):
    """
    Context-conditioned gain modulation.
    Modulates a representation based on an external context signal.

    context_signal could be:
      - prediction error magnitude (surprise)
      - task type embedding
      - spatial novelty score
    """

    def __init__(self, hidden_dim: int, context_dim: Optional[int] = None):
        super().__init__()
        context_dim = context_dim or hidden_dim

        self.gain_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Sigmoid(),                          # gain in [0, 1]
        )
        self.bias_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Tanh(),                             # bias in [-1, 1]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,             # (B, T, D) or (B, D)
        context: torch.Tensor,        # (B, context_dim)
    ) -> torch.Tensor:
        gain = self.gain_net(context)   # (B, D)
        bias = self.bias_net(context)   # (B, D)

        if x.dim() == 3:
            gain = gain.unsqueeze(1)    # (B, 1, D) — broadcast over T
            bias = bias.unsqueeze(1)

        return self.norm(x * gain + bias)


class AdaptiveGain(nn.Module):
    """
    Norepinephrine-style adaptive gain control.
    Scales the overall magnitude of activations based on uncertainty/novelty.

    High uncertainty → high gain (attend more carefully)
    Low uncertainty  → low gain (rely on prior knowledge)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),    # always positive
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            modulated: (B, ..., D) gain-adjusted representation
            uncertainty: (B, 1) estimated uncertainty score
        """
        if x.dim() == 3:
            pooled = x.mean(dim=1)  # (B, D)
        else:
            pooled = x

        uncertainty = self.uncertainty_estimator(pooled)  # (B, 1)
        gain = 1.0 + uncertainty                          # (B, 1) — boost on uncertainty

        if x.dim() == 3:
            gain = gain.unsqueeze(1)  # (B, 1, 1)

        return self.norm(x * gain), uncertainty.squeeze(-1)


class PredictionErrorGate(nn.Module):
    """
    Dopamine-inspired prediction error gate.
    Controls how much of the spatial representation flows into the LLM
    based on the magnitude of the spatial prediction error.

    High prediction error (novelty) → more spatial information flows through
    Low prediction error (familiar)  → spatial signal is damped
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        spatial_repr: torch.Tensor,    # (B, D)
        prediction_error: torch.Tensor,  # (B,) scalar error per sample
    ) -> torch.Tensor:
        """Returns gated spatial representation: (B, D)."""
        err = prediction_error.unsqueeze(-1)  # (B, 1)
        gate_input = torch.cat([spatial_repr, err], dim=-1)
        gate = self.gate(gate_input)           # (B, D)
        return self.norm(spatial_repr * gate)
