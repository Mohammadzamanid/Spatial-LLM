"""FiLM (Feature-wise Linear Modulation) Layer (Numerically Stable)."""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Applies feature-wise affine transformation conditioned on spatial context."""

    def __init__(self, d_model=512, d_condition=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.modulation_net = nn.Linear(d_condition, d_model * 2)
        nn.init.xavier_uniform_(self.modulation_net.weight, gain=0.01)
        nn.init.zeros_(self.modulation_net.bias)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, condition):
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)

        mod_params = self.modulation_net(condition)
        gamma, beta = torch.chunk(mod_params, 2, dim=-1)

        # Keep gamma near 1, beta near 0 for stability
        gamma = torch.sigmoid(gamma) * 0.2 + 0.9  # [0.9, 1.1]
        beta = torch.tanh(beta) * 0.1              # [-0.1, 0.1]

        modulated = gamma * x + beta
        modulated = self.layer_norm(modulated)
        modulated = torch.nan_to_num(modulated, nan=0.0)

        return modulated


class AdaptiveFiLMLayer(nn.Module):
    """Adaptive FiLM (delegates to FiLMLayer for stability)."""

    def __init__(self, d_model=512, d_condition=512, n_experts=4, dropout=0.1):
        super().__init__()
        self.film = FiLMLayer(d_model, d_condition, dropout)

    def forward(self, x, condition):
        return self.film(x, condition)


class ContextualFiLMLayer(nn.Module):
    """Contextual FiLM (delegates to FiLMLayer for stability)."""

    def __init__(self, d_model=512, d_condition=512, dropout=0.1):
        super().__init__()
        self.film = FiLMLayer(d_model, d_condition, dropout)

    def forward(self, x, condition):
        return self.film(x, condition)
