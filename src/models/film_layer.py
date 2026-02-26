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
    """Mixture-of-experts FiLM: routes input to specialized expert FiLM layers.

    Neuroscience analog: multiple cortical columns providing parallel
    modulation, with a gating network selecting the most relevant expert
    for the current spatial context.
    """

    def __init__(self, d_model=512, d_condition=512, n_experts=4, dropout=0.1):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            FiLMLayer(d_model, d_condition, dropout) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(d_condition, n_experts)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x, condition):
        if condition.dim() == 2:
            gate_input = condition
        else:
            gate_input = condition.mean(dim=1)  # Pool over seq dim for gating

        weights = torch.softmax(self.gate(gate_input), dim=-1)  # (batch, n_experts)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            w = weights[:, i].unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
            out = out + w * expert(x, condition)
        return out
