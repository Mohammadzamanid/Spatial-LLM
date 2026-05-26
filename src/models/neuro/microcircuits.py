"""
src/models/neuro/microcircuits.py

LOCAL CIRCUIT LEVEL — canonical cortical microcircuit computations.

  - DivisiveNormalization: the canonical cortical operation — each unit's
                           response is divided by the pooled activity of its
                           neighbours (Carandini & Heeger, 2012)
  - LateralInhibition:     surround suppression / competition between units
  - EIBalanceLayer:        separate excitatory & inhibitory populations with
                           Dale's law (a neuron is either E or I, not both)
  - CorticalColumn:        a 6-layer canonical microcircuit (L4→L2/3→L5/6)

References:
  Carandini & Heeger (2012) "Normalization as a canonical neural computation",
    Nature Reviews Neuroscience 13:51-62
  Douglas & Martin (2004) "Neuronal circuits of the neocortex"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DivisiveNormalization(nn.Module):
    """
    Divisive normalization — the canonical cortical computation.

        y_i = x_i^2 / (sigma^2 + sum_j x_j^2)

    Implements gain control, contrast normalization, and attention-like
    competition. Present everywhere from retina to cortex.
    """

    def __init__(self, dim: int, sigma: float = 1.0):
        super().__init__()
        self.sigma_sq = nn.Parameter(torch.tensor(float(sigma) ** 2))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sq = x ** 2
        pool = x_sq.sum(dim=-1, keepdim=True)               # (..., 1)
        return x_sq / (self.sigma_sq.abs() + pool + 1e-6)


class LateralInhibition(nn.Module):
    """
    Lateral (surround) inhibition.

    Each unit excites itself and inhibits its neighbours via a learned
    inhibitory kernel. Produces sharpening / contrast enhancement and
    soft winner-take-all competition.
    """

    def __init__(self, dim: int, inhibition_strength: float = 0.5):
        super().__init__()
        self.dim = dim
        # Inhibitory weight matrix: off-diagonal negative, diagonal positive
        w = -inhibition_strength * torch.ones(dim, dim) / dim
        w.fill_diagonal_(1.0)
        self.inhib = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.inhib)
        return F.relu(out)                                  # rectify (no negative rates)


class EIBalanceLayer(nn.Module):
    """
    Excitatory / Inhibitory balanced layer obeying DALE'S LAW.

    Biological constraint: a neuron releases either excitatory OR inhibitory
    transmitter — never both. We enforce this by keeping separate E and I
    populations with sign-constrained weights.

    ~80% excitatory, ~20% inhibitory (cortical ratio).
    """

    def __init__(self, in_dim: int, out_dim: int, inhib_ratio: float = 0.2):
        super().__init__()
        self.n_inhib = max(1, int(out_dim * inhib_ratio))
        self.n_excit = out_dim - self.n_inhib
        self.out_dim = out_dim

        self.exc = nn.Linear(in_dim, self.n_excit)
        self.inh = nn.Linear(in_dim, self.n_inhib)
        # Recurrent E↔I coupling
        self.e_to_i = nn.Linear(self.n_excit, self.n_inhib, bias=False)
        self.i_to_e = nn.Linear(self.n_inhib, self.n_excit, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sign-constrain to obey Dale's law
        e = F.relu(self.exc(x))                              # excitatory rates ≥ 0
        i = F.relu(self.inh(x))                              # inhibitory rates ≥ 0

        # E gets inhibited by I (subtractive); I gets driven by E (additive)
        e_balanced = F.relu(e - F.relu(self.i_to_e.weight.abs() @ i.T).T)
        i_balanced = F.relu(i + F.relu(self.e_to_i.weight.abs() @ e.T).T)

        return torch.cat([e_balanced, i_balanced], dim=-1)  # (B, out_dim)


class CorticalColumn(nn.Module):
    """
    Canonical 6-layer cortical microcircuit.

    Information flow mirrors neocortex:
        Input → L4 (input layer)
              → L2/3 (cortico-cortical processing + lateral inhibition)
              → L5/6 (output + feedback)

    Each layer uses divisive normalization (the canonical operation).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.l4 = nn.Linear(dim, dim)
        self.l23 = nn.Linear(dim, dim)
        self.l56 = nn.Linear(dim, dim)
        self.lateral = LateralInhibition(dim)
        self.norm = DivisiveNormalization(dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L4: thalamic input layer
        h4 = F.gelu(self.l4(x))
        # L2/3: intracortical processing with lateral competition
        h23 = self.lateral(F.gelu(self.l23(h4)))
        h23 = self.norm(h23)
        # L5/6: output layer with residual from L4
        h56 = F.gelu(self.l56(h23)) + h4
        return self.out_norm(h56)
