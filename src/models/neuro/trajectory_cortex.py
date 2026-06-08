"""
src/models/neuro/trajectory_cortex.py

Recurrent, ablatable spatial cortex for 4D navigation (x, y, z over time t).

Unlike the static ConfigurableCortex (which sees one frozen coordinate), this
cortex integrates a SEQUENCE of moves — heading, speed, vertical velocity at each
timestep — to recover the final position. This is the task the navigation modules
were built for:

  - conjunctive cells  → bind head-direction x speed into a per-step velocity code
  - grid attractor     → recurrent continuous-attractor PATH INTEGRATION over time
  - theta-gamma        → ordered sequence memory of the recent path (synchronization)
  - cortical column /  → canonical microcircuit post-processing
    lateral inhibition

(Boundary and phase-precession cells are deliberately omitted here — they need a
bounded arena / within-field signal, i.e. a different task — see FINDINGS.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_cells import ConjunctiveSpatialCells
from .oscillations import ThetaGammaCoupling
from .microcircuits import CorticalColumn, LateralInhibition


TRAJ_DEFAULT_CONFIG = {
    "conjunctive": True,        # head-direction x speed -> per-step velocity code
    "grid_attractor": True,     # recurrent path integrator over the move sequence
    "theta_gamma": True,        # theta-gamma ordered sequence memory (synchronization)
    "cortical_column": True,
    "lateral_inhibition": True,
}


class _AttractorIntegrator(nn.Module):
    """Recurrent continuous-attractor path integrator. Accumulates per-step velocity
    embeddings into an activity bump on a toroidal sheet (Mexican-hat recurrence) and
    reads out the integrated position. This is the grid-cell module doing its actual
    job — integrating velocity over time, not encoding a static coordinate."""

    def __init__(self, embed_dim: int, grid_size: int = 16, settle: int = 2):
        super().__init__()
        self.N = grid_size * grid_size
        self.settle = settle
        self.vel_to_sheet = nn.Linear(embed_dim, self.N)
        g = grid_size
        cells = torch.stack(torch.meshgrid(
            torch.arange(g), torch.arange(g), indexing="ij"), dim=-1).reshape(-1, 2).float()
        d = cells.unsqueeze(0) - cells.unsqueeze(1)
        d = torch.minimum(d.abs(), g - d.abs())          # toroidal distance
        dist_sq = (d ** 2).sum(-1)
        self.register_buffer("W", torch.exp(-dist_sq / 8.0) - 0.6 * torch.exp(-dist_sq / 72.0))
        self.readout = nn.Linear(self.N, embed_dim)

    def forward(self, vel_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = vel_seq.shape
        u = torch.zeros(B, self.N, device=vel_seq.device, dtype=vel_seq.dtype)
        for t in range(T):
            # signed velocity drive ACCUMULATES (this is the integration — no
            # per-step renormalisation, which would wipe the accumulated distance)
            u = u + self.vel_to_sheet(vel_seq[:, t])
            for _ in range(self.settle):                          # gentle attractor coupling
                u = u + 0.1 * F.linear(torch.tanh(u), self.W)
        return self.readout(u / T)


class TrajectoryCortex(nn.Module):
    """Ablatable recurrent cortex. Inputs are per-timestep movement signals
    heading/speed/vz of shape (B, T); output is the predicted final (x, y, z)."""

    def __init__(self, embed_dim: int = 64, config: dict | None = None,
                 aux_heads: bool = False, dims: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.dims = dims
        self.cfg = {**TRAJ_DEFAULT_CONFIG, **(config or {})}
        self.aux_heads = aux_heads

        if self.cfg["conjunctive"]:
            self.conjunctive = ConjunctiveSpatialCells(embed_dim=embed_dim)
            self.vert = nn.Linear(1, embed_dim)               # vertical-velocity channel (z)
        if self.cfg["grid_attractor"]:
            self.integrator = _AttractorIntegrator(embed_dim)
        else:
            self.pool_proj = nn.Linear(embed_dim, embed_dim)  # fallback: order-agnostic pooling
        if self.cfg["theta_gamma"]:
            self.theta_gamma = ThetaGammaCoupling(embed_dim, num_slots=7)

        self.column = CorticalColumn(dim=embed_dim) if self.cfg["cortical_column"] else None
        self.lateral = LateralInhibition(dim=embed_dim) if self.cfg["lateral_inhibition"] else None
        self.out_norm = nn.LayerNorm(embed_dim)
        self.readout = nn.Linear(embed_dim, dims)

        if aux_heads:
            names = [n for n in ("conjunctive", "grid_attractor", "theta_gamma") if self.cfg[n]]
            self.aux = nn.ModuleDict({n: nn.Linear(embed_dim, dims) for n in names})

    def forward(self, heading: torch.Tensor, speed: torch.Tensor, vz: torch.Tensor,
                return_aux: bool = False):
        B, T = heading.shape
        device = heading.device
        step = torch.zeros(B, T, self.embed_dim, device=device)

        ve = None
        if self.cfg["conjunctive"]:
            ve = (self.conjunctive(heading.reshape(B * T), speed.reshape(B * T)).view(B, T, -1)
                  + self.vert(vz.reshape(B * T, 1)).view(B, T, -1))
            step = step + ve

        if self.cfg["grid_attractor"]:
            integ = self.integrator(step)                     # recurrent path integration
        else:
            integ = self.pool_proj(step.mean(dim=1))          # order-agnostic fallback
        h = integ

        tg = None
        if self.cfg["theta_gamma"]:
            tg = self.theta_gamma(step)
            h = h + tg

        if self.column is not None:
            h = self.column(h) + h
        if self.lateral is not None:
            h = self.lateral(h) + h
        h = self.out_norm(h)
        pred = self.readout(h)

        if return_aux and self.aux_heads:
            aux_out = {}
            if ve is not None and "conjunctive" in self.aux:
                aux_out["conjunctive"] = self.aux["conjunctive"](ve.mean(dim=1))
            if "grid_attractor" in self.aux:
                aux_out["grid_attractor"] = self.aux["grid_attractor"](integ)
            if tg is not None and "theta_gamma" in self.aux:
                aux_out["theta_gamma"] = self.aux["theta_gamma"](tg)
            return pred, aux_out
        return pred
