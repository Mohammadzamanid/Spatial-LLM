"""
src/models/neuro/trajectory_cortex.py

Recurrent, ablatable spatial cortex for 4D navigation (x, y, z over time t).

Two tasks (the cortex supports both):
  - "pathint": integrate a sequence of moves -> final (x,y,z). Order-INDEPENDENT
               (a commutative sum), so it needs only velocity encoding + integration.
  - "recall":  "where were you at step k?" -> position at a queried timestep. This is
               order/ history-DEPENDENT: a sum cannot answer it. The model must keep a
               running per-step position (recurrent integration) and retrieve the k-th,
               so the recurrent integrator becomes load-bearing.

Modules (head-direction x speed = conjunctive; grid-attractor path integrator;
theta-gamma sequence memory; cortical microcircuits) can each be toggled (ablation)
and, with gated=True, each OPTIONAL module gets a learned gate so the network turns
modules it doesn't need DOWN on its own (task-dependent complexity).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_cells import ConjunctiveSpatialCells
from .oscillations import ThetaGammaCoupling
from .microcircuits import CorticalColumn, LateralInhibition


TRAJ_DEFAULT_CONFIG = {
    "conjunctive": True,        # head-direction x speed -> per-step velocity code (structural)
    "grid_attractor": True,     # recurrent path integrator over the move sequence (structural)
    "theta_gamma": True,        # theta-gamma ordered sequence memory (gateable add-on)
    "cortical_column": True,    # canonical microcircuit (gateable add-on)
    "lateral_inhibition": True, # surround inhibition (gateable add-on)
}
# Modules whose usefulness is task-dependent get a learned gate; the two structural
# ones (velocity encoder + integrator) are the I/O path and are not gated.
GATEABLE = ("theta_gamma", "cortical_column", "lateral_inhibition")


class _AttractorIntegrator(nn.Module):
    """Recurrent continuous-attractor path integrator. Accumulates per-step velocity
    embeddings into a bump on a toroidal sheet and reads out the integrated position.
    With return_sequence=True it returns the running position at EVERY step (needed for
    recall); otherwise just the final position (path integration)."""

    def __init__(self, embed_dim: int, grid_size: int = 16, settle: int = 2):
        super().__init__()
        self.N = grid_size * grid_size
        self.settle = settle
        self.vel_to_sheet = nn.Linear(embed_dim, self.N)
        g = grid_size
        cells = torch.stack(torch.meshgrid(
            torch.arange(g), torch.arange(g), indexing="ij"), dim=-1).reshape(-1, 2).float()
        d = cells.unsqueeze(0) - cells.unsqueeze(1)
        d = torch.minimum(d.abs(), g - d.abs())
        dist_sq = (d ** 2).sum(-1)
        self.register_buffer("W", torch.exp(-dist_sq / 8.0) - 0.6 * torch.exp(-dist_sq / 72.0))
        self.readout = nn.Linear(self.N, embed_dim)

    def forward(self, vel_seq: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        B, T, _ = vel_seq.shape
        u = torch.zeros(B, self.N, device=vel_seq.device, dtype=vel_seq.dtype)
        outs = []
        for t in range(T):
            u = u + self.vel_to_sheet(vel_seq[:, t])          # signed accumulate = integrate
            for _ in range(self.settle):
                u = u + 0.1 * F.linear(torch.tanh(u), self.W)  # gentle attractor coupling
            if return_sequence:
                outs.append(self.readout(u / T))               # running position at step t
        if return_sequence:
            return torch.stack(outs, dim=1)                    # (B, T, D)
        return self.readout(u / T)                             # final position (B, D)


class TrajectoryCortex(nn.Module):
    def __init__(self, embed_dim: int = 64, config: dict | None = None,
                 aux_heads: bool = False, dims: int = 3,
                 task: str = "pathint", gated: bool = False, max_T: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.dims = dims
        self.task = task
        self.gated = gated
        self.cfg = {**TRAJ_DEFAULT_CONFIG, **(config or {})}
        self.aux_heads = aux_heads

        if self.cfg["conjunctive"]:
            self.conjunctive = ConjunctiveSpatialCells(embed_dim=embed_dim)
            self.vert = nn.Linear(1, embed_dim)
        if self.cfg["grid_attractor"]:
            self.integrator = _AttractorIntegrator(embed_dim)
        else:
            self.pool_proj = nn.Linear(embed_dim, embed_dim)
        if self.cfg["theta_gamma"]:
            self.theta_gamma = ThetaGammaCoupling(embed_dim, num_slots=7)
        self.column = CorticalColumn(dim=embed_dim) if self.cfg["cortical_column"] else None
        self.lateral = LateralInhibition(dim=embed_dim) if self.cfg["lateral_inhibition"] else None

        if task == "recall":
            self.step_key = nn.Embedding(max_T, embed_dim)     # shared key/query per timestep

        self.out_norm = nn.LayerNorm(embed_dim)
        self.readout = nn.Linear(embed_dim, dims)

        # learned gates on the optional modules (sigmoid; init ~open at sigmoid(2)=0.88)
        self.gate_names = [m for m in GATEABLE if self.cfg[m]]
        if gated and self.gate_names:
            self.gates = nn.Parameter(torch.full((len(self.gate_names),), 2.0))

        if aux_heads:
            names = [n for n in ("conjunctive", "grid_attractor", "theta_gamma") if self.cfg[n]]
            self.aux = nn.ModuleDict({n: nn.Linear(embed_dim, dims) for n in names})

    def _g(self, name: str):
        if not self.gated or name not in self.gate_names:
            return 1.0
        return torch.sigmoid(self.gates[self.gate_names.index(name)])

    def gate_values(self) -> dict:
        if not self.gated or not self.gate_names:
            return {}
        return {n: round(torch.sigmoid(self.gates[i]).item(), 3) for i, n in enumerate(self.gate_names)}

    def gate_l1(self):
        if not self.gated or not self.gate_names:
            return torch.tensor(0.0)
        return torch.sigmoid(self.gates).sum()

    def forward(self, heading, speed, vz, k=None, return_aux=False):
        B, T = heading.shape
        device = heading.device
        step = torch.zeros(B, T, self.embed_dim, device=device)
        if self.cfg["conjunctive"]:
            step = step + (self.conjunctive(heading.reshape(B * T), speed.reshape(B * T)).view(B, T, -1)
                           + self.vert(vz.reshape(B * T, 1)).view(B, T, -1))

        if self.task == "recall":
            # need per-step RUNNING positions, then retrieve the queried step k
            if self.cfg["grid_attractor"]:
                states = self.integrator(step, return_sequence=True)   # (B,T,D) running positions
            else:
                states = step                                          # no integration -> velocities
            idx = torch.arange(T, device=device)
            # tied key/query: step k's query IS its own key, so attention is self-similar
            # and peaks at t=k from the start (then refines).
            attn = torch.softmax(self.step_key(k) @ self.step_key(idx).t()
                                 / math.sqrt(self.embed_dim), dim=-1)   # (B,T) peaks at step k
            position = (attn.unsqueeze(-1) * states).sum(dim=1)         # (B,D) retrieved state
        else:  # pathint: final position only
            if self.cfg["grid_attractor"]:
                position = self.integrator(step)
            else:
                position = self.pool_proj(step.mean(dim=1))

        h = position
        tg = None
        if self.cfg["theta_gamma"]:
            tg = self.theta_gamma(step)
            h = h + self._g("theta_gamma") * tg
        if self.column is not None:
            h = h + self._g("cortical_column") * self.column(h)
        if self.lateral is not None:
            h = h + self._g("lateral_inhibition") * self.lateral(h)
        h = self.out_norm(h)
        pred = self.readout(h)

        if return_aux and self.aux_heads:
            aux_out = {}
            if "conjunctive" in self.aux:
                aux_out["conjunctive"] = self.aux["conjunctive"](step.mean(dim=1))
            if "grid_attractor" in self.aux:
                aux_out["grid_attractor"] = self.aux["grid_attractor"](position)
            if tg is not None and "theta_gamma" in self.aux:
                aux_out["theta_gamma"] = self.aux["theta_gamma"](tg)
            return pred, aux_out
        return pred
