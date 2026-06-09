"""
src/models/neuro/trajectory_cortex.py

Recurrent, ablatable spatial cortex for 4D navigation (x, y, z over time t).

Three tasks:
  - "pathint":   integrate moves -> FINAL (x,y,z). Commutative sum: only needs the
                 velocity encoder + integration.
  - "recall":    "where were you at step k?" with FULL-sequence attention. Order-
                 dependent, so the recurrent integrator (running position) is essential.
  - "memrecall": same query, but through a fixed-size MEMORY BOTTLENECK — the trajectory
                 is multiplexed into ONE vector and the answer is read back from that.
                 This needs an ORDER-preserving memory (theta-gamma, ~7 slots); a mean-
                 pool bottleneck collapses order and fails. Makes the whole stack
                 (velocity encoder + integrator + theta-gamma memory) load-bearing.

With gated=True each OPTIONAL module gets a learned gate (+L1) so the network turns
down modules it doesn't need (task-dependent complexity).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_cells import ConjunctiveSpatialCells
from .oscillations import ThetaGammaCoupling, ThetaGammaMemory
from .microcircuits import CorticalColumn, LateralInhibition


TRAJ_DEFAULT_CONFIG = {
    "conjunctive": True,        # head-direction x speed -> per-step velocity code (structural)
    "grid_attractor": True,     # recurrent path integrator over the move sequence (structural)
    "theta_gamma": True,        # theta-gamma ordered sequence memory
    "cortical_column": True,    # canonical microcircuit (gateable add-on)
    "lateral_inhibition": True, # surround inhibition (gateable add-on)
}
GATEABLE = ("theta_gamma", "cortical_column", "lateral_inhibition")


class _AttractorIntegrator(nn.Module):
    """Recurrent continuous-attractor path integrator. With return_sequence=True it
    returns the running position at every step (needed for recall/memrecall)."""

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
            u = u + self.vel_to_sheet(vel_seq[:, t])
            for _ in range(self.settle):
                u = u + 0.1 * F.linear(torch.tanh(u), self.W)
            if return_sequence:
                outs.append(self.readout(u / T))
        if return_sequence:
            return torch.stack(outs, dim=1)
        return self.readout(u / T)


class TrajectoryCortex(nn.Module):
    def __init__(self, embed_dim: int = 64, config: dict | None = None,
                 aux_heads: bool = False, dims: int = 3,
                 task: str = "pathint", gated: bool = False, max_T: int = 64,
                 mem_slots: int = 8):
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

        # theta-gamma plays two roles depending on task:
        #  - pathint/recall: optional additive sequence-summary (ThetaGammaCoupling)
        #  - memrecall:      the fixed-size ORDERED memory bottleneck (ThetaGammaMemory)
        self.tg_addon = (ThetaGammaCoupling(embed_dim, num_slots=7)
                         if self.cfg["theta_gamma"] and task != "memrecall" else None)
        self.tg_mem = (ThetaGammaMemory(embed_dim, num_slots=mem_slots)
                       if self.cfg["theta_gamma"] and task == "memrecall" else None)

        self.column = CorticalColumn(dim=embed_dim) if self.cfg["cortical_column"] else None
        self.lateral = LateralInhibition(dim=embed_dim) if self.cfg["lateral_inhibition"] else None

        if task == "recall":
            self.step_key = nn.Embedding(max_T, embed_dim)   # tied key/query positional retrieval
        if task == "memrecall":
            self.q_embed = nn.Embedding(max_T, embed_dim)              # query for the mean-pool fallback
            self.bottleneck_read = nn.Linear(embed_dim * 2, embed_dim)  # read item k from order-less mem

        self.out_norm = nn.LayerNorm(embed_dim)
        self.readout = nn.Linear(embed_dim, dims)

        # theta_gamma is structural (not gated) on memrecall; gateable add-on otherwise
        self.gate_names = [m for m in GATEABLE if self.cfg[m]
                           and not (task == "memrecall" and m == "theta_gamma")]
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

    def _hidden(self, heading, speed, vz, k=None):
        """Compute the integrated hidden representation h (B, embed_dim) plus the
        intermediates needed for aux heads. Shared by encode() and forward()."""
        B, T = heading.shape
        device = heading.device
        step = torch.zeros(B, T, self.embed_dim, device=device)
        if self.cfg["conjunctive"]:
            step = step + (self.conjunctive(heading.reshape(B * T), speed.reshape(B * T)).view(B, T, -1)
                           + self.vert(vz.reshape(B * T, 1)).view(B, T, -1))

        if self.task == "pathint":
            position = self.integrator(step) if self.cfg["grid_attractor"] else self.pool_proj(step.mean(1))
        else:
            # recall / memrecall need per-step running positions
            states = self.integrator(step, return_sequence=True) if self.cfg["grid_attractor"] else step
            if self.task == "recall":
                idx = torch.arange(T, device=device)
                attn = torch.softmax(self.step_key(k) @ self.step_key(idx).t()
                                     / math.sqrt(self.embed_dim), dim=-1)
                position = (attn.unsqueeze(-1) * states).sum(dim=1)
            else:  # memrecall — through a fixed-size memory bottleneck
                if self.tg_mem is not None:
                    m = self.tg_mem.store(states)                 # ordered multiplex
                    position = self.tg_mem.retrieve(m, k)         # read slot k
                else:
                    m = states.mean(dim=1)                        # order-less bottleneck
                    position = self.bottleneck_read(torch.cat([m, self.q_embed(k)], dim=-1))

        h = position
        tg = None
        if self.tg_addon is not None:
            tg = self.tg_addon(step)
            h = h + self._g("theta_gamma") * tg
        if self.column is not None:
            h = h + self._g("cortical_column") * self.column(h)
        if self.lateral is not None:
            h = h + self._g("lateral_inhibition") * self.lateral(h)
        h = self.out_norm(h)
        return h, step, position, tg

    def encode(self, heading, speed, vz, k=None):
        """Integrated trajectory representation (B, embed_dim) — the spatial summary
        the LLM consumes in Milestone 2 (TrajectoryLLM)."""
        return self._hidden(heading, speed, vz, k)[0]

    def forward(self, heading, speed, vz, k=None, return_aux=False):
        h, step, position, tg = self._hidden(heading, speed, vz, k)
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
