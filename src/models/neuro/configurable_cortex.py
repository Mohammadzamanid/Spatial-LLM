"""
src/models/neuro/configurable_cortex.py

A configurable, ablatable version of BrainSpatialCortex.

Two purposes:
  1. ABLATION: each cell-type and circuit can be toggled on/off, so the
     ablation harness can measure each module's marginal contribution.
  2. SYNCHRONIZATION: each module can expose an auxiliary prediction head,
     giving it its OWN learning signal. This is the key to making complexity
     *help* — modules specialize instead of training as one tangled blob.

Config is a dict of booleans, e.g.:
    {"grid_attractor": True, "conjunctive": False, "phase": True, ...}
"""

import torch
import torch.nn as nn

from .attractor import GridAttractorNetwork
from .spatial_cells import ConjunctiveSpatialCells, BoundaryVectorCells
from .oscillations import PhasePrecession
from .microcircuits import CorticalColumn, LateralInhibition


DEFAULT_CONFIG = {
    "grid_attractor": True,
    "conjunctive": True,
    "phase": True,
    "boundary": True,
    "cortical_column": True,
    "lateral_inhibition": True,
}


class ConfigurableCortex(nn.Module):
    """
    Ablatable spatial cortex with optional per-module auxiliary heads.

    Args:
        embed_dim:   hidden dimension
        config:      dict of module on/off flags (see DEFAULT_CONFIG)
        aux_heads:   if True, each active cell module gets a small coordinate-
                     reconstruction head, returned for auxiliary loss
    """

    def __init__(self, embed_dim: int = 64, config: dict | None = None,
                 aux_heads: bool = False, num_tokens: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.aux_heads = aux_heads

        # Cell-type encoders (only build the active ones)
        self.cell_modules = nn.ModuleDict()
        if self.cfg["grid_attractor"]:
            self.cell_modules["grid_attractor"] = GridAttractorNetwork(
                grid_size=16, embed_dim=embed_dim)
        if self.cfg["conjunctive"]:
            self.cell_modules["conjunctive"] = ConjunctiveSpatialCells(embed_dim=embed_dim)
        if self.cfg["phase"]:
            self.cell_modules["phase"] = PhasePrecession(embed_dim=embed_dim)
        if self.cfg["boundary"]:
            self.cell_modules["boundary"] = BoundaryVectorCells(embed_dim=embed_dim)

        n_active = max(1, len(self.cell_modules))

        # Optional auxiliary coord-reconstruction heads (one per active module)
        if aux_heads:
            self.aux = nn.ModuleDict({
                name: nn.Linear(embed_dim, 2) for name in self.cell_modules
            })

        # Integration: combine active cell outputs
        self.integrate = nn.Sequential(
            nn.Linear(embed_dim * n_active, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # Microcircuit processing (optional)
        self.column = CorticalColumn(dim=embed_dim) if self.cfg["cortical_column"] else None
        self.lateral = LateralInhibition(dim=embed_dim) if self.cfg["lateral_inhibition"] else None

        self.to_tokens = nn.Linear(embed_dim, embed_dim * num_tokens)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, coords, heading=None, speed=None, return_aux=False):
        B = coords.shape[0]
        device = coords.device
        if heading is None:
            heading = torch.zeros(B, device=device)
        if speed is None:
            speed = torch.zeros(B, device=device)

        # Derived geometry for boundary/phase cells
        frac_lat = coords[:, 0] - coords[:, 0].floor()
        frac_lon = coords[:, 1] - coords[:, 1].floor()
        bdist = torch.minimum(frac_lat, frac_lon)
        bangle = torch.atan2(frac_lat, frac_lon + 1e-6)

        outs, aux_out = [], {}
        for name, mod in self.cell_modules.items():
            if name == "grid_attractor":
                o = mod(coords)
            elif name == "conjunctive":
                o = mod(heading, speed)
            elif name == "phase":
                o = mod(frac_lon)
            elif name == "boundary":
                o = mod(bdist, bangle)
            outs_o = o
            outs.append(outs_o)
            if self.aux_heads and return_aux:
                aux_out[name] = self.aux[name](outs_o)

        combined = torch.cat(outs, dim=-1)
        h = self.integrate(combined)

        if self.column is not None:
            h = self.column(h) + h
        if self.lateral is not None:
            h = self.lateral(h) + h
        h = self.out_norm(h)

        tokens = self.to_tokens(h).view(B, self.num_tokens, self.embed_dim)
        if self.num_tokens == 1:
            tokens = tokens.squeeze(1)

        if return_aux:
            return tokens, aux_out
        return tokens
