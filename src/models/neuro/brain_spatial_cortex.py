"""
src/models/neuro/brain_spatial_cortex.py

INTEGRATION — assembles the full multi-scale neural stack into one module.

This is the "spatial cortex" that ties together every level of organization:

    coordinates ─┬─→ GridAttractorNetwork  (entorhinal toroidal attractor)
                 ├─→ ConjunctiveSpatialCells (head-direction + speed)
                 ├─→ PhasePrecession         (theta-phase position code)
                 └─→ BoundaryVectorCells     (geometry of surrounding space)
                          │
                          ▼
                 CorticalColumn  (canonical L4→L2/3→L5/6 microcircuit)
                          │
                 DivisiveNormalization + LateralInhibition (gain control)
                          │
                 DendriticNeuron (multi-compartment integration)
                          │
                          ▼
              unified spatial cortex embedding  →  fed to the LLM

The output is a single (B, embed_dim) vector (or token sequence) that the
SpatialLLM fuses into the language model via cross-attention.
"""

import math
import torch
import torch.nn as nn

from .attractor import GridAttractorNetwork
from .spatial_cells import ConjunctiveSpatialCells, BoundaryVectorCells
from .oscillations import PhasePrecession
from .microcircuits import CorticalColumn, DivisiveNormalization, LateralInhibition
from .spiking_neurons import DendriticNeuron


class BrainSpatialCortex(nn.Module):
    """
    Full neuroscience-inspired spatial encoder.

    Given (lat, lon) — and optionally heading & speed for path integration —
    produces a unified spatial representation built from every level of neural
    organization (cells → circuits → attractor dynamics → oscillations).
    """

    def __init__(self, embed_dim: int = 256, num_tokens: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # ── Cell-type encoders ──────────────────────────────────────────
        self.grid_attractor = GridAttractorNetwork(grid_size=16, embed_dim=embed_dim)
        self.conjunctive = ConjunctiveSpatialCells(embed_dim=embed_dim)
        self.phase = PhasePrecession(embed_dim=embed_dim)
        self.boundary = BoundaryVectorCells(embed_dim=embed_dim)

        # ── Microcircuit processing ─────────────────────────────────────
        self.column = CorticalColumn(dim=embed_dim)
        self.divnorm = DivisiveNormalization(dim=embed_dim)
        self.lateral = LateralInhibition(dim=embed_dim)

        # ── Dendritic integration ───────────────────────────────────────
        self.dendrite = DendriticNeuron(in_dim=embed_dim * 4, out_dim=embed_dim,
                                        num_branches=6)

        # ── Token projection for cross-attention ────────────────────────
        self.to_tokens = nn.Linear(embed_dim, embed_dim * num_tokens)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        coords: torch.Tensor,
        heading: torch.Tensor | None = None,
        speed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            coords:  (B, 2) lat/lon in degrees
            heading: (B,) optional heading in radians (default 0)
            speed:   (B,) optional normalised speed (default 0)
        Returns:
            (B, num_tokens, embed_dim) spatial cortex tokens
        """
        B = coords.shape[0]
        device = coords.device

        if heading is None:
            heading = torch.zeros(B, device=device)
        if speed is None:
            speed = torch.zeros(B, device=device)

        # Derive a coarse boundary geometry from coordinates (proxy):
        # distance to nearest 1° gridline, bearing toward it.
        frac_lat = coords[:, 0] - coords[:, 0].floor()
        frac_lon = coords[:, 1] - coords[:, 1].floor()
        bdist = torch.minimum(frac_lat, frac_lon)
        bangle = torch.atan2(frac_lat, frac_lon + 1e-6)

        # Position-within-cell for phase precession (proxy from lon fraction)
        pos_in_field = frac_lon

        # ── Encode via each cell type ───────────────────────────────────
        g = self.grid_attractor(coords)                       # (B, D)
        c = self.conjunctive(heading, speed)                  # (B, D)
        p = self.phase(pos_in_field)                          # (B, D)
        b = self.boundary(bdist, bangle)                      # (B, D)

        # ── Dendritic integration of all streams ────────────────────────
        combined = torch.cat([g, c, p, b], dim=-1)            # (B, 4D)
        integrated = self.dendrite(combined)                  # (B, D)

        # ── Microcircuit processing ─────────────────────────────────────
        processed = self.column(integrated)                   # (B, D)
        processed = self.lateral(processed)                   # competition
        processed = processed + integrated                    # residual
        processed = self.out_norm(processed)

        # ── Expand to token sequence ────────────────────────────────────
        tokens = self.to_tokens(processed)                    # (B, D*num_tokens)
        return tokens.view(B, self.num_tokens, self.embed_dim)
