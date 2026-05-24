"""
src/models/grid_cell_encoder.py

Grid Cell Encoder — inspired by the entorhinal cortex.

Biological basis:
  Grid cells in the medial entorhinal cortex fire in hexagonal lattice
  patterns as an animal moves through space. Different grid modules have
  different spatial scales and orientations, providing a multi-resolution
  coordinate system — the brain's GPS.

Implementation:
  We simulate multiple grid modules, each with a different scale and
  rotation. Each module produces a 2D hexagonal activation pattern for
  a given (lat, lon) pair. The patterns are concatenated and projected
  into the model's hidden dimension.

  This outperforms plain Fourier embeddings because:
  - Hexagonal tiling has optimal packing (covers 2D space with minimal overlap)
  - Multi-scale modules naturally encode spatial hierarchy
  - Learned rotations adapt to geographic coordinate distributions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GridModule(nn.Module):
    """
    A single grid cell module with fixed scale and learnable orientation.
    Produces hexagonal activation patterns for (lat, lon) inputs.
    """

    def __init__(self, scale: float, embed_dim: int, num_cells: int = 64):
        super().__init__()
        self.scale = scale
        self.num_cells = num_cells

        # Learnable rotation angle — scalar parameter
        self.rotation = nn.Parameter(torch.zeros(()))  # scalar, not (1,)

        # Fixed hexagonal basis angles (60° apart)
        self.register_buffer("basis_angle1", torch.tensor(0.0))
        self.register_buffer("basis_angle2", torch.tensor(math.pi / 3))

        # Input: sin/cos × u/v = num_cells * 2 features
        self.proj = nn.Linear(num_cells * 2, embed_dim)

    def _hex_basis(self) -> torch.Tensor:
        """Compute rotated hexagonal basis — always (2, 2), no batch dims."""
        a1 = self.basis_angle1 + self.rotation   # scalar
        a2 = self.basis_angle2 + self.rotation   # scalar
        b1 = torch.stack([a1.cos(), a1.sin()])   # (2,)
        b2 = torch.stack([a2.cos(), a2.sin()])   # (2,)
        return torch.stack([b1, b2], dim=0)       # (2, 2)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, 2) lat/lon in degrees
        Returns:
            activations: (B, embed_dim)
        """
        coords_norm = coords / self.scale                         # (B, 2)
        basis = self._hex_basis()                                  # (2, 2)
        projected = torch.mm(coords_norm, basis.t())               # (B, 2)

        freqs = torch.arange(
            1, self.num_cells + 1, device=coords.device, dtype=coords.dtype
        )  # (num_cells,)

        # (B, 1) * (1, num_cells) → (B, num_cells) — explicit unsqueeze
        u = projected[:, 0].unsqueeze(1) * freqs.unsqueeze(0)
        v = projected[:, 1].unsqueeze(1) * freqs.unsqueeze(0)

        activations = torch.cat([
            torch.cos(2 * math.pi * u) * torch.cos(2 * math.pi * v),
            torch.sin(2 * math.pi * u) + torch.sin(2 * math.pi * v),
        ], dim=-1)  # (B, num_cells * 2)

        return self.proj(activations)  # (B, embed_dim)


class GridCellEncoder(nn.Module):
    """
    Multi-scale grid cell encoder.
    Uses N grid modules at exponentially increasing scales,
    mirroring the biological hierarchy from fine to coarse.

    For geographic coordinates, scales range from ~0.01° (≈1km) to ~10° (≈1000km).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_modules: int = 6,
        base_scale: float = 0.01,
        scale_factor: float = 3.0,
        num_cells: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        scales = [base_scale * (scale_factor ** i) for i in range(num_modules)]
        self.modules_list = nn.ModuleList([
            GridModule(scale=s, embed_dim=embed_dim, num_cells=num_cells)
            for s in scales
        ])

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * num_modules, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, 2) lat/lon in degrees
        Returns:
            grid_embedding: (B, embed_dim)
        """
        module_outputs = [m(coords) for m in self.modules_list]  # N × (B, D)
        combined = torch.cat(module_outputs, dim=-1)              # (B, N*D)
        return self.fusion(combined)                              # (B, D)


class GridCellEncoderWithTokens(GridCellEncoder):
    """
    Returns one token per grid module for cross-attention with the LLM.
    Each token represents a different spatial scale.
    """

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            tokens: (B, num_modules, embed_dim)
        """
        module_outputs = [m(coords) for m in self.modules_list]  # N × (B, D)
        return torch.stack(module_outputs, dim=1)                 # (B, N, D)
