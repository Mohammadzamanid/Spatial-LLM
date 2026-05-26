"""
src/models/neuro/spatial_cells.py

NETWORK LEVEL — the brain's dedicated spatial navigation cell types.

Beyond grid cells (entorhinal) and place cells (hippocampus), the navigation
system contains several specialised cell types. This module implements them:

  - HeadDirectionCells:   ring attractor encoding heading/orientation
                          (Taube et al., 1990) — fires for a preferred direction
  - BoundaryVectorCells:  fire at a preferred distance & direction from
                          environmental boundaries (Lever et al., 2009)
  - SpeedCells:           encode running speed, drive grid-cell path integration
                          (Kropff et al., 2015)
  - ConjunctiveGridCells: combine position + head direction (Sargolini 2006)

Together with grid + place cells these form a complete cognitive map.

References:
  Taube, Muller & Ranck (1990) "Head-direction cells recorded from the
    postsubiculum in freely moving rats"
  Lever et al. (2009) "Boundary vector cells in the subiculum of the
    hippocampal formation"
  Kropff et al. (2015) "Speed cells in the medial entorhinal cortex", Nature
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadDirectionCells(nn.Module):
    """
    Head-direction cell population modelled as a RING ATTRACTOR.

    N cells tile the 0–360° circle; each has a preferred direction. A given
    heading activates cells via a circular (von Mises) tuning curve. The
    population vector decodes heading — a continuous attractor on a ring.
    """

    def __init__(self, num_cells: int = 64, kappa: float = 4.0, embed_dim: int = 64):
        super().__init__()
        self.num_cells = num_cells
        self.kappa = kappa                                   # tuning sharpness
        preferred = torch.linspace(0, 2 * math.pi, num_cells + 1)[:-1]
        self.register_buffer("preferred_dirs", preferred)    # (num_cells,)
        self.proj = nn.Linear(num_cells, embed_dim)

    def forward(self, heading_rad: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heading_rad: (B,) or (B, 1) heading in radians
        Returns:
            (B, embed_dim) head-direction embedding
        """
        if heading_rad.dim() == 1:
            heading_rad = heading_rad.unsqueeze(1)           # (B, 1)
        # von Mises tuning: exp(kappa * cos(theta - preferred))
        diff = heading_rad - self.preferred_dirs.unsqueeze(0)  # (B, num_cells)
        activity = torch.exp(self.kappa * torch.cos(diff))
        activity = activity / activity.sum(dim=-1, keepdim=True)  # normalise
        return self.proj(activity)

    def decode_heading(self, heading_rad: torch.Tensor) -> torch.Tensor:
        """Population-vector decode: returns reconstructed heading (B,)."""
        if heading_rad.dim() == 1:
            heading_rad = heading_rad.unsqueeze(1)
        diff = heading_rad - self.preferred_dirs.unsqueeze(0)
        activity = torch.exp(self.kappa * torch.cos(diff))
        sin = (activity * self.preferred_dirs.sin()).sum(-1)
        cos = (activity * self.preferred_dirs.cos()).sum(-1)
        return torch.atan2(sin, cos)


class BoundaryVectorCells(nn.Module):
    """
    Boundary Vector Cells — fire when a boundary is at a preferred distance
    AND allocentric direction from the animal.

    We model B cells, each tuned to (preferred_distance, preferred_angle),
    using a 2D Gaussian over (distance, angle) space. Given the nearest
    boundary's (distance, bearing), the population encodes the geometry of
    the surrounding space — critical for place-field formation.
    """

    def __init__(self, num_cells: int = 32, embed_dim: int = 64,
                 max_distance: float = 1.0):
        super().__init__()
        self.num_cells = num_cells
        # Learnable preferred distances and angles
        self.pref_dist = nn.Parameter(torch.rand(num_cells) * max_distance)
        self.pref_angle = nn.Parameter(torch.rand(num_cells) * 2 * math.pi)
        self.log_sigma_d = nn.Parameter(torch.zeros(num_cells))
        self.log_sigma_a = nn.Parameter(torch.zeros(num_cells))
        self.proj = nn.Linear(num_cells, embed_dim)

    def forward(self, boundary_dist: torch.Tensor,
                boundary_angle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boundary_dist:  (B,) distance to nearest boundary
            boundary_angle: (B,) allocentric bearing to boundary (radians)
        Returns:
            (B, embed_dim)
        """
        bd = boundary_dist.unsqueeze(1)                      # (B, 1)
        ba = boundary_angle.unsqueeze(1)                     # (B, 1)
        sig_d = self.log_sigma_d.exp().clamp(min=0.05)
        sig_a = self.log_sigma_a.exp().clamp(min=0.05)

        dist_term = ((bd - self.pref_dist) ** 2) / (2 * sig_d ** 2)
        # Circular difference for angle
        angle_diff = torch.atan2(torch.sin(ba - self.pref_angle),
                                 torch.cos(ba - self.pref_angle))
        angle_term = (angle_diff ** 2) / (2 * sig_a ** 2)

        activity = torch.exp(-(dist_term + angle_term))      # (B, num_cells)
        return self.proj(activity)


class SpeedCells(nn.Module):
    """
    Speed cells — firing rate proportional to running speed. They provide the
    velocity signal that drives grid-cell path integration.
    """

    def __init__(self, num_cells: int = 16, embed_dim: int = 64):
        super().__init__()
        self.pref_speed = nn.Parameter(torch.linspace(0, 1, num_cells))
        self.log_sigma = nn.Parameter(torch.zeros(num_cells))
        self.proj = nn.Linear(num_cells, embed_dim)

    def forward(self, speed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            speed: (B,) normalised speed in [0, 1]
        Returns:
            (B, embed_dim)
        """
        s = speed.unsqueeze(1)                               # (B, 1)
        sigma = self.log_sigma.exp().clamp(min=0.05)
        activity = torch.exp(-((s - self.pref_speed) ** 2) / (2 * sigma ** 2))
        return self.proj(activity)


class ConjunctiveSpatialCells(nn.Module):
    """
    Conjunctive cells that bind POSITION + HEAD DIRECTION + SPEED into a single
    representation, as found in deeper entorhinal layers. This is the substrate
    for path integration — predicting the next position from the current state
    plus movement.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.hd = HeadDirectionCells(embed_dim=embed_dim)
        self.speed = SpeedCells(embed_dim=embed_dim)
        self.bind = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, heading_rad: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        hd_emb = self.hd(heading_rad)
        speed_emb = self.speed(speed)
        return self.bind(torch.cat([hd_emb, speed_emb], dim=-1))
