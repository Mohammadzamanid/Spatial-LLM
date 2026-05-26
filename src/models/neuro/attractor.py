"""
src/models/neuro/attractor.py

NETWORK DYNAMICS LEVEL — continuous attractor networks (CANs).

Grid cells, place cells, and head-direction cells are believed to arise from
CONTINUOUS ATTRACTOR dynamics: recurrent networks whose stable states form a
continuous manifold (a line, ring, or torus). Activity bumps move along the
manifold to track the animal's state via path integration.

  - ContinuousAttractorNetwork: 1D/2D bump attractor with local excitation +
                                global inhibition (the "Mexican-hat" kernel)
  - GridAttractorNetwork:       toroidal attractor producing grid-like activity

References:
  Amari (1977) "Dynamics of pattern formation in lateral-inhibition type
    neural fields"
  Burak & Fiete (2009) "Accurate path integration in continuous attractor
    network models of grid cells", PLoS Comput Biol
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousAttractorNetwork(nn.Module):
    """
    1D continuous attractor (ring) network.

    N units arranged on a ring with Mexican-hat connectivity:
      - short-range excitation (nearby units reinforce)
      - long-range inhibition (distant units suppress)
    A localized "bump" of activity forms and persists, encoding a continuous
    variable (e.g. heading or 1D position). External input shifts the bump.
    """

    def __init__(self, num_units: int = 128, sigma_exc: float = 5.0,
                 sigma_inh: float = 20.0, dt: float = 0.1, steps: int = 10):
        super().__init__()
        self.num_units = num_units
        self.dt = dt
        self.steps = steps

        # Precompute Mexican-hat recurrent kernel on the ring
        idx = torch.arange(num_units)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        dist = torch.minimum(dist, num_units - dist).float()   # circular distance
        exc = torch.exp(-dist ** 2 / (2 * sigma_exc ** 2))
        inh = torch.exp(-dist ** 2 / (2 * sigma_inh ** 2))
        kernel = exc - 0.5 * inh
        self.register_buffer("W", kernel)

    def forward(self, external_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            external_input: (B, num_units) drive to the network
        Returns:
            (B, num_units) settled bump activity
        """
        B = external_input.shape[0]
        u = external_input.clone()
        for _ in range(self.steps):
            recurrent = F.linear(u, self.W)                    # (B, N)
            u = u + self.dt * (-u + F.relu(recurrent + external_input))
            u = F.relu(u)
            # Normalize to prevent runaway
            denom = u.sum(dim=-1, keepdim=True) + 1e-6
            u = u / denom * self.num_units * 0.1
        return u


class GridAttractorNetwork(nn.Module):
    """
    2D toroidal continuous attractor producing grid-cell-like activity.

    Units live on a 2D sheet with periodic (torus) boundary conditions and
    Mexican-hat connectivity. The bump tiles the sheet, and because of the
    toroidal topology, mapping physical space onto the sheet yields the
    characteristic hexagonal grid firing pattern.

    Here we provide a tractable approximation: a learnable projection from
    coordinates onto the toroidal sheet, followed by attractor settling.
    """

    def __init__(self, grid_size: int = 16, embed_dim: int = 64, steps: int = 5):
        super().__init__()
        self.grid_size = grid_size
        self.num_units = grid_size * grid_size
        self.steps = steps

        # Coordinate → sheet drive
        self.coord_to_sheet = nn.Linear(2, self.num_units)

        # 2D Mexican-hat kernel with toroidal wrapping
        coords = torch.stack(torch.meshgrid(
            torch.arange(grid_size), torch.arange(grid_size), indexing="ij"
        ), dim=-1).reshape(-1, 2).float()                      # (num_units, 2)
        d = coords.unsqueeze(0) - coords.unsqueeze(1)          # (N, N, 2)
        d = torch.minimum(d.abs(), grid_size - d.abs())        # toroidal
        dist_sq = (d ** 2).sum(-1)                             # (N, N)
        exc = torch.exp(-dist_sq / (2 * 2.0 ** 2))
        inh = torch.exp(-dist_sq / (2 * 6.0 ** 2))
        self.register_buffer("W", exc - 0.6 * inh)

        self.readout = nn.Linear(self.num_units, embed_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, 2) lat/lon
        Returns:
            (B, embed_dim) grid-attractor embedding
        """
        drive = F.relu(self.coord_to_sheet(coords))            # (B, N)
        u = drive.clone()
        for _ in range(self.steps):
            recurrent = F.linear(u, self.W)
            u = F.relu(0.5 * u + 0.5 * (recurrent + drive))
            u = u / (u.sum(dim=-1, keepdim=True) + 1e-6) * self.num_units * 0.1
        return self.readout(u)
