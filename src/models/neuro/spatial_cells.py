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


class EgocentricObjectVectorCells(nn.Module):
    """
    Object-vector cells (Høydal, Skytøen, Andersson, Moser & Moser, *Nature* 2019) — fire when a discrete
    object / landmark is at a preferred DISTANCE and a preferred EGOCENTRIC BEARING (relative to the
    animal's own heading). The egocentric tuning is the defining contrast with BoundaryVectorCells
    (allocentric): an object-vector cell's field stays locked to the object and rotates with the animal, so
    downstream cells can REANCHOR a map to the object's reference frame and translate the grid pattern with
    the object (the multi-reference-frame map; cf. grid reanchoring to objects/rewards, Butler 2019,
    Boccara 2019). Optionally conditioned on object identity.
    """

    def __init__(self, num_cells: int = 32, embed_dim: int = 64,
                 max_distance: float = 3.0, num_objects: int = 1):
        super().__init__()
        self.num_cells = num_cells
        self.pref_dist = nn.Parameter(torch.rand(num_cells) * max_distance)
        self.pref_bearing = nn.Parameter(torch.rand(num_cells) * 2 * math.pi)   # EGOCENTRIC preferred bearing
        self.log_sigma_d = nn.Parameter(torch.zeros(num_cells))
        self.log_sigma_b = nn.Parameter(torch.zeros(num_cells))
        self.id_emb = nn.Embedding(num_objects, num_cells) if num_objects > 1 else None
        self.proj = nn.Linear(num_cells, embed_dim)

    def forward(self, object_dist: torch.Tensor, object_bearing: torch.Tensor,
                object_id: torch.Tensor = None) -> torch.Tensor:
        """object_dist (B,), object_bearing (B,) EGOCENTRIC bearing to the object (rad); -> (B, embed_dim)."""
        bd = object_dist.unsqueeze(1); bb = object_bearing.unsqueeze(1)
        sig_d = self.log_sigma_d.exp().clamp(min=0.05); sig_b = self.log_sigma_b.exp().clamp(min=0.05)
        dist_term = ((bd - self.pref_dist) ** 2) / (2 * sig_d ** 2)
        ang = torch.atan2(torch.sin(bb - self.pref_bearing), torch.cos(bb - self.pref_bearing))
        ang_term = (ang ** 2) / (2 * sig_b ** 2)
        act = torch.exp(-(dist_term + ang_term))                                # (B, num_cells)
        if self.id_emb is not None and object_id is not None:
            act = act * torch.sigmoid(self.id_emb(object_id))
        return self.proj(act)


class EgocentricCenterCells(nn.Module):
    """
    Egocentric center-bearing / center-distance cells (medial entorhinal cortex; Nat Commun 2025). Encode the
    geometric CENTRE of the environment in self-centred polar coordinates (distance + egocentric bearing).
    Unlike object-vector cells (a movable object) the anchor is the FIXED room centre (in vivo inferred from
    boundary geometry), giving a stable egocentric reference that coexists with the allocentric grid and with
    object / boundary egocentric frames — evidence that MEC transforms between self-centred and world-centred
    codes. Computes the egocentric centre vector internally from the agent's position and heading.
    """

    def __init__(self, num_cells: int = 32, embed_dim: int = 64,
                 max_distance: float = 3.0, center=(0.0, 0.0)):
        super().__init__()
        self.register_buffer("center", torch.tensor(center, dtype=torch.float))
        self.pref_dist = nn.Parameter(torch.rand(num_cells) * max_distance)
        self.pref_bearing = nn.Parameter(torch.rand(num_cells) * 2 * math.pi)   # EGOCENTRIC preferred bearing
        self.log_sigma_d = nn.Parameter(torch.zeros(num_cells))
        self.log_sigma_b = nn.Parameter(torch.zeros(num_cells))
        self.proj = nn.Linear(num_cells, embed_dim)

    def forward(self, pos: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        """pos (B,2), heading (B,) -> (B, embed_dim). Egocentric (distance, bearing) to the room centre."""
        vrel = self.center.unsqueeze(0) - pos                                   # (B,2) vector to centre
        dist = vrel.norm(dim=1).unsqueeze(1)
        bearing = (torch.atan2(vrel[:, 1], vrel[:, 0]) - heading).unsqueeze(1)  # egocentric bearing to centre
        sig_d = self.log_sigma_d.exp().clamp(min=0.05); sig_b = self.log_sigma_b.exp().clamp(min=0.05)
        dist_term = ((dist - self.pref_dist) ** 2) / (2 * sig_d ** 2)
        ang = torch.atan2(torch.sin(bearing - self.pref_bearing), torch.cos(bearing - self.pref_bearing))
        ang_term = (ang ** 2) / (2 * sig_b ** 2)
        return self.proj(torch.exp(-(dist_term + ang_term)))


def _periodic_cdist3(a, b):
    d = (a.unsqueeze(1) - b.unsqueeze(0)).abs(); d = torch.minimum(d, 1 - d)
    return d.norm(dim=2)


def _blue_noise_pool(n, rmin, gen, max_tries: int = 150000):
    """Poisson-disk (blue-noise) packing in the unit cube: points at least ``rmin`` apart (toroidal). Gives a
    regular nearest-neighbor spacing (LOCAL order) with NO global lattice — the bat 3D field arrangement."""
    pts = torch.empty(0, 3); t = 0
    while pts.shape[0] < n and t < max_tries:
        c = torch.rand(1, 3, generator=gen)
        if pts.shape[0] == 0 or _periodic_cdist3(c, pts).min().item() > rmin:
            pts = torch.cat([pts, c], 0)
        t += 1
    return pts


class LocalOrder3DGrid(nn.Module):
    """
    3D grid-cell population with LOCAL order but NO global lattice — the bat MEC regime (Ginosar, Aljadeff,
    Las, Derdikman & Ulanovsky, *Nature* 2021). Freely-flying bats have 3D multi-field "grid-like" MEC neurons
    whose fields sit at a characteristic nearest-neighbor distance (local order) but do NOT form a periodic 3D
    crystal (no long-range lattice) — unlike a naive cubic/FCC lattice. We realize that faithfully: a shared
    blue-noise (Poisson-disk) packing supplies the field centers (local order, no lattice), and each cell fires
    at a random subset of them (multiple fields → grid-like, not place-like). The population PATH-INTEGRATES 3D
    self-motion and localizes in full 3D — a biologically-grounded 3D entorhinal code that replaces the 1D
    vertical place-code stub. ``lattice=True`` builds the cubic-lattice control (a global lattice — non-bat).
    """

    def __init__(self, embed_dim: int = 64, n_cells: int = 128, fields_per_cell: int = 15,
                 pool_size: int = 350, rmin: float = 0.095, sigma: float = 0.3, box: float = 3.0,
                 seed: int = 0, lattice: bool = False):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        if lattice:
            side = max(2, round(pool_size ** (1.0 / 3.0)))
            g = (torch.arange(side).float() + 0.5) / side
            pool = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)   # cubic lattice (control)
        else:
            pool = _blue_noise_pool(pool_size, rmin, gen)                                    # blue-noise (bat-like)
        self.register_buffer("pool_unit", pool)                                             # [0,1]^3, for the metric
        centers = (pool * 2 - 1) * box                                                       # world coords [-box,box]
        P = pool.shape[0]; fpc = min(fields_per_cell, P)
        idx = torch.stack([torch.randperm(P, generator=gen)[:fpc] for _ in range(n_cells)])  # each cell: a subset
        self.register_buffer("centers", centers[idx])                                       # (n_cells, fpc, 3)
        self.sigma = sigma; self.n_cells = n_cells; self.box = box
        self.readout = nn.Linear(n_cells, embed_dim)

    def code_at(self, p3d: torch.Tensor) -> torch.Tensor:
        """Population activity at 3D position(s) ``p3d`` (B,3) -> (B, n_cells). Each cell sums Gaussian bumps
        over its field centers (multi-field, grid-like)."""
        d2 = ((p3d.unsqueeze(1).unsqueeze(2) - self.centers.unsqueeze(0)) ** 2).sum(-1)     # (B, n_cells, fpc)
        return torch.exp(-d2 / (2 * self.sigma ** 2)).sum(-1)

    def forward(self, v3d: torch.Tensor, return_sequence: bool = False, noise_std: float = 0.0) -> torch.Tensor:
        """Path-integrate 3D self-motion v3d (B,T,3) -> embed_dim readout of the population code (at the final
        position, or per-step with return_sequence). noise_std adds self-motion noise (3D drift)."""
        B, T, _ = v3d.shape
        p = torch.zeros(B, 3, device=v3d.device, dtype=v3d.dtype); outs = []
        for t in range(T):
            step = v3d[:, t]
            if noise_std > 0:
                step = step + torch.randn_like(step) * noise_std
            p = p + step
            if return_sequence:
                outs.append(self.readout(self.code_at(p)))
        return torch.stack(outs, 1) if return_sequence else self.readout(self.code_at(p))

    def field_centers_unit(self) -> torch.Tensor:
        """The field centers in the unit cube — pass to the local-order / global-lattice metric."""
        return self.pool_unit


class ConjunctiveGridDirectionCells(nn.Module):
    """
    Conjunctive grid × movement-direction cells (Sargolini, Fyhn, Hafting, McNaughton, Witter, Moser & Moser
    2006) — the cellular substrate of the human HEXADIRECTIONAL entorhinal signal (Doeller, Barry & Burgess
    2010; Kunz 2019), the read-out by which a grid code becomes visible in fMRI as a 6-fold-symmetric modulation
    of activity by movement direction — including movement through ABSTRACT 2-D "concept" spaces (Constantinescu,
    O'Keefe & Behrens 2016; the grid code as the brain's general cognitive-map engine).

    Each cell conjoins a grid cell with a preferred movement direction. CRUCIALLY the preferred directions here
    are UNIFORM (not clustered at the grid axes): the 6-fold signal is NOT put in. It EMERGES because the
    population is more strongly DRIVEN — its activity is modulated more — for runs aligned to the hexagonal
    lattice, read out through a movement-sensitive NONLINEARITY (Bush & Burgess 2015; Stemmler 2015). A LINEAR
    read-out of the same grid is direction-invariant (flat); a SQUARE lattice yields a 4-fold signal. So the
    directional symmetry is inherited from the grid's spatial lattice symmetry — measured, not imposed.
    """

    def __init__(self, n_dir: int = 12, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("pref_dir", torch.rand(n_dir, generator=g) * 2 * math.pi)   # UNIFORM preferred dirs

    def conjunctive(self, grid_act: torch.Tensor, move_dir: torch.Tensor) -> torch.Tensor:
        """Conjunctive population activity: grid rate x rectified movement-direction tuning.
        grid_act (T, n_grid), move_dir (T,) -> (T, n_grid, n_dir)."""
        tune = torch.clamp(torch.cos(move_dir.unsqueeze(1) - self.pref_dir.unsqueeze(0)), min=0.0)   # (T, n_dir)
        return grid_act.unsqueeze(-1) * tune.unsqueeze(1)

    @staticmethod
    def direction_signal(grid_along_run: torch.Tensor) -> float:
        """The movement-driven activity POWER a movement-sensitive population signals over one straight run:
        the per-cell temporal variance of the grid activity along the run, summed. This is the nonlinearity that
        makes the hexadirectional signal appear; a LINEAR summary (the mean) would be direction-invariant.
        grid_along_run (T, n_grid) -> scalar."""
        return grid_along_run.var(dim=0).sum().item()


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
