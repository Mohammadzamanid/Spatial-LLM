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

    def forward(self, external_input: torch.Tensor, recurrent_gain: float = 1.0) -> torch.Tensor:
        """
        Args:
            external_input: (B, num_units) drive to the network
            recurrent_gain: scalar gate on the RECURRENT term only (afferent `external_input`
                is spared). 1.0 = default; an acetylcholine-style encode/retrieve switch sets it
                to ~0 during ENCODING (suppress recall so the bump follows the raw input) and ~1
                during RETRIEVAL (Hasselmo 2006). See AcetylcholineGate in models/neuromodulation.
        Returns:
            (B, num_units) settled bump activity
        """
        B = external_input.shape[0]
        u = external_input.clone()
        for _ in range(self.steps):
            recurrent = F.linear(u, self.W)                    # (B, N)
            u = u + self.dt * (-u + F.relu(recurrent_gain * recurrent + external_input))
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

    def forward(self, coords: torch.Tensor, recurrent_gain: float = 1.0) -> torch.Tensor:
        """
        Args:
            coords: (B, 2) lat/lon
            recurrent_gain: scalar gate on the RECURRENT term only (afferent `drive` spared) — the
                acetylcholine encode/retrieve hook (see ContinuousAttractorNetwork.forward).
        Returns:
            (B, embed_dim) grid-attractor embedding
        """
        drive = F.relu(self.coord_to_sheet(coords))            # (B, N)
        u = drive.clone()
        for _ in range(self.steps):
            recurrent = F.linear(u, self.W)
            u = F.relu(0.5 * u + 0.5 * (recurrent_gain * recurrent + drive))
            u = u / (u.sum(dim=-1, keepdim=True) + 1e-6) * self.num_units * 0.1
        return self.readout(u)


class HopfieldAssociativeMemory(nn.Module):
    """
    Discrete AUTO-ASSOCIATIVE (Hopfield / Marr-Willshaw / Treves-Rolls CA3) memory.

    Unlike the CONTINUOUS attractor networks above — whose stable states are a fixed continuum of
    bump positions set by a Mexican-hat kernel — this network stores a SET OF DISCRETE PATTERNS in a
    Hebbian-written recurrent weight matrix. A partial or noisy cue is then pattern-completed to the
    nearest stored pattern by recurrent settling, and overlapping patterns INTERFERE (proactive /
    retroactive) — the CA3 auto-associator that Hasselmo's acetylcholine encode/retrieve switch acts
    upon. It shares only the relu-settle FORM with ContinuousAttractorNetwork, not its (continuous)
    attractor structure.

    References:
      Marr (1971) "Simple memory: a theory for archicortex"
      Hopfield (1982) "Neural networks and physical systems with emergent collective computational
        abilities", PNAS
      Treves & Rolls (1994) "Computational analysis of the role of the hippocampus in memory"
    """

    def __init__(self, num_units: int, steps: int = 8):
        super().__init__()
        self.num_units = num_units
        self.steps = steps
        # Recurrent auto-associative weights, written by a one-shot Hebbian rule (a buffer, not
        # backprop-trained — it is written by `store`, exactly like a plateau writes BTSP weights).
        self.register_buffer("W", torch.zeros(num_units, num_units))

    def reset(self):
        """Clear all stored patterns (W -> 0). A locus-coeruleus reset re-seeds the code on top of this."""
        self.W.zero_()

    def store(self, pattern: torch.Tensor, rate: float = 1.0) -> float:
        """One-shot Hebbian outer-product write of a pattern into the recurrent weights.

        Args:
            pattern: (N,) or (B, N) activity pattern(s) to imprint as attractor(s).
            rate:    plasticity rate (an acetylcholine-enhanced encoding gain scales THIS, leaving the
                     recurrent transmission gain in `settle` as a separate knob — the two are decoupled).
        Returns:
            ||ΔW|| — the synaptic change this write induced (the storage-energy control: it lets an
            experiment prove a pattern was written with EQUAL strength across conditions, so that a
            recall difference reflects WHAT was stored, not HOW MUCH).
        """
        p = pattern if pattern.dim() == 2 else pattern.unsqueeze(0)
        dW = rate * torch.einsum("bi,bj->ij", p, p) / p.shape[0]   # symmetric outer product
        dW = dW - torch.diag(torch.diagonal(dW))                   # no self-connections (Hopfield)
        self.W += dW
        return dW.norm().item()

    def settle(self, drive: torch.Tensor, recurrent_gain: float = 1.0,
               steps: int | None = None, drive_decay: float = 1.0) -> torch.Tensor:
        """Recurrent settling from an afferent drive. Mirrors the CAN update form; the RECURRENT term
        is gated by `recurrent_gain` (the acetylcholine hook) while the afferent `drive` is spared.

        Args:
            drive:          (B, N) afferent input (the cue, or the to-be-encoded pattern).
            recurrent_gain: gate on recurrent recall. ~0 => ENCODING (state follows the raw drive, not
                            pulled toward stored attractors); ~1 => RETRIEVAL (recurrent completion).
            drive_decay:    per-step multiplier on the afferent drive. 1.0 clamps the cue on throughout
                            (the cue keeps re-injecting itself); <1.0 makes the cue TRANSIENT so that
                            any cleanup of the state is attributable to the recurrent weights, not to
                            the cue echoing itself (the completion-requires-W_rec control).
        Returns:
            (B, N) settled activity.
        """
        steps = steps or self.steps
        x = F.relu(drive.clone())
        d = drive.clone()
        for _ in range(steps):
            recurrent = F.linear(x, self.W)                         # x @ W^T (W symmetric)
            x = F.relu(recurrent_gain * recurrent + d)
            x = x / (x.sum(dim=-1, keepdim=True) + 1e-6) * self.num_units * 0.1
            d = d * drive_decay
        return x
