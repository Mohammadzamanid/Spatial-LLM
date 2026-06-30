"""
src/models/neuro/theta_sweep.py

ThetaSweepSampler — online theta-cycle "look-around" of the entorhinal-hippocampal map (Vollan, Gardner,
Moser & Moser, *Nature* 2025). In each theta cycle, decoded grid activity does not sit at the current
position or replay an offline trajectory: it sweeps OUTWARD from the animal's location, ALTERNATING left and
right around the heading across successive cycles, sampling the surrounding space — including never-visited
or currently-inaccessible points. The sweep is carried strongly by grid cells, is aligned across modules,
and its length scales with grid-module spacing (Vollan report ~19.7% of spacing).

This is distinct from the repo's existing theta machinery (phase precession, theta-gamma ordered memory,
sharp-wave replay): those are gating / ordered working memory / offline compressed replay. The theta SWEEP
is an ONLINE active sampler — a look-ahead interface that queries the cognitive map at points ahead of the
agent, which downstream readouts (or the LLM) can use to answer "what is probably to my left if I keep
walking?" and to route around hazards before reaching them.
"""
import math

import torch
import torch.nn as nn


class ThetaSweepSampler(nn.Module):
    def __init__(self, sweep_frac: float = 0.197, angle_deg: float = 25.0, steps: int = 8):
        super().__init__()
        self.sweep_frac = sweep_frac                 # sweep length as a fraction of module spacing (Vollan ~0.197)
        self.angle = math.radians(angle_deg)         # left/right offset from heading
        self.steps = steps                           # sample points along the sweep

    def spacings(self, grid_modules) -> torch.Tensor:
        """Real-space period of each grid module (side / gain)."""
        return grid_modules.side / grid_modules.gains

    def sweep_positions(self, pos, heading, cycle_index, length):
        """Positions along ONE theta sweep: outward from `pos` along (heading + alternating-side * angle),
        extending `length` ahead in `steps` points. The side alternates left/right across theta cycles."""
        side = -1.0 if cycle_index % 2 == 0 else 1.0                 # left/right alternation
        direction = heading + side * self.angle
        d = torch.tensor([math.cos(direction), math.sin(direction)], dtype=torch.float)
        ks = (torch.arange(1, self.steps + 1, dtype=torch.float) / self.steps)
        return pos.view(1, 2) + ks.view(-1, 1) * length * d.view(1, 2), side, direction

    def forward(self, pos, heading, grid_modules, cycle_index):
        """A full theta-cycle look-around: returns (positions, grid_codes, side, direction).
        Sweep length = sweep_frac * mean module spacing (the population sweep); per-module lengths
        (sweep_frac * each spacing) are multi-scale and available via `spacings()`. The grid codes along the
        sweep are the look-ahead tokens a readout / LLM consumes."""
        length = self.sweep_frac * self.spacings(grid_modules).mean()
        positions, side, direction = self.sweep_positions(pos, heading, cycle_index, length)
        codes = grid_modules.grid_code_at(positions)                # (steps, K*M) grid activity along the sweep
        return positions, codes, side, direction
