"""
src/models/neuro/temporal_cortex.py

A generic recurrent SUBSTRATE for the temporal domain — the time analogue of TrajectoryCortex.
Deliberately UN-special: a leaky, rectified rate RNN (continuous-time dynamics) with LEARNED
recurrent weights, a SINGLE uniform time-constant (NO hand-set spectrum of timescales), and private
membrane noise. Nothing here encodes "time cells", field widening, or scalar timing — those are not
built in; they are left to EMERGE from training a task, and are measured afterwards
(src/eval/time_cells.py), exactly as grid cells are left to emerge in TrajectoryCortex.

Biological priors used (all generic, none specific to the timing signatures we test for):
  - rectified non-negative rates r = relu(h)            (neurons fire non-negative rates)
  - leaky continuous-time integration with uniform tau  (membrane/synaptic dynamics; one timescale)
  - private Gaussian membrane noise                      (neurons are noisy)
A metabolic activity penalty (sparse, efficient coding) is applied by the trainer, not here.
"""
import math

import torch
import torch.nn as nn


class TemporalCortex(nn.Module):
    def __init__(self, hidden: int = 128, n_in: int = 2, n_out: int = 1,
                 gain: float = 1.4, alpha: float = 0.25):
        super().__init__()
        self.H = hidden
        self.alpha = alpha                                     # = dt/tau, a SINGLE uniform timescale
        self.Wr = nn.Parameter(torch.randn(hidden, hidden) * (gain / math.sqrt(hidden)))
        self.Wi = nn.Parameter(torch.randn(hidden, n_in) * 0.5)
        self.b = nn.Parameter(torch.zeros(hidden))
        self.readout = nn.Linear(hidden, n_out)

    def dynamics(self, x_seq: torch.Tensor, noise: float = 0.0,
                 gen: torch.Generator | None = None) -> torch.Tensor:
        """Run the recurrent substrate over an input sequence. x_seq (B,T,n_in) -> rates (B,T,H)."""
        B, T, _ = x_seq.shape
        h = torch.zeros(B, self.H, device=x_seq.device, dtype=x_seq.dtype)
        rates = []
        for t in range(T):
            r = torch.relu(h)
            h = (1 - self.alpha) * h + self.alpha * (r @ self.Wr.t()
                                                     + x_seq[:, t] @ self.Wi.t() + self.b)
            if noise > 0:
                h = h + noise * torch.randn(h.shape, generator=gen, device=h.device, dtype=h.dtype)
            rates.append(torch.relu(h))
        return torch.stack(rates, dim=1)                       # (B,T,H) non-negative rates

    def forward(self, x_seq, noise: float = 0.0, gen=None):
        """Rates plus the per-step scalar readout (B,T,n_out)."""
        rates = self.dynamics(x_seq, noise=noise, gen=gen)
        return self.readout(rates), rates
