"""
src/models/neuro/spiking_temporal_cortex.py

A SPIKING, multi-timescale recurrent substrate for the temporal domain — the spiking successor to
TemporalCortex, built to narrow the gap between "reproducing the time-cell signature" (rate units) and
"reproducing the organ" (spikes + biophysical timescales). Grounded in recent spiking-network timing
work: heterogeneous, LEARNABLE neuronal time-constants give multi-timescale dynamics (Perez-Nieves 2021;
Yin 2023; "dynamic time constants", Sci Rep 2025), and spike-frequency adaptation is the cellular basis
of transient, sequential firing (adaptive LIF; Bellec 2020).

Deliberately UN-special, as with TemporalCortex: a recurrent adaptive-LIF network with surrogate-gradient
spikes. The two things that matter for timing are left FREE and HETEROGENEOUS, not hand-set:
  - per-unit membrane time-constant  (alpha_i = sigmoid(a_mem_i), learnable)
  - per-unit adaptation time-constant (rho_i  = sigmoid(a_adp_i), learnable)
so the multi-timescale SPECTRUM is something to MEASURE after training (src/eval/spiking_time_cells.py),
not a built-in basis. Adaptation strength (kappa) is fixed and strong — a biophysical property, not a
timing knob — which makes neurons fire transiently (the lifetime sparsity that yields single-peaked time
cells). `homogeneous=True` ties all units to ONE shared membrane time-constant (the control that removes
the spectrum).

Biological priors (generic, none timing-signature-specific): non-negative spikes, leaky membrane,
spike-frequency adaptation, private membrane noise, a short synaptic-filter readout trace.
"""
import math

import torch
import torch.nn as nn

from .spiking_neurons import spike_fn   # surrogate-gradient Heaviside spike


class SpikingTemporalCortex(nn.Module):
    def __init__(self, hidden: int = 128, n_in: int = 2, gain: float = 0.6,
                 kappa: float = 0.8, r_decay: float = 0.6, homogeneous: bool = False):
        super().__init__()
        self.N = hidden
        self.kappa = kappa                  # FIXED strong spike-frequency adaptation -> transient firing
        self.r_decay = r_decay              # short synaptic filter: time carried by CURRENT firing
        self.homogeneous = homogeneous
        if homogeneous:
            # control: ONE shared membrane time-constant (no spectrum)
            self.a_mem = nn.Parameter(torch.logit(torch.tensor(0.8)))
        else:
            # heterogeneous, learnable membrane time-constants (the multi-timescale substrate)
            self.a_mem = nn.Parameter(torch.logit(torch.empty(hidden).uniform_(0.6, 0.95)))
        self.a_adp = nn.Parameter(torch.logit(torch.empty(hidden).uniform_(0.90, 0.995)))  # adapt timescales
        self.thr = nn.Parameter(torch.ones(hidden))
        self.Wr = nn.Parameter(torch.randn(hidden, hidden) * (gain / math.sqrt(hidden)))
        self.Wi = nn.Parameter(torch.randn(hidden, n_in) * 0.5)

    def timescales(self):
        """Effective per-unit membrane time-constant tau = 1/(1-alpha) (steps)."""
        alpha = torch.sigmoid(self.a_mem)
        if self.homogeneous:
            alpha = alpha.expand(self.N)
        return 1.0 / (1.0 - alpha + 1e-3)

    def dynamics(self, x_seq, noise: float = 0.0, gen: torch.Generator | None = None):
        """x_seq (B,T,n_in) -> (filtered spike traces R (B,T,N), spike_rate scalar, spikes (B,T,N))."""
        B, T, _ = x_seq.shape
        dev = x_seq.device
        alpha = torch.sigmoid(self.a_mem)
        if self.homogeneous:
            alpha = alpha.expand(self.N)
        rho = torch.sigmoid(self.a_adp)
        v = torch.zeros(B, self.N, device=dev); b = torch.zeros(B, self.N, device=dev)
        r = torch.zeros(B, self.N, device=dev); s = torch.zeros(B, self.N, device=dev)
        traces = []; spikes = []
        for t in range(T):
            v = alpha * v + s @ self.Wr.t() + x_seq[:, t] @ self.Wi.t()
            if noise > 0:
                v = v + noise * torch.randn(v.shape, generator=gen, device=dev)
            s = spike_fn(v, self.thr + b)            # surrogate-gradient spike
            v = v * (1 - s)                          # reset
            b = rho * b + self.kappa * s             # spike-frequency adaptation
            r = self.r_decay * r + s                 # synaptic-filtered readout trace
            traces.append(r); spikes.append(s)
        R = torch.stack(traces, 1)
        S = torch.stack(spikes, 1)
        return R, S.mean(), S

    def forward(self, x_seq, noise: float = 0.0, gen=None):
        return self.dynamics(x_seq, noise=noise, gen=gen)
