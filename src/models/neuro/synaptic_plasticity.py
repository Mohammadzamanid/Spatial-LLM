"""
src/models/neuro/synaptic_plasticity.py

SYNAPSE LEVEL — biologically-grounded synaptic plasticity rules.

  - HebbianLayer:        "Cells that fire together wire together" with Oja's
                         normalization to prevent unbounded weight growth
  - STDPLayer:           Spike-Timing-Dependent Plasticity — weights strengthen
                         when pre-synaptic spikes precede post-synaptic ones,
                         weaken otherwise (Bi & Poo, 1998)
  - ShortTermPlasticity: Tsodyks-Markram facilitation & depression — synapses
                         dynamically change strength on a fast timescale based
                         on recent presynaptic activity

These give the network local, activity-dependent learning signals that
complement global backpropagation.

References:
  Oja (1982) "A simplified neuron model as a principal component analyzer"
  Bi & Poo (1998) "Synaptic modifications in cultured hippocampal neurons"
  Tsodyks & Markram (1997) "The neural code between neocortical pyramidal
    neurons depends on neurotransmitter release probability"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HebbianLayer(nn.Module):
    """
    Linear layer with a Hebbian plasticity trace (Oja's rule).

    The backprop-trained weight `W` is augmented by a fast Hebbian trace `H`
    updated locally during forward passes:
        ΔH = lr * (post ⊗ pre  -  post² * H)     # Oja normalization term
    The Oja term keeps weights bounded (approximates PCA of the input).

    The trace is a buffer (not backprop-trained); it adapts online.
    """

    def __init__(self, in_dim: int, out_dim: int, hebb_lr: float = 0.01):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.hebb_lr = hebb_lr
        self.register_buffer("hebb_trace", torch.zeros(out_dim, in_dim))

    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim)
            update: whether to update the Hebbian trace this step
        Returns:
            (B, out_dim)
        """
        out = self.linear(x) + F.linear(x, self.hebb_trace)

        if update and self.training:
            with torch.no_grad():
                pre = x.mean(dim=0)                      # (in_dim,)
                post = out.mean(dim=0)                   # (out_dim,)
                # Oja's rule: ΔH = lr (post·preᵀ − post² · H)
                outer = torch.outer(post, pre)           # (out, in)
                decay = (post ** 2).unsqueeze(1) * self.hebb_trace
                self.hebb_trace += self.hebb_lr * (outer - decay)
                self.hebb_trace.clamp_(-1.0, 1.0)
        return out

    def reset_trace(self):
        self.hebb_trace.zero_()


class STDPLayer(nn.Module):
    """
    Spike-Timing-Dependent Plasticity layer.

    Maintains pre- and post-synaptic eligibility traces. When a post-synaptic
    spike follows a pre-synaptic spike → potentiation (LTP). When the order
    reverses → depression (LTD).

    Operates on spike trains (B, T, D). Returns the post-synaptic spike train
    plus the accumulated weight change for inspection / consolidation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        tau_pre: float = 0.9,
        tau_post: float = 0.9,
        a_plus: float = 0.01,
        a_minus: float = 0.012,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.a_plus = a_plus
        self.a_minus = a_minus

    def forward(self, pre_spikes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pre_spikes: (B, T, in_dim) presynaptic spike train
        Returns:
            post_spikes: (B, T, out_dim)
            dw: (out_dim, in_dim) accumulated STDP weight change
        """
        B, T, _ = pre_spikes.shape
        device = pre_spikes.device
        in_dim = self.weight.shape[1]
        out_dim = self.weight.shape[0]

        pre_trace = torch.zeros(B, in_dim, device=device)
        post_trace = torch.zeros(B, out_dim, device=device)
        dw = torch.zeros_like(self.weight)
        post_spikes = []

        for t in range(T):
            pre = pre_spikes[:, t, :]                         # (B, in)
            current = F.linear(pre, self.weight)             # (B, out)
            post = (current > 0.5).float()                   # simple threshold

            # Update traces
            pre_trace = self.tau_pre * pre_trace + pre
            post_trace = self.tau_post * post_trace + post

            # STDP: LTP on post spike (uses pre_trace), LTD on pre spike (uses post_trace)
            ltp = torch.einsum("bo,bi->oi", post, pre_trace)   # (out, in)
            ltd = torch.einsum("bo,bi->oi", post_trace, pre)   # (out, in)
            dw = dw + self.a_plus * ltp - self.a_minus * ltd

            post_spikes.append(post)

        return torch.stack(post_spikes, dim=1), dw / B


class ShortTermPlasticity(nn.Module):
    """
    Tsodyks-Markram short-term plasticity.

    Synaptic efficacy changes on a fast timescale:
      - Facilitation (u): each spike increases release probability
      - Depression (R):   each spike depletes available resources

    Effective synaptic current = R * u * input.
    This gives the network temporal filtering — bursts are emphasised or
    suppressed depending on facilitation/depression balance.
    """

    def __init__(self, dim: int, U: float = 0.2, tau_f: float = 0.9, tau_d: float = 0.8):
        super().__init__()
        self.dim = dim
        self.U = U                # baseline release probability
        self.tau_f = tau_f        # facilitation recovery
        self.tau_d = tau_d        # depression recovery

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) presynaptic activity
        Returns:
            (B, T, D) plasticity-modulated activity
        """
        B, T, D = x.shape
        u = torch.full((B, D), self.U, device=x.device)     # facilitation
        R = torch.ones(B, D, device=x.device)               # available resources
        out = []
        for t in range(T):
            xt = x[:, t, :]
            # Update facilitation and depression
            u = u + self.U * (1.0 - u) * xt
            eff = R * u * xt                                 # effective current
            R = R + (1.0 - R) * (1.0 - self.tau_d) - R * u * xt
            R = R.clamp(0.0, 1.0)
            out.append(eff)
        return torch.stack(out, dim=1)
