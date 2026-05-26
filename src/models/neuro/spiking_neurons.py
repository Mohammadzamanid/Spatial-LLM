"""
src/models/neuro/spiking_neurons.py

SINGLE-NEURON LEVEL — biophysically-inspired spiking neuron models.

Implements the computational primitives of individual biological neurons:
  - LIFNeuron:          Leaky Integrate-and-Fire dynamics (membrane potential,
                        threshold, reset, leak) with surrogate-gradient spikes
  - AdaptiveLIFNeuron:  LIF + spike-frequency adaptation (adapting threshold,
                        as seen in cortical pyramidal cells)
  - DendriticNeuron:    Multi-compartment dendrites with local NMDA-style
                        nonlinearities — a single neuron as a 2-layer network
                        (Poirazi & Mel, 2003; Gidon et al., Science 2020)

These let the network compute with temporal, event-driven signals rather than
only static activations — the substrate biological brains actually use.

References:
  Gerstner & Kistler (2002) "Spiking Neuron Models"
  Neftci et al. (2019) "Surrogate Gradient Learning in Spiking Neural Networks"
  Gidon et al. (2020) "Dendritic action potentials and computation in human
    layer 2/3 cortical neurons", Science 367:83-87
"""

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    """
    Heaviside spike (forward) with a smooth surrogate gradient (backward).
    The biological spike is a hard threshold; the surrogate (fast sigmoid
    derivative) makes it differentiable for backprop.
    """

    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(membrane, threshold)
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold = ctx.saved_tensors
        # Fast-sigmoid surrogate: 1 / (1 + |v - thr| * beta)^2
        beta = 10.0
        diff = membrane - threshold
        surrogate = 1.0 / (1.0 + beta * diff.abs()) ** 2
        return grad_output * surrogate, None


spike_fn = SurrogateSpike.apply


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer.

    Membrane dynamics (discrete):
        v[t] = leak * v[t-1] + input[t]
        spike = 1 if v[t] >= threshold else 0
        v[t] = v[t] * (1 - spike)        # reset after spiking

    Processes an input sequence (B, T, D) one timestep at a time and returns
    the spike train. Hidden state is carried across timesteps within forward.
    """

    def __init__(self, dim: int, leak: float = 0.9, threshold: float = 1.0):
        super().__init__()
        self.dim = dim
        self.leak = leak
        self.threshold = nn.Parameter(torch.full((dim,), float(threshold)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) input current sequence
        Returns:
            spikes: (B, T, D) binary spike train
            v_final: (B, D) final membrane potential
        """
        B, T, D = x.shape
        v = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        spikes = []
        for t in range(T):
            v = self.leak * v + x[:, t, :]
            s = spike_fn(v, self.threshold)
            v = v * (1.0 - s)            # soft reset
            spikes.append(s)
        spike_train = torch.stack(spikes, dim=1)   # (B, T, D)
        return spike_train, v


class AdaptiveLIFNeuron(nn.Module):
    """
    LIF neuron with SPIKE-FREQUENCY ADAPTATION.

    After each spike the effective threshold rises (adaptation current), then
    decays back. This makes neurons fire less under sustained input — a
    hallmark of cortical pyramidal cells that implements gain control and
    novelty detection in time.
    """

    def __init__(
        self,
        dim: int,
        leak: float = 0.9,
        base_threshold: float = 1.0,
        adapt_increment: float = 0.5,
        adapt_decay: float = 0.95,
    ):
        super().__init__()
        self.dim = dim
        self.leak = leak
        self.adapt_increment = adapt_increment
        self.adapt_decay = adapt_decay
        self.base_threshold = nn.Parameter(torch.full((dim,), float(base_threshold)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        v = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        adapt = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        spikes = []
        for t in range(T):
            v = self.leak * v + x[:, t, :]
            thr = self.base_threshold + adapt
            s = spike_fn(v, thr)
            v = v * (1.0 - s)
            adapt = self.adapt_decay * adapt + self.adapt_increment * s
            spikes.append(s)
        return torch.stack(spikes, dim=1), adapt


class DendriticNeuron(nn.Module):
    """
    Multi-compartment DENDRITIC neuron.

    A biological neuron is not a point: each dendritic branch performs a local
    nonlinear computation (NMDA spike) before the soma integrates the branches.
    This makes a single neuron equivalent to a small 2-layer network.

    Architecture:
        input → K dendritic branches (each: linear + local nonlinearity)
              → soma integration (weighted sum + global nonlinearity)
    """

    def __init__(self, in_dim: int, out_dim: int, num_branches: int = 4):
        super().__init__()
        self.num_branches = num_branches
        # Each branch reads the input independently
        self.branches = nn.Linear(in_dim, out_dim * num_branches)
        # Soma integrates the branch outputs
        self.soma = nn.Linear(out_dim * num_branches, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_dim)
        Returns:
            (..., out_dim)
        """
        branch_pre = self.branches(x)                       # (..., out*K)
        # Local dendritic nonlinearity (NMDA-style supralinear): tanh saturates,
        # then we square the positive part for supralinear dendritic boosting
        branch_act = torch.tanh(branch_pre)
        branch_act = branch_act + 0.5 * torch.relu(branch_pre) ** 2 * 0.1
        soma = self.soma(branch_act)                        # (..., out_dim)
        return torch.relu(soma)                             # somatic spike rate
