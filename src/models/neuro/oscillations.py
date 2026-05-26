"""
src/models/neuro/oscillations.py

NETWORK DYNAMICS LEVEL — brain rhythms and temporal coding.

The brain doesn't just use firing RATES — it uses the TIMING of spikes
relative to ongoing oscillations to encode information.

  - ThetaOscillator:    generates a theta rhythm (4–8 Hz) that gates processing
  - PhasePrecession:     hippocampal place cells fire at progressively earlier
                         theta phases as the animal crosses a place field —
                         encodes position WITHIN the rhythm (O'Keefe & Recce 1993)
  - ThetaGammaCoupling:  nested gamma cycles within theta encode ordered
                         sequences (Lisman & Idiart, 1995) — a working-memory
                         buffer of ~7 items
  - SharpWaveRipple:     offline replay events that consolidate memories from
                         hippocampus to neocortex (during rest/sleep)

References:
  O'Keefe & Recce (1993) "Phase relationship between hippocampal place units
    and the EEG theta rhythm"
  Lisman & Idiart (1995) "Storage of 7 ± 2 short-term memories in oscillatory
    subcycles", Science
  Buzsáki (2015) "Hippocampal sharp wave-ripple"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThetaOscillator(nn.Module):
    """
    Generates a theta-band oscillatory gate over a sequence.
    The oscillation modulates which timesteps are emphasised, mimicking how
    theta rhythm gates information flow in the hippocampus.
    """

    def __init__(self, dim: int, freq: float = 6.0, learnable_phase: bool = True):
        super().__init__()
        self.dim = dim
        self.freq = freq
        if learnable_phase:
            self.phase = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("phase", torch.zeros(dim))

    def forward(self, x: torch.Tensor, dt: float = 0.02) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) sequence
            dt: timestep in seconds (0.02 = 50 Hz sampling)
        Returns:
            (B, T, D) theta-gated sequence
        """
        B, T, D = x.shape
        t = torch.arange(T, device=x.device, dtype=x.dtype) * dt   # (T,)
        # theta wave per dimension: 0.5*(1+cos) keeps gate in [0,1]
        wave = 0.5 * (1.0 + torch.cos(
            2 * math.pi * self.freq * t.unsqueeze(1) + self.phase.unsqueeze(0)
        ))  # (T, D)
        return x * wave.unsqueeze(0)


class PhasePrecession(nn.Module):
    """
    Theta phase precession encoder.

    Encodes a continuous position-within-field as a theta PHASE. As position
    advances 0→1 through a field, the encoded phase precesses 2π→0. The output
    is a (sin, cos) phase embedding projected to the model dimension — giving
    the LLM a temporal-phase code for fine spatial position.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(2, embed_dim)

    def forward(self, position_in_field: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position_in_field: (B,) values in [0, 1]
        Returns:
            (B, embed_dim)
        """
        # Phase precesses from 2π down to 0 across the field
        phase = 2 * math.pi * (1.0 - position_in_field.clamp(0, 1))   # (B,)
        feats = torch.stack([phase.sin(), phase.cos()], dim=-1)        # (B, 2)
        return self.proj(feats)


class ThetaGammaCoupling(nn.Module):
    """
    Theta-gamma phase-amplitude coupling for sequence working memory.

    Implements the Lisman-Idiart model: ~7 gamma sub-cycles nested in one
    theta cycle, each holding one item. We slot a sequence of up to
    `num_slots` items into distinct gamma phases, providing an ordered
    short-term memory buffer.
    """

    def __init__(self, dim: int, num_slots: int = 7):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        # Learnable gamma-phase embeddings (one per slot)
        self.slot_phase = nn.Parameter(torch.randn(num_slots, dim) * 0.1)
        self.readout = nn.Linear(dim, dim)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """
        Args:
            items: (B, S, D) sequence of up to num_slots items
        Returns:
            (B, D) bound working-memory representation
        """
        B, S, D = items.shape
        s = min(S, self.num_slots)
        items = items[:, :s, :]
        # Tag each item with its gamma-slot phase, then sum (multiplexing)
        tagged = items + self.slot_phase[:s].unsqueeze(0)              # (B, s, D)
        bound = tagged.sum(dim=1)                                      # (B, D)
        return self.readout(bound)


class SharpWaveRipple(nn.Module):
    """
    Sharp-wave ripple REPLAY for memory consolidation.

    During rest, the hippocampus replays recent experience in compressed bursts,
    transferring memories to neocortex. Here, given a buffer of stored episodic
    states, we generate replayed sequences (optionally reversed, as seen
    biologically) and produce a consolidated representation that can be merged
    into slower cortical weights.
    """

    def __init__(self, dim: int, compression: int = 4):
        super().__init__()
        self.dim = dim
        self.compression = compression
        self.consolidator = nn.GRU(dim, dim, batch_first=True)
        self.out = nn.Linear(dim, dim)

    def forward(self, episodic_buffer: torch.Tensor,
                reverse: bool = False) -> torch.Tensor:
        """
        Args:
            episodic_buffer: (B, M, D) stored recent episodes
            reverse: replay in reverse order (reverse replay is common in SWRs)
        Returns:
            (B, D) consolidated memory trace
        """
        seq = episodic_buffer
        if reverse:
            seq = torch.flip(seq, dims=[1])
        # Temporal compression: subsample
        if seq.shape[1] > self.compression:
            idx = torch.linspace(0, seq.shape[1] - 1, self.compression,
                                 device=seq.device).long()
            seq = seq[:, idx, :]
        _, h = self.consolidator(seq)                                 # h: (1, B, D)
        return self.out(h.squeeze(0))                                 # (B, D)
