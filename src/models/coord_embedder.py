"""
src/models/coord_embedder.py
Fourier-feature positional encoding for geographic coordinates.

Supports 2D (lat, lon) or 3D (lat, lon, elevation). Each physical dimension is
normalized according to its nature BEFORE the Fourier mapping:
  - lat, lon : angular  -> radians
  - elevation: linear m -> divided by a reference scale, then scaled to radian-like range
Applying the lat/lon radian conversion to elevation (a naive extension) would
corrupt that channel, so normalization is per-channel.

Note: "time" is intentionally NOT a coordinate channel here. Time-as-history
(learning across a sequence of events) is a separate mechanism (recurrent state /
curriculum), not a static input dimension.
"""

import torch
import torch.nn as nn
import math


class CoordinateEmbedder(nn.Module):
    """
    Encodes geographic coordinates using Fourier feature mapping.
    input_dim=2 -> (lat, lon);  input_dim=3 -> (lat, lon, elevation).
    Produces (B, embed_dim).
    """

    def __init__(self, embed_dim: int = 256, num_freqs: int = 64,
                 input_dim: int = 2, elev_scale: float = 8849.0,
                 learnable: bool = False):
        super().__init__()
        assert input_dim in (2, 3), "input_dim must be 2 (lat,lon) or 3 (lat,lon,elev)"
        self.embed_dim = embed_dim
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        self.elev_scale = elev_scale  # ~Everest; normalizes elevation to ~[0,1]

        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

        # sin & cos per channel per frequency
        in_features = num_freqs * 2 * input_dim
        self.proj = nn.Sequential(
            nn.Linear(in_features, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def _normalize(self, coords: torch.Tensor) -> torch.Tensor:
        """Per-channel normalization to comparable ranges before Fourier mapping."""
        lat = coords[:, 0:1] * (math.pi / 180.0)
        lon = coords[:, 1:2] * (math.pi / 180.0)
        if self.input_dim == 2:
            return torch.cat([lat, lon], dim=-1)
        z = (coords[:, 2:3] / self.elev_scale) * math.pi
        return torch.cat([lat, lon, z], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, >=input_dim) tensor; columns [lat, lon, (elev)]
                    Extra columns are ignored; missing ones are zero-padded.
        Returns:
            (B, embed_dim)
        """
        if coords.shape[-1] < self.input_dim:
            pad = torch.zeros(coords.shape[0], self.input_dim - coords.shape[-1],
                              device=coords.device, dtype=coords.dtype)
            coords = torch.cat([coords, pad], dim=-1)

        norm = self._normalize(coords)                              # (B, input_dim)
        scaled = norm.unsqueeze(-1) * self.freqs.view(1, 1, -1)     # (B, input_dim, F)
        scaled = scaled.flatten(1)                                  # (B, input_dim*F)
        features = torch.cat([scaled.sin(), scaled.cos()], dim=-1)  # (B, 2*input_dim*F)
        return self.proj(features)


class CoordinateEmbedderWithTokens(CoordinateEmbedder):
    """
    Produces a sequence of tokens (B, num_tokens, embed_dim) for cross-attention.
    """

    def __init__(self, embed_dim: int = 256, num_freqs: int = 64,
                 num_tokens: int = 4, input_dim: int = 2,
                 elev_scale: float = 8849.0):
        super().__init__(embed_dim, num_freqs, input_dim=input_dim,
                         elev_scale=elev_scale)
        self.num_tokens = num_tokens
        self.token_proj = nn.Linear(embed_dim, embed_dim * num_tokens)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        emb = super().forward(coords)                  # (B, embed_dim)
        tokens = self.token_proj(emb)                  # (B, embed_dim * num_tokens)
        return tokens.view(tokens.shape[0], self.num_tokens, self.embed_dim)
