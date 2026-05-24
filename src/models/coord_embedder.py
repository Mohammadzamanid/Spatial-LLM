"""
src/models/coord_embedder.py
Fourier-feature positional encoding for lat/lon coordinate pairs.
Treats coordinates as continuous values, NOT raw text tokens.
"""

import torch
import torch.nn as nn
import math


class CoordinateEmbedder(nn.Module):
    """
    Encodes (lat, lon) pairs using Fourier feature mapping.
    Produces embeddings of shape (B, embed_dim) that can be fused
    with LLM hidden states via the SpatialFusionLayer.

    Why Fourier features?
    - Standard linear projection can't represent high-freq spatial patterns.
    - Random Fourier features approximate an RBF kernel, enabling the model
      to learn fine-grained spatial distinctions at any scale.
    """

    def __init__(self, embed_dim: int = 256, num_freqs: int = 64, learnable: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_freqs = num_freqs

        # Frequency bands: 2^0 ... 2^(num_freqs-1)
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)

        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

        # Input: sin/cos × lat/lon = num_freqs * 4 features
        self.proj = nn.Sequential(
            nn.Linear(num_freqs * 4, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, 2) tensor of [lat, lon] in degrees
        Returns:
            embeddings: (B, embed_dim)
        """
        # Normalize to radians
        coords_rad = coords * (math.pi / 180.0)

        lat = coords_rad[:, 0:1] * self.freqs.unsqueeze(0)  # (B, num_freqs)
        lon = coords_rad[:, 1:2] * self.freqs.unsqueeze(0)

        features = torch.cat(
            [lat.sin(), lat.cos(), lon.sin(), lon.cos()], dim=-1
        )  # (B, num_freqs * 4)

        return self.proj(features)  # (B, embed_dim)


class CoordinateEmbedderWithTokens(CoordinateEmbedder):
    """
    Extends CoordinateEmbedder to produce a sequence of tokens
    (compatible with cross-attention over text tokens).
    Returns (B, num_tokens, embed_dim).
    """

    def __init__(self, embed_dim: int = 256, num_freqs: int = 64, num_tokens: int = 4):
        super().__init__(embed_dim, num_freqs)
        self.num_tokens = num_tokens
        self.token_proj = nn.Linear(embed_dim, embed_dim * num_tokens)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            tokens: (B, num_tokens, embed_dim)
        """
        emb = super().forward(coords)  # (B, embed_dim)
        tokens = self.token_proj(emb)  # (B, embed_dim * num_tokens)
        return tokens.view(tokens.shape[0], self.num_tokens, self.embed_dim)
