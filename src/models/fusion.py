"""
src/models/fusion.py
Cross-attention fusion layer that injects spatial tokens
(from tile encoder + coordinate embedder) into LLM hidden states.
"""

import torch
import torch.nn as nn


class SpatialFusionLayer(nn.Module):
    """
    Injects spatial context into LLM hidden states via cross-attention.

    text_hidden  (B, T, D)  — from LLM layers
    spatial_tokens (B, S, D) — from tile encoder + coord embedder

    The text tokens attend OVER the spatial tokens, so the model
    can selectively pull in visual/geographic context per token.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # Normalize spatial tokens before they enter cross-attention. Without this
        # they arrive ~27x larger than the LLM's input embeddings and bury the text.
        self.norm_spatial = nn.LayerNorm(hidden_dim)
        # Flamingo-style tanh gates, initialized at 0 so at the start of training the
        # block is an identity (fused == text), keeping generation coherent. The model
        # learns to "open" these gates only as far as the spatial signal actually helps.
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ffn_gate = nn.Parameter(torch.zeros(1))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        text_hidden: torch.Tensor,
        spatial_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            text_hidden:     (B, T, D) LLM hidden states
            spatial_tokens:  (B, S, D) spatial feature tokens
            key_padding_mask: (B, S) optional mask for spatial tokens
        Returns:
            fused: (B, T, D) enriched text hidden states
        """
        # Cross-attention: text queries spatial (spatial normalized to text scale)
        spatial_norm = self.norm_spatial(spatial_tokens)
        attended, _ = self.cross_attn(
            query=self.norm1(text_hidden),
            key=spatial_norm,
            value=spatial_norm,
            key_padding_mask=key_padding_mask,
        )
        # Gated residual: at init (gate=0) this adds nothing, so the LLM sees its
        # own pristine embeddings and generates coherently; training opens the gate.
        text_hidden = text_hidden + self.attn_gate.tanh() * attended

        # Gated feed-forward (same reasoning — don't swamp the tiny input embeddings)
        text_hidden = text_hidden + self.ffn_gate.tanh() * self.ffn(self.norm2(text_hidden))

        return text_hidden


class MultiScaleSpatialFusion(nn.Module):
    """
    Stacks N fusion layers for deeper spatial-language integration.
    Useful when spatial context needs multiple rounds of refinement.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            SpatialFusionLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        text_hidden: torch.Tensor,
        spatial_tokens: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            text_hidden = layer(text_hidden, spatial_tokens)
        return text_hidden
