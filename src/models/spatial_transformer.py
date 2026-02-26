"""Embodied Spatiotemporal LLM with Gain Field Transform."""

import torch
import torch.nn as nn

from .gain_field import GainFieldTransform, DualStreamGating


class TransformerBlock(nn.Module):
    """Standard transformer block with multi-head attention and FFN."""

    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attention_mask=None):
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class EmbodiedSpatiotemporalLLM(nn.Module):
    """Spatial LLM with gain-field coordinate transformations.

    Maintains dual egocentric/allocentric streams fused via learned gating.
    """

    def __init__(
        self,
        vocab_size=50257,
        d_model=512,
        n_layers=12,
        n_heads=8,
        n_reference_frames=8,
        max_seq_len=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Spatial components
        self.gain_field = GainFieldTransform(d_model, n_reference_frames, dropout)
        self.dual_stream_gating = DualStreamGating(d_model, dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(
        self,
        input_ids,
        schema_types=None,
        spatial_features=None,
        temporal_distances=None,
        spatial_distances=None,
        reference_frame_idx=None,
        attention_mask=None,
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        egocentric = self.dropout(token_embeds + pos_embeds)

        # Gain field: egocentric -> allocentric
        allocentric, gain = self.gain_field(egocentric, reference_frame_idx)

        # Fuse dual streams
        x, gate_vals = self.dual_stream_gating(egocentric, allocentric)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        aux_outputs = {
            "gain": gain,
            "gate_values": gate_vals,
            "attention_weights": [],
            "egocentric_stream": egocentric,
            "allocentric_stream": allocentric,
        }

        return logits, aux_outputs
