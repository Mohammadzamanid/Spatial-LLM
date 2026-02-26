"""Embodied Spatiotemporal LLM with hippocampal-inspired components."""

import torch
import torch.nn as nn

from .gain_field import GainFieldTransform, DualStreamGating
from .hippocampal_core import HippocampalCore
from .episodic_memory import EpisodicMemory


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
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class EmbodiedSpatiotemporalLLM(nn.Module):
    """Spatial LLM with gain-field and hippocampal-inspired memory dynamics."""

    def __init__(
        self,
        vocab_size=50257,
        d_model=512,
        n_layers=12,
        n_heads=8,
        n_reference_frames=8,
        max_seq_len=2048,
        dropout=0.1,
        n_schema_types=8,
        memory_slots=256,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.schema_embedding = nn.Embedding(n_schema_types, d_model)

        # Spatio-temporal feature projections
        self.spatial_proj = nn.Linear(4, d_model)
        self.temporal_proj = nn.Linear(1, d_model)
        self.distance_proj = nn.Linear(1, d_model)

        # Spatial components
        self.gain_field = GainFieldTransform(d_model, n_reference_frames, dropout)
        self.dual_stream_gating = DualStreamGating(d_model, dropout)

        # Hippocampal-inspired core and memory
        self.hippocampal_core = HippocampalCore(d_model=d_model)
        self.episodic_memory = EpisodicMemory(d_model=d_model, memory_slots=memory_slots)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

        # Auxiliary prediction heads
        self.velocity_head = nn.Linear(d_model, 2)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.schema_embedding.weight, std=0.02)
        for proj in [self.spatial_proj, self.temporal_proj, self.distance_proj, self.velocity_head]:
            nn.init.xavier_uniform_(proj.weight, gain=0.1)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
        self,
        input_ids,
        schema_types=None,
        spatial_features=None,
        temporal_distances=None,
        spatial_distances=None,
        velocity=None,
        reference_frame_idx=None,
        attention_mask=None,
        memory_state=None,
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)

        if schema_types is None:
            schema_types = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        schema_embeds = self.schema_embedding(schema_types.clamp_min(0).clamp_max(self.schema_embedding.num_embeddings - 1))

        if spatial_features is None:
            spatial_features = torch.zeros(batch_size, seq_len, 4, device=device)
        spatial_embeds = self.spatial_proj(spatial_features)

        if temporal_distances is None:
            temporal_distances = torch.zeros(batch_size, seq_len, device=device)
        temporal_embeds = self.temporal_proj(temporal_distances.unsqueeze(-1))

        if spatial_distances is None:
            spatial_distances = torch.zeros(batch_size, seq_len, device=device)
        distance_embeds = self.distance_proj(spatial_distances.unsqueeze(-1) / 50.0)

        egocentric = self.dropout(
            token_embeds + pos_embeds + schema_embeds + spatial_embeds + temporal_embeds + distance_embeds
        )

        # Use velocity from input if present; otherwise derive from spatial_features[...,:2]
        if velocity is None:
            velocity = spatial_features[..., :2]

        dt = temporal_distances.unsqueeze(-1)
        if torch.all(dt == 0):
            dt = torch.ones_like(dt)

        hippo_embed, hippo_aux = self.hippocampal_core(velocity=velocity, dt=dt, correction=egocentric)
        egocentric = egocentric + hippo_embed

        # Gain field: egocentric -> allocentric
        allocentric, gain = self.gain_field(egocentric, reference_frame_idx)

        # Fuse dual streams
        fused, gate_vals = self.dual_stream_gating(egocentric, allocentric)

        # Episodic memory read/write
        x, new_memory_state, memory_aux = self.episodic_memory(fused, memory_state=memory_state)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        pred_velocity = self.velocity_head(x)

        aux_outputs = {
            "gain": gain,
            "gate_values": gate_vals,
            "attention_weights": [],
            "egocentric_stream": egocentric,
            "allocentric_stream": allocentric,
            "hippocampal": hippo_aux,
            "memory": memory_aux,
            "pred_velocity": pred_velocity,
            "memory_state": new_memory_state,
        }

        return logits, aux_outputs
