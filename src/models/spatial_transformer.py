"""Embodied Spatiotemporal LLM with hippocampal-inspired and multimodal components."""

import torch
import torch.nn as nn

from .gain_field import GainFieldTransform, DualStreamGating
from .hippocampal_core import HippocampalCore
from .episodic_memory import EpisodicMemory


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attention_mask=None):
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class EmbodiedSpatiotemporalLLM(nn.Module):
    """Spatial LLM with gain-field, hippocampal memory, and vector<->text mappings."""

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
        visual_dim=512,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.schema_embedding = nn.Embedding(n_schema_types, d_model)

        self.spatial_proj = nn.Linear(4, d_model)
        self.temporal_proj = nn.Linear(1, d_model)
        self.distance_proj = nn.Linear(1, d_model)
        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.text_to_vector = nn.Linear(d_model, visual_dim)

        self.gain_field = GainFieldTransform(d_model, n_reference_frames, dropout)
        self.dual_stream_gating = DualStreamGating(d_model, dropout)
        self.hippocampal_core = HippocampalCore(d_model=d_model)
        self.episodic_memory = EpisodicMemory(d_model=d_model, memory_slots=memory_slots)

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.velocity_head = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.schema_embedding.weight, std=0.02)

    def forward(
        self,
        input_ids,
        schema_types=None,
        spatial_features=None,
        temporal_distances=None,
        spatial_distances=None,
        velocity=None,
        visual_features=None,
        reference_frame_idx=None,
        attention_mask=None,
        memory_state=None,
    ):
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        if schema_types is None:
            schema_types = torch.zeros(bsz, seq_len, dtype=torch.long, device=device)
        x = x + self.schema_embedding(schema_types.clamp(0, self.schema_embedding.num_embeddings - 1))

        if spatial_features is None:
            spatial_features = torch.zeros(bsz, seq_len, 4, device=device)
        x = x + self.spatial_proj(spatial_features)

        if temporal_distances is None:
            temporal_distances = torch.ones(bsz, seq_len, device=device)
        x = x + self.temporal_proj(temporal_distances.unsqueeze(-1))

        if spatial_distances is None:
            spatial_distances = torch.zeros(bsz, seq_len, device=device)
        x = x + self.distance_proj((spatial_distances / 50.0).unsqueeze(-1))

        if visual_features is not None:
            if visual_features.dim() == 2:
                visual_features = visual_features.unsqueeze(1).expand(-1, seq_len, -1)
            x = x + self.visual_proj(visual_features)

        x = self.dropout(x)

        if velocity is None:
            velocity = spatial_features[..., 2:4]
        dt = temporal_distances.unsqueeze(-1)
        hippo_embed, hippo_aux = self.hippocampal_core(velocity=velocity, dt=dt, correction=x)
        egocentric = x + hippo_embed

        allocentric, gain = self.gain_field(egocentric, reference_frame_idx)
        fused, gate_vals = self.dual_stream_gating(egocentric, allocentric)
        x, new_memory_state, memory_aux = self.episodic_memory(fused, memory_state=memory_state)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        pred_velocity = self.velocity_head(x)
        pred_vector = self.text_to_vector(x)

        aux_outputs = {
            "gain": gain,
            "gate_values": gate_vals,
            "egocentric_stream": egocentric,
            "allocentric_stream": allocentric,
            "hippocampal": hippo_aux,
            "memory": memory_aux,
            "pred_velocity": pred_velocity,
            "pred_vector": pred_vector,
            "memory_state": new_memory_state,
        }
        return logits, aux_outputs
