"""Gain Field Coordinate Transformation Module (Numerically Stable)."""

import torch
import torch.nn as nn


class GainFieldTransform(nn.Module):
    """Transforms egocentric representations to allocentric via gain field modulation."""

    def __init__(self, d_model=512, n_reference_frames=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_frames = n_reference_frames

        self.reference_frames = nn.Embedding(n_reference_frames, d_model)
        nn.init.normal_(self.reference_frames.weight, mean=0.0, std=0.02)

        self.gain_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
        )
        for module in self.gain_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.ego_to_allo = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.ego_to_allo.weight, gain=0.01)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, egocentric_repr, reference_frame_idx=None, return_attention=False):
        batch_size, seq_len, _ = egocentric_repr.shape

        if reference_frame_idx is None:
            reference_frame_idx = torch.zeros(
                batch_size, dtype=torch.long, device=egocentric_repr.device
            )

        ref_embed = self.reference_frames(reference_frame_idx)
        ref_embed = ref_embed.unsqueeze(1).expand(-1, seq_len, -1)

        concat = torch.cat([egocentric_repr, ref_embed], dim=-1)
        gain = self.gain_network(concat)
        gain = torch.sigmoid(gain) + 0.5  # Output range: [0.5, 1.5]
        gain = torch.clamp(gain, 0.5, 1.5)

        modulated = egocentric_repr * gain
        allocentric = self.ego_to_allo(modulated)
        allocentric = self.layer_norm(allocentric)
        allocentric = self.dropout(allocentric)
        allocentric = torch.nan_to_num(allocentric, nan=0.0, posinf=1.0, neginf=-1.0)

        if return_attention:
            frame_probs = torch.zeros(
                batch_size, self.n_frames, device=egocentric_repr.device
            )
            frame_probs.scatter_(1, reference_frame_idx.unsqueeze(1), 1.0)
            return allocentric, gain, frame_probs

        return allocentric, gain


class DualStreamGating(nn.Module):
    """Gates between egocentric and allocentric streams."""

    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, 1),
            nn.Sigmoid(),
        )
        for module in self.gate_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.5)

        self.output_proj = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.01)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, egocentric_stream, allocentric_stream):
        concat = torch.cat([egocentric_stream, allocentric_stream], dim=-1)
        gate = self.gate_network(concat)
        gate = torch.clamp(gate, 0.01, 0.99)

        fused = gate * allocentric_stream + (1 - gate) * egocentric_stream
        fused = self.layer_norm(fused)
        fused = self.output_proj(fused)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=-1.0)

        return fused, gate
