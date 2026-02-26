"""Brain-inspired hippocampal core modules.

Implements simplified but numerically stable modules for:
- head direction coding
- grid-like periodic spatial coding
- place-cell readout
- path integration state update
"""

import math
import torch
import torch.nn as nn


class HeadDirectionModule(nn.Module):
    """Encodes 2D velocity vectors into a head-direction representation."""

    def __init__(self, d_model=256, n_directions=32):
        super().__init__()
        self.n_directions = n_directions
        angles = torch.linspace(0, 2 * math.pi, n_directions + 1)[:-1]
        self.register_buffer("dir_x", torch.cos(angles), persistent=False)
        self.register_buffer("dir_y", torch.sin(angles), persistent=False)
        self.proj = nn.Linear(n_directions, d_model)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, velocity):
        """velocity: (batch, seq, 2)"""
        vx = velocity[..., 0].unsqueeze(-1)
        vy = velocity[..., 1].unsqueeze(-1)
        alignment = vx * self.dir_x + vy * self.dir_y
        hd = torch.softmax(alignment, dim=-1)
        return self.proj(hd), hd


class GridCellModule(nn.Module):
    """Multi-scale periodic coding over integrated 2D positions."""

    def __init__(self, d_model=256, n_scales=6):
        super().__init__()
        scales = torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0, 16.0][:n_scales], dtype=torch.float32)
        self.register_buffer("scales", scales, persistent=False)
        self.proj = nn.Linear(2 * n_scales * 2, d_model)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, position):
        """position: (batch, seq, 2)"""
        phases = []
        for s in self.scales:
            scaled = position / s
            phases.extend([torch.sin(scaled), torch.cos(scaled)])
        feat = torch.cat(phases, dim=-1)
        return self.proj(feat), feat


class PlaceCellReadout(nn.Module):
    """Soft assignment to learned place prototypes."""

    def __init__(self, d_model=256, n_place_cells=128, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        self.place_prototypes = nn.Parameter(torch.randn(n_place_cells, d_model) * 0.02)
        self.proj = nn.Linear(n_place_cells, d_model)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, state):
        norm_state = nn.functional.normalize(state, dim=-1)
        norm_proto = nn.functional.normalize(self.place_prototypes, dim=-1)
        logits = torch.matmul(norm_state, norm_proto.t()) / max(self.temperature, 1e-4)
        probs = torch.softmax(logits, dim=-1)
        place_embed = self.proj(probs)
        return place_embed, probs


class PathIntegrationSSM(nn.Module):
    """Simple state-space path integrator with correction and decay."""

    def __init__(self, d_model=256, decay=0.98):
        super().__init__()
        self.decay = decay
        self.vel_proj = nn.Linear(2, d_model)
        self.corr_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.vel_proj.weight, gain=0.1)
        nn.init.zeros_(self.vel_proj.bias)
        nn.init.xavier_uniform_(self.corr_proj.weight, gain=0.1)
        nn.init.zeros_(self.corr_proj.bias)

    def forward(self, velocity, dt, correction):
        """Returns integrated latent state over sequence.

        velocity: (batch, seq, 2)
        dt: (batch, seq, 1)
        correction: (batch, seq, d_model)
        """
        bsz, seq_len, _ = velocity.shape
        state = torch.zeros(bsz, self.vel_proj.out_features, device=velocity.device, dtype=velocity.dtype)
        states = []
        for t in range(seq_len):
            delta = self.vel_proj(velocity[:, t]) * dt[:, t]
            corr = self.corr_proj(correction[:, t])
            state = self.decay * state + delta + 0.1 * corr
            state = self.norm(state)
            states.append(state)
        return torch.stack(states, dim=1)


class HippocampalCore(nn.Module):
    """Composed hippocampal-style core producing ego/allo codes."""

    def __init__(
        self,
        d_model=256,
        n_directions=32,
        n_grid_scales=6,
        n_place_cells=128,
        decay=0.98,
    ):
        super().__init__()
        self.hd = HeadDirectionModule(d_model=d_model, n_directions=n_directions)
        self.grid = GridCellModule(d_model=d_model, n_scales=n_grid_scales)
        self.integrator = PathIntegrationSSM(d_model=d_model, decay=decay)
        self.place = PlaceCellReadout(d_model=d_model, n_place_cells=n_place_cells)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, velocity, dt, correction):
        hd_embed, hd_probs = self.hd(velocity)

        position = torch.cumsum(velocity * dt, dim=1)
        grid_embed, grid_feat = self.grid(position)

        integrated = self.integrator(velocity, dt, correction)
        place_embed, place_probs = self.place(integrated)

        fused = self.fuse(torch.cat([hd_embed, grid_embed, integrated, place_embed], dim=-1))
        fused = self.norm(fused)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=-1.0)

        aux = {
            "hd_probs": hd_probs,
            "grid_features": grid_feat,
            "place_probs": place_probs,
            "integrated_state": integrated,
            "position": position,
        }
        return fused, aux
