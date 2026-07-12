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
    spatial_tokens (B, S, D) — from tile encoder + coordinate embedder

    The text tokens attend OVER the spatial tokens, so the model
    can selectively pull in visual/geographic context per token.

    Gating
    ------
    Flamingo-style tanh gates, zero-initialized, so at the start of training the
    block is an identity (fused == text) and generation stays coherent; the model
    opens the gates only as far as the spatial signal actually helps.

    ``num_spatial_groups == 1`` (default): a single ``attn_gate`` scales the whole
    spatial blend — the original shared-gate behaviour. Existing checkpoints load
    unchanged (the parameter is still shape (1,)).

    ``num_spatial_groups > 1``: each spatial *module* (coordinate/elevation, grid
    cells, place-cell memory, tile tokens …) gets its OWN gate. Every group is
    attended independently and summed, each scaled by its own scalar, so the model
    learns to weight grid cells vs elevation vs place memory per task — and the
    trained gate values become a direct read-out of which module each task leaned
    on. The feed-forward gate stays shared (the FFN transforms the text state, not
    a specific spatial module). The zero-init identity is preserved exactly, since
    every gated term is ``0 * attended == 0`` at init.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_spatial_groups: int = 1,
        gate_init: float = 0.0,
        rsc_split: bool = False,
        perforant: bool = False,
    ):
        super().__init__()
        self.num_spatial_groups = num_spatial_groups
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # Normalize spatial tokens before they enter cross-attention. Without this
        # they arrive ~27x larger than the LLM's input embeddings and bury the text.
        self.norm_spatial = nn.LayerNorm(hidden_dim)
        # One tanh gate per spatial group (a single shared gate when groups == 1).
        # gate_init=0 -> the block starts as an identity (fused == text), keeping
        # generation coherent (used by SpatialLLM). gate_init>0 opens the spatial
        # pathway from step 0 — needed when the answer depends ENTIRELY on the spatial
        # channel (e.g. TrajectoryLLM), otherwise the zero gate gives no gradient to open.
        self.attn_gate = nn.Parameter(torch.full((num_spatial_groups,), float(gate_init)))
        self.ffn_gate = nn.Parameter(torch.full((1,), float(gate_init)))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # ── RSC bifurcation (optional) — two separately-gated OUTPUT pathways ──────
        # The retrosplenial cortex does not forward one unified map: M2-projecting
        # neurons carry an egocentric ACTION-affordance stream (to motor cortex),
        # AD-projecting neurons an allocentric location-MEMORY stream (to thalamus),
        # each independently lesionable (Molecular Psychiatry 2024). rsc_split gives
        # the injection two independently-gated read-outs of the spatial map — an
        # action pathway and a memory pathway — so downstream layers can weight, and
        # a probe can lesion, each on its own. Both gates follow the same zero-init
        # convention, so the block is still an exact identity at start.
        # Emergent reference-frame dissociation: src/eval/rsc_routing.py.
        self.rsc_split = bool(rsc_split)
        if self.rsc_split:
            self.action_proj = nn.Linear(hidden_dim, hidden_dim)
            self.memory_proj = nn.Linear(hidden_dim, hidden_dim)
            self.action_gate = nn.Parameter(torch.full((1,), float(gate_init)))
            self.memory_gate = nn.Parameter(torch.full((1,), float(gate_init)))

        # ── Perforant semantic input (optional) ───────────────────────────────────
        # The perforant path projects non-spatial / behaviourally-relevant features
        # directly into grid & place assemblies, so the map is not purely geographic —
        # it warps by relevant concepts, mixed-selective to meaning and space
        # (Boccara 2019). When ``semantic_tokens`` are supplied, this gated pathway
        # lets the text pull that semantic structure (bound alongside space) from the
        # map. perforant_gate is zero-init, so with no semantic tokens (or at init)
        # it changes nothing. Emergent metric warp: src/eval/semantic_warp.py.
        self.perforant = bool(perforant)
        if self.perforant:
            self.norm_perforant = nn.LayerNorm(hidden_dim)
            self.perforant_attn = nn.MultiheadAttention(
                hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.perforant_proj = nn.Linear(hidden_dim, hidden_dim)
            self.perforant_gate = nn.Parameter(torch.full((1,), float(gate_init)))

    def _attend(
        self,
        query: torch.Tensor,
        keyval: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attended, _ = self.cross_attn(
            query=query, key=keyval, value=keyval,
            key_padding_mask=key_padding_mask,
        )
        return attended

    def forward(
        self,
        text_hidden: torch.Tensor,
        spatial_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        group_sizes: list[int] | None = None,
        semantic_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            text_hidden:     (B, T, D) LLM hidden states
            spatial_tokens:  (B, S, D) spatial feature tokens
            key_padding_mask: (B, S) optional mask for spatial tokens
            group_sizes:     optional per-module token counts summing to S. When
                             given and the layer has >1 gate, each module is
                             attended + gated independently. Ignored for a
                             single-gate (shared) layer.
            semantic_tokens: (B, S_sem, D) optional non-spatial / concept features
                             for the perforant pathway (only used when perforant=True).
        Returns:
            fused: (B, T, D) enriched text hidden states
        """
        # Cross-attention: text queries spatial (spatial normalized to text scale)
        spatial_norm = self.norm_spatial(spatial_tokens)
        query = self.norm1(text_hidden)

        if self.rsc_split:
            # RSC bifurcation: one attention read of the spatial map -> TWO
            # separately-gated output streams (action, memory). Uses the shared
            # spatial blend (the split is output-side, orthogonal to input grouping).
            # Zero-init gates keep the block an exact identity at start.
            attended = self._attend(query, spatial_norm, key_padding_mask)
            text_hidden = (
                text_hidden
                + self.action_gate.tanh() * self.action_proj(attended)
                + self.memory_gate.tanh() * self.memory_proj(attended)
            )
        elif self.num_spatial_groups > 1 and group_sizes is not None:
            # Per-module gating: attend each spatial group on its own, scale by that
            # group's gate, and sum. Every gated term is 0 at init (gate=0), so the
            # LLM still sees its pristine embeddings and generates coherently;
            # training opens each module's gate as far as it helps that task.
            attn_update = torch.zeros_like(text_hidden)
            start = 0
            for g, size in enumerate(group_sizes):
                if size <= 0:
                    continue
                end = start + size
                group = spatial_norm[:, start:end, :]
                group_mask = (
                    key_padding_mask[:, start:end]
                    if key_padding_mask is not None else None
                )
                attended = self._attend(query, group, group_mask)
                # clamp the gate index in case more groups are supplied than the
                # layer was sized for (extra groups reuse the last gate)
                gate_idx = min(g, self.num_spatial_groups - 1)
                attn_update = attn_update + self.attn_gate[gate_idx].tanh() * attended
                start = end
            text_hidden = text_hidden + attn_update
        else:
            # Shared single gate over the whole spatial blend (original behaviour).
            attended = self._attend(query, spatial_norm, key_padding_mask)
            text_hidden = text_hidden + self.attn_gate[0].tanh() * attended

        # Perforant semantic pathway (optional): a gated read of non-spatial concept
        # features, so the text pulls semantic structure bound alongside space.
        # Skipped when no semantic tokens are supplied, so it is fully backward
        # compatible; zero-init gate makes it an identity even when they are.
        if self.perforant and semantic_tokens is not None:
            sem_norm = self.norm_perforant(semantic_tokens)
            attended_sem, _ = self.perforant_attn(query=query, key=sem_norm, value=sem_norm)
            text_hidden = text_hidden + self.perforant_gate.tanh() * self.perforant_proj(attended_sem)

        # Gated feed-forward — shared, since it transforms the text state rather
        # than any one spatial module (same reasoning: don't swamp the tiny inputs).
        text_hidden = text_hidden + self.ffn_gate.tanh() * self.ffn(self.norm2(text_hidden))

        return text_hidden


class MultiScaleSpatialFusion(nn.Module):
    """
    Stacks N fusion layers for deeper spatial-language integration.
    Useful when spatial context needs multiple rounds of refinement.

    ``num_spatial_groups`` is forwarded to every layer so per-module gating can be
    toggled for the whole stack at once. ``rsc_split`` (action/memory output
    pathways) and ``perforant`` (semantic input pathway) are likewise forwarded to
    every layer; both default off, so the stack is unchanged unless requested.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        num_spatial_groups: int = 1,
        gate_init: float = 0.0,
        rsc_split: bool = False,
        perforant: bool = False,
    ):
        super().__init__()
        self.num_spatial_groups = num_spatial_groups
        self.rsc_split = rsc_split
        self.perforant = perforant
        self.layers = nn.ModuleList([
            SpatialFusionLayer(hidden_dim, num_heads, num_spatial_groups=num_spatial_groups,
                               gate_init=gate_init, rsc_split=rsc_split, perforant=perforant)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        text_hidden: torch.Tensor,
        spatial_tokens: torch.Tensor,
        group_sizes: list[int] | None = None,
        semantic_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            text_hidden = layer(text_hidden, spatial_tokens, group_sizes=group_sizes,
                                semantic_tokens=semantic_tokens)
        return text_hidden
