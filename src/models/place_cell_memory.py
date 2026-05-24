"""
src/models/place_cell_memory.py

Place Cell Memory — inspired by the hippocampus.

Biological basis:
  Hippocampal place cells fire when an animal is at a specific location.
  Each cell has a "place field" — a Gaussian bump of activity centred on
  its preferred location. The population vector of all place cells uniquely
  encodes any position. The hippocampus also supports episodic memory:
  binding spatial context to events.

Implementation:
  - A bank of learnable "place field centres" (anchor points in lat/lon space)
  - Gaussian activation: each input coord activates nearby centres strongly
  - Sparse competitive inhibition (k-WTA): only top-k cells fire (like biology)
  - Episodic buffer: stores (coord, hidden_state) pairs as short-term memory
  - Retrieval: given a new coord, retrieve the most similar past states

This gives the model a form of spatial working memory — it can recall
what it "saw" at nearby locations during the same inference session.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PlaceCellLayer(nn.Module):
    """
    Sparse place cell population encoding.
    Maps (B, 2) coordinates to (B, num_cells) sparse activations.
    """

    def __init__(
        self,
        num_cells: int = 512,
        sparsity_k: int = 50,          # top-k winner-take-all
        coord_bounds: tuple = (-90, 90, -180, 180),  # lat_min, lat_max, lon_min, lon_max
        learnable_centres: bool = True,
    ):
        super().__init__()
        self.num_cells = num_cells
        self.sparsity_k = sparsity_k

        lat_min, lat_max, lon_min, lon_max = coord_bounds
        # Initialise centres uniformly across the globe
        centres_lat = torch.FloatTensor(num_cells).uniform_(lat_min, lat_max)
        centres_lon = torch.FloatTensor(num_cells).uniform_(lon_min, lon_max)
        centres = torch.stack([centres_lat, centres_lon], dim=1)  # (num_cells, 2)

        if learnable_centres:
            self.centres = nn.Parameter(centres)
        else:
            self.register_buffer("centres", centres)

        # Learnable width per cell (in degrees)
        log_sigma = torch.zeros(num_cells)
        self.log_sigma = nn.Parameter(log_sigma) if learnable_centres else \
            self.register_buffer("log_sigma", log_sigma) or nn.Parameter(log_sigma)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, 2) lat/lon in degrees
        Returns:
            activations: (B, num_cells) sparse population code
        """
        sigma = self.log_sigma.exp().clamp(min=0.1)  # (num_cells,)

        # Euclidean distance in degree space (simplified; good enough for local regions)
        diff = coords.unsqueeze(1) - self.centres.unsqueeze(0)  # (B, num_cells, 2)
        dist_sq = (diff ** 2).sum(dim=-1)                        # (B, num_cells)

        # Gaussian activation
        activations = torch.exp(-0.5 * dist_sq / (sigma.unsqueeze(0) ** 2))  # (B, num_cells)

        # k-Winner-Take-All sparsity (biological inhibitory interneurons)
        if self.sparsity_k < self.num_cells:
            topk_vals, _ = activations.topk(self.sparsity_k, dim=-1)
            threshold = topk_vals[:, -1:].detach()  # (B, 1)
            activations = activations * (activations >= threshold).float()

        return activations  # (B, num_cells)


class HippocampalMemory(nn.Module):
    """
    Episodic spatial memory buffer.
    Stores (coord, context_vector) pairs and retrieves by spatial similarity.
    Mimics how the hippocampus binds place with content.
    """

    def __init__(
        self,
        embed_dim: int,
        num_cells: int = 512,
        buffer_size: int = 128,
        sparsity_k: int = 50,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.buffer_size = buffer_size

        self.place_cells = PlaceCellLayer(num_cells=num_cells, sparsity_k=sparsity_k)
        self.encoder = nn.Linear(num_cells, embed_dim)
        self.retrieval_proj = nn.Linear(embed_dim, embed_dim)

        # Episodic buffer (not trained — dynamic during inference)
        self.register_buffer("mem_keys", torch.zeros(buffer_size, num_cells))
        self.register_buffer("mem_vals", torch.zeros(buffer_size, embed_dim))
        self.register_buffer("mem_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("mem_filled", torch.tensor(False))

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coords to place cell embedding: (B, embed_dim)."""
        pc = self.place_cells(coords)      # (B, num_cells)
        return self.encoder(pc)            # (B, embed_dim)

    def store(self, coords: torch.Tensor, context: torch.Tensor):
        """
        Store coord+context pairs in the episodic buffer.
        Args:
            coords:  (B, 2)
            context: (B, embed_dim)
        """
        pc = self.place_cells(coords).detach()  # (B, num_cells)
        B = pc.shape[0]
        for i in range(B):
            ptr = self.mem_ptr.item()
            self.mem_keys[ptr] = pc[i]
            self.mem_vals[ptr] = context[i].detach()
            self.mem_ptr = (self.mem_ptr + 1) % self.buffer_size
        self.mem_filled = self.mem_filled | (self.mem_ptr == 0)

    def retrieve(self, coords: torch.Tensor, top_k: int = 4) -> torch.Tensor:
        """
        Retrieve most spatially similar memories.
        Args:
            coords: (B, 2)
        Returns:
            retrieved: (B, embed_dim) weighted sum of similar memories
        """
        pc_query = self.place_cells(coords)  # (B, num_cells)

        n_stored = self.buffer_size if self.mem_filled else self.mem_ptr.item()
        if n_stored == 0:
            return torch.zeros(coords.shape[0], self.embed_dim, device=coords.device)

        keys = self.mem_keys[:n_stored]   # (M, num_cells)
        vals = self.mem_vals[:n_stored]   # (M, embed_dim)

        # Cosine similarity
        sim = F.cosine_similarity(
            pc_query.unsqueeze(1),    # (B, 1, num_cells)
            keys.unsqueeze(0),        # (1, M, num_cells)
            dim=-1
        )  # (B, M)

        k = min(top_k, n_stored)
        topk_sim, topk_idx = sim.topk(k, dim=-1)    # (B, k)
        weights = F.softmax(topk_sim, dim=-1)         # (B, k)

        retrieved_vals = vals[topk_idx]               # (B, k, D)
        retrieved = (weights.unsqueeze(-1) * retrieved_vals).sum(dim=1)  # (B, D)

        return self.retrieval_proj(retrieved)          # (B, D)

    def forward(
        self,
        coords: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        store: bool = True,
    ) -> torch.Tensor:
        """
        Full forward: encode + optionally store + retrieve.
        Returns concatenated [place_encoding, retrieved_memory].
        """
        place_emb = self.encode(coords)             # (B, D)
        retrieved = self.retrieve(coords)           # (B, D)

        if store and context is not None:
            self.store(coords, context)

        return place_emb + retrieved                # (B, D)
