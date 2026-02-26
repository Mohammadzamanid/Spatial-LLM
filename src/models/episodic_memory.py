"""Differentiable episodic memory for fast write/read behavior."""

import torch
import torch.nn as nn


class EpisodicMemory(nn.Module):
    """Simple key-value memory with recency-biased writes and soft reads."""

    def __init__(self, d_model=256, memory_slots=256):
        super().__init__()
        self.d_model = d_model
        self.memory_slots = memory_slots
        self.key_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

        for layer in [self.key_proj, self.val_proj, self.query_proj, self.gate_proj]:
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def init_state(self, batch_size, device, dtype):
        keys = torch.zeros(batch_size, self.memory_slots, self.d_model, device=device, dtype=dtype)
        vals = torch.zeros_like(keys)
        age = torch.zeros(batch_size, self.memory_slots, device=device, dtype=dtype)
        return {"keys": keys, "vals": vals, "age": age}

    def forward(self, x, memory_state=None):
        """x: (batch, seq, d_model). Returns memory-augmented x and new state."""
        bsz, seq_len, d_model = x.shape
        if memory_state is None:
            memory_state = self.init_state(bsz, x.device, x.dtype)

        keys = memory_state["keys"]
        vals = memory_state["vals"]
        age = memory_state["age"]

        outputs = []
        read_weights_all = []
        write_slots_all = []

        for t in range(seq_len):
            xt = x[:, t]
            q = self.query_proj(xt)

            scores = torch.einsum("bd,bsd->bs", q, keys) / (d_model ** 0.5)
            read_weights = torch.softmax(scores, dim=-1)
            readout = torch.einsum("bs,bsd->bd", read_weights, vals)

            gate = torch.sigmoid(self.gate_proj(xt))
            out = self.norm(xt + gate * readout)
            outputs.append(out)
            read_weights_all.append(read_weights)

            # recency-biased write: replace oldest slot
            write_slot = torch.argmax(age, dim=-1)
            write_slots_all.append(write_slot)

            k_new = self.key_proj(xt)
            v_new = self.val_proj(xt)

            batch_idx = torch.arange(bsz, device=x.device)
            keys = keys.clone()
            vals = vals.clone()
            age = age + 1
            keys[batch_idx, write_slot] = k_new
            vals[batch_idx, write_slot] = v_new
            age[batch_idx, write_slot] = 0

        y = torch.stack(outputs, dim=1)
        aux = {
            "read_weights": torch.stack(read_weights_all, dim=1),
            "write_slots": torch.stack(write_slots_all, dim=1),
        }
        new_state = {"keys": keys, "vals": vals, "age": age}
        return y, new_state, aux
