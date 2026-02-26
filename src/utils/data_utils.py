"""Spatial reasoning dataset and data loading utilities."""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

TASK_TYPE_MAP = {
    "spatial_relation": 0,
    "object_location": 1,
    "navigation": 2,
    "counting": 3,
    "existence": 4,
    "unknown": 5,
}


class SpatialReasoningDataset(Dataset):
    def __init__(self, synthetic=True, max_seq_len=256, num_samples=1000, vocab_size=50257, data_path=None, visual_dim=512):
        self.max_seq_len = max(max_seq_len, 8)
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.synthetic = synthetic
        self.visual_dim = visual_dim

        if data_path is not None:
            with open(data_path, "r") as f:
                self.data = json.load(f)
            self.num_samples = len(self.data)
        else:
            self.data = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.data is not None:
            return self._load_real_sample(idx)
        return self._generate_synthetic_sample()

    def _tokenize(self, text):
        words = text.lower().split()
        ids = [hash(w) % self.vocab_size for w in words[: self.max_seq_len]]
        while len(ids) < self.max_seq_len:
            ids.append(0)
        return torch.tensor(ids[: self.max_seq_len], dtype=torch.long)

    def _pad_2d(self, arr, out_len, out_dim):
        t = torch.tensor(arr, dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        t = t[:out_len]
        if t.shape[0] < out_len:
            t = torch.cat([t, torch.zeros(out_len - t.shape[0], t.shape[1])], dim=0)
        if t.shape[1] < out_dim:
            t = torch.nn.functional.pad(t, (0, out_dim - t.shape[1]))
        return t[:, :out_dim]

    def _pad_1d(self, arr, out_len, fill=0.0):
        t = torch.tensor(arr, dtype=torch.float32)[:out_len]
        if t.shape[0] < out_len:
            t = torch.cat([t, torch.full((out_len - t.shape[0],), float(fill))])
        return t

    def _load_real_sample(self, idx):
        sample = self.data[idx]
        input_ids = self._tokenize(f"{sample['context']} {sample['question']} {sample['answer']}")
        labels = input_ids.clone()

        schema_idx = TASK_TYPE_MAP.get(sample.get("task_type", "unknown"), TASK_TYPE_MAP["unknown"])
        schema_types = torch.full((self.max_seq_len,), schema_idx, dtype=torch.long)

        velocity = self._pad_2d(sample.get("velocity", np.random.randn(self.max_seq_len, 2) * 0.05), self.max_seq_len, 2)
        coordinates = self._pad_2d(sample.get("coordinates", torch.cumsum(velocity, dim=0).tolist()), self.max_seq_len, 2)
        temporal_distances = self._pad_1d(sample.get("temporal_distances", [1.0] * self.max_seq_len), self.max_seq_len, 1.0)
        spatial_distances = self._pad_1d(sample.get("spatial_distances", torch.norm(coordinates, dim=-1).tolist()), self.max_seq_len, 0.0)

        spatial_features = torch.zeros(self.max_seq_len, 4, dtype=torch.float32)
        spatial_features[:, :2] = coordinates
        spatial_features[:, 2:] = velocity

        visual_features = self._pad_2d(sample.get("image_vector", np.zeros((1, self.visual_dim))), 1, self.visual_dim).squeeze(0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "schema_types": schema_types,
            "spatial_features": spatial_features,
            "temporal_distances": temporal_distances,
            "spatial_distances": spatial_distances,
            "velocity": velocity,
            "visual_features": visual_features,
            "attention_mask": (input_ids != 0).float(),
        }

    def _generate_synthetic_sample(self):
        seq_len = np.random.randint(min(8, self.max_seq_len), self.max_seq_len + 1)
        input_ids = torch.randint(1, self.vocab_size, (seq_len,))
        labels = input_ids.clone()
        if seq_len < self.max_seq_len:
            pad = torch.zeros(self.max_seq_len - seq_len, dtype=torch.long)
            input_ids = torch.cat([input_ids, pad])
            labels = torch.cat([labels, pad])

        velocity = torch.randn(self.max_seq_len, 2) * 0.1
        coordinates = torch.cumsum(velocity, dim=0)
        spatial_features = torch.zeros(self.max_seq_len, 4)
        spatial_features[:, :2] = coordinates
        spatial_features[:, 2:] = velocity

        return {
            "input_ids": input_ids,
            "labels": labels,
            "schema_types": torch.randint(0, 4, (self.max_seq_len,)),
            "spatial_features": spatial_features,
            "temporal_distances": torch.ones(self.max_seq_len),
            "spatial_distances": torch.norm(coordinates, dim=-1),
            "velocity": velocity,
            "visual_features": torch.randn(self.visual_dim) * 0.01,
            "attention_mask": (input_ids != 0).float(),
        }


def collate_spatial_batch(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def create_spatial_dataloader(dataset, batch_size=8, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_spatial_batch, pin_memory=True, drop_last=True)
