"""Spatial reasoning dataset and data loading utilities."""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


SPATIAL_TEMPLATES = [
    ("The {obj1} is to the left of the {obj2}.", "left"),
    ("The {obj1} is to the right of the {obj2}.", "right"),
    ("The {obj1} is above the {obj2}.", "above"),
    ("The {obj1} is below the {obj2}.", "below"),
    ("The {obj1} is inside the {obj2}.", "inside"),
    ("The {obj1} is behind the {obj2}.", "behind"),
    ("Go north from the {obj1} to reach the {obj2}.", "north"),
    ("Go south from the {obj1} to reach the {obj2}.", "south"),
    ("The {obj1} is near the {obj2}.", "near"),
    ("The {obj1} is far from the {obj2}.", "far"),
]


TASK_TYPE_MAP = {
    "spatial_relation": 0,
    "object_location": 1,
    "navigation": 2,
    "counting": 3,
    "existence": 4,
    "unknown": 5,
}


class SpatialReasoningDataset(Dataset):
    """Generates synthetic samples or loads real parsed samples from JSON."""

    def __init__(self, synthetic=True, max_seq_len=256, num_samples=1000,
                 vocab_size=50257, data_path=None):
        self.max_seq_len = max(max_seq_len, 8)
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.synthetic = synthetic

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

    def _load_real_sample(self, idx):
        sample = self.data[idx]
        text = f"{sample['context']} {sample['question']} {sample['answer']}"
        input_ids = self._tokenize(text)
        labels = input_ids.clone()

        task_type = sample.get("task_type", "unknown")
        schema_idx = TASK_TYPE_MAP.get(task_type, TASK_TYPE_MAP["unknown"])
        schema_types = torch.full((self.max_seq_len,), schema_idx, dtype=torch.long)

        # Trajectory-aware fields if present, otherwise robust defaults
        velocity_data = sample.get("velocity")
        if velocity_data is None:
            velocity = torch.randn(self.max_seq_len, 2) * 0.05
        else:
            velocity = torch.tensor(velocity_data, dtype=torch.float32)
            if velocity.ndim == 1:
                velocity = velocity.unsqueeze(0)
            velocity = velocity[: self.max_seq_len]
            if velocity.shape[0] < self.max_seq_len:
                pad = torch.zeros(self.max_seq_len - velocity.shape[0], velocity.shape[1])
                velocity = torch.cat([velocity, pad], dim=0)
            if velocity.shape[1] < 2:
                velocity = torch.nn.functional.pad(velocity, (0, 2 - velocity.shape[1]))
            velocity = velocity[:, :2]

        temporal_distances = sample.get("temporal_distances")
        if temporal_distances is None:
            temporal_distances = torch.ones(self.max_seq_len, dtype=torch.float32)
        else:
            temporal_distances = torch.tensor(temporal_distances, dtype=torch.float32)[: self.max_seq_len]
            if temporal_distances.shape[0] < self.max_seq_len:
                temporal_distances = torch.cat([
                    temporal_distances,
                    torch.ones(self.max_seq_len - temporal_distances.shape[0]),
                ])

        spatial_distances = sample.get("spatial_distances")
        if spatial_distances is None:
            spatial_distances = torch.norm(torch.cumsum(velocity, dim=0), dim=-1)
        else:
            spatial_distances = torch.tensor(spatial_distances, dtype=torch.float32)[: self.max_seq_len]
            if spatial_distances.shape[0] < self.max_seq_len:
                spatial_distances = torch.cat([
                    spatial_distances,
                    torch.zeros(self.max_seq_len - spatial_distances.shape[0]),
                ])

        coordinates = sample.get("coordinates")
        if coordinates is None:
            coordinates = torch.cumsum(velocity, dim=0)
        else:
            coordinates = torch.tensor(coordinates, dtype=torch.float32)
            coordinates = coordinates[: self.max_seq_len]
            if coordinates.shape[0] < self.max_seq_len:
                pad = torch.zeros(self.max_seq_len - coordinates.shape[0], coordinates.shape[1])
                coordinates = torch.cat([coordinates, pad], dim=0)
            if coordinates.shape[1] < 2:
                coordinates = torch.nn.functional.pad(coordinates, (0, 2 - coordinates.shape[1]))
            coordinates = coordinates[:, :2]

        spatial_features = torch.zeros(self.max_seq_len, 4, dtype=torch.float32)
        spatial_features[:, :2] = coordinates
        spatial_features[:, 2:] = velocity

        return {
            "input_ids": input_ids,
            "labels": labels,
            "schema_types": schema_types,
            "spatial_features": spatial_features,
            "temporal_distances": temporal_distances,
            "spatial_distances": spatial_distances,
            "velocity": velocity,
            "attention_mask": (input_ids != 0).float(),
        }

    def _generate_synthetic_sample(self):
        low = min(8, self.max_seq_len)
        seq_len = np.random.randint(low, self.max_seq_len + 1)
        input_ids = torch.randint(1, self.vocab_size, (seq_len,))
        labels = input_ids.clone()

        if seq_len < self.max_seq_len:
            pad = torch.zeros(self.max_seq_len - seq_len, dtype=torch.long)
            input_ids = torch.cat([input_ids, pad])
            labels = torch.cat([labels, pad])

        velocity = torch.randn(self.max_seq_len, 2) * 0.1
        coordinates = torch.cumsum(velocity, dim=0)

        attention_mask = (input_ids != 0).float()
        schema_types = torch.randint(0, 4, (self.max_seq_len,))
        spatial_features = torch.zeros(self.max_seq_len, 4)
        spatial_features[:, :2] = coordinates
        spatial_features[:, 2:] = velocity
        temporal_distances = torch.ones(self.max_seq_len)
        spatial_distances = torch.norm(coordinates, dim=-1)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "schema_types": schema_types,
            "spatial_features": spatial_features,
            "temporal_distances": temporal_distances,
            "spatial_distances": spatial_distances,
            "velocity": velocity,
            "attention_mask": attention_mask,
        }


def collate_spatial_batch(batch):
    """Collate a list of samples into a batch dict."""
    return {key: torch.stack([b[key] for b in batch]) for key in batch[0]}


def create_spatial_dataloader(dataset, batch_size=8, shuffle=True, num_workers=0):
    """Create a DataLoader for spatial reasoning data."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_spatial_batch,
        pin_memory=True,
        drop_last=True,
    )
