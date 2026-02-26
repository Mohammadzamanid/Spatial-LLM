"""Synthetic spatial reasoning dataset and data loading utilities."""

import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _deterministic_hash(word, vocab_size):
    """Deterministic word-to-id hash, consistent across Python versions and runs.

    Python's built-in hash() is randomized per process (PYTHONHASHSEED),
    making it non-reproducible. This uses SHA-256 for consistency.
    """
    return int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % vocab_size


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

OBJECTS = [
    "table", "chair", "lamp", "door", "window", "book", "cup", "box",
    "shelf", "desk", "sofa", "plant", "clock", "mirror", "rug",
]


class SpatialReasoningDataset(Dataset):
    """Generates synthetic spatial reasoning samples on the fly."""

    def __init__(self, synthetic=True, max_seq_len=256, num_samples=1000,
                 vocab_size=50257, data_path=None):
        self.max_seq_len = max(max_seq_len, 8)
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.synthetic = synthetic

        if data_path is not None:
            import json
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

    def _load_real_sample(self, idx):
        sample = self.data[idx]
        text = f"{sample['context']} {sample['question']} {sample['answer']}"
        words = text.lower().split()
        input_ids = [_deterministic_hash(w, self.vocab_size) for w in words[: self.max_seq_len]]
        while len(input_ids) < self.max_seq_len:
            input_ids.append(0)
        input_ids = torch.tensor(input_ids[: self.max_seq_len], dtype=torch.long)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "schema_types": torch.zeros(self.max_seq_len, dtype=torch.long),
            "spatial_features": torch.randn(self.max_seq_len, 4),
            "temporal_distances": torch.zeros(self.max_seq_len),
            "spatial_distances": torch.zeros(self.max_seq_len),
            "attention_mask": (input_ids != 0).float(),
        }

    def _generate_synthetic_sample(self):
        low = min(8, self.max_seq_len)
        seq_len = np.random.randint(low, self.max_seq_len + 1)
        input_ids = torch.randint(1, self.vocab_size, (seq_len,))
        labels = input_ids.clone()

        # Pad to max_seq_len
        if seq_len < self.max_seq_len:
            pad = torch.zeros(self.max_seq_len - seq_len, dtype=torch.long)
            input_ids = torch.cat([input_ids, pad])
            labels = torch.cat([labels, pad])

        attention_mask = (input_ids != 0).float()
        schema_types = torch.randint(0, 4, (self.max_seq_len,))
        spatial_features = torch.randn(self.max_seq_len, 4)
        temporal_distances = torch.rand(self.max_seq_len) * 10
        spatial_distances = torch.rand(self.max_seq_len) * 50

        return {
            "input_ids": input_ids,
            "labels": labels,
            "schema_types": schema_types,
            "spatial_features": spatial_features,
            "temporal_distances": temporal_distances,
            "spatial_distances": spatial_distances,
            "attention_mask": attention_mask,
        }


def collate_spatial_batch(batch):
    """Collate a list of samples into a batch dict.

    Handles both tensor fields (stacked) and non-tensor fields (kept as lists).
    """
    result = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values  # strings, ints, etc.
    return result


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
