"""Dataset loaders for spatial reasoning evaluation with real data files."""

import json
import torch
from torch.utils.data import Dataset, DataLoader


class SpatialTestDataset(Dataset):
    """Load spatial reasoning tasks from JSON files."""

    def __init__(self, data_path, max_length=256, vocab_size=50257):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f"{sample['context']} {sample['question']} {sample['answer']}"
        words = text.lower().split()
        input_ids = [hash(w) % self.vocab_size for w in words[: self.max_length]]
        while len(input_ids) < self.max_length:
            input_ids.append(0)
        input_ids = torch.tensor(input_ids[: self.max_length], dtype=torch.long)

        schema_type = self._get_schema_type(sample.get("task_type", "unknown"))
        spatial_features = torch.randn(self.max_length, 4)

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "schema_types": torch.full((self.max_length,), schema_type, dtype=torch.long),
            "spatial_features": spatial_features,
            "coordinates": torch.randn(self.max_length, 2),
            "timestamps": torch.arange(self.max_length, dtype=torch.float32),
            "task_type": sample.get("task_type", "unknown"),
            "difficulty": sample.get("difficulty", 1),
            "answer": sample["answer"],
        }

    def _get_schema_type(self, task_type):
        mapping = {
            "navigation": 2,
            "object_location": 1,
            "spatial_relation": 0,
            "unknown": 3,
        }
        return mapping.get(task_type, 3)


def create_eval_dataloaders(train_path, val_path, test_path, batch_size=8):
    """Create DataLoaders for evaluation splits."""
    from src.utils.data_utils import collate_spatial_batch

    loaders = {}
    for name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if path is not None:
            dataset = SpatialTestDataset(path)
            loaders[name] = DataLoader(
                dataset, batch_size=batch_size, shuffle=(name == "train"),
                collate_fn=collate_spatial_batch,
            )
    return loaders
