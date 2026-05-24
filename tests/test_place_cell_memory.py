"""Tests for HippocampalMemory — place cell + episodic memory module."""
import pytest
import torch
from src.models.place_cell_memory import PlaceCellLayer, HippocampalMemory


@pytest.fixture
def coords():
    return torch.tensor([[35.6895, 139.6917], [51.5074, -0.1278]], dtype=torch.float32)


def test_place_cell_layer_output_shape(coords):
    layer = PlaceCellLayer(num_cells=64, sparsity_k=10)
    out = layer(coords)
    assert out.shape == (2, 64)


def test_place_cell_sparsity(coords):
    """k-WTA: at most sparsity_k non-zero activations per sample."""
    k = 10
    layer = PlaceCellLayer(num_cells=64, sparsity_k=k)
    out = layer(coords)
    nonzero_counts = (out > 0).sum(dim=-1)
    for count in nonzero_counts:
        assert count.item() <= k + 1, f"Too many active cells: {count}"


def test_hippocampal_memory_encode_shape(coords):
    mem = HippocampalMemory(embed_dim=64, num_cells=64)
    out = mem.encode(coords)
    assert out.shape == (2, 64)


def test_hippocampal_memory_store_and_retrieve(coords):
    mem = HippocampalMemory(embed_dim=64, num_cells=64, buffer_size=16)
    context = torch.randn(2, 64)

    # Before storing: retrieval returns zeros
    retrieved_before = mem.retrieve(coords)
    assert retrieved_before.shape == (2, 64)

    # Store then retrieve
    mem.store(coords, context)
    retrieved_after = mem.retrieve(coords)
    assert retrieved_after.shape == (2, 64)
    # After storing, retrieval should be non-zero
    assert retrieved_after.abs().sum() > 0


def test_hippocampal_memory_forward_no_nan(coords):
    mem = HippocampalMemory(embed_dim=64, num_cells=64)
    context = torch.randn(2, 64)
    out = mem(coords, context=context, store=True)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 64)
