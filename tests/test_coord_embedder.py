"""Tests for CoordinateEmbedder and CoordinateEmbedderWithTokens."""
import pytest
import torch
from src.models.coord_embedder import CoordinateEmbedder, CoordinateEmbedderWithTokens


@pytest.fixture
def sample_coords():
    return torch.tensor([[35.6895, 139.6917], [51.5074, -0.1278]], dtype=torch.float32)


def test_coord_embedder_output_shape(sample_coords):
    model = CoordinateEmbedder(embed_dim=256, num_freqs=64)
    out = model(sample_coords)
    assert out.shape == (2, 256), f"Expected (2, 256), got {out.shape}"


def test_coord_embedder_with_tokens_shape(sample_coords):
    model = CoordinateEmbedderWithTokens(embed_dim=128, num_freqs=32, num_tokens=4)
    out = model(sample_coords)
    assert out.shape == (2, 4, 128), f"Expected (2, 4, 128), got {out.shape}"


def test_coord_embedder_no_nan(sample_coords):
    model = CoordinateEmbedder(embed_dim=256)
    out = model(sample_coords)
    assert not torch.isnan(out).any(), "NaN values in embeddings"


def test_coord_embedder_different_coords_differ():
    model = CoordinateEmbedder(embed_dim=256)
    c1 = torch.tensor([[0.0, 0.0]])
    c2 = torch.tensor([[45.0, 90.0]])
    assert not torch.allclose(model(c1), model(c2)), "Different coords should produce different embeddings"
