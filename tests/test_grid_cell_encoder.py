"""Tests for GridCellEncoder — entorhinal cortex-inspired module."""
import pytest
import torch
from src.models.grid_cell_encoder import GridModule, GridCellEncoder, GridCellEncoderWithTokens


@pytest.fixture
def coords():
    return torch.tensor([
        [35.6895, 139.6917],   # Tokyo
        [51.5074, -0.1278],    # London
        [40.7128, -74.0060],   # New York
    ], dtype=torch.float32)


def test_grid_module_output_shape(coords):
    m = GridModule(scale=1.0, embed_dim=64, num_cells=32)
    out = m(coords)
    assert out.shape == (3, 64)


def test_grid_encoder_output_shape(coords):
    enc = GridCellEncoder(embed_dim=128, num_modules=4)
    out = enc(coords)
    assert out.shape == (3, 128)


def test_grid_encoder_with_tokens_shape(coords):
    enc = GridCellEncoderWithTokens(embed_dim=64, num_modules=4)
    out = enc(coords)
    assert out.shape == (3, 4, 64), f"Got {out.shape}"


def test_grid_encoder_no_nan(coords):
    enc = GridCellEncoder(embed_dim=64, num_modules=3)
    out = enc(coords)
    assert not torch.isnan(out).any()


def test_grid_encoder_different_cities_differ(coords):
    enc = GridCellEncoder(embed_dim=128, num_modules=4)
    out = enc(coords)
    assert not torch.allclose(out[0], out[1]), "Tokyo and London should differ"
    assert not torch.allclose(out[0], out[2]), "Tokyo and NY should differ"





def test_grid_encoder_gradients_flow():
    """
    Verifies that gradients flow through the grid cell encoder,
    and that a spatial contrastive loss reduces nearby-coord distance.
    Spatial ordering emerges from training on real data; this just
    confirms the model is differentiable and trainable.
    """
    import torch.optim as optim

    enc = GridCellEncoder(embed_dim=32, num_modules=3)
    optimizer = optim.Adam(enc.parameters(), lr=1e-2)

    near = torch.tensor([[48.85, 2.35], [48.86, 2.36]])

    dist_before = (enc(near)[0] - enc(near)[1]).norm().item()

    # Minimise distance between nearby coordinates
    for _ in range(30):
        optimizer.zero_grad()
        out = enc(near)
        loss = (out[0] - out[1]).norm()
        loss.backward()
        optimizer.step()

    dist_after = (enc(near)[0] - enc(near)[1]).detach().norm().item()

    # Gradient must have flowed: distance should decrease
    assert dist_after < dist_before, (
        f"Spatial contrastive loss should reduce nearby dist: {dist_before:.3f} -> {dist_after:.3f}"
    )

    # All params should have grads
    for name, param in enc.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
