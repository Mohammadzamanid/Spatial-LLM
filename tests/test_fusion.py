"""Tests for SpatialFusionLayer and MultiScaleSpatialFusion."""
import pytest
import torch
from src.models.fusion import SpatialFusionLayer, MultiScaleSpatialFusion


@pytest.fixture
def hidden_dim():
    return 64


def test_fusion_layer_output_shape(hidden_dim):
    layer = SpatialFusionLayer(hidden_dim, num_heads=4)
    text = torch.randn(2, 10, hidden_dim)     # (B, T, D)
    spatial = torch.randn(2, 8, hidden_dim)   # (B, S, D)
    out = layer(text, spatial)
    assert out.shape == text.shape


def test_multiscale_fusion_output_shape(hidden_dim):
    fusion = MultiScaleSpatialFusion(hidden_dim, num_heads=4, num_layers=3)
    text = torch.randn(2, 10, hidden_dim)
    spatial = torch.randn(2, 8, hidden_dim)
    out = fusion(text, spatial)
    assert out.shape == text.shape


def test_fusion_no_nan(hidden_dim):
    layer = SpatialFusionLayer(hidden_dim, num_heads=4)
    text = torch.randn(2, 10, hidden_dim)
    spatial = torch.randn(2, 8, hidden_dim)
    out = layer(text, spatial)
    assert not torch.isnan(out).any()
