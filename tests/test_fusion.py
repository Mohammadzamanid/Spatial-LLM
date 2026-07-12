"""Tests for SpatialFusionLayer and MultiScaleSpatialFusion."""
import pytest
import torch
from src.models.fusion import SpatialFusionLayer, MultiScaleSpatialFusion


@pytest.fixture
def hidden_dim():
    return 64


# ── Shared-gate behaviour (original / backward compatible) ──────────────────

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


def test_default_layer_has_single_gate(hidden_dim):
    # Default stays shape (1,) so existing (shared-gate) checkpoints load unchanged.
    layer = SpatialFusionLayer(hidden_dim, num_heads=4)
    assert layer.attn_gate.shape == (1,)
    assert layer.ffn_gate.shape == (1,)


# ── Zero-init identity — the property that fixed garbage generation ─────────

@pytest.mark.parametrize("num_groups", [1, 4])
def test_zero_init_is_identity(hidden_dim, num_groups):
    """At init every gate is 0, so the block must be an exact identity (fused ==
    text) for both shared and per-module gating — otherwise spatial tokens bury
    the text and generation degrades."""
    layer = SpatialFusionLayer(hidden_dim, num_heads=4, num_spatial_groups=num_groups).eval()
    text = torch.randn(2, 10, hidden_dim)
    spatial = torch.randn(2, 9, hidden_dim)
    group_sizes = [4, 3, 1, 1] if num_groups > 1 else None
    out = layer(text, spatial, group_sizes=group_sizes)
    assert torch.allclose(out, text, atol=1e-6)


# ── Per-module gating ───────────────────────────────────────────────────────

def test_per_module_gate_count(hidden_dim):
    layer = SpatialFusionLayer(hidden_dim, num_heads=4, num_spatial_groups=4)
    assert layer.attn_gate.shape == (4,)   # one attn gate per spatial module
    assert layer.ffn_gate.shape == (1,)    # FFN gate stays shared


def test_per_module_output_shape_and_finite(hidden_dim):
    layer = SpatialFusionLayer(hidden_dim, num_heads=4, num_spatial_groups=3).eval()
    text = torch.randn(2, 10, hidden_dim)
    spatial = torch.randn(2, 9, hidden_dim)
    out = layer(text, spatial, group_sizes=[4, 3, 2])
    assert out.shape == text.shape
    assert torch.isfinite(out).all()


def test_per_module_gates_route_independently(hidden_dim):
    """Opening module 0's gate vs module 1's gate must produce different outputs —
    proof that each gate controls a distinct spatial module, not a shared blend."""
    layer = SpatialFusionLayer(hidden_dim, num_heads=4, num_spatial_groups=3).eval()
    text = torch.randn(2, 10, hidden_dim)
    spatial = torch.randn(2, 9, hidden_dim)
    group_sizes = [4, 3, 2]

    with torch.no_grad():
        layer.attn_gate.copy_(torch.tensor([2.0, 0.0, 0.0]))
    out_module0 = layer(text, spatial, group_sizes=group_sizes)

    with torch.no_grad():
        layer.attn_gate.copy_(torch.tensor([0.0, 2.0, 0.0]))
    out_module1 = layer(text, spatial, group_sizes=group_sizes)

    assert not torch.allclose(out_module0, out_module1)


def test_per_module_fewer_groups_than_gates(hidden_dim):
    """A 4-gate layer must handle the no-tile case (only 3 groups present): the
    unused (tile) gate is simply ignored."""
    layer = SpatialFusionLayer(hidden_dim, num_heads=4, num_spatial_groups=4).eval()
    text = torch.randn(2, 10, hidden_dim)
    spatial = torch.randn(2, 8, hidden_dim)
    out = layer(text, spatial, group_sizes=[4, 3, 1])   # 3 of 4 groups
    assert out.shape == text.shape
    assert torch.isfinite(out).all()


def test_per_module_falls_back_to_shared_without_group_sizes(hidden_dim):
    """If a multi-gate layer is called without group_sizes it must still run
    (shared fallback on gate 0), not crash."""
    layer = SpatialFusionLayer(hidden_dim, num_heads=4, num_spatial_groups=4)
    text = torch.randn(2, 10, hidden_dim)
    spatial = torch.randn(2, 8, hidden_dim)
    out = layer(text, spatial)   # no group_sizes
    assert out.shape == text.shape


def test_multiscale_per_module(hidden_dim):
    fusion = MultiScaleSpatialFusion(
        hidden_dim, num_heads=4, num_layers=2, num_spatial_groups=3
    )
    text = torch.randn(2, 10, hidden_dim)
    spatial = torch.randn(2, 6, hidden_dim)
    out = fusion(text, spatial, group_sizes=[2, 3, 1])
    assert out.shape == text.shape
    assert not torch.isnan(out).any()


# ── RSC action/memory split — bifurcated output pathways (Molecular Psychiatry 2024) ──

def test_rsc_split_gate_shapes(hidden_dim):
    layer = SpatialFusionLayer(hidden_dim, num_heads=4, rsc_split=True)
    assert layer.action_gate.shape == (1,) and layer.memory_gate.shape == (1,)
    assert hasattr(layer, "action_proj") and hasattr(layer, "memory_proj")


def test_rsc_split_zero_init_is_identity(hidden_dim):
    """With rsc_split on and zero-init gates the block is still an exact identity."""
    layer = SpatialFusionLayer(hidden_dim, num_heads=4, rsc_split=True).eval()
    text = torch.randn(2, 10, hidden_dim)
    out = layer(text, torch.randn(2, 8, hidden_dim))
    assert torch.allclose(out, text, atol=1e-6)


def test_rsc_split_pathways_route_independently(hidden_dim):
    """Opening the action gate vs the memory gate must give different outputs — two distinct pathways."""
    layer = SpatialFusionLayer(hidden_dim, num_heads=4, rsc_split=True).eval()
    text = torch.randn(2, 10, hidden_dim); spatial = torch.randn(2, 8, hidden_dim)
    with torch.no_grad():
        layer.action_gate.fill_(2.0); layer.memory_gate.fill_(0.0)
    out_action = layer(text, spatial)
    with torch.no_grad():
        layer.action_gate.fill_(0.0); layer.memory_gate.fill_(2.0)
    out_memory = layer(text, spatial)
    assert not torch.allclose(out_action, out_memory)   # distinct, independently-gated streams
    assert not torch.allclose(out_action, text)          # each stream is load-bearing


def test_rsc_split_lesion_removes_one_pathway(hidden_dim):
    """Lesioning one pathway's gate removes only that pathway — the double-dissociation substrate."""
    layer = SpatialFusionLayer(hidden_dim, num_heads=4, rsc_split=True).eval()
    text = torch.randn(2, 10, hidden_dim); spatial = torch.randn(2, 8, hidden_dim)
    with torch.no_grad():
        layer.action_gate.fill_(2.0); layer.memory_gate.fill_(2.0)
    both = layer(text, spatial)
    with torch.no_grad():
        layer.memory_gate.fill_(0.0)                     # lesion the memory pathway
    action_only = layer(text, spatial)
    assert not torch.allclose(both, action_only)         # the memory pathway was contributing


# ── Perforant semantic input (Boccara 2019) ─────────────────────────────────

def test_perforant_zero_init_identity_even_with_semantics(hidden_dim):
    layer = SpatialFusionLayer(hidden_dim, num_heads=4, perforant=True).eval()
    text = torch.randn(2, 10, hidden_dim)
    out = layer(text, torch.randn(2, 8, hidden_dim), semantic_tokens=torch.randn(2, 5, hidden_dim))
    assert torch.allclose(out, text, atol=1e-6)


def test_perforant_injects_only_when_semantics_present(hidden_dim):
    """The perforant pathway is skipped without semantic tokens (backward compatible) and load-bearing with them."""
    layer = SpatialFusionLayer(hidden_dim, num_heads=4, perforant=True).eval()
    text = torch.randn(2, 10, hidden_dim); spatial = torch.randn(2, 8, hidden_dim)
    with torch.no_grad():
        layer.perforant_gate.fill_(2.0)
    out_no_sem = layer(text, spatial)                                          # no semantics -> unchanged
    out_sem = layer(text, spatial, semantic_tokens=torch.randn(2, 5, hidden_dim))
    assert torch.allclose(out_no_sem, text, atol=1e-6)                         # perforant skipped without semantics
    assert not torch.allclose(out_sem, text)                                   # semantics are load-bearing


def test_multiscale_threads_both_organs(hidden_dim):
    fusion = MultiScaleSpatialFusion(hidden_dim, num_heads=4, num_layers=2, rsc_split=True, perforant=True)
    text = torch.randn(2, 10, hidden_dim)
    out = fusion(text, torch.randn(2, 6, hidden_dim), semantic_tokens=torch.randn(2, 4, hidden_dim))
    assert out.shape == text.shape and torch.isfinite(out).all()
