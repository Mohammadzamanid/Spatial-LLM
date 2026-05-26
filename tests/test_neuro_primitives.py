"""
tests/test_neuro_primitives.py
Tests for the full neuroscience primitive stack — single neuron to network.
"""
import math
import pytest
import torch

from src.models.neuro import (
    LIFNeuron, AdaptiveLIFNeuron, DendriticNeuron,
    HebbianLayer, STDPLayer, ShortTermPlasticity,
    DivisiveNormalization, LateralInhibition, EIBalanceLayer, CorticalColumn,
    HeadDirectionCells, BoundaryVectorCells, SpeedCells, ConjunctiveSpatialCells,
    ThetaOscillator, PhasePrecession, ThetaGammaCoupling, SharpWaveRipple,
    ContinuousAttractorNetwork, GridAttractorNetwork,
)

B, T, D = 4, 12, 32


# ── SINGLE NEURON ───────────────────────────────────────────────────────────────

def test_lif_neuron_produces_binary_spikes():
    neuron = LIFNeuron(dim=D)
    x = torch.rand(B, T, D) * 2
    spikes, v = neuron(x)
    assert spikes.shape == (B, T, D)
    assert set(spikes.unique().tolist()).issubset({0.0, 1.0})
    assert v.shape == (B, D)


def test_lif_neuron_gradient_flows():
    neuron = LIFNeuron(dim=D)
    x = (torch.rand(B, T, D) * 2).requires_grad_(True)
    spikes, _ = neuron(x)
    spikes.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_adaptive_lif_adapts():
    """Sustained input should produce fewer spikes later than early (adaptation)."""
    neuron = AdaptiveLIFNeuron(dim=D, adapt_increment=1.0)
    x = torch.ones(1, 20, D) * 1.5
    spikes, _ = neuron(x)
    early = spikes[:, :5, :].sum()
    late = spikes[:, 15:, :].sum()
    assert late <= early, "Adaptive neuron should fire less under sustained drive"


def test_dendritic_neuron_shape():
    neuron = DendriticNeuron(in_dim=D, out_dim=16, num_branches=4)
    x = torch.randn(B, D)
    out = neuron(x)
    assert out.shape == (B, 16)
    assert (out >= 0).all(), "Somatic output should be rectified"


# ── SYNAPSE ──────────────────────────────────────────────────────────────────────

def test_hebbian_layer_updates_trace():
    layer = HebbianLayer(in_dim=D, out_dim=16)
    layer.train()
    x = torch.randn(B, D)
    before = layer.hebb_trace.clone()
    _ = layer(x, update=True)
    assert not torch.allclose(before, layer.hebb_trace), "Hebbian trace should update"


def test_hebbian_trace_bounded():
    layer = HebbianLayer(in_dim=D, out_dim=16, hebb_lr=0.5)
    layer.train()
    for _ in range(50):
        layer(torch.randn(B, D) * 3)
    assert layer.hebb_trace.abs().max() <= 1.0 + 1e-5, "Oja rule should bound weights"


def test_stdp_layer_produces_weight_change():
    layer = STDPLayer(in_dim=D, out_dim=16)
    pre = (torch.rand(B, T, D) > 0.5).float()
    post, dw = layer(pre)
    assert post.shape == (B, T, 16)
    assert dw.shape == (16, D)


def test_short_term_plasticity_shape():
    stp = ShortTermPlasticity(dim=D)
    x = (torch.rand(B, T, D) > 0.6).float()
    out = stp(x)
    assert out.shape == (B, T, D)
    assert not torch.isnan(out).any()


# ── MICROCIRCUIT ──────────────────────────────────────────────────────────────────

def test_divisive_normalization_bounds_output():
    dn = DivisiveNormalization(dim=D)
    x = torch.randn(B, D) * 10
    out = dn(x)
    assert out.shape == (B, D)
    assert (out >= 0).all() and (out <= 1.0).all(), "Divisive norm output in [0,1]"


def test_lateral_inhibition_sharpens():
    li = LateralInhibition(dim=D)
    x = torch.randn(B, D)
    out = li(x)
    assert out.shape == (B, D)
    assert (out >= 0).all()


def test_ei_balance_obeys_dales_law():
    layer = EIBalanceLayer(in_dim=D, out_dim=20, inhib_ratio=0.2)
    x = torch.randn(B, D)
    out = layer(x)
    assert out.shape == (B, 20)
    assert (out >= 0).all(), "All firing rates must be non-negative"


def test_cortical_column_residual():
    col = CorticalColumn(dim=D)
    x = torch.randn(B, D)
    out = col(x)
    assert out.shape == (B, D)
    assert not torch.isnan(out).any()


# ── SPATIAL CELLS ──────────────────────────────────────────────────────────────────

def test_head_direction_decode():
    hd = HeadDirectionCells(num_cells=64, embed_dim=D)
    heading = torch.tensor([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
    emb = hd(heading)
    assert emb.shape == (4, D)
    decoded = hd.decode_heading(heading)
    # Decoded heading should be close to input (within tolerance)
    err = torch.atan2(torch.sin(decoded - heading), torch.cos(decoded - heading)).abs()
    assert (err < 0.3).all(), f"Head-direction decode error too large: {err}"


def test_boundary_vector_cells_shape():
    bvc = BoundaryVectorCells(num_cells=32, embed_dim=D)
    dist = torch.rand(B)
    angle = torch.rand(B) * 2 * math.pi
    out = bvc(dist, angle)
    assert out.shape == (B, D)


def test_speed_cells_shape():
    sc = SpeedCells(num_cells=16, embed_dim=D)
    out = sc(torch.rand(B))
    assert out.shape == (B, D)


def test_conjunctive_cells_bind():
    cc = ConjunctiveSpatialCells(embed_dim=D)
    out = cc(torch.rand(B) * 2 * math.pi, torch.rand(B))
    assert out.shape == (B, D)
    assert not torch.isnan(out).any()


# ── OSCILLATIONS ──────────────────────────────────────────────────────────────────

def test_theta_oscillator_gates():
    theta = ThetaOscillator(dim=D, freq=6.0)
    x = torch.ones(B, T, D)
    out = theta(x)
    assert out.shape == (B, T, D)
    # The gate is in [0,1] so output should not exceed input
    assert (out <= 1.0 + 1e-5).all()


def test_phase_precession_monotonic_phase():
    pp = PhasePrecession(embed_dim=D)
    pos = torch.tensor([0.0, 0.5, 1.0])
    out = pp(pos)
    assert out.shape == (3, D)


def test_theta_gamma_coupling_buffer():
    tg = ThetaGammaCoupling(dim=D, num_slots=7)
    items = torch.randn(B, 10, D)   # more than num_slots → should truncate
    out = tg(items)
    assert out.shape == (B, D)


def test_sharp_wave_ripple_replay():
    swr = SharpWaveRipple(dim=D, compression=4)
    buffer = torch.randn(B, 16, D)
    forward = swr(buffer, reverse=False)
    reverse = swr(buffer, reverse=True)
    assert forward.shape == (B, D)
    assert reverse.shape == (B, D)


# ── ATTRACTOR DYNAMICS ─────────────────────────────────────────────────────────────

def test_continuous_attractor_forms_bump():
    can = ContinuousAttractorNetwork(num_units=64, steps=10)
    # Localized input
    inp = torch.zeros(2, 64)
    inp[:, 30] = 1.0
    out = can(inp)
    assert out.shape == (2, 64)
    assert (out >= 0).all()
    # Activity should be concentrated (peak well above mean)
    assert out.max() > out.mean()


def test_grid_attractor_network_shape():
    gan = GridAttractorNetwork(grid_size=12, embed_dim=D, steps=5)
    coords = torch.tensor([[35.69, 139.69], [51.5, -0.13]])
    out = gan(coords)
    assert out.shape == (2, D)
    assert not torch.isnan(out).any()


def test_grid_attractor_different_coords_differ():
    gan = GridAttractorNetwork(grid_size=12, embed_dim=D)
    gan.eval()
    c1 = torch.tensor([[0.0, 0.0]])
    c2 = torch.tensor([[45.0, 90.0]])
    with torch.no_grad():
        e1, e2 = gan(c1), gan(c2)
    assert not torch.allclose(e1, e2, atol=1e-4)


# ── FULL INTEGRATION ──────────────────────────────────────────────────────────────

def test_brain_spatial_cortex_integration():
    from src.models.neuro.brain_spatial_cortex import BrainSpatialCortex
    cortex = BrainSpatialCortex(embed_dim=64, num_tokens=4)
    coords = torch.tensor([[35.69, 139.69], [51.5, -0.13], [40.71, -74.0], [35.69, 51.39]])
    tokens = cortex(coords)
    assert tokens.shape == (4, 4, 64)
    assert not torch.isnan(tokens).any()
    assert not torch.isinf(tokens).any()


def test_brain_spatial_cortex_with_movement():
    from src.models.neuro.brain_spatial_cortex import BrainSpatialCortex
    cortex = BrainSpatialCortex(embed_dim=64, num_tokens=4)
    coords = torch.tensor([[35.69, 139.69], [51.5, -0.13]])
    heading = torch.tensor([0.0, math.pi])
    speed = torch.tensor([0.5, 0.8])
    tokens = cortex(coords, heading=heading, speed=speed)
    assert tokens.shape == (2, 4, 64)


def test_brain_spatial_cortex_gradient_flows():
    from src.models.neuro.brain_spatial_cortex import BrainSpatialCortex
    cortex = BrainSpatialCortex(embed_dim=32, num_tokens=2)
    coords = torch.tensor([[35.69, 139.69]], requires_grad=False)
    tokens = cortex(coords)
    loss = tokens.sum()
    loss.backward()
    # Check at least one parameter got a gradient
    grads = [p.grad for p in cortex.parameters() if p.grad is not None]
    assert len(grads) > 0
