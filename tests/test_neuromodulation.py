"""Tests for neuromodulation modules."""
import pytest
import torch
from src.models.neuromodulation import SpatialNeuromodulator, AdaptiveGain, PredictionErrorGate


def test_neuromodulator_2d():
    mod = SpatialNeuromodulator(hidden_dim=64)
    x = torch.randn(2, 64)
    ctx = torch.randn(2, 64)
    out = mod(x, ctx)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()


def test_neuromodulator_3d():
    """Works on (B, T, D) sequence inputs."""
    mod = SpatialNeuromodulator(hidden_dim=64)
    x = torch.randn(2, 10, 64)
    ctx = torch.randn(2, 64)
    out = mod(x, ctx)
    assert out.shape == (2, 10, 64)


def test_adaptive_gain_output_shape():
    ag = AdaptiveGain(hidden_dim=64)
    x = torch.randn(2, 64)
    out, uncertainty = ag(x)
    assert out.shape == (2, 64)
    assert uncertainty.shape == (2,)
    assert (uncertainty >= 0).all(), "Uncertainty should be non-negative"


def test_adaptive_gain_no_nan():
    ag = AdaptiveGain(hidden_dim=64)
    x = torch.randn(4, 64)
    out, unc = ag(x)
    assert not torch.isnan(out).any()
    assert not torch.isnan(unc).any()


def test_prediction_error_gate():
    gate = PredictionErrorGate(hidden_dim=64)
    x = torch.randn(2, 64)
    err = torch.tensor([0.1, 2.5])   # low vs high prediction error
    out = gate(x, err)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()
