"""Tests for SpatialPredictiveCoding."""
import pytest
import torch
from src.models.predictive_coding import PredictiveCodingLevel, SpatialPredictiveCoding


def test_pc_level_output_shapes():
    level = PredictiveCodingLevel(dim=64, pred_dim=64)
    x = torch.randn(2, 64)
    state, error, td_pred = level(x)
    assert state.shape == (2, 64)
    assert error.shape == (2, 64)
    assert td_pred.shape == (2, 64)


def test_pc_level_with_top_down():
    level = PredictiveCodingLevel(dim=64, pred_dim=64)
    x = torch.randn(2, 64)
    td = torch.randn(2, 64)
    state, error, td_pred = level(x, top_down_pred=td)
    assert state.shape == (2, 64)
    assert not torch.isnan(error).any()


def test_spatial_predictive_coding():
    pc = SpatialPredictiveCoding(spatial_dim=64, llm_dim=128, num_levels=3)
    x = torch.randn(2, 64)
    out, loss = pc(x)
    assert out.shape == (2, 128)
    assert loss.item() >= 0, "PC loss should be non-negative"
    assert not torch.isnan(out).any()
    assert not torch.isnan(loss)


def test_pc_loss_is_scalar():
    pc = SpatialPredictiveCoding(spatial_dim=32, llm_dim=64)
    x = torch.randn(3, 32)
    _, loss = pc(x)
    assert loss.dim() == 0, "PC loss should be a scalar"
