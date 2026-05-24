"""Tests for loss functions."""
import pytest
import torch
from src.training.loss import HaversineLoss, SpatialLMLoss


def test_haversine_loss_zero():
    loss = HaversineLoss()
    coords = torch.tensor([[48.85, 2.35], [35.68, 139.69]])
    assert loss(coords, coords).item() == pytest.approx(0.0, abs=1e-4)


def test_haversine_loss_known():
    loss = HaversineLoss()
    # London to Paris ~340km
    pred = torch.tensor([[51.5074, -0.1278]])
    true = torch.tensor([[48.8566,  2.3522]])
    val = loss(pred, true).item()
    assert 330 < val < 350


def test_haversine_loss_gradient():
    loss = HaversineLoss()
    pred = torch.tensor([[48.0, 2.0]], requires_grad=True)
    true = torch.tensor([[48.85, 2.35]])
    l = loss(pred, true)
    l.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


def test_spatial_lm_loss_no_coords():
    loss_fn = SpatialLMLoss(coord_weight=0.0)
    logits = torch.randn(2, 10, 100)
    labels = torch.randint(0, 100, (2, 10))
    out = loss_fn(logits, labels)
    assert "lm_loss" in out
    assert "total_loss" in out
    assert out["total_loss"].item() > 0


def test_spatial_lm_loss_with_coords():
    loss_fn = SpatialLMLoss(coord_weight=0.1)
    logits = torch.randn(2, 10, 100)
    labels = torch.randint(0, 100, (2, 10))
    pred_coords = torch.tensor([[48.0, 2.0], [35.0, 139.0]])
    true_coords = torch.tensor([[48.85, 2.35], [35.68, 139.69]])
    out = loss_fn(logits, labels, pred_coords, true_coords)
    assert "geo_loss" in out
    assert out["total_loss"] > out["lm_loss"]
