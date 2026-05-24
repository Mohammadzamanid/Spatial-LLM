"""Tests for spatial evaluation metrics."""
import pytest
from src.eval.metrics import haversine_km, mean_haversine_error, bbox_iou, exact_match


def test_haversine_same_point():
    assert haversine_km(51.5, -0.1, 51.5, -0.1) == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance():
    # London to Paris ≈ 340 km
    dist = haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
    assert 330 < dist < 350, f"London-Paris expected ~340km, got {dist:.1f}km"


def test_mean_haversine_error():
    preds = [(51.5, -0.1), (48.8, 2.3)]
    trues = [(51.5, -0.1), (48.8, 2.3)]
    metrics = mean_haversine_error(preds, trues)
    assert metrics["mean_km"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["within_1km"] == pytest.approx(1.0)


def test_bbox_iou_perfect():
    box = (0.0, 0.0, 1.0, 1.0)
    assert bbox_iou(box, box) == pytest.approx(1.0)


def test_bbox_iou_no_overlap():
    assert bbox_iou((0, 0, 1, 1), (2, 2, 3, 3)) == pytest.approx(0.0)


def test_exact_match():
    preds = ["Paris", "London", "Tokyo"]
    refs  = ["Paris", "Berlin", "Tokyo"]
    assert exact_match(preds, refs) == pytest.approx(2 / 3)
