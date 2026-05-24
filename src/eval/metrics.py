"""
src/eval/metrics.py
Spatial-aware evaluation metrics.
Standard NLP metrics (BLEU/ROUGE) are insufficient for spatial reasoning.
"""

import math
from typing import Sequence
import numpy as np


def haversine_km(
    pred_lat: float, pred_lon: float,
    true_lat: float, true_lon: float,
) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    dlat = math.radians(true_lat - pred_lat)
    dlon = math.radians(true_lon - pred_lon)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(pred_lat))
        * math.cos(math.radians(true_lat))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def mean_haversine_error(
    pred_coords: list[tuple[float, float]],
    true_coords: list[tuple[float, float]],
) -> dict[str, float]:
    """
    Compute mean, median, and percentile Haversine errors over a dataset.
    Each element is a (lat, lon) tuple.
    """
    errors = [
        haversine_km(p[0], p[1], t[0], t[1])
        for p, t in zip(pred_coords, true_coords)
    ]
    arr = np.array(errors)
    return {
        "mean_km": float(arr.mean()),
        "median_km": float(np.median(arr)),
        "p75_km": float(np.percentile(arr, 75)),
        "p90_km": float(np.percentile(arr, 90)),
        "within_1km": float((arr <= 1.0).mean()),
        "within_10km": float((arr <= 10.0).mean()),
        "within_100km": float((arr <= 100.0).mean()),
    }


def bbox_iou(
    pred_box: tuple[float, float, float, float],
    true_box: tuple[float, float, float, float],
) -> float:
    """
    Intersection-over-Union for geographic bounding boxes.
    Boxes: (min_lat, min_lon, max_lat, max_lon)
    """
    p_min_lat, p_min_lon, p_max_lat, p_max_lon = pred_box
    t_min_lat, t_min_lon, t_max_lat, t_max_lon = true_box

    inter_min_lat = max(p_min_lat, t_min_lat)
    inter_min_lon = max(p_min_lon, t_min_lon)
    inter_max_lat = min(p_max_lat, t_max_lat)
    inter_max_lon = min(p_max_lon, t_max_lon)

    inter_area = max(0, inter_max_lat - inter_min_lat) * max(0, inter_max_lon - inter_min_lon)
    pred_area = (p_max_lat - p_min_lat) * (p_max_lon - p_min_lon)
    true_area = (t_max_lat - t_min_lat) * (t_max_lon - t_min_lon)
    union_area = pred_area + true_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Exact string match rate (case-insensitive, stripped)."""
    matches = sum(
        p.strip().lower() == r.strip().lower()
        for p, r in zip(predictions, references)
    )
    return matches / len(predictions) if predictions else 0.0
