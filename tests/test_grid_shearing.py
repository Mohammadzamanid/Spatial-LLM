"""Tests for grid shearing — the hexagonal grid deforms with environmental geometry (GAPS.md Tier 2).

The grid should stay hexagonal in a square arena but SHEAR in a trapezoid under the same boundary anchoring, and
the deformation must require BOTH the polarized geometry AND the anchoring (a double dissociation) — so the shear
emerges from the geometry-mismatched boundary fix, and is never drawn in (Krupic 2015; Stensola 2015).
"""
from src.eval.grid_shearing import run_seed


def test_grid_shears_with_geometry():
    o = run_seed(0)

    # (A) SHEARING: hexagonal in a square, deformed in a trapezoid (same anchoring)
    assert o["grid_square_anchor"] > 0.5, "the grid should be clearly hexagonal in the square baseline"
    assert o["grid_trapezoid_anchor"] < o["grid_square_anchor"] - 0.4, "it should deform in the trapezoid"
    assert o["shear_drop"] > 0.4
    # dose-response: half the shear deforms less than the full shear
    assert o["grid_trapezoid_half"] > o["grid_trapezoid_anchor"]

    # (B) DOUBLE DISSOCIATION: the deformation needs BOTH the geometry and the anchoring
    assert o["grid_trapezoid_noanchor"] > 0.5, "geometry alone (no anchoring) should NOT deform the grid"
    assert o["falsifier_gap"] > 0.4, "only trapezoid+anchoring deforms"
