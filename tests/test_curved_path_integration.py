"""Tests for non-Euclidean path integration — curvature from self-motion (GAPS.md 3-D/non-Euclidean item).

A flat head-direction / grid path-integrator on a curved manifold should inherit the Gauss-Bonnet holonomy: the
parallel-transport rotation around a closed loop equals the enclosed area x curvature (= solid angle), a
geodesic triangle with three right angles gives pi/2, the holonomy vanishes in flat space, and a flat compass
then mis-homes by the holonomy. Measured, never put into the code.
"""
import math

from src.eval.curved_path_integration import run_seed


def test_curvature_from_self_motion():
    o = run_seed(0)

    # (A) holonomy = enclosed area x curvature (Gauss-Bonnet)
    assert 0.9 < o["holonomy_vs_solidangle_slope"] < 1.1, "holonomy should equal the enclosed solid angle"
    assert o["holonomy_vs_solidangle_corr"] > 0.95, "and track it tightly across radii/sizes"
    assert o["gauss_bonnet_rel_residual"] < 0.1, "calibrated to <10% (it IS area x curvature)"
    assert 1.3 < o["triangle_right_angle_excess"] < 1.85, "3 right angles -> holonomy pi/2 (=1.571)"

    # (B) flat-space falsifier: loops close (holonomy 0); curvature dose-response monotone
    assert o["flat_holonomy"] < 0.2, "flat space has no holonomy (loops close)"
    assert o["curvature_dose_monotone"] == 1.0, "holonomy grows as curvature rises at fixed area"

    # (C) behavioural consequence: a flat compass mis-homes on a curved world, not on a flat one
    assert o["homing_miss_curved"] > 1.0, "the flat compass mis-homes after a loop on the sphere"
    assert o["homing_miss_flat"] < 0.2, "but closes in flat space (falsifier)"
