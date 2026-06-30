"""
tests/test_local_3d_order.py
Locks the bat 3D-grid regime: a local-order (blue-noise) field code has high local order but low global
lattice (the bat signature), separable from a true 3D lattice (high both) and random points (low both).
"""
import pytest
from src.eval.local_3d_order import run_seed


@pytest.fixture(scope="module")
def seed0():
    return run_seed(0)


def test_local_order_without_global_lattice(seed0):
    r = seed0
    # bat-like (local-order): high local order, low global lattice
    assert r["local_order"]["local"] > 0.85, "local-order code should have regular nearest-neighbor spacing"
    assert r["local_order"]["global"] < 0.2, "local-order code should NOT form a global lattice"
    # a true lattice: high on BOTH; random: low local order
    assert r["lattice"]["global"] > 0.5 and r["lattice"]["local"] > 0.85, "a lattice should be ordered globally and locally"
    assert r["random"]["local"] < r["local_order"]["local"] - 0.15, "random points should lack local order"
