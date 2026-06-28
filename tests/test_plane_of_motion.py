"""
tests/test_plane_of_motion.py
Locks the plane-of-motion result: PCA recovers the motion plane; the plane-aligned 2D grid localizes 3D
position orientation-invariantly (flat across tilt); a fixed-plane grid degrades at steep tilt.
(We do NOT assert a decode advantage over a naive 3D grid -- that was an honest negative; see the module.)
"""
import pytest

from src.eval.plane_of_motion import run_seed, TILTS


@pytest.fixture(scope="module")
def seed0():
    return run_seed(0)


def test_pca_recovers_motion_plane(seed0):
    for t in TILTS:
        assert seed0["recov"][t] < 0.1, f"PCA should recover the motion-plane normal at tilt {t}, got {seed0['recov'][t]:.3f}"


def test_aligned_is_orientation_invariant(seed0):
    a = [seed0["aligned"][t] for t in TILTS]
    assert max(a) < 0.3, f"plane-aligned 3D decode should be accurate, got {max(a):.3f}"
    assert max(a) - min(a) < 0.12, "plane-aligned accuracy should be ~flat across plane tilt (orientation-invariant)"


def test_fixed_plane_degrades_with_tilt(seed0):
    # at the steepest tilt the fixed (horizontal) grid should be worse than the aligned one
    assert seed0["fixed"][TILTS[-1]] > seed0["aligned"][TILTS[-1]], "fixed-plane grid should degrade as the plane tilts"
