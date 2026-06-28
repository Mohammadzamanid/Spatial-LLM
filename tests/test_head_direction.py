"""
tests/test_head_direction.py
Locks the head-direction organ: (1) HD cells + a ring attractor EMERGE from angular path integration
(trained >> untrained on HD-tuned fraction, heading decode, ring correlation); (2) the emergent HD system's
noisy integration causes heading-dominated drift that a visual reset bounds.
"""
import pytest

from src.eval.head_direction import run_seed


@pytest.fixture(scope="module")
def seed0():
    return run_seed(0)


def test_hd_cells_and_heading_coding_emerge(seed0):
    """The clean, training-specific emergence signatures: HD-cell tuning and accurate, stable heading
    maintenance. (We deliberately do NOT assert a trained>untrained ring-correlation -- a ring-shaped
    manifold is partly inherent to recurrent integration; see the module docstring.)"""
    em, un = seed0["em"], seed0["em_un"]
    assert em["decode_err"] < 15.0, f"trained should decode heading well, got {em['decode_err']:.1f}deg"
    assert em["decode_err"] < 0.4 * un["decode_err"], "training should sharply improve heading decode"
    assert em["hd_frac"] > un["hd_frac"] + 0.15, f"HD cells should emerge: {em['hd_frac']:.2f} vs {un['hd_frac']:.2f}"
    assert em["ring_corr"] > 0.7, f"trained population should trace a ring manifold, got {em['ring_corr']:.2f}"


def test_heading_drift_and_visual_reset(seed0):
    d = seed0["drift"]
    # heading-dominated drift is real, and the visual reset bounds both heading and position error
    assert d["no_reset"]["heading_err"] > d["reset"]["heading_err"] + 5.0, "visual reset should cut heading drift"
    assert d["reset"]["pos_err"] < d["no_reset"]["pos_err"], "visual reset should cut position drift"
