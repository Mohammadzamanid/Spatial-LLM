"""Tests for explicit uncertainty that drives behavior (GAPS.md #7).

A calibrated uncertainty read out of the grid population (the reconstruction residual = inter-module
inconsistency) should be calibrated to the true decode error under INDEPENDENT module drift but blind under
SHARED drift; it should DRIVE emergent Bayesian cue re-weighting (the reliability law the repo left open) where
a reliability-blind head cannot; and behavior should follow the BELIEF (inflated residual) not the truth.
Measured, never in a loss.
"""
from src.eval.uncertainty_behavior import run_seed


def test_uncertainty_is_calibrated_and_drives_behavior():
    o = run_seed(0)

    # (A) CALIBRATED population uncertainty: tracks true error under independent drift, blind under shared
    assert o["calib_corr_independent"] > 0.7, "the residual should be calibrated to the true error"
    assert o["calib_corr_shared"] < 0.4, "shared drift is confidently wrong (residual can't see coherent drift)"
    assert o["rho_growth_independent"] > 1.5, "uncertainty should rise over path integration"
    assert o["reset_drop"] > 0.5, "uncertainty should drop at a cue (re-anchor)"

    # (B) it DRIVES emergent Bayesian reliability weighting; a reliability-blind head cannot
    assert o["weight_slope_aware"] > 0.4, "the head's weight should track the inverse-variance optimum"
    assert o["weight_corr_aware"] > 0.8, "and track its shape tightly"
    assert o["weight_slope_blind"] < 0.25, "a reliability-blind head can only average (falsifier)"

    # (C) behavior follows the BELIEF (inflated residual), not the truth; the blind head does not respond
    assert o["belief_delta_aware"] > 0.1, "inflating the residual should raise landmark trust"
    assert o["belief_delta_blind"] < 0.05, "the blind head has no residual to act on (falsifier)"
    assert o["crossover_shift"] > 5, "the re-anchor threshold should move with landmark reliability"
