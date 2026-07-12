"""Tests for decoupling the map from value (GAPS.md map/policy-conflation item).

The successor representation decouples the goal-INDEPENDENT cognitive map (M) from value (V = M.R). One learned
map should serve many goals and revalue INSTANTLY when the goal moves, where a FUSED agent (value baked into its
state read-out) stays stuck on the old goal and must relearn its value from scratch. Measured.
"""
from src.eval.map_value_decouple import run_seed


def test_one_map_many_values_and_instant_revaluation():
    o = run_seed(0)

    # (A) one goal-independent map serves many goals
    assert o["sr_reuse"] > 0.9, "a single SR map should solve many goals via V = M[:, g]"

    # (B) instant revaluation when the goal moves; the fused agent stays stuck on the old goal
    assert o["sr_reval"] > 0.9, "the decoupled agent revalues for free (a lookup)"
    assert o["sr_reval"] > o["fused_stale"] + 0.3, "the fused agent cannot revalue without relearning"

    # (C) the cost of fusion: the fused agent must relearn (>0 sweeps); the decoupled one needs none
    assert o["sr_relearn_steps"] == 0, "the decoupled agent pays nothing to revalue"
    assert o["fused_relearn_steps"] > 3, "the fused agent must relearn its value for the moved goal"
