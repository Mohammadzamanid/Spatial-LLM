"""Tests for top-down feedback closing the reciprocal loop (GAPS.md unidirectional-integration item).

A learned top-down path (goal -> gain modulation of the spatial cortex, under a conserved attention budget),
trained only for goal-directed precision, should EMERGENTLY over-represent the goal (gain concentrates near it --
Dupret 2010), decode better near the goal than a FEEDFORWARD read-only model, show the attention trade-off
(better near, worse far), and collapse when fed the WRONG goal. Nothing about enhancing the goal is hardcoded.
"""
from src.eval.topdown_feedback import run_seed


def test_topdown_feedback_reshapes_the_spatial_cortex():
    o = run_seed(0)

    # (A) the map reorganises toward the goal -- emergent, never in the loss
    assert o["over_repr_corr"] > 0.15, "the learned top-down gain should concentrate on cells near the goal"

    # (B) the reciprocal loop beats the read-only (feedforward) pipeline where precision matters
    assert o["near_topdown"] < o["near_feedforward"] - 0.01, "top-down feedback should decode better near the goal"

    # (C) the attention trade-off: better where attended, worse elsewhere (a limited budget)
    assert o["far_topdown"] > o["near_topdown"] + 0.03, "attention is reallocated -- far decoding is sacrificed"

    # (D) falsifier: the feedback must MATCH the goal, not merely be present
    assert o["near_shuffled"] > o["near_topdown"] + 0.03, "the wrong goal enhances the wrong region"
