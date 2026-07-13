"""Tests for imagination -> planning (GAPS.md agency frontier, organ 4).

Planning emerges from multi-step imagination over a LEARNED forward model: building only the imagination (roll the
model forward) + a goal + a generic 'closest imagined approach' selection (MPC), with no trained policy and no
hand-coded planner, the agent solves a detour a reactive model-free agent cannot, revalues to moved goals zero-shot,
needs the multi-step rollout (H=1 is myopic), and collapses if the forward model is corrupted.
"""
from src.eval.imagination_planning import run_seed, H_LIST


def test_planning_emerges_from_multistep_imagination():
    o = run_seed(0)
    hmax, hmin = H_LIST[-1], H_LIST[0]

    # (A) planning emerges: the imagination-planner solves the detour with no trained policy
    assert o[f"h{hmax}"] > 0.8, "the imagination-planner solves the detour task"

    # (B) model-based, not habit: it beats a reactive go-straight agent and revalues to moved goals
    assert o["reactive"] < 0.35, "a reactive model-free agent is stuck at the obstacle"
    assert o[f"h{hmax}"] > o["reactive"] + 0.5, "planning clearly beats the model-free habit"
    assert o["reval_planner"] > o["reval_reactive"], "the planner revalues to moved goals more flexibly"

    # (C) planning requires the MULTI-STEP rollout: H=1 (bare one-step model) is myopic
    assert o[f"h{hmin}"] < o[f"h{hmax}"] - 0.3, "one-step imagination is myopic; success needs horizon"

    # (D) it rides on imagination accuracy: corrupt the forward model and planning collapses
    assert o["broken"] < 0.2, "a broken forward model destroys planning (it plans over garbage)"
