"""Tests for emergent epistemic foraging (GAPS.md active-inference item).

The agent is rewarded ONLY for reaching the goal (no landmark / information-gain / exploration reward). Under
path-integration DRIFT it should EMERGENTLY detour to a landmark to relocalise before committing, beating a
sigma-blind greedy agent and random; in a NO-DRIFT world it should STOP detouring (the proof it is
uncertainty-driven, not a hardcoded landmark preference); and blind to its own uncertainty it should collapse to
the greedy rate. Measured, never in the objective.
"""
import torch

from src.eval.active_inference import (LM_CHOICES, detour_fraction, greedy_policy,
                                       planner_policy, random_policy, success_rate,
                                       value_iteration)


def test_epistemic_foraging_emerges_and_is_uncertainty_driven():
    lm = LM_CHOICES[0]
    gen = torch.Generator().manual_seed(0)
    V = value_iteration(lm, drift=True)
    V_off = value_iteration(lm, drift=False)
    V_blind = value_iteration(lm, drift=True, see_u=False)

    # (A/B) epistemic foraging emerges under drift and VANISHES without it (non-hardcoding proof)
    assert detour_fraction(V, lm, drift=True) > 0.3, "the planner should detour to relocalise under drift"
    assert detour_fraction(V_off, lm, drift=False) < 0.05, "no uncertainty to reduce -> no detour (not hardcoded)"

    # (C) it pays: uncertainty-aware planner > sigma-blind greedy > random (same goal reward for planner/greedy)
    sp = success_rate(planner_policy(V, lm, True, True), lm, gen, drift=True, n=150)
    sg = success_rate(greedy_policy(lm), lm, gen, drift=True, n=150)
    sr = success_rate(random_policy(gen), lm, gen, drift=True, n=150)
    assert sp > sg + 0.1, "relocalising should reach the goal more often than beelining"
    assert sg > sr, "even a blind goal-seeker beats random"

    # (D) it needs to sense its uncertainty: blind to u, the planner collapses toward the greedy rate
    sa = success_rate(planner_policy(V_blind, lm, True, False), lm, gen, drift=True, n=150)
    assert sa < sp - 0.05, "a u-blind planner cannot time the detour"
