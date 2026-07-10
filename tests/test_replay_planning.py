"""Tests for replay that computes — reverse-replay credit assignment + forward-replay planning (GAPS.md #6).

A direction-free SCALAR priority (|TD error|, prioritized sweeping) should make value updates sweep BACKWARD
from the reward (reverse replay = credit assignment) — a clean sweep where RANDOM-order replay is at chance and
needs many times more backups. The SAME learned predictive map, read FORWARD by a greedy value-ascent rollout,
should route around the barrier to the goal (forward replay = planning) where an untrained value cannot. Measured.
"""
from src.eval.replay_planning import run_seed


def test_reverse_credit_and_forward_planning():
    o = run_seed(0)

    # A. REVERSE replay = credit assignment: emergent reverse sweep (direction never encoded)
    assert o["reverse_frac_prioritized"] > 0.9, "prioritized replay should sweep backward from the reward"
    assert 0.4 < o["reverse_frac_random"] < 0.6, "random replay has no directional structure (chance)"

    # B. FORWARD replay = planning: the same map, read forward, solves the maze
    assert o["forward_frac_planning"] > 0.9, "the planning rollout should run forward to the goal"
    assert o["plan_success_trained"] > 0.9, "the replay-trained map should be plannable"
    assert o["plan_success_untrained"] < 0.1, "an untrained value has no gradient to plan with (falsifier)"

    # D. PAYOFF: prioritized reaches a plannable map in far fewer backups than random
    assert o["backup_speedup"] > 3.0, "prioritized replay should be several times cheaper than random"
