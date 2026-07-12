"""Tests for goal generation (GAPS.md agency frontier, organ 2).

An autotelic agent proposes its own goals by learning progress over a goal space, with no handed goal and no
difficulty schedule. A developmental curriculum emerges (proposed difficulty rises over the lifetime), the agent
masters essentially all learnable goals, and it threads the zone of proximal development between two failure modes:
'always hardest' is trapped on impossible goals (masters nothing, the goal-space noisy TV) and 'always easiest'
stalls on trivial goals.
"""
from src.eval.goal_generation import run_seed


def test_curriculum_emerges_and_threads_the_zpd():
    o = run_seed(0)

    # (A) a developmental curriculum emerges with no schedule: autotelic difficulty rises; random stays flat
    assert o["curric_autotelic"] > 0.08, "self-proposed goal difficulty rises over the lifetime (curriculum)"
    assert o["curric_autotelic"] > o["curric_random"] + 0.08, "random-goal proposals show no curriculum"
    assert o["diff_late_auto"] > o["diff_early_auto"], "late-life goals are harder than early-life goals"

    # (B) the autotelic agent masters essentially all learnable goals; the fixed strategies fail
    assert o["mastered_autotelic"] >= 0.9 * o["n_learn"], "autotelic masters the learnable goal space"
    assert o["mastered_autotelic"] > o["mastered_random"], "and more than random-goal selection"

    # (C) it threads the ZPD: 'always hardest' is trapped on impossible goals; 'always easiest' stalls
    assert o["mastered_hardest"] <= 1, "always-hardest masters ~nothing (stuck on impossible goals)"
    assert o["impossible_hardest"] > 0.8, "always-hardest wastes almost all practice on impossible goals (noisy TV)"
    assert o["mastered_easiest"] <= 2, "always-easiest stalls on trivial goals"
    assert o["frontier_autotelic"] > 0.3, "the autotelic agent self-organises onto the productive frontier"
