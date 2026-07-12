"""Tests for intrinsic motivation (GAPS.md agency frontier, organ 1).

Self-directed exploration and mastery emerge from an internal drive alone (no external reward, no goal): the agent
is rewarded only by improving its own world model. A learning-progress agent masters the learnable environment
where random action does not, and escapes the noisy-TV trap (irreducible randomness) that a pure-novelty agent is
caught in -- because error REDUCTION is not fooled by noise while error itself is.
"""
from src.eval.intrinsic_motivation import run_seed


def test_self_directed_mastery_and_noisy_tv_escape():
    o = run_seed(0)

    # (A) mastery emerges from the intrinsic drive alone; random action masters far less
    assert o["mastered_progress"] > o["mastered_random"] + 10, "the drive organises exploration; random does not"
    assert o["mastered_progress"] >= 0.9 * o["n_learn"], "learning progress masters most of the learnable world"

    # (B) noisy-TV falsifier: learning progress escapes the trap that pure novelty falls into
    assert o["dwell_novelty"] > o["dwell_progress"] + 0.1, "pure novelty is trapped by irreducible noise"
    assert o["dwell_progress"] < 0.2, "learning progress samples the noise then leaves"

    # (C) and it pays off: learning progress reaches mastery faster than novelty; random never does
    assert o["steps90_progress"] < o["steps90_novelty"], "trap-avoidance makes learning progress more efficient"
    assert o["steps90_random"] >= o["steps90_progress"], "random is slowest (never reaches 90% in the horizon)"
