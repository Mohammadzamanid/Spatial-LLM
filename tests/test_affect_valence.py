"""Tests for affect / valence (GAPS.md agency frontier, organ 5).

A global mood state emerges from reward-prediction-error momentum (Eldar-Dayan 2016): mood is a leaky integral of
RPE and biases perceived reward. Emergent: mood tracks surprise not level; its self-reinforcing feedback produces
slow swings above a critical gain (bipolar-like); those swings colour the agent's valuation of a stationary world;
and cutting the feedback (gain 0) leaves a passive read-out with no swings.
"""
from src.eval.affect_valence import run_seed


def test_mood_momentum_and_emergent_swings():
    o = run_seed(0)

    # (A) mood is the momentum of surprise: spikes at a streak's onset, decays as it becomes expected
    assert o["mom_onset"] > 0.1, "a better-than-expected streak lifts mood"
    assert abs(o["mom_steady"]) < 0.06, "mood decays back toward zero once the streak is expected"
    assert o["mom_bad_onset"] < -0.1, "a worse-than-expected streak drops mood"

    # (B) self-reinforcing swings emerge above a critical feedback gain
    assert o["moodstd_f0"] < 0.1, "with no feedback, mood is a small fast tracker"
    assert o["moodstd_f5"] > 0.5, "above the critical gain, mood self-amplifies into large swings"
    assert o["moodstd_f5"] > 4 * o["moodstd_f1"], "a clear dose-response in the feedback gain"
    assert o["autocorr_swing"] > 0.3 and o["autocorr_stable"] < 0.2, "the swings are slow (mood cycles), not fast noise"

    # (C) the swings colour valuation of a stationary world
    assert o["valuestd_swing"] > 3 * o["valuestd_stable"], "affect injects spurious value swings into a fixed world"

    # (D) falsifier: without the feedback loop there are no swings and no value distortion
    assert o["valuestd_stable"] < 0.5, "cutting the feedback leaves valuation stable (no mood distortion)"
