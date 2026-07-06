"""Tests for volatility-adaptive meta-learning (GAPS.md Tier 5, #B3).

A GRU meta-trained ONLY to predict the next observation should, at frozen weights, reveal a learning rate that
RISES with volatility and — the non-circular signature — FALLS under pure stochasticity (highest variance),
while an untrained net stays flat. All measured post-hoc, never trained. (Trains one net; ~40-60s.)
"""
from src.eval.meta_learning import run_seed


def test_volatility_adaptive_learning_rate():
    o = run_seed(0, iters=1800)

    # (A) tracks volatility: learning rate higher when the world jumps than when it is stable
    assert o["vol_gain"] > -0.01, "learning rate should not DROP going stable -> volatile"

    # (B) THE DISSOCIATION (falsifier): under pure stochasticity the rate is LOWER than under volatility, and
    # lower than under stability, even though the stochastic block has the HIGHEST observation variance --
    # a naive 'learn faster when errors are big' account would predict the opposite.
    assert o["dissoc"] > 0.03, "stochasticity should LOWER the learning rate vs volatility (dissociation)"
    assert o["stable_vs_stoch"] > 0.0, "stochastic learning rate should sit below the stable one"

    # (C) LEARNED, not architectural: an untrained net does not differentiate the blocks
    assert abs(o["untr_vol_gain"]) < 0.05, "untrained control should be ~flat across blocks"

    # (D) FUNCTIONAL: the adaptive net beats the best single fixed learning rate on the mixed session
    assert o["perf_ratio"] < 1.05, "adaptive net should roughly match or beat the best fixed-alpha predictor"
