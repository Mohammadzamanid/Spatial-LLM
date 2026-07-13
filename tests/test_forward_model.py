"""Tests for the forward model + efference copy (GAPS.md agency frontier, organ 3).

One self-supervised forward model (predict the next sensation from sensation + efference copy) grounds BOTH a sense
of agency (self- vs world-caused, from its prediction error, never labelled) and motor control (a Smith predictor
through the sensory delay). The efference copy is the load-bearing cause: remove it and agency collapses to chance.
Self and world perturbations are magnitude-matched, so only predictability-given-the-efference-copy separates them.
"""
from src.eval.forward_model import run_seed


def test_agency_and_motor_control_emerge_from_one_forward_model():
    o = run_seed(0)

    # (A) a sense of agency emerges: self-caused change is predicted (low error), world-caused is not
    assert o["err_world"] > 5 * o["err_self"], "self-caused change is predicted; world-caused is not"
    assert o["agency_auc"] > 0.85, "self/world is recoverable from the prediction error alone (never labelled)"

    # (B) sensory attenuation: the self-caused sensation is predicted away
    assert o["attenuation"] < 0.2, "self-caused sensation is attenuated (you can't tickle yourself)"

    # (C) the efference copy is the cause: without it, the errors equalise and agency collapses to chance
    assert 0.4 < o["agency_auc_noEC"] < 0.6, "no efference copy -> agency collapses to chance"
    assert o["err_self_noEC"] > 3 * o["err_self"], "without the efference copy the self-caused change is unpredictable too"

    # (D) the same model controls the body: equal at zero delay, forward model pulls ahead as delay grows
    assert abs(o["track_fm_d0"] - o["track_stale_d0"]) < 0.02, "at zero delay the two controllers are equivalent"
    assert o["track_stale_d6"] > o["track_stale_d0"] + 0.5, "stale-feedback control degrades badly with delay"
    assert o["track_fm_d6"] < 0.5 * o["track_stale_d6"], "the forward model compensates for the delay (Smith predictor)"
