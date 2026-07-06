"""Tests for astrocyte-gated slow plasticity (GAPS.md Tier 5, #B4).

A slow astrocyte gate that throttles plasticity at importance-tagged synapses should retain old tasks in a
continual stream better than a UNIFORM plasticity reduction of the SAME total ‖Δw‖ (it is WHERE plasticity is
throttled, not less of it), and a FAST astrocyte should give no advantage over its matched uniform control (the
gain needs the slow timescale). All measured, never trained. (Trains a few small recurrent nets; ~20s.)
"""
from src.eval.astrocyte_plasticity import run_seed


def test_astrocyte_targeted_plasticity_signatures():
    o = run_seed(0)

    # ordering: the slow astrocyte forgets least; a matched uniform reduction is in between; ungated forgets most
    assert o["ret_slow"] < o["ret_uniform"] < o["ret_ungated"], "slow astrocyte < matched uniform < ungated"

    # (A) TARGETING beats a MATCHED-budget uniform reduction — the gain is from WHERE plasticity is throttled
    assert o["targeting_gain"] > 0.005, "targeted throttle should beat the matched uniform reduction"

    # matched plasticity: the slow and uniform conditions applied the same total ‖Δw‖ (so it is not 'less learning')
    assert abs(o["dw_slow"] - o["dw_uniform"]) / (o["dw_slow"] + 1e-9) < 0.05, "budgets must be matched"

    # (B) TIMESCALE FALSIFIER: a fast astrocyte gives ~no advantage over its matched uniform (needs the slow gate)
    assert abs(o["falsifier_gain"]) < 0.02, "fast astrocyte should match its uniform control"

    # (C) the astrocyte helps vs FULL plasticity too
    assert o["load_gain"] > 0.03, "slow astrocyte should retain better than ungated e-prop"
