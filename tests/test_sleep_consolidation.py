"""Tests for the sleep triple-coupling (GAPS.md Tier 5, #C7).

Nesting replay in the SO->spindle windows should make consolidation SELECTIVE (tagged memories win the scarce
windows by competition — well above the proportional floor) and TIMED (every replay consolidates vs random
replay wasting DOWN-state events), and removing the SO structure should collapse the selection to the floor. All
measured, never imposed.
"""
from src.eval.sleep_consolidation import run_seed


def test_sleep_coupling_selects_and_times():
    o = run_seed(0)

    # (A) SELECTIVITY (emergent, not by construction): coupled consolidation is far more tagged-selective than
    # uncoupled, and well above the proportional (no-selection) floor
    assert o["sel_coupled"] > o["proportional"] + 0.10, "the coupling should select tagged memories above the proportional floor"
    assert o["selectivity_gap"] > 0.05, "coupled should be more selective than uncoupled"

    # (B) COORDINATION: at matched replay count, coupled replay consolidates far more events than random-timed
    assert o["coordination_gap"] > 0.20 and o["coord_coupled"] > o["coord_uncoupled"] + 0.20

    # (C) FALSIFIER: without the SO structure, the selection collapses to the proportional floor
    assert o["sel_noso"] < o["sel_coupled"] - 0.10, "selection should need the SO nesting"
    assert abs(o["sel_noso"] - o["proportional"]) < 0.07, "no-SO consolidation is indiscriminate (proportional)"
