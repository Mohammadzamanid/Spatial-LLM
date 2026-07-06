"""Tests for the faithfulness capstone: grid cells under a non-backprop rule (GAPS.md Tier 5 capstone).

A recurrent cortex trained by RFLO (eligibility trace × fixed-random-feedback learning signal — no weight
transport, no backprop-through-time) should LEARN path integration and grow the emergent periodic grid code,
comparable to backprop; a shuffled-feedback control should fall to the untrained floor on the grid code. All
measured, never trained. (Trains a few small recurrent nets; ~30s.)
"""
from src.eval.emergent_grid_bio import run_seed


def test_grid_cells_emerge_under_non_backprop_rule():
    o = run_seed(0)

    # (A) RFLO LEARNS path integration without weight transport: place-loss far below the untrained readout,
    # and comparable to backprop
    assert o["rflo_place_loss"] < 0.5 * o["untrained_place_loss"], "RFLO should learn the place-prediction task"
    assert o["rflo_place_loss"] < 3.0 * o["backprop_place_loss"], "RFLO should roughly match backprop on the task"

    # (B) the EMERGENT grid code (rate-map periodicity — never in the loss) appears under RFLO, above the
    # untrained floor and comparable to backprop
    assert o["rflo_mean_periodicity"] > o["untrained_mean_periodicity"] + 0.02, "grid periodicity should emerge under RFLO"
    assert o["rflo_frac_periodic"] > o["untrained_frac_periodic"] + 0.08, "more units become periodic under RFLO"
    assert o["rflo_mean_periodicity"] > 0.9 * o["backprop_mean_periodicity"], "RFLO grid code ~matches backprop"

    # (C) FALSIFIER: shuffled feedback develops no grid code beyond the untrained floor (needs consistent feedback)
    assert o["rflo_mean_periodicity"] > o["shuffled_mean_periodicity"] + 0.02, "consistent feedback grows the grid code"
    assert abs(o["shuffled_mean_periodicity"] - o["untrained_mean_periodicity"]) < 0.04, "shuffled ~ untrained floor"
