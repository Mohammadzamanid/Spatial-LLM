"""Tests for the unified agent (GAPS.md integration capstone).

One agent whose only objective is to survive should compose a grid position sense, an uncertainty read-out,
landmark relocalisation and interoceptive drives into a coherent animal: an N-organ lesion dissociation (all four
load-bearing, each ablation failing in its own way) and an emergent super-additive interaction (the uncertainty
organ is worthless without the landmark organ). Nothing about which resource or when to relocalise is hardcoded.
"""
from src.eval.unified_agent import run_seed


def test_organs_are_jointly_necessary_and_form_a_circuit():
    o = run_seed(0)
    intact = o["drive_intact"]

    # (A) N-organ lesion dissociation: removing ANY organ raises drive (lower drive = healthier)
    assert o["drive_no_grid"] > intact + 10, "no position sense -> can't navigate (catastrophic)"
    assert o["drive_no_uncertainty"] > intact + 2, "no uncertainty read-out -> can't tell when it's lost"
    assert o["drive_no_landmark"] > intact + 2, "no landmark -> can't undo drift"
    assert o["drive_no_drive"] > intact + 1, "no interoception -> can't tell which deficit is killing it"

    # (B) emergent complementarity: the uncertainty organ helps only WITH the landmark organ (super-additive)
    assert o["cost_unc_with_lm"] > o["cost_unc_without_lm"] + 2, "uncertainty helps more when landmarks are present"
    assert o["cost_unc_without_lm"] < 3, "knowing you're lost is worthless if you cannot re-anchor"
