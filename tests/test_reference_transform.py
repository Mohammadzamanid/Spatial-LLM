"""Tests for the RSC/PPC egocentric->allocentric transform + emergent gain fields (GAPS.md Tier 1/2).

A plain MLP trained only on allocentric output should learn the head-direction-gated rotation (generalize to
head directions held out of training), fail without the correct heading (shuffled/removed), and develop
multiplicative ego*HD gain-field units (Zipser-Andersen) — measured, not imposed.
"""
from src.eval.reference_transform import run_seed


def test_ego_to_allo_transform_and_gain_fields():
    o = run_seed(0)

    # (A) LEARNED THE TRANSFORM: generalizes to head directions never seen in training (near-zero error)
    assert o["rmse_heldout_norm"] < 0.15, "should generalize to unseen head directions (learned the rotation)"

    # (C) FALSIFIERS: wrong or missing head direction breaks the transform
    assert o["falsifier_gap"] > 0.3, "a shuffled head direction should wreck the transform"
    assert o["rmse_no_hd"] > o["rmse_heldout_hd"] + 0.5, "removing head direction should make it impossible"

    # (B) GAIN FIELDS EMERGE: multiplicative ego*HD tuning develops with training, above the untrained baseline
    assert o["gain_emergence"] > 0.03 and o["gain_field_frac"] > 0.1, "gain-field units should emerge, not be at init"
