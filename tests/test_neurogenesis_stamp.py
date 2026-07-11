"""Tests for adult neurogenesis — temporal stamping + reduced interference (GAPS.md neurogenesis item).

With NO time encoded and random per-event content, a dentate gyrus whose newborn cells pass a hyper-excitable /
hyper-plastic maturation window before freezing should EMERGENTLY (A) stamp time -- code overlap tracks temporal
proximity, near-vs-far decodable from the code -- and (B) reduce interference -- old memories retained, recall
flat across age -- where a STATIC DG (no turnover) shows neither (no stamp, catastrophic recency). Measured.
"""
from src.eval.neurogenesis_stamp import run_seed


def test_temporal_stamping_and_reduced_interference_emerge():
    o = run_seed(0)

    # (A) temporal stamping emerges only with cohort turnover (content carries no time)
    assert o["stamp_corr_neuro"] < -0.3, "neurogenic code overlap should track temporal proximity"
    assert abs(o["stamp_corr_static"]) < 0.15, "a static DG carries content, not time"
    assert o["near_far_auc_neuro"] > 0.75, "near-vs-far in time should be decodable from the neurogenic code"
    assert 0.4 < o["near_far_auc_static"] < 0.6, "static: near/far at chance"

    # (B) reduced interference: neurogenic retains old memories and is age-flat; static forgets catastrophically
    assert o["old_recall_neuro"] > o["old_recall_static"] + 0.03, "neurogenesis should retain old memories better"
    assert o["retention_gap_neuro"] < 0.2, "neurogenic recall is flat across memory age"
    assert o["retention_gap_static"] > 0.3, "static recall is dominated by recent memories (catastrophic recency)"
