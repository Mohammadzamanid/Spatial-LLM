"""Tests for semantic warping of the cognitive map (GAPS.md purely-geographic-map item).

A capacity-limited code reconstructing [position, value] stays a spatial map yet its metric WARPS by a non-spatial
concept (mixed selectivity, Boccara 2019) ONLY when the concept is behaviourally relevant AND the perforant
(semantic) projection is present. The warp is never in the loss. Double dissociation: remove the path -> no warp;
make the concept irrelevant -> no warp. Payoff: a downstream linear probe reads the concept off the warped map,
not off the map without the perforant path.
"""
from src.eval.semantic_warp import run_seed


def test_map_warps_by_relevant_concept_via_perforant_path():
    o = run_seed(0)

    # (A) the map warps by concept while staying spatial (mixed selectivity)
    assert o["warp_rel"] > 0.12, "a relevant concept warps the representational metric (control for space)"
    assert o["spatial_rel"] > 0.4, "yet the code is still strongly spatial (a warped spatial map, not a concept map)"

    # (B) double dissociation: no warp without the perforant path, and no warp without behavioural relevance
    assert o["warp_np"] < 0.08, "no perforant projection -> the map cannot warp, even though the concept is relevant"
    assert o["warp_rel"] > o["warp_np"] + 0.1, "the perforant path is what carries the concept into the map"
    assert o["dose_00"] < 0.08, "path present but concept irrelevant -> no warp (relevance is required too)"

    # (C) dose-response: warp grows with behavioural relevance
    assert o["dose_00"] < o["dose_10"] < o["dose_20"], "the map is attracted to concepts in proportion to relevance"

    # (D) payoff: the concept is readable off the warped map, but at chance without the perforant path
    assert o["probe_rel"] > o["probe_np"] + 0.15, "a downstream reader gets the concept off the warped map for free"
    assert o["probe_np"] < o["chance"] + 0.18, "without the perforant path the concept is ~unreadable (near chance)"
