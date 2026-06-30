"""
tests/test_egocentric_anchors.py
Locks coexisting egocentric anchor frames: the combined population decodes the egocentric vector to the
centre, an object, and the nearest boundary simultaneously (coexistence), and each frame decodes from its
own cells but not from another anchor's (specificity).

The claim is about the RELATIVE structure (combined ~ own; other >> own), which is independent of anchor
magnitude. We do NOT assert a single absolute error threshold across anchors: the object's egocentric
vector spans ~2x the range of the centre/boundary vectors (up to 2*R*sqrt(2) vs R*sqrt(2)), so its absolute
decode error is irreducibly larger regardless of cell count -- a fixed threshold would be a magnitude
artefact, not a test of the science.
"""
import pytest
from src.eval.egocentric_anchors import run_seed, ANCHORS


@pytest.fixture(scope="module")
def seed0():
    return run_seed(0)


def test_anchors_coexist(seed0):
    # Coexistence: the combined population retains every anchor -- decoding an anchor's egocentric vector
    # from the combined population is about as good as from its own cells, and far better than from a
    # wrong single population. (Magnitude-independent: compares combined to that same anchor's own/other.)
    for anc in ANCHORS:
        c = seed0[anc]
        assert c["combined"] < c["own"] + 0.2, f"{anc}: combined population should retain the anchor (coexistence)"
        assert c["combined"] < c["other"], f"{anc}: combined should beat a wrong single population"


def test_anchor_specificity(seed0):
    # Specificity: each anchor decodes from its OWN cells, far better than from another anchor's cells.
    for anc in ANCHORS:
        c = seed0[anc]
        assert c["other"] > c["own"] + 0.2, f"{anc}: another anchor's cells should NOT decode it (specificity)"
        assert c["own"] < 0.5 * c["other"], f"{anc} should decode from its own cells, not another's"


def test_anchors_genuinely_decodable(seed0):
    # Absolute sanity floor (magnitude-aware): the small-range anchors decode tightly; the object's vector
    # spans ~2x the range so it gets a 2x-looser bound. No anchor is undecodable.
    assert seed0["center"]["own"] < 0.5
    assert seed0["boundary"]["own"] < 0.5
    assert seed0["object"]["own"] < 1.0
