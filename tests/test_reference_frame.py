"""
tests/test_reference_frame.py
Locks the multi-reference-frame map: object-vector cells encode the object vector; an object-frame agent
solves an object-relative goal whose object moves (a global map can't), needing the HD transform; the grid
reanchors by translating with the object; and the landmark cue is used under reliability.
"""
import pytest

from src.eval.reference_frame import run_seed, NOISES


@pytest.fixture(scope="module")
def seed0():
    return run_seed(0, episodes=200)


def test_object_vector_code(seed0):
    assert seed0["decode"] < 0.3, f"OVC population should encode the object vector, got err {seed0['decode']:.3f}"


def test_reference_frame_dissociation(seed0):
    d = seed0["diss"]
    assert d["objvec"] > 0.8, f"object-frame agent should reach the moving object-relative goal, got {d['objvec']:.2f}"
    assert d["objvec"] > d["global"] + 0.4, "global map cannot track a moving object-relative goal"
    assert d["objvec"] > d["lesion_hd"] + 0.2, "lesioning HD should break the egocentric->allocentric transform"


def test_grid_reanchors_by_translation(seed0):
    r = seed0["reanchor"]
    assert r["match"] < 0.01, f"object-frame code should equal grid translated by object displacement, got {r['match']:.3f}"
    assert r["unshift"] > 10 * r["match"] + 0.01, "the un-shifted code must NOT match (the grid genuinely translates)"


def test_robust_to_object_cue_noise(seed0):
    # honest: object-relative nav re-senses and temporally averages the (unbiased) object cue, so it is
    # ROBUST to moderate cue noise (not a graceful down-weighting -- that would need biased/single-shot cues)
    rel = seed0["rel"]
    assert rel[NOISES[-1]] > 0.8, f"object-relative nav should stay robust to moderate object-cue noise, got {rel[NOISES[-1]]:.2f}"
