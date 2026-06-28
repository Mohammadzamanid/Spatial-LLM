"""
tests/test_landmark_anchoring.py
Locks the dynamic landmark-reanchoring result: reanchoring the grid phase to a landmark corrects allocentric
drift; allocentric (global) and egocentric (landmark-relative) positions coexist; and a noisier landmark
helps less (reliability matters).
"""
import pytest

from src.eval.landmark_anchoring import run_seed, NOISES


@pytest.fixture(scope="module")
def seed0():
    return run_seed(0)


def test_reanchoring_corrects_allocentric_drift(seed0):
    assert seed0["anchor_allo"] < 0.6 * seed0["pi"], (
        f"landmark reanchoring should bound allocentric drift: pi {seed0['pi']:.2f} -> "
        f"anchored {seed0['anchor_allo']:.2f}")


def test_allocentric_and_egocentric_coexist(seed0):
    # both frames are read at the same time and are usable (bounded), not at arena-scale chance
    assert seed0["anchor_allo"] < 1.5, "allocentric (global) position should be usable"
    assert seed0["anchor_ego"] < 1.5, "egocentric (landmark-relative) position should be usable"


def test_reliability_dependence(seed0):
    rel = seed0["rel"]
    assert rel[NOISES[-1]] >= rel[NOISES[0]] - 0.05, "a noisier landmark should not help more than a clean one"
