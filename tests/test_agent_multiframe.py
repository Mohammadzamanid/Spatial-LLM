"""
tests/test_agent_multiframe.py
Locks the unified multi-reference-frame agent's double dissociation: one agent navigates in both a global
(grid) and an object (object-vector + HD) frame; lesioning the grid kills the global frame only, lesioning
the object-vector cells kills the object frame only, and lesioning head-direction kills both.
"""
import pytest

from src.eval.agent_multiframe import run_seed


@pytest.fixture(scope="module")
def seed0():
    return run_seed(0, episodes=120)


def test_intact_navigates_both_frames(seed0):
    assert seed0["global"]["none"] > 0.8 and seed0["object"]["none"] > 0.8


def test_double_dissociation(seed0):
    # grid lesion: global fails, object intact
    assert seed0["global"]["grid"] < 0.5 and seed0["object"]["grid"] > 0.8
    # object-vector lesion: object fails, global intact
    assert seed0["object"]["object"] < 0.5 and seed0["global"]["object"] > 0.8


def test_hd_lesion_breaks_both(seed0):
    assert seed0["global"]["hd"] < 0.5 and seed0["object"]["hd"] < 0.5
