"""
tests/test_agent_deadreckoning.py
Locks the unified dead-reckoning stack: the agent estimates heading (HD) and position (grid) from
self-motion alone. Oracle is the floor; the HD organ in the loop inflates position error; correcting BOTH
organs bounds it; lesioning either is catastrophic; homing works intact and is abolished by lesions.
"""
import pytest

from src.eval.agent_deadreckoning import run_seed


@pytest.fixture(scope="module")
def seed0():
    return run_seed(0)


def test_localization_stack(seed0):
    loc = seed0["loc"]
    assert loc["oracle"] < 0.3, f"oracle (true heading) should be near-perfect, got {loc['oracle']:.3f}"
    assert loc["none"] > loc["oracle"] + 0.3, "HD drift in the loop should inflate position error"
    assert loc["both"] < loc["none"], "correcting both organs should beat no correction"
    assert loc["both"] < 0.5, f"with both corrections position should be bounded, got {loc['both']:.3f}"
    assert loc["lesion_hd"] > loc["both"] + 0.5, "lesioning HD should be catastrophic"
    assert loc["lesion_grid"] > loc["both"] + 0.5, "lesioning grid should be catastrophic"


def test_homing(seed0):
    home = seed0["home"]
    assert home["both"] < 1.0, f"intact agent should home accurately, got {home['both']:.3f}"
    assert home["lesion_hd"] > home["both"], "lesioning HD should abolish homing"
    assert home["lesion_grid"] > home["both"], "lesioning grid should abolish homing"
