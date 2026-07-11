"""Tests for the unified agent learning its world (GAPS.md integration capstone, learning).

The agent is dropped in NOT knowing where resources are and must LEARN them from experience (nothing hardcoded).
REPLAY should propagate a discovered resource's value across the map far faster than plain online learning (map
accuracy a fixed window after discovery), and CLS consolidation should make a familiar world survive a
hippocampal lesion where without it the lesion is fatal. Measured.
"""
from src.eval.unified_agent_learn import run_seed


def test_agent_learns_world_with_replay_and_cls():
    o = run_seed(0)

    # (A) it learns its world: late-life drive is lower than early-life
    assert o["drive_late"] < o["drive_early"], "the agent should learn where resources are (drive falls)"

    # (B) replay propagates the discovered value across the map far faster than online-only learning
    assert o["map_acc_replay"] > 0.8, "with replay the world map is learned quickly after discovery"
    assert o["map_acc_replay"] > o["map_acc_noreplay"] + 0.1, "replay beats no-replay on map-learning speed"

    # (C) CLS consolidation makes the familiar world survive a hippocampal lesion
    assert o["postlesion_cls"] < o["postlesion_nocls"] - 8, "the consolidated world is retained through the lesion"
