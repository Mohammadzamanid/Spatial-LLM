"""Tests for the integrated embodied agent (GAPS.md agency frontier, integration capstone).

The five agency organs wire into ONE autonomous loop: with no scripted goal the full agent keeps its drives bounded
(survives) where a null random-action agent's drives run away, and each organ is load-bearing on the axis where it
acts -- planning reaches goals behind the obstacle a reactive controller cannot; intrinsic motivation discovers the
resources early and covers the world; goal generation arbitrates to the needed resource when a drive is urgent.
"""
from src.eval.embodied_agent import run_seed


def test_integrated_agent_is_autonomous_and_each_organ_acts():
    o = run_seed(0)

    # (A) autonomous & competent: the full loop survives (bounded drive) where random action floods
    assert o["drive_full"] < o["drive_null"] - 1.0, "the integrated agent keeps its drives far lower than random action"

    # (B) each organ on its own axis
    # planning: reach goals behind the obstacle
    assert o["reach_plan"] > 0.8, "world-model rollouts reach goals behind the obstacle"
    assert o["reach_greedy"] < 0.2, "a reactive go-straight controller is stuck at the obstacle"
    # intrinsic motivation: discover early + cover the world, vs undirected random action
    assert o["disc_intrinsic"] < o["disc_null"], "directed exploration discovers the resources earlier than random"
    assert o["cover_intrinsic"] > o["cover_null"] + 0.1, "and covers more of the world"
    # goal generation: arbitrate to the needed resource when a drive is urgent
    assert o["arb_full"] > 0.3, "when a drive is urgent the agent heads to the needed resource"
