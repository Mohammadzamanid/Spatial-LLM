"""
tests/test_goal_reward.py
Locks GAPS.md #3 (goal/reward-vector coding):
 A) goal_vector: a goal-DIRECTION code emerges from navigation and is GOAL-SPECIFIC (untrained + shuffle at
    the false-positive floor). (Reduced training iters for test speed.)
 B) reward_map: reward-triggered BTSP builds an ANTICIPATORY reward field (asymmetric kernel shifts it upstream;
    symmetric kernel does not), and the fields CONCENTRATE at the reward far more than a yoked random control.
"""
from src.eval.goal_vector import run_seed as goal_seed
from src.eval.reward_map import run_seed as reward_seed


def test_goal_direction_code_emerges_and_is_goal_specific():
    r = goal_seed(0, iters=700)
    assert r["nav_success"] > 0.6, f"navigation should work ({r['nav_success']:.2f})"
    assert r["frac_allo_dir"] > 0.30, "a goal-direction code should emerge"
    assert r["untr_allo"] < 0.20 and r["shuf_allo"] < 0.20, "untrained + shuffle nulls at the floor (goal-specific)"


def test_reward_fields_anticipate_and_concentrate():
    r = reward_seed(0)
    # anticipatory shift only with the asymmetric kernel
    assert r["shift_asym"] < -0.10, f"asymmetric BTSP should shift the field upstream ({r['shift_asym']:.2f})"
    assert r["shift_symm"] > r["shift_asym"] + 0.10, "the symmetric-kernel control should not shift (dissociation)"
    # reward-specific concentration vs a yoked random-location control
    assert r["over_rep_reward"] > 5.0, "fields should concentrate at the reward"
    assert r["over_rep_reward"] > 3.0 * max(r["over_rep_yoked"], 0.5), "far above the yoked random control"
