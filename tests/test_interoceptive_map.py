"""Tests for the interoceptive cognitive map (GAPS.md interoceptive-anchoring item).

With NO 'thirsty->water' rule -- only a homeostatic reward (reduce total drive) and the body's physiology
(deficits grow; water resets thirst, food resets hunger) -- the planner should EMERGENTLY navigate to the
resource matching its dominant deficit, value the same place differently under different drives, and keep the
body regulated far better than a DRIVE-BLIND planner that cannot read its own deficits. Measured, never in a rule.
"""
from src.eval.interoceptive_map import run_seed


def test_drive_remaps_value_and_navigation():
    o = run_seed(0)

    # (A) interoceptive navigation emerges; a drive-blind planner cannot choose by deficit
    assert o["congruent_full"] > 0.75, "the planner should head to the drive-matched resource"
    assert o["congruent_blind"] < 0.4, "blind to its deficits, it cannot pick the drive-matched resource"

    # (B) drive-dependent value remapping: the same place is valued oppositely under thirst vs hunger
    assert o["remap_corr"] < -0.3, "drive-specific value residuals should anti-correlate (remapping)"
    assert o["resource_value_gain"] > 0.1, "each resource is worth more under its own deficit"

    # (C) homeostatic regulation payoff + (D) shuttling emerge
    assert o["mean_drive_full"] < o["mean_drive_blind"] - 5, "interoception keeps the body better regulated"
    assert o["mean_drive_full"] < o["mean_drive_random"], "and better than random"
    assert o["switches_full"] > 2, "it shuttles between resources as its deficits cycle"
