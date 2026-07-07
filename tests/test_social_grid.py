"""Tests for the #9 social-grid CPU de-risk (GAPS.md Tier 3, #9).

The FROZEN space cortex should expose a DISSOCIABLE 2-D social map: DOMINANCE reads the POWER axis cleanly
(social transitive inference), a genuine 2-D SOCIAL-DISTANCE metric shows up in OFF-AXIS "socially closer",
and the two social axes are separately readable (power predicts dominance, affiliation does not). Shuffled
positions collapse dominance to chance. Averaged over 3 seeds.
"""
from src.eval.social_grid_cortex import run_seed


def test_social_grid_dissociable_map():
    per = [run_seed(s) for s in (0, 1, 2)]

    def m(k):
        return sum(p[k] for p in per) / len(per)

    # (A) DOMINANCE — the power axis reads out cleanly (social transitive inference)
    assert m("dominance_power") > 0.75, "held-out dominance from the decoded power axis should be well above chance"

    # (C) AXIS DISSOCIATION — dominance is read from POWER, not affiliation
    assert m("dissociation_gap") > 0.2, "power should predict dominance far better than affiliation"
    assert m("dominance_affil") < m("dominance_power") - 0.2

    # FALSIFIER — shuffle the agent<->position map and dominance collapses to chance
    assert m("dominance_power_shuffled") < 0.6, "dominance should need the true agent<->position map"

    # (B) SOCIAL DISTANCE — a genuine 2-D metric (off-axis, where a power-only read is <=0.5 by construction)
    assert m("social_offaxis_free") > 0.53, "off-axis 'socially closer' should beat chance (genuine 2-D)"
    assert m("social_gap") > 0.03, "social-distance off-axis should need the true positions"
