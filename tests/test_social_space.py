"""
tests/test_social_space.py
Locks the social-space result (self + other-agent maps in one population; Danjo 2018, Omer 2018):
both pure-self and pure-other place cells emerge, and a self/other lesion double dissociation holds.
(Reduced iters for test speed; the effect is robust well before full convergence.)
"""
from src.eval.social_space import run_seed


def test_self_and_other_maps_dissociate():
    o = run_seed(0, iters=800)
    # both maps emerge as distinct sub-populations
    assert o["frac_self"] > 0.03, "pure self-place cells should emerge"
    assert o["frac_other"] > 0.03, "pure other-place cells should emerge"
    # double dissociation: lesioning OTHER cells hurts OTHER decoding much more than SELF
    assert o["other_mae_lesion_other"] > o["self_mae_lesion_other"] + 0.05
    assert o["other_mae_lesion_other"] > o["other_mae"] + 0.05
    # ...and lesioning SELF cells hurts SELF decoding much more than OTHER
    assert o["self_mae_lesion_self"] > o["other_mae_lesion_self"] + 0.05
    assert o["self_mae_lesion_self"] > o["self_mae"] + 0.05
