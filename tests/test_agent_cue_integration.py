"""
tests/test_agent_cue_integration.py
Locks the near-optimal cue-integration correction: a generic learned fuser (no hand-coded gate) beats both
single cues AND the old fixed gate and tracks Kalman (A); it genuinely USES the boundary (ablation) and
relies on it in proportion to its reliability (C).
"""
from src.eval.agent_cue_integration import run_seed, NOISES, BNS


def test_learned_fuser_beats_cues_and_old_gate():
    r = run_seed(0)
    hi = NOISES[-1]
    A = r["A"][hi]
    assert A["learned"] < A["pi"], f"learned {A['learned']:.3f} should beat PI-only {A['pi']:.3f}"
    assert A["learned"] < A["boundary"], f"learned {A['learned']:.3f} should beat boundary-only {A['boundary']:.3f}"
    assert A["learned"] < A["fixed"], f"learned {A['learned']:.3f} should beat the old fixed gate {A['fixed']:.3f}"
    assert A["learned"] < 1.6 * A["kalman"], f"learned {A['learned']:.3f} should track Kalman {A['kalman']:.3f}"


def test_boundary_is_used_and_reliability_weighted():
    r = run_seed(0)
    C = r["C"]
    # (B) ablating the boundary collapses the fuser back toward PI-only (so it genuinely integrates it)
    lo = C[BNS[0]]
    assert lo["ablated"] > lo["full"] + 0.1, "ablating the boundary should raise error (it is used)"
    # (C) the boundary's contribution shrinks as it gets noisier (reliability weighting)
    contrib_lo = C[BNS[0]]["ablated"] - C[BNS[0]]["full"]
    contrib_hi = C[BNS[-1]]["ablated"] - C[BNS[-1]]["full"]
    assert contrib_lo > contrib_hi, f"reliance should fall with reliability: {contrib_lo:.3f} -> {contrib_hi:.3f}"
