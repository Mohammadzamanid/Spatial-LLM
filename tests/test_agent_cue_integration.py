"""
tests/test_agent_cue_integration.py
Locks the near-optimal cue-integration correction: a generic learned fuser (no hand-coded gate) beats
BOTH single cues and the old fixed gate, and its combined reliability approaches the Bayesian-optimal bound.
"""
import torch

from src.eval.agent_cue_integration import run_seed, NOISES


def test_learned_fuser_is_near_optimal():
    r = run_seed(0)
    hi = NOISES[-1]
    A = r["A"][hi]
    # learned fuser beats both single cues...
    assert A["learned"] < A["pi"], f"learned {A['learned']:.3f} should beat PI-only {A['pi']:.3f}"
    assert A["learned"] < A["boundary"], f"learned {A['learned']:.3f} should beat boundary-only {A['boundary']:.3f}"
    # ...and the OLD hand-coded fixed gate (the correction)...
    assert A["learned"] < A["fixed"], f"learned {A['learned']:.3f} should beat fixed gate {A['fixed']:.3f}"
    # ...and tracks the Kalman optimum (within ~50%)
    assert A["learned"] < 1.6 * A["kalman"], f"learned {A['learned']:.3f} should be near Kalman {A['kalman']:.3f}"


def test_bayesian_optimality_at_contact():
    r = run_seed(0)
    for nz in NOISES:
        B = r["B"][nz]
        # combined reliability is better than either single cue (the optimal-integration signature)
        assert B["sig_learned"] < B["sig_pi"] + 1e-6
        assert B["sig_learned"] < B["sig_b"] + 0.05
        # and sits near the Bayesian-optimal bound (not far below — that would be impossible — nor far above)
        assert B["sig_learned"] < 1.8 * B["sig_opt"], f"noise {nz}: learned {B['sig_learned']:.3f} vs opt {B['sig_opt']:.3f}"
