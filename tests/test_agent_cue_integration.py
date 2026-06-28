"""
tests/test_agent_cue_integration.py
Locks the near-optimal cue-integration correction: a generic learned fuser (no hand-coded gate) beats both
single cues AND the old fixed gate and tracks Kalman (A); it genuinely USES the boundary (ablation) and
integrates it robustly as the cue degrades (C). We do NOT assert the strict reliability-weighting law
(confounded by temporal averaging of unbiased cues; see the module docstring).
"""
import pytest

from src.eval.agent_cue_integration import run_seed, NOISES, BNS


@pytest.fixture(scope="module")
def seed0():
    return run_seed(0)


def test_learned_fuser_beats_cues_and_old_gate(seed0):
    A = seed0["A"][NOISES[-1]]
    assert A["learned"] < A["pi"], f"learned {A['learned']:.3f} should beat PI-only {A['pi']:.3f}"
    assert A["learned"] < A["boundary"], f"learned {A['learned']:.3f} should beat boundary-only {A['boundary']:.3f}"
    assert A["learned"] < A["fixed"], f"learned {A['learned']:.3f} should beat the old fixed gate {A['fixed']:.3f}"
    assert A["learned"] < 1.6 * A["kalman"], f"learned {A['learned']:.3f} should track Kalman {A['kalman']:.3f}"


def test_boundary_is_used_and_integration_is_robust(seed0):
    C = seed0["C"]
    # (B) ablating the boundary collapses the fuser back toward PI-only -> it genuinely integrates the cue
    lo = C[BNS[0]]
    assert lo["ablated"] > lo["full"] + 0.1, "ablating the boundary should raise error (it is used)"
    # (C) robust integration: full error stays bounded even as the boundary becomes very noisy (it averages
    #     many unbiased observations) -- we do NOT assert strict down-weighting (confounded; see docstring)
    assert C[BNS[-1]]["full"] < 1.3 * C[BNS[0]]["full"], "full error should stay bounded as the boundary degrades"
    assert C[BNS[-1]]["full"] < C[BNS[-1]]["pi"], "even with a noisy boundary the fuser beats PI-only"
