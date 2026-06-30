"""
tests/test_theta_sweep.py
Locks the theta-cycle look-around result: theta-sweep look-ahead beats a reactive agent at reaching goals in
a concave-trap field (it samples space ahead and avoids dead-ends), and the sampler reproduces the Vollan
signatures (left/right alternation; sweep length ~20% of spacing, multi-scale).
"""
import pytest

from src.eval.theta_sweep import run_seed
from src.models.neuro import ThetaSweepSampler
from src.eval.agent_grid_cortex import build_cortex


@pytest.fixture(scope="module")
def seed0():
    return run_seed(0, trials=100)


def test_lookahead_beats_reactive(seed0):
    assert seed0["theta_sweep"]["success"] > seed0["reactive"]["success"] + 0.08, (
        f"theta-sweep look-ahead should beat reactive: "
        f"{seed0['theta_sweep']['success']:.2f} vs {seed0['reactive']['success']:.2f}")


def test_vollan_signatures(seed0):
    sig = seed0["sig"]
    assert sig["alternates"], "the sweep must alternate left/right across theta cycles"
    assert 0.15 < sig["sweep_frac"] < 0.25, f"sweep length ~20% of spacing (Vollan 19.7%), got {sig['sweep_frac']}"
    assert sig["multiscale_r"] > 0.99, "per-module sweep length must scale with module spacing (multi-scale)"


def test_sampler_api():
    import torch
    mod = build_cortex(0); ts = ThetaSweepSampler()
    pos, codes, side, _ = ts(torch.tensor([0.0, 0.0]), 0.0, mod, 0)
    assert pos.shape == (ts.steps, 2) and codes.shape[0] == ts.steps
    _, _, side1, _ = ts(torch.tensor([0.0, 0.0]), 0.0, mod, 1)
    assert side != side1                                              # left/right alternation across cycles
