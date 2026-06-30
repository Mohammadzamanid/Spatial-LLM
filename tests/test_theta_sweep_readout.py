"""
tests/test_theta_sweep_readout.py
Locks the look-ahead readout result + the sweep geometry (CPU, no LLM stack):
(1) theta-sweep points lie AHEAD of the agent within ~one sweep length;
(2) in a novel per-episode layout, real sweep tokens predict the blocked cone well above chance, while
    ablating them (or feeding a wrong-heading sweep) collapses to near chance — the tokens are load-bearing.
"""
import math

import torch

from src.models.neuro.trajectory_cortex import _HexGridModules
from src.models.neuro.theta_sweep import ThetaSweepSampler
from src.eval.theta_sweep_readout import batched_sweep, run_seed


def test_sweep_points_are_ahead():
    mod = _HexGridModules(embed_dim=32, n_modules=6, base_spacing=1.6)
    sampler = ThetaSweepSampler()
    pos = torch.zeros(6, 2)
    head = torch.zeros(6)                                   # facing east
    length = sampler.sweep_frac * sampler.spacings(mod).mean()
    for cyc in (0, 1):
        swept = batched_sweep(mod, sampler, pos, head, cyc)         # (6,steps,2)
        ahead = swept[..., 0]                                       # east component
        assert (ahead >= -1e-6).all(), "sweep must extend ahead of the agent"
        assert swept.norm(dim=-1).max() <= length + 1e-5, "sweep stays within ~one sweep length"


def test_sweep_tokens_are_load_bearing():
    r = run_seed(0)
    assert r["real"] > 0.8, f"real sweep tokens should predict blocked-ahead well (got {r['real']:.2f})"
    assert r["real"] > r["ablated"] + 0.2, "ablating the sweep should hurt a lot (load-bearing)"
    assert r["real"] > r["shuffled"] + 0.2, "a wrong-heading sweep should not substitute for the real one"
    assert r["ablated"] < 0.7, f"without look-ahead the novel layout is near chance (got {r['ablated']:.2f})"
