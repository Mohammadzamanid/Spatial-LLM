"""
tests/test_grid_capacity.py
Locks the grid coding-capacity (Fiete) result: the analytic Fisher-information forms must match
autograd exactly, and the capacity scaling must hold (grid resolution ~flat, place ~linear in arena size).
"""
import math
import torch

from src.eval.grid_capacity import (
    grid_F, grid_code, grid_fisher, place_centers, place_code, place_fisher,
    det_res, loglog_slope, run_seed, ARENAS,
)


def test_grid_fisher_matches_autograd():
    g = torch.Generator().manual_seed(1)
    F = grid_F(4.0, g)
    x = torch.tensor([1.3, -0.7], requires_grad=True)
    Jr = torch.autograd.functional.jacobian(lambda p: grid_code(p.unsqueeze(0), F).squeeze(0), x)
    J_auto = Jr.t() @ Jr
    assert torch.allclose(J_auto, grid_fisher(F), atol=1e-3)


def test_place_fisher_matches_autograd():
    C, sig = place_centers(4.0)
    x = torch.tensor([0.5, 0.3], requires_grad=True)
    Jr = torch.autograd.functional.jacobian(lambda p: place_code(p.unsqueeze(0), C, sig).squeeze(0), x)
    J_auto = Jr.t() @ Jr
    assert torch.allclose(J_auto, place_fisher(x.detach(), C, sig), atol=1e-3)


def test_grid_fisher_is_position_independent():
    g = torch.Generator().manual_seed(0)
    F = grid_F(2.0, g)
    j = grid_fisher(F)
    # cos/sin code: Fisher = F^T F regardless of where you evaluate it -> identical det at any x
    assert det_res(j) == det_res(F.t() @ F)


def test_capacity_scaling_grid_flat_place_linear():
    """The Fiete signature: at fixed budget, grid resolution stays ~flat with arena size while place
    grows ~linearly (log-log slope ~0 vs ~1)."""
    r = run_seed(0)
    grid_res = [r[L]["grid_res"] for L in ARENAS]
    place_res = [r[L]["place_res"] for L in ARENAS]
    sg, sp = loglog_slope(grid_res), loglog_slope(place_res)
    assert sg < 0.35, f"grid resolution should be ~flat, got slope {sg:.2f}"
    assert sp > 0.85, f"place resolution should grow ~linearly, got slope {sp:.2f}"
    # and the grid advantage should grow with arena (largest > smallest ratio)
    ratio_small = place_res[0] / grid_res[0]
    ratio_big = place_res[-1] / grid_res[-1]
    assert ratio_big > ratio_small > 1.0
