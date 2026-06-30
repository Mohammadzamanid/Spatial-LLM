"""
tests/test_theta_sweep_llm.py
Locks the theta-sweep token machinery added to TrajectoryLLM, WITHOUT loading the 1.5B base LLM: the sweep
methods only touch cortex.integrator / theta_sweep / sweep_to_tokens, so we exercise them on a light mock.
"""
import types

import pytest
import torch
import torch.nn as nn

pytest.importorskip("peft", reason="TrajectoryLLM imports the LLM stack (peft/transformers); GPU/Kaggle only")

from src.models.trajectory_llm import TrajectoryLLM           # noqa: E402
from src.models.neuro.trajectory_cortex import _HexGridModules  # noqa: E402
from src.models.neuro.theta_sweep import ThetaSweepSampler      # noqa: E402


def _mock(llm_dim=64, cycles=2, steps=8):
    gm = _HexGridModules(embed_dim=32, n_modules=6, base_spacing=1.6)
    return types.SimpleNamespace(
        cortex=types.SimpleNamespace(integrator=gm),
        theta_sweep=ThetaSweepSampler(steps=steps),
        sweep_to_tokens=nn.Linear(gm.K * gm.M, llm_dim),
        n_sweep_cycles=cycles,
        _current_pos_heading=TrajectoryLLM._current_pos_heading,
    )


def test_current_pos_heading_path_integrates():
    # A straight east walk: x = sum(speed), y = 0, final heading 0.
    heading = torch.zeros(2, 5); speed = torch.full((2, 5), 0.2); vz = torch.zeros(2, 5)
    pos, h = TrajectoryLLM._current_pos_heading(heading, speed, vz)
    assert torch.allclose(pos[:, 0], torch.full((2,), 1.0), atol=1e-5)
    assert torch.allclose(pos[:, 1], torch.zeros(2), atol=1e-5)
    assert torch.allclose(h, torch.zeros(2), atol=1e-6)


def test_sweep_tokens_shapes_and_modes():
    m = _mock(llm_dim=64, cycles=2, steps=8)
    B, T = 4, 6
    heading = torch.rand(B, T) * 6.28; speed = torch.rand(B, T) * 0.2; vz = torch.zeros(B, T)
    real = TrajectoryLLM._sweep_tokens(m, heading, speed, vz, mode="real")
    assert real.shape == (B, 2 * 8, 64), "n_cycles*steps look-ahead tokens of llm_dim"
    ablated = TrajectoryLLM._sweep_tokens(m, heading, speed, vz, mode="ablated")
    assert ablated.shape == real.shape and torch.count_nonzero(ablated) == 0, "ablated sweep is zeros"
    shuffled = TrajectoryLLM._sweep_tokens(m, heading, speed, vz, mode="shuffled")
    # a wrong heading samples a different region -> different look-ahead tokens
    assert not torch.allclose(real, shuffled, atol=1e-4)


def test_sweep_positions_are_ahead():
    # The swept points must lie ahead of the agent (positive projection on the heading), within ~sweep reach.
    m = _mock()
    B = 8
    heading = torch.zeros(B, 4); speed = torch.full((B, 4), 0.2); vz = torch.zeros(B, 4)
    pos, head = TrajectoryLLM._current_pos_heading(heading, speed, vz)
    gm = m.cortex.integrator; s = m.theta_sweep
    length = s.sweep_frac * s.spacings(gm).mean()
    ks = torch.arange(1, s.steps + 1, dtype=torch.float) / s.steps
    for cyc in range(2):
        side = -1.0 if cyc % 2 == 0 else 1.0
        direction = head + side * s.angle
        d = torch.stack([direction.cos(), direction.sin()], -1)
        swept = pos.unsqueeze(1) + ks.view(1, -1, 1) * length * d.unsqueeze(1)
        ahead = (swept - pos.unsqueeze(1))[..., 0]      # east component; heading is east
        assert (ahead >= -1e-6).all(), "sweep extends ahead of the agent"
