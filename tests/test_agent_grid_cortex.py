"""
tests/test_agent_grid_cortex.py
Locks the closed-loop grid-cortex agent: the public grid_code_at() must equal the recurrent integrator's
path-integrated grid code (so reading the code at a position == having walked there), the nonlinear readout
must localize, and a -grid lesion must abolish closed-loop navigation.
"""
import math
import torch

from src.models.neuro.trajectory_cortex import _HexGridModules
from src.eval.agent_grid_cortex import build_cortex, train_decoder, decode_err, navigate, R, STEP


def test_grid_code_at_matches_path_integration():
    """Walking a velocity path through the recurrent integrator must land on the same grid code as
    reading grid_code_at() at the final position (exact noiseless path integration)."""
    torch.manual_seed(0)
    mod = _HexGridModules(embed_dim=64, n_modules=6, base_spacing=1.6)
    vel = torch.tensor([[[0.2, 0.1, 0.0], [-0.1, 0.3, 0.0], [0.15, -0.2, 0.0]]])  # (1,T,3)
    _, last_grid = mod(vel, return_cells=True)                     # recurrent path integration
    final_xy = vel[0, :, :2].sum(0, keepdim=True)                  # displacement = sum of velocities
    assert torch.allclose(last_grid, mod.grid_code_at(final_xy), atol=1e-5)


def test_grid_code_at_shape_and_range():
    mod = build_cortex(0)
    pos = torch.tensor([[0.5, -0.3], [1.2, 0.8]])
    code = mod.grid_code_at(pos)
    assert code.shape == (2, mod.K * mod.M)
    assert (code >= 0).all() and (code <= 1.0001).all()           # Gaussian bumps in [0,1]


def test_nonlinear_readout_localizes_and_lesion_abolishes_nav():
    mod = build_cortex(1)
    gen = torch.Generator().manual_seed(5)
    dec = train_decoder(mod, gen, nonlinear=True, iters=800)
    assert decode_err(mod, dec, gen) < 0.15                        # grid code -> position is decodable

    def nav_dist(lesion):
        ds = []
        for _ in range(40):
            start = (torch.rand(2, generator=gen) * 2 - 1) * R
            goal = (torch.rand(2, generator=gen) * 2 - 1) * R
            traj = navigate(mod, dec, start, mod.grid_code_at(goal.unsqueeze(0)), gen, lesion_grid=lesion)
            ds.append((traj[-1] - goal).norm().item())
        return sum(d < 0.4 for d in ds) / len(ds)

    assert nav_dist(False) > 0.8                                   # intact: reaches the goal
    assert nav_dist(True) < 0.2                                    # -grid lesion: navigation abolished
