"""
tests/test_grid_3d.py
Locks the biologically-grounded 3D grid code (LocalOrder3DGrid) and its integration into _HexGridModules:
(1) its field centers are in the BAT regime (high local order, ~zero global lattice) and separable from a
    cubic-lattice control (high global lattice); (2) it path-integrates and localizes in full 3D;
(3) _HexGridModules(grid_3d=True) swaps the 1-D z stub for the 3D code without breaking the default path.
"""
import torch
import torch.nn as nn

from src.models.neuro.spatial_cells import LocalOrder3DGrid
from src.models.neuro.trajectory_cortex import _HexGridModules
from src.eval.local_3d_order import local_order, global_lattice


def test_field_centers_are_in_the_bat_regime():
    lo = LocalOrder3DGrid(seed=0)
    la = LocalOrder3DGrid(seed=0, lattice=True)
    # local-order code: regular spacing (high local) but NO global lattice (low global)
    assert local_order(lo.field_centers_unit()) > 0.8
    assert global_lattice(lo.field_centers_unit()) < 0.2
    # the cubic-lattice control has a global lattice (the non-biological crystal)
    assert global_lattice(la.field_centers_unit()) > 0.8


def test_localizes_in_3d():
    g = LocalOrder3DGrid(seed=0)
    gen = torch.Generator().manual_seed(1)
    dec = nn.Sequential(nn.Linear(g.n_cells, 256), nn.ReLU(), nn.Linear(256, 3))
    opt = torch.optim.Adam(dec.parameters(), 3e-3)
    for _ in range(900):
        p = (torch.rand(256, 3, generator=gen) * 2 - 1) * 2.5
        loss = ((dec(g.code_at(p)) - p) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = (torch.rand(1500, 3, generator=gen) * 2 - 1) * 2.5
        err = (dec(g.code_at(p)) - p).pow(2).sum(-1).sqrt().mean().item()
    assert err < 0.5, f"the 3D code should localize in full 3D (err {err:.2f})"


def test_grid_3d_mode_replaces_the_z_stub():
    # grid_3d=True swaps the 1-D z place code for the 3D grid code; default path is unaffected.
    m3 = _HexGridModules(embed_dim=32, n_modules=6, base_spacing=1.6, grid_3d=True, grid3d_seed=0)
    assert hasattr(m3, "grid3d") and not hasattr(m3, "z_centers")
    assert m3.readout.in_features == m3.K * m3.M + m3.grid3d.n_cells
    out = m3.forward(torch.randn(3, 6, 3) * 0.1)
    assert out.shape == (3, 32)
    m2 = _HexGridModules(embed_dim=32, n_modules=4)             # default still 2D grid + 1-D z stub
    assert hasattr(m2, "z_centers") and not getattr(m2, "grid_3d", False)
    assert m2.forward(torch.zeros(2, 5, 3)).shape == (2, 32)
