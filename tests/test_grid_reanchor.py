"""
tests/test_grid_reanchor.py
Locks object/landmark reanchoring of the grid phase INSIDE the core cortex (_HexGridModules.forward(object_obs=...)):
(1) the egocentric->allocentric transform recovers the agent's position from an object sighting;
(2) object reanchoring bounds path-integration drift, and a shuffled-anchor control does not (load-bearing);
(3) the boundary pathway is preserved.
"""
import math

import torch

from src.models.neuro.trajectory_cortex import _HexGridModules


def test_ego_to_allo_inverts_the_sighting():
    # An anchor at world offset `vrel` from the agent, observed at egocentric (dist, bearing) under `heading`,
    # must transform back to that same world offset — the bridge that makes p_hat = anchor - ego_allo == pos.
    torch.manual_seed(0)
    for _ in range(20):
        pos = (torch.rand(2) * 2 - 1) * 2.0
        anchor = (torch.rand(2) * 2 - 1) * 2.0
        heading = torch.rand(1) * 2 * math.pi
        vrel = anchor - pos
        dist = vrel.norm().unsqueeze(0)
        bearing = (torch.atan2(vrel[1], vrel[0]) - heading)
        ego_allo = _HexGridModules._ego_to_allo(dist, bearing, heading)[0]
        assert torch.allclose(ego_allo, vrel, atol=1e-4), "ego->allo should recover the world offset"
        assert torch.allclose(anchor - ego_allo, pos, atol=1e-4), "anchor - ego_allo should recover the position"


def _decoder(mod, iters=600):
    import torch.nn as nn
    gen = torch.Generator().manual_seed(7)
    dec = nn.Sequential(nn.Linear(mod.K * mod.M, 128), nn.ReLU(), nn.Linear(128, 2))
    opt = torch.optim.Adam(dec.parameters(), 3e-3)
    for _ in range(iters):
        pos = (torch.rand(256, 2, generator=gen) * 2 - 1) * 2.5
        loss = ((dec(mod.grid_code_at(pos)) - pos) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


def _walk(mod, dec, gen, cue):
    """A central-field drift walk; cue in {'pi','object','shuffle'}. Returns final allocentric decode error."""
    th = 0.5; pos = torch.zeros(2); v, hd, oobs, truth = [], [], [], None
    LM = torch.tensor([1.6, 1.6]); BAD = torch.tensor([-1.7, -0.6])
    for t in range(130):
        th = th + 0.25 * math.sin(t * 0.3) + torch.randn(1, generator=gen).item() * 0.1
        vel = 0.18 * torch.tensor([math.cos(th), math.sin(th)])
        nxt = (pos + vel).clamp(-0.8, 0.8); vel = nxt - pos; pos = nxt
        v.append(torch.tensor([vel[0], vel[1], 0.0])); hd.append(th); truth = pos.clone()
        vrel = LM - pos; r = vrel.norm(); beta = math.atan2(vrel[1].item(), vrel[0].item()) - th
        anchor = BAD if cue == "shuffle" else LM
        oobs.append(torch.tensor([r, beta, anchor[0], anchor[1], 1.0]))
    v = torch.stack(v).unsqueeze(0); hd = torch.tensor(hd).unsqueeze(0); oobs = torch.stack(oobs).unsqueeze(0)
    use_o = oobs if cue in ("object", "shuffle") else None
    g = mod.forward(v, object_obs=use_o, heading=hd, return_grid_seq=True)
    return (dec(g[0, -1]) - truth).norm().item()


def test_object_reanchoring_is_load_bearing():
    torch.manual_seed(0)
    mod = _HexGridModules(embed_dim=64, n_modules=6, base_spacing=1.6, noise_std=0.06, object_anchor=True)
    for p in mod.parameters():
        p.requires_grad_(False)
    dec = _decoder(mod)
    gen = torch.Generator().manual_seed(3)
    pi = sum(_walk(mod, dec, gen, "pi") for _ in range(8)) / 8
    obj = sum(_walk(mod, dec, gen, "object") for _ in range(8)) / 8
    shuf = sum(_walk(mod, dec, gen, "shuffle") for _ in range(8)) / 8
    assert obj < 0.5 * pi, f"object reanchoring should bound drift (object {obj:.2f} vs path-int {pi:.2f})"
    assert shuf > pi, f"a shuffled-anchor control should NOT help (shuffle {shuf:.2f} vs path-int {pi:.2f})"


def test_object_obs_is_optional():
    # Existing callers that pass no object_obs are unaffected (the object block is skipped).
    mod = _HexGridModules(embed_dim=32, n_modules=4, object_anchor=True)
    out, grid = mod.forward(torch.zeros(2, 5, 3), return_cells=True)
    assert out.shape == (2, 32) and grid.shape == (2, mod.K * mod.M)
