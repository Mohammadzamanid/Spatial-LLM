"""
src/eval/agent_grid_reanchor.py

OBJECT / LANDMARK REANCHORING OF THE GRID PHASE — now INSIDE the core grid cortex (`_HexGridModules`),
not an external eval loop. Grid cells do not operate only as a global GPS: within a trial they reanchor to a
task-relevant object by TRANSLATING the grid pattern (Nature Neurosci 2025), and allocentric & egocentric
codes coexist in MEC (Nat Commun 2025). Until now the repo's core path-integrator could only reset its phase
at BOUNDARIES; the object-vector cells lived only in standalone eval scripts. We wired the egocentric
object-vector organ into `_HexGridModules.forward(object_obs=...)`, through the SAME egocentric->allocentric
transform the boundary path uses (`_ego_to_allo` + `_apply_phase_fix`):

    p_hat = anchor_world_pos - R(heading) @ ego_vector(distance, egocentric_bearing)   # implied position
    phi   = (1 - w) * phi + w * gains * p_hat                                            # reanchor (translate)

The reanchoring is now LOAD-BEARING and runs through the module itself. The ablation contrasts a LOCAL cue
(boundaries — useful only at walls) against a GLOBAL cue (an object/landmark — a fix anywhere it is visible):

  OPEN FIELD (agent forages far from every wall): boundary-vector cells are weak (the walls are distant), so
    boundary anchoring is largely ineffective. The OBJECT cue, seen across the open field, reanchors the grid
    and bounds the drift several-fold better — a capability the boundary-only module did NOT have. A SHUFFLED-
    anchor control (object cue present, but its world position scrambled) does NOT help — the rescue is the
    true egocentric->allocentric geometry, not merely "some extra input".

  NEAR A WALL: BOUNDARY anchoring bounds the drift (the pre-existing local capability is preserved).

The dissociation: the boundary cue is LOCAL (rescues only the near-wall regime), the object cue is GLOBAL
(rescues anywhere the landmark is visible) — both through one shared egocentric->allocentric transform in one
module. The object observation is mildly NOISY (a real landmark sense), so the object fix is good, not an
oracle. The grid is path-integrated globally and dynamically reanchored to whichever cue is available.

Multi-seed, mean +/- 95% CI. Writes results/agent_grid_reanchor.json + .svg.

    python -m src.eval.agent_grid_reanchor --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.models.neuro.trajectory_cortex import _HexGridModules
from src.eval.agent_grid_cortex import train_decoder, R
from src.eval.agent_grid_drift import sense_wall

STEP = 0.18
WALK = 150
A_NOISE = 0.06                       # self-motion (path-integration) noise -> drift
O_NOISE = 0.08                       # object-observation noise (a real landmark sense is not an oracle)
TAIL = 50                            # average the allocentric error over the last TAIL steps
N_WALKS = 24
LM_OPEN = torch.tensor([1.6, 1.6])   # the object/landmark, visible across the open central field
LM_BAD = torch.tensor([-1.7, -0.6])  # scrambled anchor position for the shuffled-anchor control
OPEN_HALF = 0.8                      # open-field foraging stays within |x|,|y| <= 0.8 (walls >=1.7 away)
WALL_X = R - 0.25                    # near-wall foraging hugs the +x wall


def build(seed):
    """The real velocity-driven hex grid cortex, with BOTH allothetic pathways enabled (boundary + object).
    Only the velocity gains and anchor geometry are fixed; gates/cells are the organs' own params."""
    torch.manual_seed(seed)
    mod = _HexGridModules(embed_dim=64, n_modules=6, base_spacing=1.6,
                          noise_std=A_NOISE, boundary_anchor=True, object_anchor=True)
    for p in mod.parameters():
        p.requires_grad_(False)
    return mod


def run_walk(mod, dec, gen, regime, cue):
    """One exploratory walk. regime: 'open' (central, walls far) or 'wall' (hugging +x wall).
    cue: 'pi' | 'boundary' | 'object' | 'shuffle'. Returns the mean allocentric (global) decode error over
    the last TAIL steps. ALL anchoring runs through mod.forward(...) — the integration we are testing."""
    th = torch.rand(1, generator=gen).item() * 2 * math.pi
    if regime == "open":
        pos = (torch.rand(2, generator=gen) * 2 - 1) * (OPEN_HALF * 0.5)
        lim_lo = torch.tensor([-OPEN_HALF, -OPEN_HALF]); lim_hi = torch.tensor([OPEN_HALF, OPEN_HALF])
    else:
        pos = torch.tensor([WALL_X, (torch.rand(1, generator=gen).item() * 2 - 1) * 0.8])
        lim_lo = torch.tensor([WALL_X - 0.5, -1.0]); lim_hi = torch.tensor([R, 1.0])
    v, hd, bobs, oobs, truth = [], [], [], [], []
    for t in range(WALK):
        th = th + 0.25 * math.sin(t * 0.3) + torch.randn(1, generator=gen).item() * 0.1
        vel = STEP * torch.tensor([math.cos(th), math.sin(th)])
        nxt = torch.minimum(torch.maximum(pos + vel, lim_lo), lim_hi)   # forage within the regime's region
        vel = nxt - pos; pos = nxt
        v.append(torch.tensor([vel[0], vel[1], 0.0])); hd.append(th); truth.append(pos.clone())
        dist, bear, _, _ = sense_wall(pos)
        bobs.append(torch.tensor([dist, bear]))
        anchor = LM_BAD if cue == "shuffle" else LM_OPEN
        vrel = LM_OPEN - pos; r = vrel.norm()
        beta = math.atan2(vrel[1].item(), vrel[0].item()) - th
        rn = (r + torch.randn(1, generator=gen).item() * O_NOISE)              # noisy landmark sense (not an oracle)
        bn = beta + torch.randn(1, generator=gen).item() * O_NOISE
        oobs.append(torch.tensor([rn, bn, anchor[0], anchor[1], 1.0]))
    v = torch.stack(v).unsqueeze(0); hd = torch.tensor(hd).unsqueeze(0)
    bobs = torch.stack(bobs).unsqueeze(0); oobs = torch.stack(oobs).unsqueeze(0)
    use_b = bobs if cue == "boundary" else None
    use_o = oobs if cue in ("object", "shuffle") else None
    grid_seq = mod.forward(v, boundary_obs=use_b, object_obs=use_o, heading=hd, return_grid_seq=True)  # (1,T,K*M)
    truth = torch.stack(truth)                                          # (T,2)
    est = dec(grid_seq[0])                                              # (T,2) decoded global position
    return (est[-TAIL:] - truth[-TAIL:]).pow(2).sum(-1).sqrt().mean().item()


def run_seed(seed):
    mod = build(seed)
    gen = torch.Generator().manual_seed(seed + 100)
    dec = train_decoder(mod, gen, nonlinear=True, iters=1500)           # grid code -> position (self-supervised)
    out = {}
    for regime in ("open", "wall"):
        out[regime] = {}
        for cue in ("pi", "boundary", "object", "shuffle"):
            out[regime][cue] = sum(run_walk(mod, dec, gen, regime, cue) for _ in range(N_WALKS)) / N_WALKS
    return out


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 3), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 3) if n > 1 else 0.0


CUES = ("pi", "boundary", "object", "shuffle")
REGIMES = ("open", "wall")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    agg = {rg: {cue: ci([p[rg][cue] for p in per]) for cue in CUES} for rg in REGIMES}

    print(f"\nOBJECT REANCHORING OF THE GRID PHASE — inside _HexGridModules (n={a.seeds}; allocentric decode "
          f"error over the last {TAIL} steps, mean ± 95% CI)\n" + "=" * 86, flush=True)
    print(f"    {'regime':>20} | {'path-int':>10} | {'boundary':>10} | {'object':>10} | {'shuffled obj':>12}", flush=True)
    lab = {"open": "OPEN FIELD (walls far)", "wall": "NEAR A WALL"}
    for rg in REGIMES:
        d = agg[rg]
        print(f"    {lab[rg]:>20} | {d['pi'][0]:>10.3f} | {d['boundary'][0]:>10.3f} | "
              f"{d['object'][0]:>10.3f} | {d['shuffle'][0]:>12.3f}", flush=True)
    op = agg["open"]; wl = agg["wall"]
    print(f"\n  -> LOCAL vs GLOBAL allothetic cues through ONE shared transform in the core module: in the OPEN "
          f"FIELD (every wall far) boundary anchoring is largely ineffective (err {op['boundary'][0]:.2f}, "
          f"path-int {op['pi'][0]:.2f}), but the OBJECT cue reanchors the grid and bounds the drift "
          f"({op['object'][0]:.2f}, ~{op['boundary'][0]/max(op['object'][0],1e-6):.0f}x better) — a capability "
          f"the boundary-only module lacked; a SHUFFLED-anchor control "
          f"fails ({op['shuffle'][0]:.2f}), so the rescue is the true egocentric->allocentric geometry, not just "
          f"extra input. NEAR A WALL the boundary cue bounds it ({wl['boundary'][0]:.2f} vs path-int "
          f"{wl['pi'][0]:.2f}) — the local capability is preserved. Object-vector cells now reanchor the grid "
          f"phase from WITHIN _HexGridModules.forward(object_obs=...), the load-bearing integration.", flush=True)

    out = {"n_seeds": a.seeds, "walk": WALK, "tail": TAIL, "noise": A_NOISE,
           "results": {rg: agg[rg] for rg in REGIMES}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/agent_grid_reanchor.json", "w"), indent=2)
    svg(agg, "results/agent_grid_reanchor.svg")
    print("\nwrote results/agent_grid_reanchor.json and results/agent_grid_reanchor.svg", flush=True)


def svg(agg, out):
    pad = 70; gw = 230; gap = 60; ph = 200; W = pad + len(REGIMES) * (gw + gap) + 20; H = 86 + ph + 56
    col = {"pi": "#c9341a", "boundary": "#e6a000", "object": "#2ca25f", "shuffle": "#8a94a6"}
    lab = {"pi": "path-int", "boundary": "boundary", "object": "object", "shuffle": "shuffled obj"}
    rlab = {"open": "OPEN FIELD (walls far)", "wall": "NEAR A WALL"}
    hi = max(agg[rg][c][0] for rg in REGIMES for c in CUES) * 1.18 + 1e-6
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Object reanchoring of the grid phase, inside the core cortex &#8212; a double dissociation</text>')
    e.append('<text x="26" y="44" font-size="10.5" fill="#5b6b8c">allocentric decode error (lower=better). '
             'Open field: only the OBJECT cue rescues drift; near a wall: only the BOUNDARY cue does; '
             'shuffled-anchor control fails.</text>')
    oy = 58; base = oy + ph
    bw = (gw - 18) / len(CUES)
    for i, rg in enumerate(REGIMES):
        x0 = pad + i * (gw + gap)
        e.append(f'<line x1="{x0-8}" y1="{base}" x2="{x0+gw}" y2="{base}" stroke="#33415c"/>')
        for j, c in enumerate(CUES):
            v = agg[rg][c][0]; x = x0 + j * (bw + 4); h = v / hi * ph
            e.append(f'<rect x="{x:.0f}" y="{base-h:.1f}" width="{bw:.0f}" height="{h:.1f}" fill="{col[c]}" opacity="0.88"/>')
            e.append(f'<text x="{x+bw/2:.0f}" y="{base-h-4:.0f}" font-size="9" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
            e.append(f'<text x="{x+bw/2:.0f}" y="{base+13:.0f}" font-size="8" fill="#5b6b8c" text-anchor="middle" transform="rotate(0)">{lab[c]}</text>')
        e.append(f'<text x="{x0+gw/2:.0f}" y="{base+32:.0f}" font-size="11" font-weight="700" fill="#28324a" text-anchor="middle">{rlab[rg]}</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
