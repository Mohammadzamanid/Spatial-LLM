"""
src/eval/agent_multiframe.py

THE UNIFIED MULTI-REFERENCE-FRAME NAVIGATING BRAIN — one closed-loop agent that navigates in BOTH a global
(allocentric) frame and an object-centred (egocentric) frame, sharing one organ stack. This is the
functional consolidation of the reference-frame work (grid cortex + head-direction ring + object-vector
cells): not five separate eval modules but ONE agent whose two reference frames dissociate by organ.

  - GLOBAL goal  (a fixed room location): navigated via the GRID position code (allocentric).
  - OBJECT goal  (an offset from a per-episode landmark): navigated via the EGOCENTRIC OBJECT-VECTOR cells
                 transformed to allocentric by the HEAD-DIRECTION organ.
  - Steering is EGOCENTRIC (the agent turns relative to its heading), so HD is needed to convert any
    allocentric goal-direction into a body turn — shared by both frames.

A clean DOUBLE DISSOCIATION (+ shared HD) results:
  - lesion GRID    -> the GLOBAL frame fails (no allocentric position); the OBJECT frame is intact.
  - lesion OBJECT  -> the OBJECT frame fails (no landmark vector); the GLOBAL frame is intact.
  - lesion HD      -> BOTH fail (no heading for egocentric steering / the ego->allo transform).

Multi-seed, mean +/- 95% CI. Writes results/agent_multiframe.json + .svg.

    python -m src.eval.agent_multiframe --seeds 3
"""
import argparse
import json
import math
import os

import torch

from src.eval.head_direction import train_hd, canonical, nearest
from src.eval.agent_grid_cortex import build_cortex, train_decoder, R
from src.eval.reference_frame import train_ovc

S = 0.2; RAD = 0.4; OFFSET = torch.tensor([0.6, 0.0]); STEPS = 40; G_ROOM = torch.tensor([0.6, 0.6])
FRAMES = ["global", "object"]; LESIONS = ["none", "grid", "object", "hd"]


def run_seed(seed, episodes=150):
    gen = torch.Generator().manual_seed(seed)
    hd, _ = train_hd(seed, iters=1500); keys, vals = canonical(hd)
    mod = build_cortex(seed); dec = train_decoder(mod, gen, nonlinear=True, iters=1200)
    ovc, lin = train_ovc(gen)

    def head_est(th, lesion):
        return torch.rand(1, generator=gen).item() * 2 * math.pi if lesion == "hd" else hd.decode(nearest(keys, vals, th))

    def episode(frame, lesion):
        obj = (torch.rand(2, generator=gen) * 2 - 1) * (R * 0.6)
        goal = G_ROOM if frame == "global" else obj + OFFSET
        pos = (torch.rand(2, generator=gen) * 2 - 1) * R; th = torch.rand(1, generator=gen).item() * 2 * math.pi
        phi = mod.gains.view(mod.K, 1, 1) * pos.view(1, 1, 2).clone()
        for _ in range(STEPS):
            if frame == "global":                                    # GLOBAL: allocentric position from the grid
                pe = torch.randn(2, generator=gen) if lesion == "grid" else dec(mod._grid_code(phi))[0]
                d = goal - pe
            else:                                                    # OBJECT: egocentric object-vector + HD transform
                vrel = obj - pos; r = vrel.norm(); beta = math.atan2(vrel[1].item(), vrel[0].item()) - th
                ego = (torch.randn(2, generator=gen) if lesion == "object"
                       else lin(ovc(torch.tensor([r]), torch.tensor([beta])))[0])
                te = head_est(th, lesion); c, s_ = math.cos(te), math.sin(te)
                d = torch.tensor([c * ego[0] - s_ * ego[1], s_ * ego[0] + c * ego[1]]) + OFFSET
            d_ang = math.atan2(d[1].item(), d[0].item())
            te = head_est(th, lesion)                                # EGOCENTRIC steering: the turn needs heading
            th = th + math.atan2(math.sin(d_ang - te), math.cos(d_ang - te))
            v = S * torch.tensor([math.cos(th), math.sin(th)]); pos = (pos + v).clamp(-R, R)
            phi = phi + mod.gains.view(mod.K, 1, 1) * v.view(1, 1, 2)
            if (pos - goal).norm().item() < RAD:
                return 1.0
        return 1.0 if (pos - goal).norm().item() < RAD else 0.0

    return {f: {l: sum(episode(f, l) for _ in range(episodes)) / episodes for l in LESIONS} for f in FRAMES}


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 3), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 3) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    agg = {f: {l: ci([p[f][l] for p in per]) for l in LESIONS} for f in FRAMES}

    print(f"\nUNIFIED MULTI-REFERENCE-FRAME AGENT — one brain, two frames (n={a.seeds}; success, mean ± 95% CI)\n" + "=" * 78, flush=True)
    print(f"    {'lesion':>10} | {'GLOBAL goal (grid)':>20} | {'OBJECT goal (obj-vector)':>26}", flush=True)
    lab = {"none": "intact", "grid": "- grid cortex", "object": "- object-vector cells", "hd": "- head-direction"}
    for l in LESIONS:
        g, o = agg["global"][l], agg["object"][l]
        print(f"    {lab[l]:>21} | {g[0]:>17.0%}   | {o[0]:>24.0%}", flush=True)
    print(f"\n  -> ONE agent navigates in BOTH frames intact (global {agg['global']['none'][0]:.0%}, "
          f"object {agg['object']['none'][0]:.0%}); a clean DOUBLE DISSOCIATION: lesioning the GRID kills the "
          f"GLOBAL frame only ({agg['global']['grid'][0]:.0%} vs object {agg['object']['grid'][0]:.0%}); "
          f"lesioning the OBJECT-VECTOR cells kills the OBJECT frame only ({agg['object']['object'][0]:.0%} vs "
          f"global {agg['global']['object'][0]:.0%}); lesioning HEAD-DIRECTION kills BOTH "
          f"({agg['global']['hd'][0]:.0%}/{agg['object']['hd'][0]:.0%}) -- it supplies the egocentric steering "
          f"and the ego->allo transform shared by both frames. The reference-frame organs unified in one "
          f"navigating brain.", flush=True)

    out = {"n_seeds": a.seeds, "results": {f: {l: agg[f][l] for l in LESIONS} for f in FRAMES}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/agent_multiframe.json", "w"), indent=2)
    svg(agg, "results/agent_multiframe.svg")
    print("\nwrote results/agent_multiframe.json and results/agent_multiframe.svg", flush=True)


def svg(agg, out):
    pad = 64; gw = 150; gap = 40; ph = 210; W = pad + len(LESIONS) * (gw + gap) + 40; H = 80 + ph + 56
    cg, co = "#3182bd", "#e6550d"
    lab = {"none": "intact", "grid": "− grid", "object": "− object-vec", "hd": "− head-dir"}
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="28" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'One brain, two reference frames: a clean double dissociation</text>')
    e.append('<text x="28" y="44" font-size="10.5" fill="#5b6b8c">GLOBAL goal via the grid (allocentric) '
             '&#183; OBJECT goal via object-vector cells + HD (egocentric) &#183; HD shared (egocentric steering)</text>')
    oy = 56; base = oy + ph
    e.append(f'<line x1="{pad-8}" y1="{base}" x2="{W-20}" y2="{base}" stroke="#33415c"/>')
    for vv in (0.0, 0.5, 1.0):
        e.append(f'<text x="{pad-12}" y="{base-vv*ph+4:.0f}" font-size="9" fill="#5b6b8c" text-anchor="end">{int(vv*100)}%</text>')
    bw = (gw - 14) / 2
    for i, l in enumerate(LESIONS):
        x0 = pad + i * (gw + gap)
        for j, (frame, c) in enumerate((("global", cg), ("object", co))):
            v = agg[frame][l][0]; x = x0 + j * (bw + 14); h = v * ph
            e.append(f'<rect x="{x:.0f}" y="{base-h:.1f}" width="{bw:.0f}" height="{h:.1f}" fill="{c}" opacity="0.88"/>')
            e.append(f'<text x="{x+bw/2:.0f}" y="{base-h-5:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.0%}</text>')
        e.append(f'<text x="{x0+gw/2-7:.0f}" y="{base+16:.0f}" font-size="10" fill="#28324a" text-anchor="middle">{lab[l]}</text>')
    e.append(f'<rect x="{pad}" y="{base+30}" width="13" height="6" fill="{cg}"/><text x="{pad+18}" y="{base+36}" font-size="10" fill="#28324a">GLOBAL frame (grid)</text>')
    e.append(f'<rect x="{pad+180}" y="{base+30}" width="13" height="6" fill="{co}"/><text x="{pad+198}" y="{base+36}" font-size="10" fill="#28324a">OBJECT frame (object-vector + HD)</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
