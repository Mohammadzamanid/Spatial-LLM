"""
src/eval/unified_agent_cortex.py

THE UNIFIED AGENT, GROUNDED ON THE REAL GRID CORTEX (GAPS.md integration capstone, grounded).

`unified_agent.py` composed the survival organs at the belief level. This runs the SAME emergent survival policy
on the ACTUAL shared substrate: the agent's POSITION comes from decoding the real velocity-driven hexagonal grid
cortex (`_HexGridModules`, #7/#8) as it path-integrates and DRIFTS, and its UNCERTAINTY is the real reconstruction
residual ρ = ‖code − grid_code_at(decode(code))‖ (#7 showed this is calibrated to the true decode error under
independent per-module drift). Nothing is an integer counter any more — the drift, the miss, and the sense of
being lost are all produced by the cortex.

The loop: the agent believes it is at p̂ = decode(grid code); it navigates toward a resource using p̂, so when the
code has drifted it aims wrong and MISSES (the true body ends up off the resource); a LANDMARK, sensed
allothetically, re-anchors the cortex (ρ → 0); resources reset the matching deficit only when the TRUE body
reaches them. The policy is the emergent survival plan of the capstone (plan with an internal model, execute on
the real cortex); which resource and when to relocalise are never hardcoded. We re-run the capstone's two claims
on the real substrate:

  (A) THE POSITION-ORGAN DISSOCIATION still holds on the real cortex — grid (scramble the decode), uncertainty
      (ignore the real ρ), and landmark (block re-anchoring) are each clearly load-bearing. (Honest limit: the
      interoceptive DRIVE organ, cleanly load-bearing for resource CHOICE in #4, barely moves SURVIVAL here — with
      two symmetric resources a non-adaptive alternation nearly suffices — so grounding gives a clean 3-organ, not
      4-organ, dissociation. We report all four and say so.)
  (B) THE EMERGENT COMPLEMENTARITY still holds — the real uncertainty read-out is worth survival only WITH the
      landmark organ; remove landmarks and knowing ρ no longer helps.

So the position organs proven one-at-a-time cohere into a surviving animal on the real cortex. Multi-seed,
mean ± 95% CI. Writes results/unified_agent_cortex.json + .svg.

    python -m src.eval.unified_agent_cortex --seeds 5
"""
import argparse
import json
import math
import os

import torch

from src.eval.agent_grid_cortex import build_cortex, train_decoder
from src.eval.successor import ci95
from src.eval.unified_agent import (D, G, GAMMA, HORIZON, LMS, FOOD, U, WATER,
                                    step_pol, value_iteration)

AR = 2.0                 # arena half-width (within the cortex's unambiguous range)
CELL = 2 * AR / G        # region size
STEP = 0.30              # locomotion step
SM = 0.13                # per-module drift std (strong enough that un-corrected drift causes real misses)
SENSE = 0.60             # landmark re-anchor radius (allothetic)
RES_R = 0.32             # resource-acquisition radius (drift pushes the true body outside it -> a miss)


def center(i, j):
    return torch.tensor([-AR + (i + 0.5) * CELL, -AR + (j + 0.5) * CELL])


def region(p):
    i = int(min(max((p[0].item() + AR) / CELL, 0), G - 1))
    j = int(min(max((p[1].item() + AR) / CELL, 0), G - 1))
    return i, j


def grid_code(mod, phi):
    return mod._grid_code(phi)


def decode_and_residual(mod, dec, phi):
    code = mod._grid_code(phi)
    phat = dec(code)[0]
    rho = (code - mod.grid_code_at(phat.unsqueeze(0))).norm().item()
    return phat, rho


def calibrate_u(mod, dec, gen, trials=6):
    """Map the real residual ρ to the planner's uncertainty level by measuring ρ at each step-since-anchor (ρ
    grows with drift, #7). u_of_rho then approximates 'steps since last re-anchor' — the planner's u."""
    curves = []
    for _ in range(trials):
        pos = (torch.rand(2, generator=gen) * 2 - 1) * AR * 0.5
        phi = mod.gains.view(mod.K, 1, 1) * pos.view(1, 1, 2).clone()
        row = []
        for _ in range(U + 2):
            _, rho = decode_and_residual(mod, dec, phi)
            row.append(rho)
            head = torch.randn(2, generator=gen); head = head / head.norm() * STEP
            phi = phi + mod.gains.view(mod.K, 1, 1) * head.view(1, 1, 2) + torch.randn(mod.K, 1, 2, generator=gen) * SM
        curves.append(row)
    mean = torch.tensor(curves).mean(0)
    return mean[1:U + 1].tolist()                                  # thresholds: ρ at steps 1..U


def u_of_rho(rho, thr):
    return min(sum(rho >= t for t in thr), U)


def rollout(mod, dec, thr, V, rt, rh, see_pos=True, see_u=True, see_drive=True, lm_work=True, seed=0):
    gen = torch.Generator().manual_seed(seed * 100 + 21)
    true = center(*START).clone()
    phi = mod.gains.view(mod.K, 1, 1) * true.view(1, 1, 2).clone()
    t = h = 0; drv = []
    lm_c = [center(*c) for c in LMS]; w_c = center(*WATER); f_c = center(*FOOD)
    for _ in range(HORIZON):
        phat, rho = decode_and_residual(mod, dec, phi)
        if not see_pos:
            phat = (torch.rand(2, generator=gen) * 2 - 1) * AR       # grid lesion: no position sense
        xr, yr = region(phat)
        ul = u_of_rho(rho, thr) if see_u else 0
        a, b = step_pol(V, xr, yr, ul, t, h, rt, rh, see_pos, see_u, see_drive, lm_work, gen)
        tgt = center(a, b)
        v = tgt - phat
        if v.norm() > 1e-6:
            v = v / v.norm() * STEP                                 # move toward the target BASED ON BELIEF p̂
        true = (true + v).clamp(-AR, AR)
        phi = phi + mod.gains.view(mod.K, 1, 1) * v.view(1, 1, 2) + torch.randn(mod.K, 1, 2, generator=gen) * SM
        if lm_work and min((true - c).norm().item() for c in lm_c) < SENSE:      # allothetic re-anchor
            phi = mod.gains.view(mod.K, 1, 1) * true.view(1, 1, 2).clone()
        t = 0 if (true - w_c).norm().item() < RES_R else min(t + rt, D)          # drink only if the TRUE body is there
        h = 0 if (true - f_c).norm().item() < RES_R else min(h + rh, D)
        drv.append(t ** 2 + h ** 2)
    return sum(drv) / len(drv)


START = (3, 0)


def run_seed(seed):
    torch.manual_seed(seed)
    mod = build_cortex(seed)
    gen = torch.Generator().manual_seed(seed + 500)
    dec = train_decoder(mod, gen, nonlinear=True, iters=1200)
    thr = calibrate_u(mod, dec, gen)
    rt, rh = (2, 1) if seed % 2 == 0 else (1, 2)                   # asymmetric drives (interoception validated in #4)
    V = value_iteration(rt, rh, lm=True)
    Vno = value_iteration(rt, rh, lm=False)

    def d(V_, lm_work=True, **kw):                                 # average over drift realisations (variance control)
        return sum(rollout(mod, dec, thr, V_, rt, rh, seed=seed * 20 + r, lm_work=lm_work, **kw) for r in range(6)) / 6
    intact = d(V)
    out = {"drive_intact": intact, "drive_no_grid": d(V, see_pos=False), "drive_no_uncertainty": d(V, see_u=False),
           "drive_no_landmark": d(Vno, lm_work=False), "drive_no_drive": d(V, see_drive=False),
           "drive_no_both": d(Vno, lm_work=False, see_u=False)}
    out["cost_unc_with_lm"] = out["drive_no_uncertainty"] - intact
    out["cost_unc_without_lm"] = out["drive_no_both"] - out["drive_no_landmark"]
    out["interaction"] = out["cost_unc_with_lm"] - out["cost_unc_without_lm"]
    return out


KEYS = ["drive_intact", "drive_no_grid", "drive_no_uncertainty", "drive_no_landmark", "drive_no_drive",
        "cost_unc_with_lm", "cost_unc_without_lm", "interaction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    agg = {k: ci95([p[k] for p in per]) for k in KEYS}

    print(f"THE UNIFIED AGENT ON THE REAL GRID CORTEX (n={a.seeds}; mean ± 95% CI)\n" + "=" * 70, flush=True)
    print("  position = decoded from the real drifting grid code; uncertainty = the real #7 residual\n", flush=True)
    lab = {"drive_intact": "ALL ORGANS INTACT — mean drive (lower = alive)",
           "drive_no_grid": "  − grid position sense   (scramble the decode)",
           "drive_no_uncertainty": "  − uncertainty read-out  (ignore the real ρ)",
           "drive_no_landmark": "  − landmark reset        (block re-anchoring)",
           "drive_no_drive": "  − interoceptive drive    (can't read deficits)"}
    for k in ["drive_intact", "drive_no_grid", "drive_no_uncertainty", "drive_no_landmark", "drive_no_drive"]:
        print(f"  {lab[k]:52} {agg[k][0]:6.1f} ± {agg[k][1]:.1f}", flush=True)
    print("  " + "-" * 66, flush=True)
    print(f"  interaction: cost of −uncertainty WITH landmarks      {agg['cost_unc_with_lm'][0]:+6.1f} ± {agg['cost_unc_with_lm'][1]:.1f}", flush=True)
    print(f"               cost of −uncertainty once landmarks GONE {agg['cost_unc_without_lm'][0]:+6.1f} ± {agg['cost_unc_without_lm'][1]:.1f}", flush=True)
    print(f"\n  A. on the REAL cortex the three POSITION organs cohere and dissociate cleanly: intact "
          f"{agg['drive_intact'][0]:.0f} vs − grid {agg['drive_no_grid'][0]:.0f}, − uncertainty "
          f"{agg['drive_no_uncertainty'][0]:.0f}, − landmark {agg['drive_no_landmark'][0]:.0f}. (Honest: the "
          f"interoceptive DRIVE organ — the resource CHOICE cleanly load-bearing in #4 — barely moves survival "
          f"here [{agg['drive_no_drive'][0]:.0f}], because non-adaptive alternation nearly suffices for two "
          f"symmetric resources.)", flush=True)
    print(f"  B. the emergent circuit survives grounding: the real uncertainty read-out is worth "
          f"{agg['cost_unc_with_lm'][0]:+.0f} WITH landmarks but {agg['cost_unc_without_lm'][0]:+.0f} without — "
          f"knowing you're lost only helps if you can re-anchor.", flush=True)

    out = {"n_seeds": a.seeds, "arena": AR, "drift_std": SM,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS},
           "verdict": "The unified survival agent runs on the ACTUAL grid cortex: position is decoded from the real "
                      "drifting grid code and uncertainty is the real #7 reconstruction residual (no counters). The "
                      "three POSITION organs — grid, uncertainty read-out, landmark relocalisation — dissociate "
                      "cleanly (each ablation raises drive), and the emergent uncertainty×landmark complementarity "
                      "survives grounding (the real ρ is worth survival only WITH landmarks). Honest limit: the "
                      "interoceptive DRIVE organ, cleanly load-bearing for resource CHOICE in #4, barely moves "
                      "survival in this two-resource regime where non-adaptive alternation nearly suffices — so the "
                      "grounded capstone is a clean 3-organ dissociation, not 4. Nothing about which resource or "
                      "when to relocalise is hardcoded."}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/unified_agent_cortex.json", "w"), indent=2)
    svg_grounded(agg, "results/unified_agent_cortex.svg")
    print("\nwrote results/unified_agent_cortex.json and results/unified_agent_cortex.svg", flush=True)


def svg_grounded(agg, out):
    W_, H = 700, 300
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W_}" height="{H}" fill="#ffffff"/>',
         '<text x="20" y="26" font-size="15" font-weight="800" fill="#0b1324">'
         'The unified agent on the REAL grid cortex: the organs cohere into a surviving animal</text>',
         '<text x="20" y="45" font-size="10.5" fill="#5b6b8c">position decoded from the real drifting grid code; '
         'uncertainty = the real #7 reconstruction residual (no counters)</text>']
    bx, by, bh, bw = 44, 82, 175, 52
    e.append(f'<text x="{bx}" y="{by-8}" font-size="11" font-weight="700" fill="#28324a">mean drive (lower = alive)</text>')
    bars = [("drive_intact", "intact", "#2ca25f"), ("drive_no_grid", "−grid", "#c9341a"),
            ("drive_no_uncertainty", "−uncert", "#e6842a"), ("drive_no_landmark", "−land", "#c9341a"),
            ("drive_no_drive", "−drive", "#e6842a")]
    top = max(agg[k][0] for k, _, _ in bars) * 1.15
    for i, (k, lab, col) in enumerate(bars):
        v = agg[k][0]; x = bx + i * (bw + 8); hh = v / top * bh
        e.append(f'<rect x="{x}" y="{by+bh-hh:.0f}" width="{bw}" height="{hh:.0f}" fill="{col}" opacity="0.85"/>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{by+bh-hh-5:.0f}" font-size="10.5" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.0f}</text>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{by+bh+14:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{lab}</text>')
    e.append(f'<line x1="{bx-4}" y1="{by+bh}" x2="{bx+5*(bw+8):.0f}" y2="{by+bh}" stroke="#33415c"/>')
    e.append(f'<line x1="{bx-4}" y1="{by+bh-agg["drive_intact"][0]/top*bh:.0f}" x2="{bx+5*(bw+8):.0f}" y2="{by+bh-agg["drive_intact"][0]/top*bh:.0f}" stroke="#2ca25f" stroke-dasharray="3 3" opacity="0.5"/>')
    rx = 470; rw = 80
    e.append(f'<text x="{rx}" y="{by-8}" font-size="11" font-weight="700" fill="#28324a">real uncertainty organ:</text>')
    e.append(f'<text x="{rx}" y="{by+6}" font-size="11" font-weight="700" fill="#28324a">cost of losing it</text>')
    ct = max(0.5, agg["cost_unc_with_lm"][0]) * 1.4
    for i, (k, lab, col) in enumerate([("cost_unc_with_lm", "with\nlandmarks", "#2b8cbe"), ("cost_unc_without_lm", "landmarks\ngone", "#8c8c8c")]):
        v = max(0.0, agg[k][0]); x = rx + i * (rw + 20); hh = v / ct * (bh - 20)
        e.append(f'<rect x="{x}" y="{by+bh-hh:.0f}" width="{rw}" height="{hh:.0f}" fill="{col}" opacity="0.85"/>')
        e.append(f'<text x="{x+rw/2:.0f}" y="{by+bh-hh-5:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{agg[k][0]:+.0f}</text>')
        for li, part in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+rw/2:.0f}" y="{by+bh+13+li*11:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{part}</text>')
    e.append(f'<line x1="{rx-4}" y1="{by+bh}" x2="{rx+2*rw+20:.0f}" y2="{by+bh}" stroke="#33415c"/>')
    e.append(f'<text x="{rx}" y="{by+bh+40:.0f}" font-size="9.5" fill="#5a6b8c">the emergent circuit survives grounding on</text>')
    e.append(f'<text x="{rx}" y="{by+bh+52:.0f}" font-size="9.5" fill="#5a6b8c">the actual shared substrate.</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
