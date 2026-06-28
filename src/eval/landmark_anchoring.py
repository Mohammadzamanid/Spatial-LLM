"""
src/eval/landmark_anchoring.py

DYNAMIC REFERENCE-FRAME ANCHORING — reliability-gated reanchoring of the grid phase to a landmark, with
ALLOCENTRIC and EGOCENTRIC codes coexisting. The capstone of the reference-frame work: not a global GPS, but
a map that is path-integrated globally AND dynamically reanchored to a task-relevant landmark under cue
reliability (Nature Neurosci 2025: grid cells reanchor by translating the pattern; Nature Comms 2025:
allocentric & egocentric representations coexist in MEC).

The mechanism is the review's exact formula, applied to a LANDMARK (a point cue visible from a distance),
like boundary anchoring but anywhere the landmark is seen:

    grid_phase = integrate(self_motion)                          # allocentric path integration (drifts)
    ego = EgocentricObjectVectorCells(landmark)                  # egocentric distance + bearing (OVC organ)
    p_hat = anchor_pos - R(heading) @ ego                        # landmark-implied position
    w = reliability_gate(landmark_precision)                     # down-weight a far / noisy landmark
    grid_phase = (1 - w) * grid_phase + w * gains * p_hat        # reanchor

We measure:
  (A) REANCHORING CORRECTS ALLOCENTRIC DRIFT. global (allocentric) position error: pure path integration
      drifts unbounded; landmark reanchoring bounds it (a landmark gives an allothetic fix anywhere it is
      seen, not only at walls).
  (B) ALLOCENTRIC AND EGOCENTRIC COEXIST. at every step the agent reads BOTH global position (allocentric,
      from the grid) AND landmark-relative position (egocentric, from the object-vector cells) — both
      accurate simultaneously (the MEC coexistence).
  (C) RELIABILITY. as the landmark cue is made noisier, the reanchoring's benefit falls and the gate
      down-weights it. (Honest: the strictly-optimal combiner is the *learned* fuser of
      agent_cue_integration; a hand-coded Kalman gate is mis-calibrated here, so we use a reliability gate
      and report the dependence, not an optimal-weighting claim.)

Multi-seed, mean +/- 95% CI. Writes results/landmark_anchoring.json + .svg.

    python -m src.eval.landmark_anchoring --seeds 3
"""
import argparse
import json
import math
import os

import torch

from src.eval.head_direction import train_hd, canonical, nearest
from src.eval.agent_grid_cortex import build_cortex, train_decoder, R
from src.eval.reference_frame import train_ovc

S = 0.2; LM = torch.tensor([1.4, 1.4]); WALK = 160; A_NOISE = 0.05
W0 = 0.7; REF = 0.08                          # reanchor strength + reliability reference precision
NOISES = [0.05, 0.15, 0.30]                   # landmark-observation base noise (panel C)


def run_walk(organs, mode, gen, lm_noise=0.05):
    """mode: 'pi' (no landmark) or 'anchor' (reliability-gated landmark reanchoring). Returns the
    last-segment allocentric (global) error and egocentric (landmark-relative) error, plus a drift trace."""
    hd, keys, vals, mod, dec, ovc, lin = organs
    gains = mod.gains
    pos = (torch.rand(2, generator=gen) * 2 - 1) * R
    phi = gains.view(mod.K, 1, 1) * pos.view(1, 1, 2).clone()
    th = torch.rand(1, generator=gen).item() * 2 * math.pi
    aerr, eerr, trace = [], [], []
    for t in range(WALK):
        th = th + 0.3 * math.sin(t * 0.3) + torch.randn(1, generator=gen).item() * 0.15
        v = S * torch.tensor([math.cos(th), math.sin(th)]); pos = (pos + v).clamp(-R, R)
        phi = phi + gains.view(mod.K, 1, 1) * (v + torch.randn(2, generator=gen) * A_NOISE).view(1, 1, 2)
        if mode == "anchor":
            vrel = LM - pos
            r = vrel.norm(); beta = math.atan2(vrel[1].item(), vrel[0].item()) - th
            eff = lm_noise * (0.4 + 1.6 * r.item() / R)                       # far landmark -> noisier
            rn = torch.tensor([(r + torch.randn(1, generator=gen) * eff).clamp(min=0).item()])
            bn = torch.tensor([beta + torch.randn(1, generator=gen).item() * eff])
            ego = lin(ovc(rn, bn))[0]                                         # EGOCENTRIC object-vector
            th_est = hd.decode(nearest(keys, vals, th))
            c, sn = math.cos(th_est), math.sin(th_est)
            ego_allo = torch.tensor([c * ego[0] - sn * ego[1], sn * ego[0] + c * ego[1]])
            p_hat = LM - ego_allo                                            # anchor - R(heading) @ ego
            w = W0 * REF ** 2 / (REF ** 2 + eff ** 2)                        # reliability gate: down-weight noisy/far
            phi = (1 - w) * phi + w * (gains.view(mod.K, 1, 1) * p_hat.view(1, 1, 2))
            eerr.append((ego_allo - vrel).norm().item())                     # egocentric (landmark-relative) error
        ae = (dec(mod._grid_code(phi))[0] - pos).norm().item()              # allocentric (global) error
        aerr.append(ae); trace.append(ae)
    return (sum(aerr[-40:]) / 40, (sum(eerr[-40:]) / 40 if eerr else float("nan")), trace)


def run_seed(seed):
    gen = torch.Generator().manual_seed(seed)
    hd, _ = train_hd(seed, iters=1500); keys, vals = canonical(hd)
    mod = build_cortex(seed); dec = train_decoder(mod, gen, nonlinear=True, iters=1200)
    ovc, lin = train_ovc(gen)
    organs = (hd, keys, vals, mod, dec, ovc, lin)
    N = 30
    pi = sum(run_walk(organs, "pi", gen)[0] for _ in range(N)) / N
    an_a, an_e = [], []
    tr_pi = run_walk(organs, "pi", gen)[2]; tr_an = run_walk(organs, "anchor", gen)[2]
    for _ in range(N):
        a, e, _ = run_walk(organs, "anchor", gen); an_a.append(a); an_e.append(e)
    rel = {nz: sum(run_walk(organs, "anchor", gen, lm_noise=nz)[0] for _ in range(N)) / N for nz in NOISES}
    return {"pi": pi, "anchor_allo": sum(an_a) / N, "anchor_ego": sum(an_e) / N,
            "rel": rel, "trace_pi": tr_pi, "trace_an": tr_an}


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 4), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 4) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    pi = ci([p["pi"] for p in per]); allo = ci([p["anchor_allo"] for p in per]); ego = ci([p["anchor_ego"] for p in per])
    rel = {nz: ci([p["rel"][nz] for p in per]) for nz in NOISES}
    tr_pi = [sum(p["trace_pi"][t] for p in per) / a.seeds for t in range(WALK)]
    tr_an = [sum(p["trace_an"][t] for p in per) / a.seeds for t in range(WALK)]

    print(f"\nDYNAMIC LANDMARK REANCHORING — allocentric+egocentric coexistence (n={a.seeds}; mean ± 95% CI)\n" + "=" * 82, flush=True)
    print(f"(A) reanchoring corrects allocentric drift:  PI-only {pi[0]:.3f} -> landmark-anchored {allo[0]:.3f}", flush=True)
    print(f"(B) COEXISTENCE: allocentric (global) err {allo[0]:.3f}  AND  egocentric (landmark-relative) err {ego[0]:.3f}  -- simultaneously", flush=True)
    print("(C) reliability — anchored allocentric err vs landmark-observation noise:", flush=True)
    for nz in NOISES:
        print(f"    lm_noise={nz}: {rel[nz][0]:.3f} ± {rel[nz][1]:.3f}", flush=True)
    print(f"\n  -> the grid phase is dynamically REANCHORED to a landmark (anchor - R(heading)@ego, "
          f"reliability-gated): pure path integration drifts ({pi[0]:.2f}) but a landmark seen anywhere bounds "
          f"the allocentric error ({allo[0]:.2f}); and the agent reads BOTH global (allocentric {allo[0]:.2f}) "
          f"AND landmark-relative (egocentric {ego[0]:.2f}) position at once -- the two MEC frames coexisting. "
          f"Reliability matters (err {rel[NOISES[0]][0]:.2f}->{rel[NOISES[-1]][0]:.2f} as the landmark gets "
          f"noisier); the strictly-optimal combiner is the learned fuser (agent_cue_integration). The map is a "
          f"reference-frame transformer: path-integrated globally, reanchored to landmarks on demand.", flush=True)

    out = {"n_seeds": a.seeds, "pi": pi, "anchor_allo": allo, "anchor_ego": ego,
           "reliability": {str(nz): rel[nz] for nz in NOISES}, "trace_pi": tr_pi, "trace_an": tr_an}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/landmark_anchoring.json", "w"), indent=2)
    svg(pi, allo, ego, rel, tr_pi, tr_an, "results/landmark_anchoring.svg")
    print("\nwrote results/landmark_anchoring.json and results/landmark_anchoring.svg", flush=True)


def svg(pi, allo, ego, rel, tr_pi, tr_an, out):
    pad = 56; pw = 300; ph = 200; gap = 92; W = pad + 2 * pw + gap + 24; H = 86 + ph + 46
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'Dynamic landmark reanchoring: allocentric drift corrected, egocentric code coexists</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">grid phase reanchored to a landmark '
             '(anchor &#8722; R(heading)&#183;ego, reliability-gated); global + landmark-relative position read at once</text>')
    oy = 60
    # Panel A: allocentric error over the walk (PI drifts vs anchored bounded)
    oxA = pad; tmax = max(max(tr_pi), max(tr_an)) * 1.1 + 1e-6
    def XA(t): return oxA + (t / (WALK - 1)) * pw
    def YA(v): return oy + ph - (v / tmax) * ph
    e.append(f'<text x="{oxA}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(A) allocentric (global) error over a walk</text>')
    e.append(f'<line x1="{oxA}" y1="{oy+ph}" x2="{oxA+pw}" y2="{oy+ph}" stroke="#33415c"/>'
             f'<line x1="{oxA}" y1="{oy}" x2="{oxA}" y2="{oy+ph}" stroke="#33415c"/>')
    for who, tr, c in (("pi", tr_pi, "#c9341a"), ("an", tr_an, "#2ca25f")):
        pts = " ".join(f"{XA(t):.1f},{YA(tr[t]):.1f}" for t in range(WALK))
        e.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.2"/>')
    e.append(f'<text x="{oxA+pw/2:.0f}" y="{oy+ph+16:.0f}" font-size="9.5" fill="#5b6b8c" text-anchor="middle">step &#8594;</text>')
    e.append(f'<rect x="{oxA+pw-150}" y="{oy+6}" width="13" height="4" fill="#c9341a"/><text x="{oxA+pw-133}" y="{oy+11}" font-size="9" fill="#28324a">PI-only (drifts {pi[0]:.2f})</text>')
    e.append(f'<rect x="{oxA+pw-150}" y="{oy+22}" width="13" height="4" fill="#2ca25f"/><text x="{oxA+pw-133}" y="{oy+27}" font-size="9" fill="#28324a">landmark-anchored ({allo[0]:.2f})</text>')
    # Panel B: coexistence + reliability
    oxB = pad + pw + gap
    e.append(f'<text x="{oxB}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(B) coexistence + (C) reliability</text>')
    e.append(f'<text x="{oxB}" y="{oy+22}" font-size="11" fill="#28324a">allocentric (global): <tspan font-weight="700" fill="#2ca25f">{allo[0]:.2f}</tspan></text>')
    e.append(f'<text x="{oxB}" y="{oy+40}" font-size="11" fill="#28324a">egocentric (landmark-rel): <tspan font-weight="700" fill="#3182bd">{ego[0]:.2f}</tspan></text>')
    e.append(f'<text x="{oxB}" y="{oy+56}" font-size="9" fill="#7787a6">both read simultaneously (the two MEC frames)</text>')
    bx = oxB; by = oy + 80; bw = (pw - 40) / len(NOISES); hmax = max(rel[nz][0] for nz in NOISES) * 1.25
    e.append(f'<text x="{oxB}" y="{by-6}" font-size="9.5" fill="#5b6b8c">anchored allocentric err vs landmark noise:</text>')
    base = by + 100
    e.append(f'<line x1="{bx}" y1="{base}" x2="{bx+len(NOISES)*bw}" y2="{base}" stroke="#33415c"/>')
    for i, nz in enumerate(NOISES):
        v = rel[nz][0]; h = v / hmax * 90; x = bx + i * bw + 8
        e.append(f'<rect x="{x:.0f}" y="{base-h:.1f}" width="{bw-16:.0f}" height="{h:.1f}" fill="#e6a000" opacity="0.85"/>')
        e.append(f'<text x="{x+(bw-16)/2:.0f}" y="{base-h-4:.0f}" font-size="9" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        e.append(f'<text x="{x+(bw-16)/2:.0f}" y="{base+13:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{nz}</text>')
    e.append(f'<text x="{bx}" y="{base+28:.0f}" font-size="8" fill="#9aa6bd">(optimal combiner = learned fuser, agent_cue_integration)</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
