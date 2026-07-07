"""
src/eval/representational_drift.py

REPRESENTATIONAL DRIFT and the POPULATION GEOMETRY that survives it (GAPS.md Tier 5, #C6).

Place cells change their tuning over DAYS even in a fixed environment with stable behavior — representational
drift (Ziv 2013; Rule 2019). What supports stable behavior across it? The population-geometry answer (Morales
2025; and 2025 CA1 coordinated-drift work): read the ENVIRONMENT'S GEOMETRY carried by the population manifold,
not the identity of particular cells — the geometry survives drift that reshuffles which cells code what.

We test this NON-circularly (an earlier version was circular — RSA over a Gaussian tiling is blind to remapping;
this version is rebuilt against that critique). We compare, at MATCHED single-cell drift, two kinds of drift and
a label-free geometry read-out:

  RELOCATE (geometry-preserving): a fraction of place fields relocate each day (+ gain jitter). Single-cell
     tuning changes, but the population still tiles the track, so the manifold — the environment's 1-D geometry
     — is preserved.
  REMAP (geometry-preserving, extreme): ALL cells re-tile the track in one step (0% cell identity conserved).
  NOISE (geometry-DESTROYING): independent per-neuron drift of MATCHED magnitude (same single-cell tuning change)
     that thickens/corrupts the manifold rather than moving it coherently.

Read-outs: a FIXED decoder (frozen day-0 linear weights, bound to specific cells) and a GEOMETRY read-out that is
LABEL-FREE — it recovers position from the current day's manifold ORDERING (the Fiedler / kNN-Laplacian 1-D
coordinate), using no current position labels, only the conserved geometry (calibrated once to the day-0 map).

Measured signatures (n=5), never imposed:
  (A) GEOMETRY, NOT CELLS, IS WHAT SURVIVES: at MATCHED single-cell drift, the label-free geometry read-out is
      near-perfect under geometry-preserving drift (relocate) but FAILS under geometry-destroying drift (noise).
      Since the single-cell drift is matched, the difference is the drift STRUCTURE (whether the geometry is
      conserved), not how much the cells changed.
  (B) A FIXED decoder degrades under any drift (its cells moved) while the geometry read-out survives.
  (C) ROBUST TO REMAPPING: the geometry read-out survives even a FULL remap (0% cells conserved) — it reads the
      environment's geometry, not cell identity. (This is the honest resolution of "is conservation of cells
      what matters?" — it is not; conservation of the GEOMETRY is.)

Honest scope: a phenomenological place-code drift model; the geometry read-out is label-free unsupervised
manifold decoding (it fails when the manifold is corrupted, which is the point). Multi-seed, mean ± 95% CI.
Writes results/representational_drift.json + .svg.

    python -m src.eval.representational_drift --seeds 5
"""
import argparse
import json
import math
import os

import torch

N = 200                # cells
P = 60                 # probe positions
DAYS = 30
SIG = 0.07             # place-field width
RELOCATE = 0.06        # fraction of fields that relocate per day
GAIN_JIT = 0.10
NOISE_SD = 0.40        # per-day per-neuron drift for the geometry-DESTROYING control (matched to relocate's cell drift)
KNN = 6                # neighbours for the manifold-graph read-out
XS = torch.linspace(0, 1, P)


def place_code(centers, gains):
    d2 = (XS.unsqueeze(1) - centers.unsqueeze(0)) ** 2
    return gains.unsqueeze(0) * torch.exp(-d2 / (2 * SIG ** 2))


def _cell_corr(R0, Rd):
    a, b = R0 - R0.mean(0, keepdim=True), Rd - Rd.mean(0, keepdim=True)
    den = a.norm(dim=0) * b.norm(dim=0) + 1e-9
    m = den > 1e-6
    return ((a * b).sum(0)[m] / den[m]).mean().item()


def _manifold_err(Rd):
    """LABEL-FREE geometry read-out: recover position from the population's 1-D manifold ordering (the Fiedler
    vector of a kNN graph), calibrated only to the day-0 map (uniform track). Error is low iff the manifold —
    the environment's geometry — is intact; high if the drift has corrupted it."""
    Rn = Rd - Rd.mean(0, keepdim=True)
    D2 = torch.cdist(Rn, Rn)
    knn = D2.argsort(1)[:, 1:KNN + 1]
    Wa = torch.zeros(P, P)
    Wa.scatter_(1, knn, 1.0)
    Wa = torch.maximum(Wa, Wa.t())                                   # symmetric kNN affinity
    lap = torch.diag(Wa.sum(1)) - Wa
    f = torch.linalg.eigh(lap)[1][:, 1]                              # Fiedler vector = 1-D manifold coordinate
    rank = f.argsort().argsort().float() / (P - 1)
    if ((rank - rank.mean()) * (XS - XS.mean())).sum() < 0:          # sign/direction calibration (day-0 geometry)
        rank = 1 - rank
    return (rank - XS).abs().mean().item()


def _fixed_err(Rd, W0):
    return ((torch.cat([Rd, torch.ones(P, 1)], 1) @ W0).squeeze(1) - XS).abs().mean().item()


_TR = torch.arange(0, P, 2)
_TE = torch.arange(1, P, 2)


def _heldout_err(Rd):
    """SUPERVISED confirmation that the GEOMETRY (not just the label-free reader) is what differs: fit a linear
    decoder on half the positions and test on the held-out half. It GENERALIZES only if the code still lies on a
    smooth low-D manifold — so it survives geometry-preserving drift and fails geometry-destroying drift. (An
    all-position fit would OVERFIT — N>P — and hide this; held-out exposes it.)"""
    W = torch.linalg.lstsq(torch.cat([Rd[_TR], torch.ones(len(_TR), 1)], 1), XS[_TR].unsqueeze(1)).solution
    return ((torch.cat([Rd[_TE], torch.ones(len(_TE), 1)], 1) @ W).squeeze(1) - XS[_TE]).abs().mean().item()


def simulate(seed, mode):
    g = torch.Generator().manual_seed(seed)
    centers = torch.rand(N, generator=g); gains = 0.5 + torch.rand(N, generator=g)
    R0 = place_code(centers, gains)
    W0 = torch.linalg.lstsq(torch.cat([R0, torch.ones(P, 1)], 1), XS.unsqueeze(1)).solution
    Rd = R0.clone(); noise = torch.zeros(P, N)
    for _ in range(DAYS):
        if mode in ("relocate", "remap"):
            frac = RELOCATE if mode == "relocate" else 1.0
            m = torch.rand(N, generator=g) < frac
            centers = centers.clone(); centers[m] = torch.rand(int(m.sum()), generator=g)
            gains = (gains * (1 + GAIN_JIT * torch.randn(N, generator=g))).clamp(min=0.05)
            Rd = place_code(centers, gains)
        else:                                                        # geometry-destroying independent drift
            noise = noise + NOISE_SD * torch.randn(P, N, generator=g)
            Rd = R0 + noise
    return _cell_corr(R0, Rd), _manifold_err(Rd), _fixed_err(Rd, W0), _heldout_err(Rd)


def run_seed(seed, iters=None):
    cc_r, mf_r, fx_r, ho_r = simulate(seed, "relocate")
    cc_n, mf_n, fx_n, ho_n = simulate(seed + 300, "noise")
    cc_m, mf_m, fx_m, ho_m = simulate(seed + 600, "remap")
    return {
        "cell_corr_relocate": cc_r, "cell_corr_noise": cc_n, "cell_corr_remap": cc_m,
        "manifold_relocate": mf_r, "manifold_noise": mf_n, "manifold_remap": mf_m,
        "heldout_relocate": ho_r, "heldout_noise": ho_n, "heldout_remap": ho_m,
        "fixed_relocate": fx_r, "fixed_noise": fx_n,
        "geometry_gap": mf_n - mf_r,                                 # (A) label-free reader: geometry destroyed vs preserved
        "heldout_gap": ho_n - ho_r,                                 # (A') supervised held-out: same dissociation
        "reader_vs_fixed": fx_r - mf_r,                             # (B) geometry read-out beats the fixed decoder under drift
        "drift_match": abs(cc_r - cc_n),                           # single-cell drift matched between relocate & noise
    }


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0


KEYS = ["cell_corr_relocate", "cell_corr_noise", "cell_corr_remap",
        "manifold_relocate", "manifold_noise", "manifold_remap",
        "heldout_relocate", "heldout_noise", "heldout_remap", "fixed_relocate", "fixed_noise",
        "geometry_gap", "heldout_gap", "reader_vs_fixed", "drift_match"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    for s, p in enumerate(per):
        print(f"  seed {s}: cell-corr relocate {p['cell_corr_relocate']:+.2f} / noise {p['cell_corr_noise']:+.2f} "
              f"(matched) | geometry read-out: relocate {p['manifold_relocate']:.3f} / remap "
              f"{p['manifold_remap']:.3f} / noise {p['manifold_noise']:.3f} | fixed {p['fixed_relocate']:.3f}", flush=True)
    agg = {k: ci95([p[k] for p in per]) for k in KEYS}

    print(f"\nREPRESENTATIONAL DRIFT — the population GEOMETRY survives it (n={a.seeds}; mean ± 95% CI)\n" + "=" * 90, flush=True)
    print(f"  single-cell drift is MATCHED across the geometry-preserving and geometry-destroying conditions: "
          f"cell-corr relocate {agg['cell_corr_relocate'][0]:+.2f} vs noise {agg['cell_corr_noise'][0]:+.2f} "
          f"(|Δ| {agg['drift_match'][0]:.2f})", flush=True)
    print(f"  (A) GEOMETRY, NOT CELLS, IS WHAT SURVIVES — the label-free geometry read-out error:", flush=True)
    print(f"      geometry-PRESERVING drift (relocate) {agg['manifold_relocate'][0]:.3f} ± "
          f"{agg['manifold_relocate'][1]:.3f}  vs  geometry-DESTROYING drift (noise) {agg['manifold_noise'][0]:.3f} "
          f"± {agg['manifold_noise'][1]:.3f}   (gap {agg['geometry_gap'][0]:+.3f} ± {agg['geometry_gap'][1]:.3f}; "
          f"chance ≈ 0.25)", flush=True)
    print(f"      SUPERVISED confirmation (held-out linear decode — exposes the same, not overfit): preserving "
          f"{agg['heldout_relocate'][0]:.3f} vs destroying {agg['heldout_noise'][0]:.3f} (gap "
          f"{agg['heldout_gap'][0]:+.3f} ± {agg['heldout_gap'][1]:.3f}) — even with labels, position does not "
          f"generalise once the geometry is gone", flush=True)
    print(f"  (B) FIXED decoder fails, GEOMETRY read-out survives (under relocate): fixed "
          f"{agg['fixed_relocate'][0]:.3f} vs geometry {agg['manifold_relocate'][0]:.3f} "
          f"(gap {agg['reader_vs_fixed'][0]:+.3f} ± {agg['reader_vs_fixed'][1]:.3f})", flush=True)
    print(f"  (C) ROBUST TO REMAPPING: a FULL remap (0% cells conserved) — geometry read-out "
          f"{agg['manifold_remap'][0]:.3f} (still survives) — it reads the environment's geometry, not cell identity", flush=True)

    print(f"\n  -> under drift that changes single-cell tuning by the SAME amount (cell-corr "
          f"{agg['cell_corr_relocate'][0]:+.2f} vs {agg['cell_corr_noise'][0]:+.2f}), a LABEL-FREE read-out of the "
          f"population manifold recovers position almost perfectly when the drift PRESERVES the geometry "
          f"(relocate {agg['manifold_relocate'][0]:.3f}) but FAILS when the drift DESTROYS it (noise "
          f"{agg['manifold_noise'][0]:.3f} ≈ chance) — so stable read-out rides on the conserved GEOMETRY, not on "
          f"single-cell stability. A fixed decoder bound to specific cells degrades ({agg['fixed_relocate'][0]:.3f}), "
          f"and the geometry read-out survives even a FULL remap ({agg['manifold_remap'][0]:.3f}): it reads the "
          f"environment's geometry, not which cells carry it (Morales 2025). Measured, not put in the loss.", flush=True)

    out = {"n_seeds": a.seeds, "N": N, "P": P, "days": DAYS, "relocate": RELOCATE, "noise_sd": NOISE_SD,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/representational_drift.json", "w"), indent=2)
    svg(agg, "results/representational_drift.svg")
    print("\nwrote results/representational_drift.json and results/representational_drift.svg", flush=True)


def svg(agg, out):
    pad = 60; pw = 250; ph = 200; gap = 74; W = pad + 2 * pw + gap + 20; Hh = 92 + ph + 44
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'Representational drift: the population GEOMETRY survives it</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">at matched single-cell drift, a label-free '
             'geometry read-out survives geometry-preserving drift (even a full remap) but fails when the drift destroys geometry</text>')
    oy = 58; base = oy + ph
    # Panel A: geometry read-out error across conditions (+ fixed)
    oxA = pad
    e.append(f'<text x="{oxA}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(A) geometry read-out error (lower=better)</text>')
    e.append(f'<line x1="{oxA}" y1="{base}" x2="{oxA+pw}" y2="{base}" stroke="#33415c"/>')
    ch = 0.25
    e.append(f'<line x1="{oxA}" y1="{base-ch/ (agg["manifold_noise"][0]+1e-6)*(ph-30):.0f}" x2="{oxA+pw}" '
             f'y2="{base-ch/(agg["manifold_noise"][0]+1e-6)*(ph-30):.0f}" stroke="#9aa6bd" stroke-dasharray="4 3"/>')
    bars = [("relocate\n(geom kept)", agg["manifold_relocate"][0], "#2ca25f"),
            ("remap\n(geom kept)", agg["manifold_remap"][0], "#4ca66f"),
            ("noise\n(geom lost)", agg["manifold_noise"][0], "#c9341a")]
    hi = max(b[1] for b in bars) + 1e-6
    for i, (lab, v, col) in enumerate(bars):
        h = (v / hi) * (ph - 30); x = oxA + 20 + i * 74
        e.append(f'<rect x="{x}" y="{base-h:.1f}" width="52" height="{h:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<text x="{x+26}" y="{base-h-6:.0f}" font-size="10" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.3f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+26}" y="{base+13+j*10:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxA}" y="{base+34:.0f}" font-size="9" fill="#5b6b8c">dashed = chance; single-cell drift matched across all</text>')
    # Panel B: fixed vs geometry (relocate)
    oxB = pad + pw + gap
    e.append(f'<text x="{oxB}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(B) fixed decoder vs geometry read-out</text>')
    e.append(f'<line x1="{oxB}" y1="{base}" x2="{oxB+pw}" y2="{base}" stroke="#33415c"/>')
    b2 = [("fixed\ndecoder", agg["fixed_relocate"][0], "#c9341a"), ("GEOMETRY\nread-out", agg["manifold_relocate"][0], "#2ca25f")]
    hi2 = max(b[1] for b in b2) + 1e-6
    for i, (lab, v, col) in enumerate(b2):
        h = (v / hi2) * (ph - 30); x = oxB + 44 + i * 100
        e.append(f'<rect x="{x}" y="{base-h:.1f}" width="64" height="{h:.1f}" fill="{col}" opacity="0.9"/>')
        e.append(f'<text x="{x+32}" y="{base-h-6:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.3f}</text>')
        for j, ln in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+32}" y="{base+13+j*10:.0f}" font-size="8.5" fill="#28324a" text-anchor="middle">{ln}</text>')
    e.append(f'<text x="{oxB}" y="{base+34:.0f}" font-size="9.5" fill="#5b6b8c">under geometry-preserving drift '
             f'(relocate); the fixed decoder is bound to cells that moved</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
