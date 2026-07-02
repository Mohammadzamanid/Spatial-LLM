"""
src/eval/btsp.py

BEHAVIORAL-TIMESCALE PLASTICITY (BTSP) — the biological one-shot learning rule, and its PREDICTIVE place field.

The model's one-shot memory (agent_memory.py) writes a place code into an episodic store — a functional
abstraction. The hippocampus does it differently and more interestingly: a single dendritic PLATEAU potential
imprints a complete place field in ONE traversal, through a SECONDS-WIDE, TEMPORALLY ASYMMETRIC plasticity
kernel (Bittner, Milstein, Lu, Turi & Magee, Science 2017; Grienberger & Magee 2022). Because the animal
occupied UPSTREAM positions in the seconds before the plateau, those inputs are potentiated most, so the new
field peaks UPSTREAM of the induction site — an anticipatory / PREDICTIVE field. None of that is put in: we set
only the kernel (the biology) and MEASURE the field, its shift, and its speed-dependence.

We drive a bank of position-tuned input cells along a linear track at running speed v, fire ONE plateau at the
track centre, apply the `BTSPPlasticity` organ once, and read out the resulting place field. We measure:

  (A) ONE-SHOT FIELD needs a SECONDS-scale kernel. BTSP (seconds) and a symmetric-seconds control both imprint
      a strong, broad field in one pass; a millisecond STDP-scale kernel imprints almost nothing (a sliver).
  (B) The PREDICTIVE SHIFT needs the ASYMMETRY. Only the asymmetric BTSP kernel shifts the field UPSTREAM of
      the plateau (shift < 0 — the cell fires BEFORE the induction site on the next same-direction lap); the
      symmetric control sits on the plateau (shift ~ 0).
  (C) The shift SCALES WITH RUNNING SPEED (a temporal kernel -> a spatial shift = kernel-offset x speed), a
      specific Bittner prediction — measured across speeds.

Multi-seed, mean +/- 95% CI. Writes results/btsp.json + .svg.

    python -m src.eval.btsp --seeds 5
"""
import argparse
import json
import math
import os

import torch

from src.models.neuro import BTSPPlasticity

L = 300.0            # track length (cm-ish; long enough that the seconds-wide kernel never runs off the ends)
N_IN = 151           # position-tuned input cells
SIG = 4.0            # input-cell tuning width
DT = 0.02            # time step (s)
X_STAR = 150.0       # plateau induction site (track centre)
SPEEDS = [15.0, 25.0, 40.0]   # running speeds (cm/s); Bittner regime
XGRID = torch.linspace(0, L, 600)


def field_from_weights(w, pref):
    """Place field f(x) = sum_i w_i * tuning(x - pref_i), evaluated on XGRID -> (400,)."""
    t = torch.exp(-((XGRID.unsqueeze(1) - pref.unsqueeze(0)) ** 2) / (2 * SIG ** 2))   # (400, N)
    return t @ w


def run_once(kernel, speed, pref, gen, noise=0.03):
    """One plateau-induced traversal; returns (field strength, predictive shift, field width)."""
    T = int((L / speed) / DT)
    times = torch.arange(T, dtype=torch.float) * DT
    pos = (speed * times).clamp(max=L)
    pre = torch.exp(-((pos.unsqueeze(1) - pref.unsqueeze(0)) ** 2) / (2 * SIG ** 2))    # (T, N)
    pre = (pre + torch.randn(pre.shape, generator=gen) * noise).clamp(min=0.0)
    w = kernel.induce(pre, times, X_STAR / speed)                                       # (N,) one-shot weights
    f = field_from_weights(w, pref)
    strength = w.clamp(min=0).sum().item()                                             # total potentiation
    fp = f.clamp(min=0)
    if fp.sum() <= 1e-6:
        return strength, 0.0, 0.0
    com = (XGRID * fp).sum().item() / fp.sum().item()                                  # field centre of mass
    shift = com - X_STAR                                                               # upstream (predictive) if < 0
    width = math.sqrt(((XGRID - com) ** 2 * fp).sum().item() / fp.sum().item())
    return strength, shift, width


def run_seed(seed):
    gen = torch.Generator().manual_seed(seed)
    pref = torch.linspace(0, L, N_IN) + torch.randn(N_IN, generator=gen) * 0.4          # jittered input positions
    btsp = BTSPPlasticity(tau_pre=1.3, tau_post=0.55)                                   # asymmetric (biological)
    symm = BTSPPlasticity(tau_pre=0.9, tau_post=0.9, symmetric=True)                    # seconds, symmetric (control)
    stdp = BTSPPlasticity(tau_pre=0.02, tau_post=0.02, symmetric=True)                  # millisecond scale (STDP-like)
    v0 = SPEEDS[1]
    out = {"conditions": {}}
    for name, k in (("btsp", btsp), ("symmetric", symm), ("stdp", stdp)):
        s, sh, wd = run_once(k, v0, pref, gen)
        out["conditions"][name] = {"strength": s, "shift": sh, "width": wd}
    # normalise field strength to BTSP = 1
    base = out["conditions"]["btsp"]["strength"]
    for c in out["conditions"].values():
        c["strength_norm"] = c["strength"] / base if base > 0 else 0.0
    # (C) speed-dependence of the predictive shift (BTSP)
    out["speed_shift"] = {v: run_once(btsp, v, pref, gen)[1] for v in SPEEDS}
    return out


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 3), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 3) if n > 1 else 0.0


CONDS = ("btsp", "symmetric", "stdp")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    agg = {c: {m: ci([p["conditions"][c][m] for p in per]) for m in ("strength_norm", "shift", "width")} for c in CONDS}
    spd = {v: ci([p["speed_shift"][v] for p in per]) for v in SPEEDS}

    print(f"\nBEHAVIORAL-TIMESCALE PLASTICITY (BTSP) — one-shot predictive place field (n={a.seeds}; "
          f"mean ± 95% CI; plateau at x={X_STAR:.0f}, track {L:.0f})\n" + "=" * 84, flush=True)
    print(f"    {'kernel':>22} | {'field strength':>15} | {'predictive shift':>17} | {'field width':>12}", flush=True)
    lab = {"btsp": "BTSP (asym, seconds)", "symmetric": "symmetric (seconds)", "stdp": "STDP (millisecond)"}
    for c in CONDS:
        d = agg[c]
        print(f"    {lab[c]:>22} | {d['strength_norm'][0]:>13.2f}   | {d['shift'][0]:>+15.2f}   | {d['width'][0]:>12.2f}", flush=True)
    print(f"\n  (C) predictive shift vs running speed (BTSP):  " +
          "   ".join(f"v={v:.0f}: {spd[v][0]:+.2f}" for v in SPEEDS), flush=True)
    b = agg["btsp"]; s = agg["symmetric"]; st = agg["stdp"]
    print(f"\n  -> ONE plateau imprints a place field in ONE pass — but only a SECONDS-scale kernel does it "
          f"(BTSP strength {b['strength_norm'][0]:.2f} and symmetric {s['strength_norm'][0]:.2f} vs the "
          f"millisecond STDP kernel's {st['strength_norm'][0]:.2f}). The PREDICTIVE shift needs the ASYMMETRY: "
          f"only BTSP shifts the field UPSTREAM of the plateau ({b['shift'][0]:+.2f}, i.e. the cell fires before "
          f"the induction site) while the symmetric control sits on it ({s['shift'][0]:+.2f}). And the shift "
          f"SCALES WITH SPEED ({spd[SPEEDS[0]][0]:+.2f}→{spd[SPEEDS[-1]][0]:+.2f} as v goes {SPEEDS[0]:.0f}→"
          f"{SPEEDS[-1]:.0f}) — a temporal kernel read out as a spatial shift. All three are MEASURED from the "
          f"kernel, not trained: the biological one-shot rule, with its predictive signature, not an episodic store.", flush=True)

    out = {"n_seeds": a.seeds, "plateau_x": X_STAR, "track": L, "speeds": SPEEDS,
           "conditions": {c: agg[c] for c in CONDS}, "speed_shift": {str(v): spd[v] for v in SPEEDS}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/btsp.json", "w"), indent=2)
    svg(agg, spd, per[0], "results/btsp.svg")
    print("\nwrote results/btsp.json and results/btsp.svg", flush=True)


def svg(agg, spd, sample, out):
    pad = 60; pw = 300; ph = 210; gap = 70; W = pad + 2 * pw + gap + 20; H = 92 + ph + 40
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'BTSP: one plateau, a one-shot PREDICTIVE place field</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">a seconds-wide asymmetric kernel shifts the '
             'field UPSTREAM of the plateau (predictive); measured, not trained</text>')
    oy = 58
    # Panel A: the induced fields (BTSP vs symmetric vs STDP) over the track
    oxA = pad
    pref = torch.linspace(0, L, N_IN)
    col = {"btsp": "#2ca25f", "symmetric": "#3182bd", "stdp": "#c9341a"}
    lab = {"btsp": "BTSP (asym)", "symmetric": "symmetric", "stdp": "STDP (ms)"}
    fields = {}
    for c in CONDS:
        k = (BTSPPlasticity(1.3, 0.55) if c == "btsp" else
             BTSPPlasticity(0.9, 0.9, symmetric=True) if c == "symmetric" else
             BTSPPlasticity(0.02, 0.02, symmetric=True))
        gen = torch.Generator().manual_seed(0)
        T = int((L / 30.0) / DT); times = torch.arange(T, dtype=torch.float) * DT; pos = (30.0 * times).clamp(max=L)
        pre = torch.exp(-((pos.unsqueeze(1) - pref.unsqueeze(0)) ** 2) / (2 * SIG ** 2))
        w = k.induce(pre, times, X_STAR / 30.0); fields[c] = field_from_weights(w, pref).clamp(min=0)
    fmax = max(f.max().item() for f in fields.values()) + 1e-9
    def XA(x): return oxA + (x / L) * pw
    def YA(v): return oy + ph - (v / fmax) * ph
    e.append(f'<line x1="{oxA}" y1="{oy+ph}" x2="{oxA+pw}" y2="{oy+ph}" stroke="#33415c"/>')
    e.append(f'<line x1="{XA(X_STAR):.0f}" y1="{oy}" x2="{XA(X_STAR):.0f}" y2="{oy+ph}" stroke="#9aa6bd" stroke-dasharray="4 3"/>')
    e.append(f'<text x="{XA(X_STAR):.0f}" y="{oy-3:.0f}" font-size="9" fill="#7787a6" text-anchor="middle">plateau x={X_STAR:.0f}</text>')
    for c in CONDS:
        f = fields[c]
        pts = " ".join(f"{XA(XGRID[i].item()):.1f},{YA(f[i].item()):.1f}" for i in range(0, len(XGRID), 2))
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col[c]}" stroke-width="2.2" opacity="0.9"/>')
    e.append(f'<text x="{oxA+pw/2:.0f}" y="{oy+ph+16:.0f}" font-size="9.5" fill="#5b6b8c" text-anchor="middle">track position &#8594; (run direction)</text>')
    ly = oy + 8
    for c in CONDS:
        e.append(f'<rect x="{oxA+pw-96}" y="{ly-8}" width="12" height="4" fill="{col[c]}"/><text x="{oxA+pw-80}" y="{ly-4}" font-size="9" fill="#28324a">{lab[c]}</text>'); ly += 14
    e.append(f'<text x="{oxA}" y="{oy+ph+30:.0f}" font-size="9" fill="#2ca25f">BTSP field peaks UPSTREAM of the plateau (predictive)</text>')
    # Panel B: shift vs speed
    oxB = pad + pw + gap
    e.append(f'<text x="{oxB}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(C) predictive shift &#8733; running speed</text>')
    bw = (pw - 40) / len(SPEEDS); base = oy + ph - 20
    smin = min(spd[v][0] for v in SPEEDS) * 1.25 - 1e-6
    e.append(f'<line x1="{oxB}" y1="{oy+8}" x2="{oxB}" y2="{base}" stroke="#33415c"/>')
    e.append(f'<line x1="{oxB}" y1="{oy+8}" x2="{oxB+len(SPEEDS)*bw}" y2="{oy+8}" stroke="#33415c" stroke-dasharray="3 3"/>')
    e.append(f'<text x="{oxB+4}" y="{oy+6}" font-size="8.5" fill="#7787a6">0 (on plateau)</text>')
    for i, v in enumerate(SPEEDS):
        sh = spd[v][0]; h = (sh / smin) * (base - oy - 20) if smin != 0 else 0; x = oxB + i * bw + 12
        e.append(f'<rect x="{x:.0f}" y="{oy+8:.0f}" width="{bw-24:.0f}" height="{abs(h):.1f}" fill="#2ca25f" opacity="0.85"/>')
        e.append(f'<text x="{x+(bw-24)/2:.0f}" y="{oy+8+abs(h)+12:.0f}" font-size="9" font-weight="700" fill="#0b1324" text-anchor="middle">{sh:+.1f}</text>')
        e.append(f'<text x="{x+(bw-24)/2:.0f}" y="{base+14:.0f}" font-size="9" fill="#28324a" text-anchor="middle">v={v:.0f}</text>')
    e.append(f'<text x="{oxB}" y="{base+30:.0f}" font-size="9" fill="#5b6b8c">upstream shift grows with speed (kernel is in TIME)</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
