"""
src/eval/head_direction.py

A HEAD-DIRECTION ORGAN — emergent HD cells + a ring attractor, and the heading-dominated path-integration
drift it causes (with its visual correction). This closes the deepest gap flagged by agent_cue_integration:
biological PI drift is dominated by HEADING (angular) error from the head-direction system, not the
translational noise our earlier drift module injected.

Same emergence methodology as grid/time cells: train a GENERIC recurrent substrate on a task, then MEASURE
brain signatures that were never in the loss.

  (1) EMERGENCE. A generic rate-RNN is trained only to track heading from angular velocity (angular path
      integration). We then measure, vs an untrained control:
        - HD CELLS: units tuned to a single preferred heading (mean resultant length > 0.4).
        - A FUNCTIONAL RING ATTRACTOR: the trained net maintains and updates a single heading bump and reads
          heading out accurately and stably (population decode error in degrees); its activity lies on a 1-D
          ring manifold (visualised in the figure). Honest nuance: a ring-SHAPED manifold appears even in
          the UNTRAINED recurrent net (it is partly inherent to recurrent integration), so the
          training-specific emergence is the HD tuning and the accurate, stable maintenance — the attractor
          FUNCTION — not the manifold shape per se (we report the PC-angle~heading correlation for both and
          do not claim it as a trained>untrained signature).
      HD cells and accurate heading maintenance are NOT built in (the recurrent weights are learned,
      generic) — they emerge, as in Drosophila/rodent compass circuits (Zhang 1996; Kim et al. 2017; Taube 1990).

  (2) HEADING-DOMINATED DRIFT + VISUAL RESET (Knierim, Kudrimoti & McNaughton 1995). The emergent HD net
      integrates NOISY angular velocity, so heading drifts; the agent path-integrates POSITION using that
      heading, so heading error (not translational noise) drives position drift. A visual landmark pins the
      ring bump to the true heading -> bounded heading -> bounded position. The biologically-correct drift
      source and its allothetic correction.

Multi-seed, mean +/- 95% CI. Writes results/head_direction.json + .svg.

    python -m src.eval.head_direction --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

HID = 64; T = 50; ALPHA = 0.2
NB = 36                       # heading bins for tuning curves
WALK_STEPS = 140; N_WALKS = 30; A_NOISE = 0.06; RESET_PERIOD = 12


class HDNet(nn.Module):
    """Generic leaky rate-RNN: integrates angular velocity -> heading (cos, sin). Nothing HD-specific."""

    def __init__(self):
        super().__init__()
        self.U = nn.Linear(1, HID); self.W = nn.Linear(HID, HID); self.V = nn.Linear(HID, 2)

    def run(self, omega, noise=0.0, gen=None):
        B = omega.shape[0]; h = torch.zeros(B, HID); rs = []
        for t in range(T):
            n = torch.randn(B, HID, generator=gen) * noise if noise > 0 else 0.0
            h = (1 - ALPHA) * h + ALPHA * (self.W(torch.relu(h)) + self.U(omega[:, t:t+1]) + n)
            rs.append(torch.relu(h))
        R = torch.stack(rs, 1)
        return self.V(R), R

    def step(self, h, omega, noise=0.0, gen=None):
        n = torch.randn(1, HID, generator=gen) * noise if noise > 0 else 0.0
        return (1 - ALPHA) * h + ALPHA * (self.W(torch.relu(h)) + self.U(omega.view(1, 1)) + n)

    def decode(self, h):
        o = self.V(torch.relu(h)); return math.atan2(o[0, 1].item(), o[0, 0].item())


def train_hd(seed, iters=2000):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    net = HDNet(); opt = torch.optim.Adam(net.parameters(), 3e-3)
    for _ in range(iters):
        omega = torch.randn(64, T, generator=g) * 0.35
        theta = torch.cumsum(omega, 1); tgt = torch.stack([theta.cos(), theta.sin()], -1)
        pred, R = net.run(omega, noise=0.02, gen=g)
        loss = ((pred - tgt) ** 2).mean() + 1e-3 * R.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net, g


def circ_corr(a, b):
    a0 = a - torch.atan2(a.sin().mean(), a.cos().mean()); b0 = b - torch.atan2(b.sin().mean(), b.cos().mean())
    num = (a0.sin() * b0.sin()).sum()
    den = torch.sqrt((a0.sin() ** 2).sum() * (b0.sin() ** 2).sum()) + 1e-9
    return (num / den).abs().item()


def emergence_metrics(net, gen):
    omega = torch.randn(200, T, generator=gen) * 0.35
    theta = torch.cumsum(omega, 1)
    with torch.no_grad():
        pred, R = net.run(omega, noise=0.0)
    err = torch.atan2(pred[..., 1], pred[..., 0]) - theta
    decode_err = torch.atan2(err.sin(), err.cos()).abs().mean().item() * 180 / math.pi
    th = torch.atan2(theta.sin(), theta.cos()).reshape(-1)
    rr = R.reshape(-1, HID)
    bins = ((th + math.pi) / (2 * math.pi) * NB).long().clamp(0, NB - 1)
    tc = torch.stack([rr[bins == b].mean(0) if (bins == b).any() else torch.zeros(HID) for b in range(NB)])  # (NB,HID)
    ang = torch.linspace(-math.pi, math.pi, NB)
    dirs = torch.stack([ang.cos(), ang.sin()], -1)
    vec = torch.einsum('bh,bd->hd', tc, dirs)
    strength = vec.norm(dim=1) / (tc.sum(0) + 1e-6)
    hd_frac = (strength > 0.4).float().mean().item()
    X = tc - tc.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(X, full_matrices=False)
    pc = X @ Vt[:2].t()
    ring_corr = circ_corr(torch.atan2(pc[:, 1], pc[:, 0]), ang)
    return {"decode_err": decode_err, "hd_frac": hd_frac, "ring_corr": ring_corr,
            "pc": pc.tolist(), "ang": ang.tolist()}


def canonical(net):
    h = torch.zeros(1, HID); keys = []; vals = []
    om = torch.tensor(2 * math.pi / 120)
    for _ in range(480):
        h = net.step(h, om); keys.append(net.decode(h)); vals.append(h.clone())
    return torch.tensor(keys), vals


def nearest(keys, vals, th):
    d = torch.atan2((keys - th).sin(), (keys - th).cos()).abs()
    return vals[int(d.argmin())].clone()


def drift_walk(net, keys, vals, gen, do_reset):
    h = nearest(keys, vals, 0.0); th_true = 0.0
    est = torch.zeros(2); true = torch.zeros(2); herr = []; perr = []
    for t in range(WALK_STEPS):
        om = 0.3 * math.sin(t * 0.3) + torch.randn(1, generator=gen).item() * 0.15
        th_true += om
        h = net.step(h, torch.tensor(om) + torch.randn(1, generator=gen) * A_NOISE, gen=gen)
        th_est = net.decode(h)
        true = true + 0.2 * torch.tensor([math.cos(th_true), math.sin(th_true)])
        est = est + 0.2 * torch.tensor([math.cos(th_est), math.sin(th_est)])
        if do_reset and t % RESET_PERIOD == 0:
            h = nearest(keys, vals, th_true)                      # visual landmark pins the bump
        he = abs(math.atan2(math.sin(th_est - th_true), math.cos(th_est - th_true))) * 180 / math.pi
        herr.append(he); perr.append((est - true).norm().item())
    return herr, perr


def run_seed(seed):
    net, g = train_hd(seed)
    em = emergence_metrics(net, g)
    un = HDNet(); em_un = emergence_metrics(un, torch.Generator().manual_seed(seed + 1))
    keys, vals = canonical(net)
    drift = {}
    traces = {}
    for tag, do_reset in (("no_reset", False), ("reset", True)):
        hs, ps = [], []
        htr = [0.0] * WALK_STEPS
        for _ in range(N_WALKS):
            he, pe = drift_walk(net, keys, vals, g, do_reset)
            hs.append(sum(he[-30:]) / 30); ps.append(sum(pe[-30:]) / 30)
            for t in range(WALK_STEPS):
                htr[t] += he[t] / N_WALKS
        drift[tag] = {"heading_err": sum(hs) / len(hs), "pos_err": sum(ps) / len(ps)}
        traces[tag] = htr
    return {"em": em, "em_un": em_un, "drift": drift, "traces": traces}


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 3), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 3) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    em = {k: ci([p["em"][k] for p in per]) for k in ("decode_err", "hd_frac", "ring_corr")}
    em_un = {k: ci([p["em_un"][k] for p in per]) for k in ("decode_err", "hd_frac", "ring_corr")}
    drift = {tag: {m: ci([p["drift"][tag][m] for p in per]) for m in ("heading_err", "pos_err")}
             for tag in ("no_reset", "reset")}
    traces = {tag: [sum(p["traces"][tag][t] for p in per) / a.seeds for t in range(WALK_STEPS)]
              for tag in ("no_reset", "reset")}

    print(f"\nHEAD-DIRECTION ORGAN — emergent ring attractor + heading-dominated drift (n={a.seeds}; mean ± 95% CI)\n" + "=" * 84, flush=True)
    print("(1) EMERGENCE from angular path integration (trained vs untrained control):", flush=True)
    print(f"    {'metric':>26} | {'TRAINED':>16} | {'untrained':>16}", flush=True)
    print(f"    {'heading decode err (deg)':>26} | {em['decode_err'][0]:>14.1f}   | {em_un['decode_err'][0]:>14.1f}", flush=True)
    print(f"    {'HD-tuned units':>26} | {em['hd_frac'][0]:>14.0%}   | {em_un['hd_frac'][0]:>14.0%}", flush=True)
    print(f"    {'ring corr (PC-angle~heading)':>26} | {em['ring_corr'][0]:>14.2f}   | {em_un['ring_corr'][0]:>14.2f}"
          f"   (NOT a clean discriminator: a ring manifold is inherent to recurrent integration)", flush=True)
    print("\n(2) HEADING-DOMINATED DRIFT + VISUAL RESET (Knierim 1995):", flush=True)
    print(f"    {'condition':>18} | {'heading err (deg)':>18} | {'position err':>14}", flush=True)
    for tag, lbl in (("no_reset", "no reset (drift)"), ("reset", "VISUAL reset")):
        d = drift[tag]
        print(f"    {lbl:>18} | {d['heading_err'][0]:>13.1f}±{d['heading_err'][1]:<3.1f} | {d['pos_err'][0]:>9.3f}±{d['pos_err'][1]:<.3f}", flush=True)
    print(f"\n  -> (1) HD cells ({em['hd_frac'][0]:.0%} vs {em_un['hd_frac'][0]:.0%}) and a FUNCTIONAL ring "
          f"attractor -- accurate, stable heading maintenance (decoded to {em['decode_err'][0]:.0f}° vs "
          f"{em_un['decode_err'][0]:.0f}°; the untrained net does not hold heading) -- EMERGE from angular path "
          f"integration; nothing HD-specific is built in (the ring-shaped manifold itself appears even "
          f"untrained, so the emergent signatures are HD tuning + accurate maintenance, not the manifold shape). "
          f"(2) the emergent HD system's noisy integration makes heading "
          f"DRIFT ({drift['no_reset']['heading_err'][0]:.0f}°), which drives POSITION drift "
          f"({drift['no_reset']['pos_err'][0]:.1f}); a VISUAL landmark pinning the ring bump bounds both "
          f"(heading {drift['reset']['heading_err'][0]:.0f}°, position {drift['reset']['pos_err'][0]:.1f}) -- the "
          f"biologically-correct, heading-dominated drift and its allothetic correction.", flush=True)

    out = {"n_seeds": a.seeds, "emergence": {"trained": em, "untrained": em_un}, "drift": drift}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/head_direction.json", "w"), indent=2)
    svg(per[0]["em"], em, em_un, traces, drift, "results/head_direction.svg")
    print("\nwrote results/head_direction.json and results/head_direction.svg", flush=True)


def svg(em0, em, em_un, traces, drift, out):
    pad = 56; pw = 300; ph = 220; gap = 100; W = pad + 2 * pw + gap + 20; H = 84 + ph + 44
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'A head-direction organ: a ring attractor emerges; heading drift drives position drift, reset by vision</text>')
    e.append(f'<text x="26" y="42" font-size="10.5" fill="#5b6b8c">trained only to track heading from angular '
             f'velocity &#183; HD cells {em["hd_frac"][0]:.0%} (vs {em_un["hd_frac"][0]:.0%}), heading decoded to '
             f'{em["decode_err"][0]:.0f}&#176; (vs {em_un["decode_err"][0]:.0f}&#176;)</text>')
    oy = 60
    # Panel A: the emergent ring (top-2 PCs of population state, ordered by heading -> a loop)
    oxA = pad + pw / 2 + 10; cyA = oy + ph / 2 + 6
    pc = torch.tensor(em0["pc"]); angs = torch.tensor(em0["ang"])
    rad = pc.norm(dim=1).mean().item() + 1e-6; scale = (ph / 2 - 18) / rad
    e.append(f'<text x="{pad}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(1) emergent ring attractor (population PCA)</text>')
    npc = pc.shape[0]                                          # heading bins are already in ascending order
    pts = " ".join(f"{oxA + pc[i,0].item()*scale:.1f},{cyA - pc[i,1].item()*scale:.1f}" for i in range(npc))
    e.append(f'<polyline points="{pts} {oxA + pc[0,0].item()*scale:.1f},{cyA - pc[0,1].item()*scale:.1f}" '
             f'fill="none" stroke="#3182bd" stroke-width="2"/>')
    for i in range(pc.shape[0]):
        hue = int((angs[i].item() + math.pi) / (2 * math.pi) * 330)
        e.append(f'<circle cx="{oxA + pc[i,0].item()*scale:.1f}" cy="{cyA - pc[i,1].item()*scale:.1f}" r="3.4" '
                 f'fill="hsl({hue},70%,50%)"/>')
    e.append(f'<text x="{oxA:.0f}" y="{cyA:.0f}" font-size="8.5" fill="#7787a6" text-anchor="middle">colour = heading</text>')
    e.append(f'<text x="{pad}" y="{oy+ph+14:.0f}" font-size="9" fill="#5b6b8c">the trained net&#8217;s heading bump '
             f'traces a 1-D ring manifold (HD cells {em["hd_frac"][0]:.0%} vs {em_un["hd_frac"][0]:.0%} untrained)</text>')
    # Panel B: heading error vs time (drift vs visual reset)
    oxB = pad + pw + gap
    tmax = max(max(traces["no_reset"]), max(traces["reset"])) * 1.12 + 1e-6
    def XB(t): return oxB + (t / (WALK_STEPS - 1)) * pw
    def YB(v): return oy + ph - (v / tmax) * ph
    e.append(f'<text x="{oxB}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(2) heading drift &amp; visual reset</text>')
    e.append(f'<line x1="{oxB}" y1="{oy+ph}" x2="{oxB+pw}" y2="{oy+ph}" stroke="#33415c"/>'
             f'<line x1="{oxB}" y1="{oy}" x2="{oxB}" y2="{oy+ph}" stroke="#33415c"/>')
    for vv in (0, 30, 60):
        if vv <= tmax:
            e.append(f'<text x="{oxB-6}" y="{YB(vv)+3:.0f}" font-size="8" fill="#5b6b8c" text-anchor="end">{vv}&#176;</text>')
    for tag, c in (("no_reset", "#c9341a"), ("reset", "#2ca25f")):
        pp = " ".join(f"{XB(t):.1f},{YB(traces[tag][t]):.1f}" for t in range(WALK_STEPS))
        e.append(f'<polyline points="{pp}" fill="none" stroke="{c}" stroke-width="2.2"/>')
    e.append(f'<text x="{oxB+pw/2:.0f}" y="{oy+ph+16:.0f}" font-size="9.5" fill="#5b6b8c" text-anchor="middle">step &#8594;</text>')
    e.append(f'<rect x="{oxB+pw-150}" y="{oy+6}" width="13" height="4" fill="#c9341a"/>'
             f'<text x="{oxB+pw-133}" y="{oy+11}" font-size="9" fill="#28324a">no reset (drift) &#183; pos err {drift["no_reset"]["pos_err"][0]:.1f}</text>')
    e.append(f'<rect x="{oxB+pw-150}" y="{oy+22}" width="13" height="4" fill="#2ca25f"/>'
             f'<text x="{oxB+pw-133}" y="{oy+27}" font-size="9" fill="#28324a">visual reset &#183; pos err {drift["reset"]["pos_err"][0]:.1f}</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
