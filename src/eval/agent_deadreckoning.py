"""
src/eval/agent_deadreckoning.py

THE DEAD-RECKONING BRAIN — one closed self-localization stack, HD -> grid -> place, driven only by
self-motion. The culmination of the spatial organs: instead of being given its heading (as
agent_grid_cortex was), the agent now estimates BOTH heading and position from its own motor commands:

    motor command (turn, step) -> HEAD-DIRECTION ring attractor (heading, drifts)
        -> GRID cortex path-integrates POSITION using that heading (drifts more)
        -> PLACE read-out (position estimate) -> behaviour (homing).

This makes the drift biologically correct: it originates as HEADING error in the HD organ and propagates
into position error through the grid integrator (the path integrator accumulates each actual displacement
ROTATED by the heading error theta_est - theta_true). Two allothetic corrections fix two organs: a VISUAL
landmark resets the HD ring bump; BOUNDARY input resets the grid phase.

We measure:
  (1) LOCALIZATION error of the full stack vs condition: oracle (true heading) is the floor; with the HD
      organ in the loop heading drift inflates position error; visual reset alone does NOT rescue position
      (the grid integrator's accumulated error persists) -- only correcting BOTH organs (visual + boundary)
      bounds it; lesioning HD or grid is catastrophic. So the unified stack needs BOTH corrections, each for
      a different organ.
  (2) HOMING (path integration return; desert-ant homing, Wehner): the agent wanders out, then returns to
      the origin using ONLY its integrated position estimate. Intact (both corrections) homes accurately;
      lesioning HD or grid abolishes homing.

Multi-seed, mean +/- 95% CI. Writes results/agent_deadreckoning.json + .svg.

    python -m src.eval.agent_deadreckoning --seeds 3
"""
import argparse
import json
import math
import os

import torch

from src.eval.head_direction import train_hd, canonical, nearest
from src.eval.agent_grid_cortex import build_cortex, train_decoder, R

S = 0.2                       # step length
A_NOISE = 0.05               # angular (vestibular) self-motion noise -> HD drift
RESET_PERIOD = 12            # visual landmark encounter period
BSCALE = 0.35                # boundary-proximity gate width
LOC_STEPS = 120
N_WALKS = 24
OUT_STEPS = 60; HOME_STEPS = 90; MAXTURN = 0.5

# condition -> (heading source, visual reset, boundary reset, grid lesion)
LOC_CONDS = {
    "oracle":      ("true", False, False, False),
    "none":        ("hd",   False, False, False),
    "visual":      ("hd",   True,  False, False),
    "boundary":    ("hd",   False, True,  False),
    "both":        ("hd",   True,  True,  False),
    "lesion_hd":   ("frozen", False, False, False),
    "lesion_grid": ("hd",   True,  True,  True),
}
HOME_CONDS = {"both": ("hd", True, True, False), "none": ("hd", False, False, False),
              "lesion_hd": ("frozen", False, False, False), "lesion_grid": ("hd", True, True, True)}


class Stack:
    """The unified HD->grid->place self-localization stack for one agent."""

    def __init__(self, organs, gen):
        self.hd, self.keys, self.vals, self.mod, self.dec = organs
        self.gen = gen
        self.gains = self.mod.gains
        self.th_true = 0.0
        self.true = torch.zeros(2)
        self.h = nearest(self.keys, self.vals, 0.0)
        self.phi = self.gains.view(self.mod.K, 1, 1) * self.true.view(1, 1, 2).clone()

    def step(self, om, cond, t):
        head_src, visual, boundary, lesion_grid = cond
        self.th_true += om
        new_true = (self.true + S * torch.tensor([math.cos(self.th_true), math.sin(self.th_true)])).clamp(-R, R)
        actual = new_true - self.true; self.true = new_true
        # heading estimate from the HD organ (or oracle / lesion)
        if head_src == "true":
            th_est = self.th_true
        elif head_src == "frozen":
            th_est = 0.0
        else:
            self.h = self.hd.step(self.h, torch.tensor(om) + torch.randn(1, generator=self.gen) * A_NOISE, gen=self.gen)
            th_est = self.hd.decode(self.h)
        # path integration: actual displacement ROTATED by the heading error (this is how HD drift corrupts PI)
        rot = th_est - self.th_true; c, sn = math.cos(rot), math.sin(rot)
        disp = torch.tensor([c * actual[0] - sn * actual[1], sn * actual[0] + c * actual[1]])
        self.phi = self.phi + self.gains.view(self.mod.K, 1, 1) * disp.view(1, 1, 2)
        if visual and t % RESET_PERIOD == 0:                          # visual landmark -> HD reset
            self.h = nearest(self.keys, self.vals, self.th_true)
        if boundary:                                                  # boundary input -> grid reset
            for ax in (0, 1):
                w = math.exp(-(R - abs(self.true[ax].item())) / BSCALE)
                self.phi[:, 0, ax] = (1 - w) * self.phi[:, 0, ax] + w * (self.gains * self.true[ax])
        if lesion_grid:
            pos_est = self.dec(torch.zeros(1, self.mod.K * self.mod.M))[0]   # grid code destroyed
        else:
            pos_est = self.dec(self.mod._grid_code(self.phi))[0]
        return pos_est, th_est


def wander_om(t, gen):
    return 0.3 * math.sin(t * 0.3) + torch.randn(1, generator=gen).item() * 0.15


def localization_walk(organs, cond, gen):
    st = Stack(organs, gen); errs = []
    for t in range(LOC_STEPS):
        pos_est, _ = st.step(wander_om(t, gen), cond, t)
        errs.append((pos_est - st.true).norm().item())
    return sum(errs[-30:]) / 30


def homing(organs, cond, gen, record=False):
    st = Stack(organs, gen); traj = [st.true.clone()]
    pos_est = th_est = None
    for t in range(OUT_STEPS):                                        # outbound wander
        pos_est, th_est = st.step(wander_om(t, gen), cond, t); traj.append(st.true.clone())
    for k in range(HOME_STEPS):                                       # home using the position estimate
        home_dir = math.atan2(-pos_est[1].item(), -pos_est[0].item())
        dh = math.atan2(math.sin(home_dir - th_est), math.cos(home_dir - th_est))
        om = max(-MAXTURN, min(MAXTURN, dh))
        pos_est, th_est = st.step(om, cond, OUT_STEPS + k)
        traj.append(st.true.clone())
        if pos_est.norm().item() < 0.3:                              # agent believes it is home
            break
    err = st.true.norm().item()
    return (err, [p.tolist() for p in traj]) if record else err


def run_seed(seed):
    g = torch.Generator().manual_seed(seed + 31)
    hd, _ = train_hd(seed, iters=2000)
    keys, vals = canonical(hd)
    mod = build_cortex(seed); dec = train_decoder(mod, g, nonlinear=True, iters=1200)
    organs = (hd, keys, vals, mod, dec)
    loc = {c: sum(localization_walk(organs, LOC_CONDS[c], g) for _ in range(N_WALKS)) / N_WALKS for c in LOC_CONDS}
    home = {c: sum(homing(organs, HOME_CONDS[c], g) for _ in range(N_WALKS)) / N_WALKS for c in HOME_CONDS}
    trajs = {c: homing(organs, HOME_CONDS[c], g, record=True)[1] for c in ("both", "lesion_hd")}
    return {"loc": loc, "home": home, "trajs": trajs}


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 3), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 3) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    loc = {c: ci([p["loc"][c] for p in per]) for c in LOC_CONDS}
    home = {c: ci([p["home"][c] for p in per]) for c in HOME_CONDS}

    print(f"\nTHE DEAD-RECKONING BRAIN — unified HD->grid->place stack (n={a.seeds}; mean ± 95% CI)\n" + "=" * 78, flush=True)
    lab = {"oracle": "oracle heading (floor)", "none": "HD in loop, no correction", "visual": "+ visual reset (HD only)",
           "boundary": "+ boundary reset (grid only)", "both": "+ BOTH corrections", "lesion_hd": "lesion HD organ",
           "lesion_grid": "lesion grid organ"}
    print("(1) LOCALIZATION error of the full stack (lower = better):", flush=True)
    for c in LOC_CONDS:
        print(f"    {lab[c]:30} {loc[c][0]:.3f} ± {loc[c][1]:.3f}", flush=True)
    print("\n(2) HOMING error (path-integration return to origin):", flush=True)
    hlab = {"both": "intact (both corrections)", "none": "no correction", "lesion_hd": "lesion HD", "lesion_grid": "lesion grid"}
    for c in HOME_CONDS:
        print(f"    {hlab[c]:28} {home[c][0]:.3f} ± {home[c][1]:.3f}", flush=True)
    print(f"\n  -> the agent estimates BOTH heading (HD ring attractor) and position (grid) from self-motion "
          f"alone. Oracle floor {loc['oracle'][0]:.2f}; with the HD organ in the loop heading drift inflates "
          f"position error ({loc['none'][0]:.2f}), and correcting heading ALONE (visual {loc['visual'][0]:.2f}) "
          f"does not rescue it -- only correcting BOTH organs (visual+boundary {loc['both'][0]:.2f}) bounds "
          f"position; lesioning HD ({loc['lesion_hd'][0]:.2f}) or grid ({loc['lesion_grid'][0]:.2f}) is "
          f"catastrophic. (2) Homing works intact ({home['both'][0]:.2f}) and is abolished by lesioning HD "
          f"({home['lesion_hd'][0]:.2f}) or grid ({home['lesion_grid'][0]:.2f}) -- the dead-reckoning brain.", flush=True)

    out = {"n_seeds": a.seeds, "localization": {c: loc[c] for c in LOC_CONDS}, "homing": {c: home[c] for c in HOME_CONDS}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/agent_deadreckoning.json", "w"), indent=2)
    svg(loc, home, per[0]["trajs"], "results/agent_deadreckoning.svg")
    print("\nwrote results/agent_deadreckoning.json and results/agent_deadreckoning.svg", flush=True)


def svg(loc, home, trajs, out):
    pad = 58; pw = 330; ph = 210; gap = 96; W = pad + pw + gap + 300 + 20; Hh = 84 + ph + 56
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="15" font-weight="800" fill="#0b1324">'
             'The dead-reckoning brain: one HD&#8594;grid&#8594;place stack from self-motion alone</text>')
    e.append('<text x="26" y="42" font-size="10.5" fill="#5b6b8c">heading drift (HD) propagates to position '
             'drift (grid); BOTH allothetic corrections (visual&#8594;HD, boundary&#8594;grid) are needed</text>')
    oy = 60
    # Panel A: localization error bars
    order = ["oracle", "none", "visual", "boundary", "both", "lesion_hd", "lesion_grid"]
    col = {"oracle": "#2ca25f", "none": "#e6a000", "visual": "#e6a000", "boundary": "#e6a000",
           "both": "#2ca25f", "lesion_hd": "#c9341a", "lesion_grid": "#c9341a"}
    short = {"oracle": "oracle", "none": "no corr", "visual": "+vis", "boundary": "+bnd", "both": "+both",
             "lesion_hd": "−HD", "lesion_grid": "−grid"}
    hi = max(loc[c][0] for c in order) * 1.12
    bw = pw / len(order) - 12; base = oy + ph
    e.append(f'<text x="{pad}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(1) localization error of the stack</text>')
    e.append(f'<line x1="{pad-4}" y1="{base}" x2="{pad+pw}" y2="{base}" stroke="#33415c"/>')
    for i, c in enumerate(order):
        x = pad + i * (pw / len(order)); h = loc[c][0] / hi * ph
        e.append(f'<rect x="{x:.1f}" y="{base-h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{col[c]}" opacity="0.88"/>')
        e.append(f'<text x="{x+bw/2:.1f}" y="{base-h-3:.0f}" font-size="8.5" font-weight="700" fill="#0b1324" text-anchor="middle">{loc[c][0]:.2f}</text>')
        e.append(f'<text x="{x+bw/2:.1f}" y="{base+13:.0f}" font-size="8" fill="#28324a" text-anchor="middle">{short[c]}</text>')
    # Panel B: homing trajectories (intact vs lesion HD)
    oxB = pad + pw + gap; sz = 230; cx = oxB + sz / 2; cyB = oy + sz / 2
    sc = (sz / 2 - 12) / R
    e.append(f'<text x="{oxB}" y="{oy-4}" font-size="11.5" font-weight="700" fill="#0b1324">(2) homing (path-integration return)</text>')
    e.append(f'<rect x="{oxB}" y="{oy}" width="{sz}" height="{sz}" fill="none" stroke="#c7d0e0"/>')
    e.append(f'<circle cx="{cx:.0f}" cy="{cyB:.0f}" r="4" fill="#0b1324"/>'
             f'<text x="{cx+7:.0f}" y="{cyB-5:.0f}" font-size="8.5" fill="#0b1324">home</text>')
    for c, col2 in (("both", "#2ca25f"), ("lesion_hd", "#c9341a")):
        tr = trajs[c]
        pts = " ".join(f"{cx + p[0]*sc:.1f},{cyB - p[1]*sc:.1f}" for p in tr)
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col2}" stroke-width="1.8" opacity="0.85"/>')
        last = tr[-1]
        e.append(f'<circle cx="{cx + last[0]*sc:.1f}" cy="{cyB - last[1]*sc:.1f}" r="3.5" fill="{col2}"/>')
    e.append(f'<rect x="{oxB}" y="{oy+sz+8}" width="12" height="4" fill="#2ca25f"/>'
             f'<text x="{oxB+16}" y="{oy+sz+13}" font-size="9" fill="#28324a">intact &#8594; returns home (err {home["both"][0]:.2f})</text>')
    e.append(f'<rect x="{oxB}" y="{oy+sz+24}" width="12" height="4" fill="#c9341a"/>'
             f'<text x="{oxB+16}" y="{oy+sz+29}" font-size="9" fill="#28324a">lesion HD &#8594; lost (err {home["lesion_hd"][0]:.2f})</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
