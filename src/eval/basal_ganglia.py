"""
src/eval/basal_ganglia.py

A BASAL-GANGLIA ACTION-SELECTION ORGAN (new Tier-2 system) — and the dopamine-dependence of
reward-based action learning. A faithful cortico-striatal circuit: state (cortex/place code) drives
opponent striatal pathways — Go (D1) and NoGo (D2) — and the action is selected by softmax(Go - NoGo)
(direct-minus-indirect, thalamic disinhibition). Learning is LOCAL and DOPAMINE-GATED (three-factor;
Frank's OpAL): a dopamine reward-prediction error gates plasticity — positive RPE -> Go LTP, negative RPE
-> NoGo LTP — with synaptic homeostasis (decay) for stability. No backprop.

Lesion analysis on a navigation task:
  - intact      : learns goal-directed action selection.
  - -dopamine   : RPE no longer gates striatal plasticity -> cannot learn from reward (Parkinsonian).
  - -D1 (Go)    : no Go LTP -> impaired learning to APPROACH reward.
  - -D2 (NoGo)  : no NoGo LTP -> learning to suppress is lost (Go still works).

Multi-seed, mean +/- 95% CI. Writes results/basal_ganglia.json + .svg.

    python -m src.eval.basal_ganglia --seeds 3
"""
import argparse
import json
import math
import os

import torch

BOX = 1.0; STEP = 0.2; MAXT = 40; GAMMA = 0.95; GOAL = torch.tensor([0.5, 0.5]); RAD = 0.25
DIRS = torch.tensor([[math.cos(a), math.sin(a)] for a in [i * math.pi / 4 for i in range(8)]])
_cx, _cy = torch.meshgrid(torch.linspace(-BOX, BOX, 10), torch.linspace(-BOX, BOX, 10), indexing="ij")
CENTERS = torch.stack([_cx.reshape(-1), _cy.reshape(-1)], -1); M = 100; SIG = 0.22


def place(p):
    return torch.exp(-((p.unsqueeze(0) - CENTERS) ** 2).sum(-1) / (2 * SIG ** 2))


def run_seed(seed, lesion="none", episodes=4000, lr=0.02, lrv=0.05, decay=0.04):
    g = torch.Generator().manual_seed(seed)
    Wgo = torch.zeros(8, M); Wno = torch.zeros(8, M); wv = torch.zeros(M)
    succ = []
    for ep in range(episodes):
        p = (torch.rand(2, generator=g) * 2 - 1) * BOX; done = False
        for _ in range(MAXT):
            s = place(p)
            A = (Wgo - Wno) @ s
            pi = torch.softmax(A, 0); a = int(torch.multinomial(pi, 1, generator=g))
            p2 = (p + DIRS[a] * STEP).clamp(-BOX, BOX)
            at = ((p2 - GOAL) ** 2).sum().sqrt() < RAD; r = 1.0 if at else -0.01
            s2 = place(p2)
            delta = r + GAMMA * (wv @ s2) - (wv @ s)
            wv += lrv * delta * s
            if lesion != "dopamine":
                if lesion != "D1":
                    Wgo[a] += lr * (torch.relu(delta) * s - decay * Wgo[a])     # D1: +RPE -> Go LTP
                if lesion != "D2":
                    Wno[a] += lr * (torch.relu(-delta) * s - decay * Wno[a])    # D2: -RPE -> NoGo LTP
            p = p2
            if at:
                done = True; break
        succ.append(1.0 if done else 0.0)
    nb = 8; b = episodes // nb
    return [sum(succ[i * b:(i + 1) * b]) / b for i in range(nb)]


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 3), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 3) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    conds = ["none", "dopamine", "D1", "D2"]
    curves = {c: [run_seed(s, lesion=c) for s in range(a.seeds)] for c in conds}
    mean_curve = {c: [sum(cur[i] for cur in curves[c]) / a.seeds for i in range(8)] for c in conds}
    late = {c: ci([cur[-1] for cur in curves[c]]) for c in conds}
    early = {c: ci([cur[0] for cur in curves[c]]) for c in conds}
    lab = {"none": "INTACT", "dopamine": "- dopamine (no RPE gating)",
           "D1": "- D1 / Go pathway", "D2": "- D2 / NoGo pathway"}
    print(f"\nBASAL GANGLIA — dopamine-gated action selection (n={a.seeds}; success, mean ± 95% CI)\n" + "=" * 70, flush=True)
    for c in conds:
        print(f"  {lab[c]:30} {early[c][0]:.0%} (early) -> {late[c][0]:.0%} ± {late[c][1]:.0%} (late)", flush=True)
    print(f"\n  -> a faithful Go/NoGo striatal circuit learns action selection by LOCAL dopamine-gated "
          f"plasticity (intact -> {late['none'][0]:.0%}); LESIONING DOPAMINE abolishes learning "
          f"({late['dopamine'][0]:.0%}, stays at chance) -- the dopamine-dependence of reward-based action "
          f"learning. D1(Go) lesion {late['D1'][0]:.0%} / D2(NoGo) lesion {late['D2'][0]:.0%} (opponent pathways).", flush=True)
    out = {"n_seeds": a.seeds,
           "results": {c: {"early": early[c], "late": late[c], "curve": [round(x, 3) for x in mean_curve[c]]} for c in conds}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/basal_ganglia.json", "w"), indent=2)
    svg(mean_curve, late, conds, lab, "results/basal_ganglia.svg")
    print("\nwrote results/basal_ganglia.json and results/basal_ganglia.svg", flush=True)


def svg(curve, late, conds, lab, out):
    pad = 60; pw = 380; ph = 200; W = pad + pw + 180; H = 70 + ph + 40
    col = {"none": "#2ca25f", "dopamine": "#c9341a", "D1": "#3182bd", "D2": "#e6550d"}
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="28" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'A basal-ganglia organ: dopamine-gated action learning (lesion abolishes it)</text>')
    oy = 52
    def X(i): return pad + (i / 7) * pw
    def Y(v): return oy + ph - v * ph
    e.append(f'<line x1="{pad}" y1="{Y(0):.0f}" x2="{pad+pw}" y2="{Y(0):.0f}" stroke="#33415c"/>'
             f'<line x1="{pad}" y1="{oy}" x2="{pad}" y2="{Y(0):.0f}" stroke="#33415c"/>')
    for vv in (0.0, 0.5, 1.0):
        e.append(f'<text x="{pad-8}" y="{Y(vv)+4:.0f}" font-size="9" fill="#5b6b8c" text-anchor="end">{int(vv*100)}%</text>')
    for c in conds:
        pts = " ".join(f"{X(i):.1f},{Y(curve[c][i]):.1f}" for i in range(8))
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col[c]}" stroke-width="2.4"/>')
    e.append(f'<text x="{pad+pw/2:.0f}" y="{Y(0)+16:.0f}" font-size="10" fill="#5b6b8c" text-anchor="middle">training episodes &#8594;</text>')
    ly = oy + 8
    for c in conds:
        e.append(f'<rect x="{pad+pw+14}" y="{ly}" width="14" height="4" fill="{col[c]}"/>')
        e.append(f'<text x="{pad+pw+32}" y="{ly+5}" font-size="9.5" fill="#28324a">{lab[c]} ({late[c][0]:.0%})</text>'); ly += 18
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
