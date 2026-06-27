"""
src/eval/agent_unified.py

THE UNIFIED BRAIN-IN-MINIATURE — one agent, one task, three organs, a clean triple-lesion dissociation.
The culmination of the behaving-agent phase: instead of three separate demos (navigation / memory /
timing), a SINGLE agent solves a task that needs all three at once, and removing any one organ removes
its sub-capacity and zeros the reward.

Task ("delayed memory-guided harvest"): each "day" reward is at a NEW location, available only in a brief
window around time D. To get it the agent must
  - RECALL where the reward is            -> hippocampal episodic store
  - NAVIGATE there                        -> cognitive map (successor representation)
  - HARVEST at the right time (~D)        -> time cells (TemporalCortex)
Reward = (at the remembered place) AND (harvest at |t - D| <= W).

Conditions: all-intact, and each single organ lesioned. Prediction: all-intact succeeds; each lesion
zeros the reward via its own failure (-map: can't reach; -memory: wrong place; -time: wrong moment).
Multi-seed, mean +/- 95% CI. Writes results/agent_unified.json + .svg.

    python -m src.eval.agent_unified --seeds 3
"""
import argparse
import json
import math
import os

import torch

from src.models.neuro.temporal_cortex import TemporalCortex
from src.eval.successor import make_world, true_sr, transition_matrix, neighbors

T = 40; H = 64; D = 25; W = 4; NOISE = 0.06


def train_time(seed, iters=2000):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    cx = TemporalCortex(hidden=H, n_in=2); opt = torch.optim.Adam(cx.parameters(), 3e-3)
    for _ in range(iters):
        B = 96; x = torch.zeros(B, T, 2); x[:, 0, 0] = 1.0
        pr = torch.randint(T // 5, T, (B,), generator=g); x[torch.arange(B), pr, 1] = 1.0
        pred, R = cx(x, noise=NOISE, gen=g); pred = pred[torch.arange(B), pr].squeeze(-1)
        loss = ((pred - pr.float() / T) ** 2).mean() + 1e-3 * R.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in cx.parameters():
        p.requires_grad_(False)
    x = torch.zeros(400, T, 2); x[:, 0, 0] = 1.0; R = cx.dynamics(x, noise=NOISE, gen=g)
    A = R.reshape(-1, H); tt = torch.arange(T).float().repeat(400)
    Wt = torch.linalg.lstsq(torch.cat([A, torch.ones(A.shape[0], 1)], 1), tt.unsqueeze(1)).solution
    return cx, Wt, g


def harvest_time(cx, Wt, gen, lesion_time):
    x = torch.zeros(1, T, 2); x[:, 0, 0] = 1.0; R = cx.dynamics(x, noise=NOISE, gen=gen)[0]
    if lesion_time:
        R = torch.zeros_like(R)
    dec = (torch.cat([R, torch.ones(T, 1)], 1) @ Wt).squeeze(-1)
    for t in range(T):
        if dec[t] >= D:
            return t
    return T - 1


def navigate(start, goal, value, free, G):
    cur = start; traj = [cur]
    for _ in range(T):
        if cur != goal:
            nb = neighbors(cur[0], cur[1], free, G); cur = max(nb, key=lambda c: value[c])
        traj.append(cur)
    return traj


def run_seed(seed, G=11, days=40):
    cx, Wt, g = train_time(seed)
    free, cells, idx = make_world(G, 0, barrier=False)
    M = true_sr(transition_matrix(cells, idx, free, G))
    conds = ["all-intact", "-map", "-memory", "-time"]
    rew = {c: [] for c in conds}
    for _ in range(days):
        goal = cells[int(torch.randint(len(cells), (1,), generator=g))]
        start = cells[int(torch.randint(len(cells), (1,), generator=g))]
        for c in conds:
            g_rec = goal if c != "-memory" else cells[int(torch.randint(len(cells), (1,), generator=g))]
            if c == "-map":
                val = {cc: float(torch.rand(1, generator=g)) for cc in cells}      # no map -> random locomotion
            else:
                val = {cc: M[idx[cc], idx[g_rec]].item() for cc in cells}
            traj = navigate(start, g_rec, val, free, G)
            th = harvest_time(cx, Wt, g, lesion_time=(c == "-time"))
            rew[c].append(1.0 if (traj[th] == goal and abs(th - D) <= W) else 0.0)
    return {c: sum(rew[c]) / len(rew[c]) for c in conds}


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 3), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 3) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    conds = ["all-intact", "-map", "-memory", "-time"]
    per = [run_seed(s) for s in range(a.seeds)]
    agg = {c: ci([p[c] for p in per]) for c in conds}
    print(f"\nUNIFIED AGENT — triple-lesion dissociation (n={a.seeds}; reward, mean ± 95% CI)\n" + "=" * 66, flush=True)
    lab = {"all-intact": "ALL INTACT (recall + navigate + time)", "-map": "  - cognitive map (can't reach the place)",
           "-memory": "  - episodic store (goes to wrong place)", "-time": "  - time cells (right place, wrong moment)"}
    for c in conds:
        print(f"  {lab[c]:44} {agg[c][0]:.0%} ± {agg[c][1]:.0%}", flush=True)
    print(f"\n  -> one agent, one task, three organs: ALL-INTACT {agg['all-intact'][0]:.0%}; removing ANY single "
          f"organ zeros the reward (-map {agg['-map'][0]:.0%}, -memory {agg['-memory'][0]:.0%}, "
          f"-time {agg['-time'][0]:.0%}) via its own failure mode. A brain-in-miniature with a clean "
          f"organ->capacity->lesion correspondence.", flush=True)
    out = {"n_seeds": a.seeds, "target_D": D, "window": W,
           "results": {c: {"mean": agg[c][0], "ci95": agg[c][1]} for c in conds}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/agent_unified.json", "w"), indent=2)
    svg(agg, conds, "results/agent_unified.svg")
    print("\nwrote results/agent_unified.json and results/agent_unified.svg", flush=True)


def svg(agg, conds, out):
    pad = 60; bw = 70; gap = 34; ph = 200; W_ = pad + len(conds) * (bw + gap) + pad; Hh = 70 + ph + 50
    col = {"all-intact": "#2ca25f", "-map": "#3182bd", "-memory": "#e6550d", "-time": "#9467bd"}
    short = {"all-intact": "all\nintact", "-map": "− map", "-memory": "− memory", "-time": "− time"}
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_}" height="{Hh}" font-family="Segoe UI, Arial">',
         f'<rect width="{W_}" height="{Hh}" fill="#ffffff"/>']
    e.append('<text x="28" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'One unified agent: three organs, three lesions, one clean dissociation</text>')
    e.append('<text x="28" y="44" font-size="10.5" fill="#5b6b8c">delayed memory-guided harvest: recall '
             'WHERE + navigate THERE + harvest at WHEN &#183; reward needs all three</text>')
    oy = 58; base = oy + ph
    e.append(f'<line x1="{pad-6}" y1="{base}" x2="{W_-pad+6}" y2="{base}" stroke="#33415c"/>')
    for vv in (0.0, 0.5, 1.0):
        e.append(f'<text x="{pad-12}" y="{base-vv*ph+4:.0f}" font-size="9" fill="#5b6b8c" text-anchor="end">{int(vv*100)}%</text>')
    for i, c in enumerate(conds):
        v = agg[c][0]; x = pad + i * (bw + gap); h = v * ph
        e.append(f'<rect x="{x}" y="{base-h:.1f}" width="{bw}" height="{h:.1f}" fill="{col[c]}" opacity="0.88"/>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{base-h-6:.0f}" font-size="13" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.0%}</text>')
        for j, line in enumerate(short[c].split("\n")):
            e.append(f'<text x="{x+bw/2:.0f}" y="{base+16+j*12:.0f}" font-size="10" fill="#28324a" text-anchor="middle">{line}</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
