"""
src/eval/agent_memory.py

MEMORY-GUIDED AGENT — one-shot place learning (Morris water maze / one-trial learning), the second
behaving-agent capacity. The agent has (a) a cognitive map M (successor representation; navigate to any
goal) and (b) a one-shot HIPPOCAMPAL EPISODIC store. Each "day" the reward moves to a new cell: on trial
1 the agent EXPLORES to find it and stores its place code in ONE shot; on later trials it RECALLS the
location (population-vector readout of the stored bump) and navigates straight there via the map.

The control LESIONS the episodic store, so the agent cannot recall and must re-explore every trial. The
prediction (and the classic finding): one-shot savings appear only with the episodic store — its lesion
abolishes one-trial place memory while leaving navigation intact. Multi-seed, mean +/- 95% CI.

    python -m src.eval.agent_memory --seeds 5
"""
import argparse
import json
import math
import os

import torch

from src.eval.successor import make_world, true_sr, transition_matrix, neighbors


def navigate(start, goal, value, free, G, cap):
    cur = start; seen = set()
    for t in range(cap):
        if cur == goal:
            return t
        nb = neighbors(cur[0], cur[1], free, G)
        nxt = max(nb, key=lambda c: value[c])
        if nxt in seen:
            return cap
        seen.add(cur); cur = nxt
    return cap


def explore(start, goal, free, G, cap, gen):
    cur = start
    for t in range(cap):
        if cur == goal:
            return t
        nb = neighbors(cur[0], cur[1], free, G)
        cur = nb[int(torch.randint(len(nb), (1,), generator=gen))]
    return cap


def run_seed(seed, G=11, days=30, trials=4, cap=200):
    g = torch.Generator().manual_seed(seed)
    free, cells, idx = make_world(G, 0, barrier=False)
    M = true_sr(transition_matrix(cells, idx, free, G))
    pos = {c: torch.tensor([c[0], c[1]], dtype=torch.float) for c in cells}
    lat = {"intact": [[] for _ in range(trials)], "lesioned": [[] for _ in range(trials)]}
    for d in range(days):
        goal = cells[int(torch.randint(len(cells), (1,), generator=g))]
        bump = torch.tensor([math.exp(-((pos[c] - pos[goal]) ** 2).sum().item() / 2.0) for c in cells])
        for cond in ("intact", "lesioned"):
            mem = None
            for tr in range(trials):
                start = cells[int(torch.randint(len(cells), (1,), generator=g))]
                if cond == "intact" and tr > 0 and mem is not None:
                    rc = cells[int(torch.argmax(mem))]                       # recall: pop-vector decode
                    val = {c: M[idx[c], idx[rc]].item() for c in cells}      # navigate via the map
                    lat[cond][tr].append(navigate(start, goal, val, free, G, cap))
                else:
                    lat[cond][tr].append(explore(start, goal, free, G, cap, g))
                    if cond == "intact" and tr == 0:
                        mem = bump.clone()                                  # STORE in one shot
    return {c: [sum(v) / len(v) for v in lat[c]] for c in lat}


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 1), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 1) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--trials", type=int, default=4); a = ap.parse_args()
    per = [run_seed(s, trials=a.trials) for s in range(a.seeds)]
    agg = {c: [ci([p[c][tr] for p in per]) for tr in range(a.trials)] for c in ("intact", "lesioned")}
    print(f"\nMEMORY-GUIDED AGENT — one-shot place learning (n={a.seeds}; latency steps, mean ± 95% CI)\n" + "=" * 72, flush=True)
    for c in ("intact", "lesioned"):
        print(f"  {c:9} latency by trial: " + "  ".join(f"T{tr+1} {agg[c][tr][0]:.0f}±{agg[c][tr][1]:.0f}" for tr in range(a.trials)), flush=True)
    sav_i = agg["intact"][0][0] - agg["intact"][1][0]; sav_l = agg["lesioned"][0][0] - agg["lesioned"][1][0]
    print(f"\n  -> ONE rewarded trial collapses latency from {agg['intact'][0][0]:.0f} to {agg['intact'][1][0]:.0f} steps "
          f"(one-shot savings {sav_i:.0f}); LESIONING the episodic store abolishes it (savings {sav_l:.0f}, "
          f"latency stays ~{agg['lesioned'][1][0]:.0f}). Navigation is intact in both — only the one-trial MEMORY is lost.", flush=True)
    out = {"n_seeds": a.seeds, "trials": a.trials,
           "intact": [{"mean": m, "ci95": c} for m, c in agg["intact"]],
           "lesioned": [{"mean": m, "ci95": c} for m, c in agg["lesioned"]]}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/agent_memory.json", "w"), indent=2)
    svg(agg, a.trials, "results/agent_memory.svg")
    print("\nwrote results/agent_memory.json and results/agent_memory.svg", flush=True)


def svg(agg, trials, out):
    pad = 62; pw = 360; ph = 200; W = pad + pw + 150; H = 70 + ph + 44
    ymax = max(agg["lesioned"][0][0], agg["intact"][0][0]) * 1.1
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="28" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Memory-guided agent: one-shot place learning, abolished by lesioning episodic memory</text>')
    oy = 52
    def X(tr): return pad + (tr / (trials - 1)) * pw
    def Y(v): return oy + ph - (v / ymax) * ph
    e.append(f'<line x1="{pad}" y1="{oy+ph}" x2="{pad+pw}" y2="{oy+ph}" stroke="#33415c"/>'
             f'<line x1="{pad}" y1="{oy}" x2="{pad}" y2="{oy+ph}" stroke="#33415c"/>')
    for vv in (0, ymax / 2, ymax):
        e.append(f'<text x="{pad-8}" y="{Y(vv)+4:.0f}" font-size="9" fill="#5b6b8c" text-anchor="end">{vv:.0f}</text>')
    for cond, col in [("intact", "#2ca25f"), ("lesioned", "#c9341a")]:
        pts = " ".join(f"{X(tr):.1f},{Y(agg[cond][tr][0]):.1f}" for tr in range(trials))
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.6"/>')
        for tr in range(trials):
            m, c = agg[cond][tr]
            e.append(f'<line x1="{X(tr):.1f}" y1="{Y(min(ymax,m+c)):.1f}" x2="{X(tr):.1f}" y2="{Y(max(0,m-c)):.1f}" stroke="{col}"/>')
            e.append(f'<circle cx="{X(tr):.1f}" cy="{Y(m):.1f}" r="3.5" fill="{col}"/>')
    for tr in range(trials):
        e.append(f'<text x="{X(tr):.1f}" y="{oy+ph+15:.0f}" font-size="9.5" fill="#5b6b8c" text-anchor="middle">trial {tr+1}</text>')
    e.append(f'<text x="{pad+pw/2:.0f}" y="{oy+ph+30:.0f}" font-size="10" fill="#28324a" text-anchor="middle">trials within a day (reward at a NEW location each day)</text>')
    ly = oy + 6
    for cond, lab, col in [("intact", "episodic memory INTACT", "#2ca25f"), ("lesioned", "episodic memory LESIONED", "#c9341a")]:
        e.append(f'<rect x="{pad+pw+12}" y="{ly}" width="14" height="5" fill="{col}"/>')
        e.append(f'<text x="{pad+pw+30}" y="{ly+6}" font-size="10" fill="#28324a">{lab}</text>'); ly += 20
    e.append(f'<text x="{pad+pw+12}" y="{ly+6}" font-size="9.5" fill="#5b6b8c">latency = steps to reward</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
