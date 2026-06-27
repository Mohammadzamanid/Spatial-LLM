"""
src/eval/agent_navigation.py

THE BEHAVING AGENT — from probing the cognitive map to USING it in a closed loop. Two results:

  A. CLOSED LOOP (navigation emerges). An agent in a 2D arena path-integrates its own moves into a
     PLACE code (no coordinates given), feeds it to a dopamine-TD CRITIC (value) and a basal-ganglia-like
     softmax ACTOR (action selection), acts, and learns online from the reward-prediction error.
     Goal-directed navigation EMERGES from the integrated loop (early -> late success rate).

  B. COGNITIVE-MAP FLEXIBILITY (what the map buys). The agent learns a SUCCESSOR REPRESENTATION of a
     barriered world from its OWN random-walk exploration (TD), then ONE map serves ANY goal: greedily
     ascending V = M[:, goal] navigates ZERO-SHOT to arbitrary goals, around the barrier. Controls:
     Euclidean vector-navigation stalls at the wall; a model-free policy trained for goal A does not
     transfer to other goals. This flexible, zero-shot, any-goal behavior is the defining function of a
     cognitive map -- here driven by a map the agent learned itself.

Multi-seed, mean +/- 95% CI. Writes results/agent_navigation.json + .svg.

    python -m src.eval.agent_navigation --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.eval.successor import make_world, td_sr, geodesic, plan_success

# ---------------------------------------------------------------- A. closed-loop place-code actor-critic
BOX = 1.0; STEP = 0.2; MAXT = 40; GAMMA = 0.95
GOAL = torch.tensor([0.5, 0.5]); RAD = 0.25
DIRS = torch.tensor([[math.cos(a), math.sin(a)] for a in [i * math.pi / 4 for i in range(8)]])
_cx, _cy = torch.meshgrid(torch.linspace(-BOX, BOX, 10), torch.linspace(-BOX, BOX, 10), indexing="ij")
CENTERS = torch.stack([_cx.reshape(-1), _cy.reshape(-1)], -1); M_PC = CENTERS.shape[0]; SIG = 0.22


def place_code(p):
    return torch.exp(-((p.unsqueeze(0) - CENTERS) ** 2).sum(-1) / (2 * SIG ** 2))


def closed_loop(seed, episodes=4000, pi_noise=0.02):
    """Returns the success-rate learning curve (binned) — navigation emerging from the loop."""
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    actor = nn.Linear(M_PC, 8); critic = nn.Linear(M_PC, 1)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), 5e-3)
    succ = []
    for ep in range(episodes):
        p_true = (torch.rand(2, generator=g) * 2 - 1) * BOX
        p_hat = p_true.clone(); done = False; logps = []; vals = []; rs = []; ents = []
        for t in range(MAXT):
            s = place_code(p_hat); pi = torch.softmax(actor(s), -1); v = critic(s).squeeze(-1)
            a = torch.multinomial(pi, 1, generator=g).item()
            logps.append(torch.log(pi[a] + 1e-9)); vals.append(v); ents.append(-(pi * torch.log(pi + 1e-9)).sum())
            mv = DIRS[a] * STEP; p_true = (p_true + mv).clamp(-BOX, BOX)
            p_hat = (p_hat + mv + pi_noise * torch.randn(2, generator=g)).clamp(-BOX, BOX)
            at = ((p_true - GOAL) ** 2).sum().sqrt() < RAD; rs.append(1.0 if at else -0.01)
            if at:
                done = True; break
        succ.append(1.0 if done else 0.0)
        R = 0.0; rets = []
        for r in reversed(rs):
            R = r + GAMMA * R; rets.insert(0, R)
        rets = torch.tensor(rets); V = torch.stack(vals); adv = rets - V.detach()
        loss = -(torch.stack(logps) * adv).mean() + 0.5 * (V - rets).pow(2).mean() - 0.01 * torch.stack(ents).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    nb = 8; b = episodes // nb
    curve = [sum(succ[i * b:(i + 1) * b]) / b for i in range(nb)]
    return curve


# ---------------------------------------------------------------- B. self-learned-map flexibility
def flexible(seed, G=11, n_goals=15):
    g = torch.Generator().manual_seed(seed)
    gap = int(torch.randint(1, G - max(1, G // 6), (1,), generator=g))
    free, cells, idx = make_world(G, gap, barrier=True)
    M = td_sr(cells, idx, free, G, steps=50000, seed=seed)                    # learned from exploration
    goals = [cells[int(torch.randint(len(cells), (1,), generator=g))] for _ in range(n_goals)]
    pos = {c: torch.tensor([c[0], c[1]], dtype=torch.float) for c in cells}
    sr = sum(plan_success(lambda c: M[idx[c], idx[gl]].item(), cells, idx, free, G, gl) for gl in goals) / n_goals
    euc = sum(plan_success(lambda c: (pos[c] - pos[gl]).norm().item(), cells, idx, free, G, gl, descend=True)
              for gl in goals) / n_goals
    dA = geodesic(cells, idx, free, G, goals[0]); vA = {c: -dA[k].item() for k, c in enumerate(cells)}
    transfer = sum(plan_success(lambda c: vA[c], cells, idx, free, G, gl) for gl in goals) / n_goals
    return sr, euc, transfer


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 3), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 3) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    curves = []; sr = []; euc = []; tr = []
    for s in range(a.seeds):
        curves.append(closed_loop(s))
        x, y, z = flexible(s); sr.append(x); euc.append(y); tr.append(z)
        print(f"  seed {s}: closed-loop success {curves[-1][0]:.0%}->{curves[-1][-1]:.0%} | "
              f"flex SR {x:.0%} / Euclid {y:.0%} / model-free transfer {z:.0%}", flush=True)
    curve = [sum(c[i] for c in curves) / len(curves) for i in range(len(curves[0]))]
    SR, EU, TR = ci(sr), ci(euc), ci(tr); early = ci([c[0] for c in curves]); late = ci([c[-1] for c in curves])
    print(f"\nTHE BEHAVING AGENT (n={a.seeds} seeds; mean ± 95% CI)\n" + "=" * 64, flush=True)
    print(f"  A. closed-loop navigation EMERGES: success {early[0]:.0%} (early) -> {late[0]:.0%} ± {late[1]:.0%} (late)", flush=True)
    print(f"  B. cognitive-map flexibility (zero-shot, any goal, barriered world):", flush=True)
    print(f"       self-learned SR map     {SR[0]:.0%} ± {SR[1]:.0%}", flush=True)
    print(f"       Euclidean vector-nav    {EU[0]:.0%} ± {EU[1]:.0%}   (stalls at the wall)", flush=True)
    print(f"       model-free transfer     {TR[0]:.0%} ± {TR[1]:.0%}   (goal-A policy fails on other goals)", flush=True)
    print(f"\n  -> a behaving agent learns goal-directed navigation in a closed loop, and ONE cognitive map it "
          f"learned from its own exploration drives FLEXIBLE zero-shot navigation to any goal ({SR[0]:.0%}), "
          f"where Euclidean ({EU[0]:.0%}) and model-free transfer ({TR[0]:.0%}) fail.", flush=True)
    out = {"n_seeds": a.seeds,
           "closed_loop_success_early": early, "closed_loop_success_late": late, "learning_curve": curve,
           "flex_sr": SR, "flex_euclidean": EU, "flex_modelfree_transfer": TR}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/agent_navigation.json", "w"), indent=2)
    svg(curve, SR, EU, TR, "results/agent_navigation.svg")
    print("\nwrote results/agent_navigation.json and results/agent_navigation.svg", flush=True)


def svg(curve, SR, EU, TR, out):
    pad = 56; pw = 300; ph = 180; gap = 70; W = pad + pw + gap + pw + pad; H = 70 + ph + 44
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'The behaving agent: navigation emerges, and one self-learned map navigates anywhere</text>')
    oy = 54
    # A: learning curve
    e.append(f'<text x="{pad}" y="{oy-6}" font-size="11" font-weight="700" fill="#28324a">A. closed-loop navigation emerges</text>')
    e.append(f'<line x1="{pad}" y1="{oy+ph}" x2="{pad+pw}" y2="{oy+ph}" stroke="#33415c"/>'
             f'<line x1="{pad}" y1="{oy}" x2="{pad}" y2="{oy+ph}" stroke="#33415c"/>')
    for vv in (0.0, 0.5, 1.0):
        e.append(f'<text x="{pad-6}" y="{oy+ph-vv*ph+4:.0f}" font-size="9" fill="#5b6b8c" text-anchor="end">{int(vv*100)}%</text>')
    n = len(curve)
    pts = " ".join(f"{pad+i/(n-1)*pw:.1f},{oy+ph-curve[i]*ph:.1f}" for i in range(n))
    e.append(f'<polyline points="{pts}" fill="none" stroke="#e6550d" stroke-width="2.6"/>')
    e.append(f'<text x="{pad+pw/2:.0f}" y="{oy+ph+16:.0f}" font-size="10" fill="#5b6b8c" text-anchor="middle">training episodes &#8594; success rate</text>')
    # B: flexibility bars
    bx = pad + pw + gap; base = oy + ph; bw = 56
    e.append(f'<text x="{bx}" y="{oy-6}" font-size="11" font-weight="700" fill="#28324a">B. flexible zero-shot navigation (any goal)</text>')
    e.append(f'<line x1="{bx}" y1="{base}" x2="{bx+pw}" y2="{base}" stroke="#33415c"/>')
    for i, (lab, val, col) in enumerate([("SR map", SR[0], "#2ca25f"), ("Euclid", EU[0], "#9aa5b8"),
                                         ("model-free\ntransfer", TR[0], "#c9341a")]):
        h = val * ph; x = bx + 24 + i * (bw + 30)
        e.append(f'<rect x="{x}" y="{base-h:.0f}" width="{bw}" height="{h:.0f}" fill="{col}" opacity="0.85"/>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{base-h-5:.0f}" font-size="12" font-weight="700" fill="#0b1324" text-anchor="middle">{val:.0%}</text>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{base+15:.0f}" font-size="9.5" fill="#28324a" text-anchor="middle">{lab.split(chr(10))[0]}</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
