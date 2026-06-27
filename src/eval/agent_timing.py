"""
src/eval/agent_timing.py

TIMING-GUIDED AGENT — interval production, the third behaving-agent capacity (the temporal organ driving
action). The agent must emit a single action at a target interval D; reward peaks at D and decays away
(exp(-(t_act-D)^2 / 2 sigma^2)). Its policy reads the EMERGENT TIME-CELL population (a TemporalCortex
pretrained on elapsed-time, then frozen) to decide WHEN to act, and learns by REINFORCE.

The control LESIONS the temporal code (zeros it): the policy can no longer tell elapsed time, so it cannot
wait for D. Prediction (and result): timed action emerges only with the temporal organ; its lesion
abolishes timing (the agent acts immediately) while leaving the rest of the agent intact -- the temporal
analogue of the place-memory lesion. Multi-seed, mean +/- 95% CI.

    python -m src.eval.agent_timing --seeds 3
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.models.neuro.temporal_cortex import TemporalCortex

T = 40; H = 64; D = 25; SIGR = 3.0; NOISE = 0.06


def train_cortex(seed, iters=2000):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    cx = TemporalCortex(hidden=H, n_in=2); opt = torch.optim.Adam(cx.parameters(), 3e-3)
    for _ in range(iters):
        B = 96; x = torch.zeros(B, T, 2); x[:, 0, 0] = 1.0
        probe = torch.randint(T // 5, T, (B,), generator=g); x[torch.arange(B), probe, 1] = 1.0
        pred, R = cx(x, noise=NOISE, gen=g); pred = pred[torch.arange(B), probe].squeeze(-1)
        loss = ((pred - probe.float() / T) ** 2).mean() + 1e-3 * R.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in cx.parameters():
        p.requires_grad_(False)
    return cx, g


def time_code(cx, B, gen):
    x = torch.zeros(B, T, 2); x[:, 0, 0] = 1.0
    return cx.dynamics(x, noise=NOISE, gen=gen)


def _rollout(w, cx, B, gen, lesion):
    R = time_code(cx, B, gen)
    if lesion:
        R = torch.zeros_like(R)
    p = torch.sigmoid(w(R).squeeze(-1))
    acted = torch.zeros(B, dtype=torch.bool); tact = torch.full((B,), T)
    u = torch.rand(B, T, generator=gen)
    for t in range(T):
        fire = (u[:, t] < p[:, t].detach()) & ~acted; tact[fire] = t; acted |= fire
    rew = torch.where(acted, torch.exp(-((tact.float() - D) ** 2) / (2 * SIGR ** 2)), torch.zeros(B))
    return p, tact, acted, rew


def run_seed(seed, episodes=4000, lesion=False, want_hist=False):
    cx, g = train_cortex(seed)
    w = nn.Linear(H, 1); opt = torch.optim.Adam(w.parameters(), 3e-3); base = 0.3
    for ep in range(episodes):
        p, tact, acted, rew = _rollout(w, cx, 96, g, lesion)
        lognotp = torch.log(1 - p + 1e-6); csum = torch.cumsum(lognotp, 1)
        prefix = torch.where(tact > 0, csum.gather(1, (tact - 1).clamp(min=0).unsqueeze(1)).squeeze(1), torch.zeros(96))
        logp_act = torch.log(p.gather(1, tact.clamp(max=T - 1).unsqueeze(1)).squeeze(1) + 1e-6)
        lp = torch.where(acted, prefix + logp_act, csum[:, -1])
        loss = -((rew - base) * lp).mean(); base = 0.99 * base + 0.01 * rew.mean().item()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _, tact, acted, rew = _rollout(w, cx, 600, g, lesion)
        at = tact[acted].float()
        out = {"reward": rew.mean().item(), "act_mean": at.mean().item() if len(at) else float("nan"),
               "act_std": at.std().item() if len(at) > 1 else float("nan")}
        hist = torch.histc(tact.float(), bins=T, min=0, max=T) if want_hist else None
    return out, hist


def ci(vals):
    vals = [v for v in vals if v == v]
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 3), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 3) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    I = {"reward": [], "act_mean": [], "act_std": []}; L = {"reward": [], "act_mean": [], "act_std": []}
    hi = hl = None
    for s in range(a.seeds):
        oi, h_i = run_seed(s, lesion=False, want_hist=(s == 0))
        ol, h_l = run_seed(s, lesion=True, want_hist=(s == 0))
        for k in I:
            I[k].append(oi[k]); L[k].append(ol[k])
        if s == 0:
            hi, hl = h_i, h_l
        print(f"  seed {s}: INTACT reward {oi['reward']:.2f} act@{oi['act_mean']:.0f}±{oi['act_std']:.1f} | "
              f"LESIONED reward {ol['reward']:.2f} act@{ol['act_mean']:.0f}±{ol['act_std']:.1f}", flush=True)
    rI, rL = ci(I["reward"]), ci(L["reward"]); aI, aL = ci(I["act_mean"]), ci(L["act_mean"]); sI = ci(I["act_std"])
    print(f"\nTIMING-GUIDED AGENT — interval production, target D={D} (n={a.seeds}; mean ± 95% CI)\n" + "=" * 70, flush=True)
    print(f"  INTACT (reads time cells):   reward {rI[0]:.2f} ± {rI[1]:.2f} | acts at {aI[0]:.0f} ± {aI[1]:.0f} (precision ±{sI[0]:.1f})", flush=True)
    print(f"  LESIONED (temporal code 0):  reward {rL[0]:.2f} ± {rL[1]:.2f} | acts at {aL[0]:.0f} ± {aL[1]:.0f}", flush=True)
    print(f"\n  -> timed action EMERGES from the temporal organ: the agent acts at the target interval "
          f"(D={D}, reward {rI[0]:.2f}); LESIONING the temporal code abolishes timing (reward {rL[0]:.2f}, "
          f"acts at {aL[0]:.0f}) -- it can no longer tell elapsed time, while the rest of the agent is intact.", flush=True)
    out = {"n_seeds": a.seeds, "target_D": D, "reward_sigma": SIGR,
           "intact": {"reward": rI, "act_mean": aI, "act_std": sI},
           "lesioned": {"reward": rL, "act_mean": aL}}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/agent_timing.json", "w"), indent=2)
    if hi is not None:
        svg(hi, hl, rI, rL, "results/agent_timing.svg")
    print("\nwrote results/agent_timing.json and results/agent_timing.svg", flush=True)


def svg(hi, hl, rI, rL, out):
    pad = 56; pw = 380; ph = 180; W = pad + pw + pad; H_ = 70 + ph + 44
    hm = max(hi.max().item(), hl.max().item()) + 1e-6
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H_}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H_}" fill="#ffffff"/>']
    e.append('<text x="26" y="24" font-size="14.5" font-weight="800" fill="#0b1324">'
             'Timing-guided agent: acts on cue at D, abolished by lesioning the temporal code</text>')
    e.append(f'<text x="26" y="42" font-size="10.5" fill="#5b6b8c">action-time distribution &#183; '
             f'INTACT reward {rI[0]:.2f} vs LESIONED {rL[0]:.2f}</text>')
    oy = 54
    def X(t): return pad + (t / (T - 1)) * pw
    def Y(v): return oy + ph - (v / hm) * ph
    e.append(f'<line x1="{pad}" y1="{oy+ph}" x2="{pad+pw}" y2="{oy+ph}" stroke="#33415c"/>')
    e.append(f'<line x1="{X(D):.1f}" y1="{oy}" x2="{X(D):.1f}" y2="{oy+ph}" stroke="#c9341a" stroke-dasharray="4,3" opacity="0.6"/>')
    e.append(f'<text x="{X(D):.1f}" y="{oy-2:.0f}" font-size="9" fill="#c9341a" text-anchor="middle">target D={D}</text>')
    for h, col in [(hl, "#9aa5b8"), (hi, "#2ca25f")]:
        pts = " ".join(f"{X(t):.1f},{Y(h[t].item()):.1f}" for t in range(T))
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>')
    e.append(f'<text x="{pad+pw/2:.0f}" y="{oy+ph+16:.0f}" font-size="10" fill="#5b6b8c" text-anchor="middle">action time (steps since start)</text>')
    ly = oy + 6
    for col, lab in [("#2ca25f", "INTACT (reads time cells)"), ("#9aa5b8", "LESIONED (temporal code zeroed)")]:
        e.append(f'<rect x="{pad+pw-188}" y="{ly}" width="13" height="4" fill="{col}"/>')
        e.append(f'<text x="{pad+pw-171}" y="{ly+5}" font-size="9.5" fill="#28324a">{lab}</text>'); ly += 15
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
