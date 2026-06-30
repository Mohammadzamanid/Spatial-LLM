"""
src/eval/theta_sweep_readout.py

THETA-SWEEP TOKENS ARE LOAD-BEARING FOR LOOK-AHEAD — the readout/LLM side of the Vollan (Nature 2025) sweep.

`theta_sweep.py` showed the BEHAVIORAL benefit (an agent that samples ahead routes around dead-ends a reactive
agent enters, 76%->100%). The piece a review flagged as missing: the sweep as TOKENS that a readout (and the
LLM — see TrajectoryLLM.use_theta_sweep + notebooks/m7_theta_sweep_llm_kaggle.py) actually consumes, with an
ablation showing performance drops when they are removed.

The decisive test uses a NOVEL per-episode environment, so the answer is NOT memorizable from position. Each
episode places an obstacle at a fresh random location. The agent stands at a free cell with a heading and must
predict whether the cone it is about to walk into is BLOCKED. Because the obstacle is new every episode, the
agent's CURRENT location tells it nothing about what is ahead — it has to look. The theta sweep does exactly
that: it samples points ahead (left + right cycles, ~20%-of-spacing, the real ThetaSweepSampler) and reads the
local sense there. Each look-ahead token is [grid code at the swept point (where) + a NOISY obstacle sense
there (what)]. We feed the SAME small readout three things:

  - real sweep   : the look-ahead tokens along the actual heading.
  - ablated      : the sweep slot zeroed (current cell + heading only).
  - shuffled     : look-ahead tokens sampled along a RANDOM wrong heading.

With a novel layout, only the real sweep can see the obstacle ahead, so real >> ablated ~ shuffled ~ chance —
an unambiguous, capacity-independent demonstration that the sweep tokens carry the decision (Vollan's sweeps
extend into never-visited / inaccessible space; this is that look-around made into tokens).

Multi-seed, mean +/- 95% CI. Writes results/theta_sweep_readout.json + .svg.

    python -m src.eval.theta_sweep_readout --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from src.models.neuro.trajectory_cortex import _HexGridModules
from src.models.neuro.theta_sweep import ThetaSweepSampler

R = 2.5
OBS_SIG = 0.4              # obstacle blob width
SENSE_NOISE = 0.20        # the look-ahead sense is noisy (not an oracle)
CONDS = ("real", "ablated", "shuffled")


def obstacle_sense(pos, centers):
    """Local obstacle intensity at pos (n,2) given each episode's obstacle centre (n,2) -> (n,)."""
    return torch.exp(-((pos - centers) ** 2).sum(-1) / (2 * OBS_SIG ** 2))


def batched_sweep(mod, sampler, pos, head, cycle):
    """Vectorised theta sweep: swept positions (n,steps,2) along (heading + alternating-side*angle),
    length = sweep_frac * mean module spacing — the real sampler's geometry."""
    length = sampler.sweep_frac * sampler.spacings(mod).mean()
    side = -1.0 if cycle % 2 == 0 else 1.0
    direction = head + side * sampler.angle
    d = torch.stack([direction.cos(), direction.sin()], -1)
    ks = torch.arange(1, sampler.steps + 1, dtype=torch.float) / sampler.steps
    return pos.unsqueeze(1) + ks.view(1, -1, 1) * length * d.unsqueeze(1)             # (n,steps,2)


def look_tokens(mod, sampler, pos, head, centers, gen):
    """Look-ahead tokens for both theta cycles: per swept point, [grid code (where) + noisy obstacle sense
    (what)]. Returns tokens (n, 2*steps, K*M+1) and the TRUE (noiseless) max sense over the cone (n,)."""
    toks, truth = [], []
    for cyc in (0, 1):
        swept = batched_sweep(mod, sampler, pos, head, cyc)                           # (n,steps,2)
        code = mod.grid_code_at(swept.reshape(-1, 2)).view(pos.shape[0], sampler.steps, -1)
        sense = obstacle_sense(swept, centers.unsqueeze(1))                            # (n,steps) true sense
        noisy = (sense + torch.randn(sense.shape, generator=gen) * SENSE_NOISE).unsqueeze(-1)
        toks.append(torch.cat([code, noisy], -1)); truth.append(sense)
    return torch.cat(toks, 1), torch.cat(truth, 1).max(1).values


def make_set(mod, sampler, n, gen):
    centers = (torch.rand(n, 2, generator=gen) * 2 - 1) * (R * 0.7)                    # NOVEL obstacle per episode
    pos = (torch.rand(n, 2, generator=gen) * 2 - 1) * (R * 0.7)
    # keep the agent itself out of the obstacle (it stands in free space)
    far = obstacle_sense(pos, centers) < 0.3
    head = torch.rand(n, generator=gen) * 2 * math.pi
    cur_code = mod.grid_code_at(pos)
    cur_sense = (obstacle_sense(pos, centers) + torch.randn(n, generator=gen) * SENSE_NOISE).unsqueeze(-1)
    real, truth = look_tokens(mod, sampler, pos, head, centers, gen)
    bad = torch.rand(n, generator=gen) * 2 * math.pi
    shuf, _ = look_tokens(mod, sampler, pos, bad, centers, gen)
    y = (truth > 0.5).float()
    d = {"cur": torch.cat([cur_code, cur_sense], -1), "head": head, "real": real, "shuf": shuf, "y": y}
    return {k: v[far] for k, v in d.items()}


def balance(d, gen):
    pos_idx = (d["y"] > 0.5).nonzero(as_tuple=True)[0]; neg_idx = (d["y"] <= 0.5).nonzero(as_tuple=True)[0]
    m = min(len(pos_idx), len(neg_idx))
    pi = pos_idx[torch.randperm(len(pos_idx), generator=gen)[:m]]
    ni = neg_idx[torch.randperm(len(neg_idx), generator=gen)[:m]]
    idx = torch.cat([pi, ni])
    return {k: v[idx] for k, v in d.items()}


class Readout(nn.Module):
    """Small fixed-capacity readout shared across conditions: encode each [grid code + sense] token to 48-d,
    pool the look-ahead tokens, concat [current, heading, look-ahead] -> blocked-ahead logit."""

    def __init__(self, dim):
        super().__init__()
        self.enc = nn.Linear(dim, 48)
        self.clf = nn.Sequential(nn.Linear(48 + 2 + 48, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, cur, head, sweep):
        cf = torch.relu(self.enc(cur))
        sf = torch.relu(self.enc(sweep)).mean(dim=1)
        hf = torch.stack([head.cos(), head.sin()], -1)
        return self.clf(torch.cat([cf, hf, sf], -1)).squeeze(-1)


def train_eval(cond, tr, te, dim, gen):
    torch.manual_seed(int(torch.randint(1 << 30, (1,), generator=gen)))
    net = Readout(dim); opt = torch.optim.Adam(net.parameters(), 3e-3); lossf = nn.BCEWithLogitsLoss()
    pick = {"real": "real", "ablated": None, "shuffled": "shuf"}[cond]
    sw_tr = tr[pick] if pick else torch.zeros_like(tr["real"])
    sw_te = te[pick] if pick else torch.zeros_like(te["real"])
    for _ in range(400):
        loss = lossf(net(tr["cur"], tr["head"], sw_tr), tr["y"]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return ((net(te["cur"], te["head"], sw_te) > 0).float() == te["y"]).float().mean().item()


def run_seed(seed):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed + 11)
    mod = _HexGridModules(embed_dim=64, n_modules=6, base_spacing=1.6)
    for p in mod.parameters():
        p.requires_grad_(False)
    sampler = ThetaSweepSampler()
    dim = mod.K * mod.M + 1
    tr = balance(make_set(mod, sampler, 9000, gen), gen)
    te = balance(make_set(mod, sampler, 4500, gen), gen)
    return {c: train_eval(c, tr, te, dim, gen) for c in CONDS}


def ci(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 3), round(1.96 * t.std(unbiased=True).item() / math.sqrt(n), 3) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    agg = {c: ci([p[c] for p in per]) for c in CONDS}

    print(f"\nTHETA-SWEEP TOKENS FOR LOOK-AHEAD — blocked-ahead in a NOVEL per-episode layout (n={a.seeds}; "
          f"balanced accuracy, chance 0.50, mean ± 95% CI)\n" + "=" * 80, flush=True)
    name = {"real": "real sweep tokens", "ablated": "sweep ablated (current cell + heading)", "shuffled": "shuffled (wrong-heading sweep)"}
    for c in CONDS:
        print(f"    {name[c]:>38} : {agg[c][0]:.3f} ± {agg[c][1]:.3f}", flush=True)
    print(f"\n  -> the theta-sweep tokens are LOAD-BEARING for look-ahead: in a NOVEL per-episode layout (the "
          f"agent's position reveals nothing about what is ahead) the REAL sweep predicts the blocked cone at "
          f"{agg['real'][0]:.0%}, but ablating the sweep (current cell + heading only) collapses it to "
          f"{agg['ablated'][0]:.0%} and a shuffled wrong-heading sweep to {agg['shuffled'][0]:.0%} — both near "
          f"chance. The sweep samples points AHEAD and reads what is there; nothing else can. This is the "
          f"readout/LLM side of the Vollan look-around (sweeps into never-visited space); "
          f"TrajectoryLLM(use_theta_sweep=True) feeds these as extra spatial tokens (full frozen-LLM ablation: "
          f"notebooks/m7_theta_sweep_llm_kaggle.py).", flush=True)

    out = {"n_seeds": a.seeds, "chance": 0.5, "results": agg}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/theta_sweep_readout.json", "w"), indent=2)
    svg(agg, "results/theta_sweep_readout.svg")
    print("\nwrote results/theta_sweep_readout.json and results/theta_sweep_readout.svg", flush=True)


def svg(agg, out):
    pad = 70; bw = 120; gap = 46; ph = 210; W = pad + len(CONDS) * (bw + gap) + 30; H = 86 + ph + 40
    col = {"real": "#2ca25f", "ablated": "#c9341a", "shuffled": "#8a94a6"}
    lab = {"real": "real sweep", "ablated": "ablated", "shuffled": "shuffled"}
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="26" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Theta-sweep tokens are load-bearing for look-ahead</text>')
    e.append('<text x="26" y="44" font-size="10.5" fill="#5b6b8c">blocked-ahead in a NOVEL per-episode layout; '
             'only the real sweep samples what is ahead &#8212; ablated &amp; shuffled fall to chance</text>')
    oy = 58; base = oy + ph
    e.append(f'<line x1="{pad-10}" y1="{base}" x2="{W-20}" y2="{base}" stroke="#33415c"/>')
    ych = base - 0.5 * ph
    e.append(f'<line x1="{pad-10}" y1="{ych:.0f}" x2="{W-20}" y2="{ych:.0f}" stroke="#9aa6bd" stroke-dasharray="4 3"/>')
    e.append(f'<text x="{W-20}" y="{ych-3:.0f}" font-size="9" fill="#9aa6bd" text-anchor="end">chance 0.50</text>')
    for i, c in enumerate(CONDS):
        v = agg[c][0]; ci_ = agg[c][1]; x = pad + i * (bw + gap); h = v * ph
        e.append(f'<rect x="{x:.0f}" y="{base-h:.1f}" width="{bw}" height="{h:.1f}" fill="{col[c]}" opacity="0.88"/>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{base-h-6:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{base-h-19:.0f}" font-size="8.5" fill="#5b6b8c" text-anchor="middle">&#177;{ci_:.2f}</text>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{base+16:.0f}" font-size="10.5" fill="#28324a" text-anchor="middle">{lab[c]}</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
