"""
src/eval/structural_transfer_cortex.py

FAITHFUL de-risk for the structural-transfer LLM headline (lesson from the torus null: validate the
ACTUAL frozen cortex.encode pipeline the LLM reads, not a raw-grid-code proxy).

We mirror the trainer exactly: pre-train the cortex self-supervised (Euclidean place code) and FREEZE
it. Then lay an abstract ordered structure (ranks 0..N-1) along a 1-D concept axis, encode each item by
its OWN position through the frozen cortex (a directed path reaching that position — never the signed
relative displacement, which would leak the answer), and train a comparison readout on ADJACENT pairs
only. If the frozen, space-pretrained cortex.encode exposes the concept positions well enough for
transitive inference on never-seen far pairs — and the shuffled-position control collapses it — then
the LLM version (readout = frozen Qwen+LoRA) is sound.

    python -m src.eval.structural_transfer_cortex --seeds 4
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neuro.trajectory_cortex import TrajectoryCortex


def directed_walk(x_positions, T=8):
    """Encode each 1-D concept position as a directed T-step path reaching (x, 0, 0). Returns
    (B,T) heading/speed/vz so cortex.encode integrates to net displacement (x,0,0)."""
    x = x_positions
    heading = torch.where(x >= 0, torch.zeros_like(x), torch.full_like(x, math.pi)).unsqueeze(1).expand(-1, T)
    speed = (x.abs() / T).unsqueeze(1).expand(-1, T)
    vz = torch.zeros(x.shape[0], T)
    return heading, speed, vz


def pretrain_cortex(seed, env_half=4.0, K=512, sigma=1.2, epochs=45, T=8):
    """Self-supervised Euclidean place-code pretraining over directed walks, then FREEZE (as trainer)."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    cx = TrajectoryCortex(embed_dim=128, task="pathint", constrained_velocity=True)
    centers = (torch.rand(K, 3, generator=g) * (2 * env_half) - env_half)
    sup = nn.Linear(128, K)
    opt = torch.optim.Adam(list(cx.parameters()) + list(sup.parameters()), lr=3e-3)

    def place_code(pos):
        d2 = ((pos.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
        return torch.exp(-d2 / (2 * sigma ** 2))
    cx.train()
    for _ in range(epochs):
        h = torch.rand(256, T, generator=g) * 2 * math.pi
        s = torch.rand(256, T, generator=g) * 0.8
        vz = (torch.rand(256, T, generator=g) - 0.5) * 0.4
        pos = torch.stack([(s * h.cos()).sum(1), (s * h.sin()).sum(1), vz.sum(1)], -1)
        opt.zero_grad(); F.mse_loss(sup(cx.encode(h, s, vz)), place_code(pos)).backward(); opt.step()
    for p in cx.parameters():
        p.requires_grad_(False)
    cx.eval()
    return cx


def encode_items(cx, x_positions):
    h, s, vz = directed_walk(x_positions)
    with torch.no_grad():
        return cx.encode(h, s, vz)                       # (N, 128) frozen cortex codes


def run_seed(seed, N=12, D=0.5, steps=2500):
    cx = pretrain_cortex(seed)
    ranks = torch.arange(N).float()
    xpos = ranks * D - (N - 1) * D / 2                   # ordered concept axis
    codes = encode_items(cx, xpos)
    adj = torch.tensor([(i, i + 1) for i in range(N - 1)] + [(i + 1, i) for i in range(N - 1)])
    nonadj = torch.tensor([(i, j) for i in range(N) for j in range(N) if abs(i - j) >= 2])

    def train_C(cd, seed_):
        torch.manual_seed(seed_)
        C = nn.Sequential(nn.Linear(2 * cd.shape[1], 128), nn.ReLU(), nn.Linear(128, 1))
        opt = torch.optim.Adam(C.parameters(), lr=1e-3)
        a_, b_ = adj[:, 0], adj[:, 1]; y = (ranks[a_] > ranks[b_]).float()
        for _ in range(steps):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(C(torch.cat([cd[a_], cd[b_]], -1)).squeeze(-1), y).backward()
            opt.step()
        return C

    @torch.no_grad()
    def acc(C, cd, pairs, scramble=False):
        a_, b_ = pairs[:, 0], pairs[:, 1]; y = (ranks[a_] > ranks[b_]).float()
        jb = b_[torch.randperm(len(b_))] if scramble else b_
        return ((C(torch.cat([cd[a_], cd[jb]], -1)).squeeze(-1) > 0).float() == y).float().mean().item()

    C = train_C(codes, seed)
    ti = acc(C, codes, nonadj); adj_acc = acc(C, codes, adj); scr = acc(C, codes, nonadj, scramble=True)
    # shuffled-position falsifier: ranks at RANDOM positions -> ordered metric destroyed
    codes_sh = encode_items(cx, xpos[torch.randperm(N)])
    ti_sh = acc(train_C(codes_sh, seed + 1), codes_sh, nonadj)
    return {"ti": ti, "adj": adj_acc, "ti_scrambled_2nd": scr, "ti_shuffled_pos": ti_sh}


def ci95(v):
    t = torch.tensor(v); n = len(v)
    return round(t.mean().item(), 3), round(1.96 * (t.std(unbiased=True).item() if n > 1 else 0) / math.sqrt(n), 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=4); a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    keys = ["ti", "adj", "ti_shuffled_pos", "ti_scrambled_2nd"]
    agg = {k: ci95([p[k] for p in per]) for k in keys}
    print(f"STRUCTURAL TRANSFER through the FROZEN cortex.encode pipeline (n={a.seeds})\n" + "=" * 70, flush=True)
    lab = {"ti": "transitive inference (far, untrained)", "adj": "adjacent (trained)",
           "ti_shuffled_pos": "CONTROL shuffled positions (->chance)",
           "ti_scrambled_2nd": "CONTROL scrambled 2nd item (->chance)"}
    for k in keys:
        print(f"  {lab[k]:42} {agg[k][0]:.3f} ± {agg[k][1]:.3f}", flush=True)
    verdict = "SOUND (frozen Euclidean cortex exposes the concept axis -> build the LLM cell)" \
        if agg["ti"][0] > 0.7 and agg["ti_shuffled_pos"][0] < agg["ti"][0] - 0.1 else \
        "NEEDS concept-axis pretraining (like the torus needed toroidal) before the LLM cell"
    print(f"\n  verdict: {verdict}", flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump({"n_seeds": a.seeds, "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in keys},
               "verdict": verdict}, open("results/structural_transfer_cortex.json", "w"), indent=2)
    print("wrote results/structural_transfer_cortex.json", flush=True)


if __name__ == "__main__":
    main()
