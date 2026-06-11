"""
src/eval/magnitude_frontier.py

Attacking the MAGNITUDE frontier (cortex-level, fast, no LLM).

The harder-question run showed DIRECTION (bearing) generalizes flat across path length but
MAGNITUDE (distance) degrades: the frozen-rep distance probe fell 85% -> 37% from T=8 to
T=24. Is that fixable or fundamental? We sweep the two levers on the distance probe, training
the cortex on lengths {6,8,10,12} and reading distance back at T in {8,16,24} (3 seeds):

  ARCHITECTURE (2x2):
    target = placecode : self-supervised Gaussian place code in a BOUNDED box (the M2
                         default) — cannot represent positions outside the box.
    target = position  : regress the actual (x,y,z) — a SCALE-FREE target (linear readout
                         of the scale-free integrator; the stress-test showed this
                         extrapolates magnitude).
    out_norm on/off    : the cortex's final LayerNorm normalises the rep's MAGNITUDE away.

  DISTRIBUTION:
    widen the speed range (and the bucket cap + place-cell box to match) so LARGER distances
    are in-distribution at the training lengths. If the length curve then flattens, the
    degradation was a coverage problem; if it still droops, it's the recurrent integration
    over more steps (architectural).

Metric: distance-bucket probe accuracy (exact) by length, mean over seeds. Flat across length
= the magnitude code is length-invariant (fixed); a drop with length = the frontier persists.
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn

from ..models.neuro.trajectory_cortex import TrajectoryCortex


def gen(n, T, seed, speed_hi, loop_frac, cap):
    """Random walks (+ a fraction of return-home loops). Returns H,S,V (n,T) and the
    distance bucket label (n,). Distance ~ speed*sqrt(T); speed_hi widens the range."""
    g = torch.Generator().manual_seed(seed)
    H = torch.rand(n, T, generator=g) * 2 * math.pi
    S = torch.rand(n, T, generator=g) * (speed_hi - 0.2) + 0.2
    V = (torch.rand(n, T, generator=g) - 0.5) * 0.8
    half = T // 2
    Hm, Sm, Vm = H.clone(), S.clone(), V.clone()
    for i in range(half):                      # mirror 2nd half -> returns home (loop)
        j = half + i
        Hm[:, j] = H[:, i] + math.pi; Sm[:, j] = S[:, i]; Vm[:, j] = -V[:, i]
    is_loop = (torch.rand(n, generator=g) < loop_frac).unsqueeze(1)
    H = torch.where(is_loop, Hm, H); S = torch.where(is_loop, Sm, S); V = torch.where(is_loop, Vm, V)
    dx = (S * H.cos()).sum(1); dy = (S * H.sin()).sum(1); dz = V.sum(1)
    dist = torch.sqrt(dx * dx + dy * dy + dz * dz)
    bucket = dist.round().clamp(max=cap).long()
    return H, S, V, bucket


def _place(pos, centers, sigma):
    d2 = ((pos.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
    return torch.exp(-d2 / (2 * sigma ** 2))


def _final_pos(H, S, V):
    return torch.stack([(S * H.cos()).sum(1), (S * H.sin()).sum(1), V.sum(1)], dim=-1)


def run_config(name, target, out_norm, speed_hi, env_half, sigma, cap,
               train_L, eval_L, seeds, epochs, n_per, device):
    accs = {T: [] for T in eval_L}
    for seed in seeds:
        torch.manual_seed(seed)
        cx = TrajectoryCortex(embed_dim=64, task="pathint",
                              length_norm=False, out_norm=out_norm).to(device)
        tr = {T: gen(n_per, T, 1000 + T + 7 * seed, speed_hi, 0.3, cap) for T in train_L}
        va = {T: gen(400, T, 2000 + T + 7 * seed, speed_hi, 0.3, cap) for T in eval_L}
        cg = torch.Generator().manual_seed(0)
        if target == "placecode":
            centers = (torch.rand(512, 3, generator=cg) * (2 * env_half) - env_half).to(device)
            head = nn.Linear(64, 512).to(device)
            params = list(cx.parameters()) + list(head.parameters())
        elif target == "multiscale":
            # FINE (local) + COARSE (long-range) Gaussian place codes
            cf = (torch.rand(384, 3, generator=cg) * (2 * env_half) - env_half).to(device)
            cc = (torch.rand(128, 3, generator=cg) * (6 * env_half) - 3 * env_half).to(device)
            head = nn.Linear(64, 512).to(device)
            params = list(cx.parameters()) + list(head.parameters())
        elif target == "gridcode":
            # periodic multi-scale GRID code (sin/cos of position at log-spaced spatial
            # scales) — modular, unbounded, extrapolates magnitude beyond the trained arena
            m = 256
            periods = torch.exp(torch.rand(m, generator=cg) * (math.log(40) - math.log(0.8)) + math.log(0.8))
            d = torch.randn(m, 3, generator=cg); d = d / d.norm(dim=1, keepdim=True)
            freqs = (d * (2 * math.pi / periods).unsqueeze(1)).to(device)
            head = nn.Linear(64, 2 * m).to(device)
            params = list(cx.parameters()) + list(head.parameters())
        else:  # "position" — supervised scale-free target
            params = list(cx.parameters())
        opt = torch.optim.Adam(params, lr=3e-3); mse = nn.MSELoss(); cx.train()
        order = [(T, i) for T in train_L for i in range(0, n_per, 256)]
        for _ in range(epochs):
            torch.manual_seed(torch.randint(0, 1 << 30, (1,)).item())
            for T, i in order:
                H = tr[T][0][i:i + 256].to(device); S = tr[T][1][i:i + 256].to(device)
                Vv = tr[T][2][i:i + 256].to(device)
                opt.zero_grad(); h = cx.encode(H, S, Vv); pos = _final_pos(H, S, Vv)
                if target == "placecode":
                    loss = mse(head(h), _place(pos, centers, sigma))
                elif target == "multiscale":
                    tgt = torch.cat([_place(pos, cf, sigma), _place(pos, cc, 3 * sigma)], -1)
                    loss = mse(head(h), tgt)
                elif target == "gridcode":
                    proj = pos @ freqs.t()
                    loss = mse(head(h), torch.cat([proj.sin(), proj.cos()], -1))
                else:
                    loss = mse(cx.readout(h), pos)
                loss.backward(); opt.step()
        for p in cx.parameters():
            p.requires_grad_(False)
        cx.eval()
        # multi-class distance probe on the frozen rep
        htr = torch.cat([cx.encode(tr[T][0].to(device), tr[T][1].to(device), tr[T][2].to(device))
                         for T in train_L])
        ytr = torch.cat([tr[T][3].to(device) for T in train_L])
        probe = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, cap + 1)).to(device)
        po = torch.optim.Adam(probe.parameters(), lr=1e-2); ce = nn.CrossEntropyLoss()
        for _ in range(500):
            po.zero_grad(); ce(probe(htr), ytr).backward(); po.step()
        with torch.no_grad():
            for T in eval_L:
                h = cx.encode(va[T][0].to(device), va[T][1].to(device), va[T][2].to(device))
                accs[T].append((probe(h).argmax(-1) == va[T][3].to(device)).float().mean().item())
    agg = {T: round(sum(v) / len(v), 3) for T, v in accs.items()}
    print(f"  {name:28} " + "  ".join(f"T{T}:{agg[T]:.0%}" for T in eval_L), flush=True)
    return {"target": target, "out_norm": out_norm, "speed_hi": speed_hi, "env_half": env_half,
            "cap": cap, "acc_by_len": agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train_L", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument("--eval_L", type=int, nargs="+", default=[8, 16, 24])
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--n_per", type=int, default=800)
    ap.add_argument("--out", default="results/magnitude_frontier.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  train_L={a.train_L}  eval_L={a.eval_L}  seeds={a.seeds}\n"
          f"distance-probe accuracy by length (flat = magnitude code is length-invariant):", flush=True)

    # All configs share the real selfsup environment (env_half=4, sigma=1.2) and standard
    # distance data (speed_hi=0.8, cap=5); the only variables are the TARGET and out_norm.
    common = dict(train_L=a.train_L, eval_L=a.eval_L, seeds=a.seeds,
                  epochs=a.epochs, n_per=a.n_per, device=device)
    results = {}
    results["placecode_LN"]   = run_config("placecode (selfsup, M2 default)",   "placecode",  True,  0.8, 4.0, 1.2, 5, **common)
    results["placecode_noLN"] = run_config("placecode + no-LayerNorm",          "placecode",  False, 0.8, 4.0, 1.2, 5, **common)
    results["multiscale_LN"]  = run_config("multiscale placecode (selfsup)",    "multiscale", True,  0.8, 4.0, 1.2, 5, **common)
    results["gridcode_LN"]    = run_config("GRID code (selfsup, faithful fix)", "gridcode",   True,  0.8, 4.0, 1.2, 5, **common)
    results["position_LN"]    = run_config("position-regress (supervised ref)", "position",   True,  0.8, 4.0, 1.2, 5, **common)
    results["position_noLN"]  = run_config("position-regress + no-LayerNorm",   "position",   False, 0.8, 4.0, 1.2, 5, **common)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"config": vars(a), "device": device, "results": results}, f, indent=2)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
