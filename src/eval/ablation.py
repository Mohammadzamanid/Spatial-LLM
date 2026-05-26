"""
src/eval/ablation.py

Ablation harness — measures each module's MARGINAL contribution.

Two modes:
  - leave_one_out:  start from full config, disable one module at a time,
                    measure the accuracy drop. Big drop = important module.
  - add_one_in:     start from minimal config, add one module at a time,
                    measure the gain.

Also supports an auxiliary-loss training mode to test whether giving each
module its own learning signal ("synchronization") improves the full stack.

Usage:
    python -m src.eval.ablation --mode leave_one_out
    python -m src.eval.ablation --mode add_one_in
    python -m src.eval.ablation --aux_loss   # test synchronization
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from src.models.neuro.configurable_cortex import ConfigurableCortex, DEFAULT_CONFIG

EMBED = 64
MODULES = list(DEFAULT_CONFIG.keys())


def make_fine_grid_data(n, lat0=40, lon0=0, span=10, seed=0):
    """100-class 1° grid classification — rewards spatial resolution."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        lat = rng.uniform(lat0, lat0 + span - 1e-3)
        lon = rng.uniform(lon0, lon0 + span - 1e-3)
        cls = int(lat - lat0) * span + int(lon - lon0)
        X.append([lat, lon]); y.append(cls)
    return torch.tensor(X), torch.tensor(y)


def train_eval(config, Xtr, ytr, Xte, yte, n_cls=100, epochs=40,
               aux_loss=False, lr=3e-3):
    cortex = ConfigurableCortex(embed_dim=EMBED, config=config,
                                aux_heads=aux_loss, num_tokens=1)
    head = nn.Linear(EMBED, n_cls)
    params = list(cortex.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    cortex.train(); head.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 512):
            idx = perm[i:i+512]
            opt.zero_grad()
            if aux_loss:
                emb, aux = cortex(Xtr[idx], return_aux=True)
                logits = head(emb)
                loss = ce(logits, ytr[idx])
                # Each module also predicts the coordinate (its own signal)
                for name, pred in aux.items():
                    loss = loss + 0.1 * mse(pred, Xtr[idx])
            else:
                emb = cortex(Xtr[idx])
                loss = ce(head(emb), ytr[idx])
            loss.backward(); opt.step()

    cortex.eval(); head.eval()
    with torch.no_grad():
        acc = (head(cortex(Xte)).argmax(-1) == yte).float().mean().item()
    n_params = sum(p.numel() for p in params)
    return acc, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["leave_one_out", "add_one_in"],
                    default="leave_one_out")
    ap.add_argument("--aux_loss", action="store_true",
                    help="Test module synchronization via per-module aux losses")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    Xtr, ytr = make_fine_grid_data(20000, seed=1)
    Xte, yte = make_fine_grid_data(4000, seed=2)

    # Full reference
    full_acc, full_params = train_eval(dict(DEFAULT_CONFIG), Xtr, ytr, Xte, yte,
                                       aux_loss=args.aux_loss)
    print(f"\nFULL stack: acc={full_acc:.1%}  params={full_params:,}"
          f"  (aux_loss={args.aux_loss})\n")

    results = {"full": {"acc": full_acc, "params": full_params}, "ablation": []}

    if args.mode == "leave_one_out":
        print(f"{'Disabled module':<22}{'Acc':>8}{'Δ vs full':>12}")
        print("-" * 42)
        for m in MODULES:
            cfg = dict(DEFAULT_CONFIG); cfg[m] = False
            acc, _ = train_eval(cfg, Xtr, ytr, Xte, yte, aux_loss=args.aux_loss)
            delta = acc - full_acc
            results["ablation"].append({"module": m, "acc": acc, "delta": delta})
            flag = "  ← hurts to remove" if delta < -0.02 else ("  (negligible)" if abs(delta) <= 0.02 else "  ← better without!")
            print(f"{m:<22}{acc:>7.1%}{delta:>+11.1%}{flag}")
    else:
        print(f"{'Only this module':<22}{'Acc':>8}")
        print("-" * 30)
        for m in MODULES:
            cfg = {k: False for k in DEFAULT_CONFIG}; cfg[m] = True
            acc, _ = train_eval(cfg, Xtr, ytr, Xte, yte, aux_loss=args.aux_loss)
            results["ablation"].append({"module": m, "acc": acc})
            print(f"{m:<22}{acc:>7.1%}")

    suffix = "_aux" if args.aux_loss else ""
    fname = f"ablation_{args.mode}{suffix}.json"
    json.dump(results, open(fname, "w"), indent=2)
    print(f"\nSaved → {fname}")


if __name__ == "__main__":
    main()
