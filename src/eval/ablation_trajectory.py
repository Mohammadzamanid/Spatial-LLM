"""
src/eval/ablation_trajectory.py

Ablation on a 4D NAVIGATION task — path integration.

The model is given a sequence of moves (heading, speed, vertical velocity) over T
timesteps and must predict the final (x, y, z) displacement from the start. This is
"move from one point to another through time" — the task the brain's navigation
system is built for, and the one the static grid task could not express.

Metric: mean Euclidean error (lower = better) + within-tolerance "accuracy"
(fraction of predictions within TOL of the true final position).

Usage (CPU is fine; uses GPU automatically if present):
    python -m src.eval.ablation_trajectory --mode leave_one_out
    python -m src.eval.ablation_trajectory --mode add_one_in
    python -m src.eval.ablation_trajectory --aux_loss        # synchronization test
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn

from src.models.neuro.trajectory_cortex import TrajectoryCortex, TRAJ_DEFAULT_CONFIG

EMBED = 64
MODULES = list(TRAJ_DEFAULT_CONFIG.keys())
TOL = 0.15  # a prediction counts as correct if within this distance of the true final pos
            # (full model err ~0.11; baseline "predict origin" err ~2.0, so this discriminates)


def make_trajectory_data(n, T=12, seed=0, max_speed=1.0):
    """Random 3D walks. Returns (heading, speed, vz) each (n, T) and final pos (n, 3).
    The model never sees positions — only the per-step moves — so it must integrate."""
    g = torch.Generator().manual_seed(seed)
    heading = torch.rand(n, T, generator=g) * 2 * math.pi          # azimuth
    speed = torch.rand(n, T, generator=g) * max_speed              # horizontal speed
    vz = (torch.rand(n, T, generator=g) - 0.5) * max_speed         # vertical velocity
    dx = (speed * heading.cos()).sum(1)
    dy = (speed * heading.sin()).sum(1)
    dz = vz.sum(1)
    final = torch.stack([dx, dy, dz], dim=-1)
    return heading, speed, vz, final


def train_eval(config, data_tr, data_te, epochs=40, aux_loss=False, lr=3e-3, device="cpu"):
    htr, str_, vtr, ytr = [d.to(device) for d in data_tr]
    hte, ste, vte, yte = [d.to(device) for d in data_te]
    model = TrajectoryCortex(embed_dim=EMBED, config=config, aux_heads=aux_loss, dims=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    model.train()
    n = htr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            if aux_loss:
                pred, aux = model(htr[idx], str_[idx], vtr[idx], return_aux=True)
                loss = mse(pred, ytr[idx])
                for _, ap in aux.items():                      # each module predicts the target too
                    loss = loss + 0.1 * mse(ap, ytr[idx])
            else:
                pred = model(htr[idx], str_[idx], vtr[idx])
                loss = mse(pred, ytr[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(hte, ste, vte)
        err = (pred - yte).norm(dim=-1)
        mean_err = err.mean().item()
        acc = (err < TOL).float().mean().item()
    n_params = sum(p.numel() for p in model.parameters())
    return mean_err, acc, n_params


def _runner(args, device):
    def run(config):
        errs, accs, p = [], [], 0
        for sd in args.seeds:
            torch.manual_seed(sd)
            np.random.seed(sd)
            tr = make_trajectory_data(args.n_train, T=args.T, seed=sd * 10 + 1)
            te = make_trajectory_data(4000, T=args.T, seed=sd * 10 + 2)
            e, a, p = train_eval(config, tr, te, epochs=args.epochs,
                                 aux_loss=args.aux_loss, device=device)
            errs.append(e)
            accs.append(a)
        return (float(np.mean(errs)), float(np.std(errs)),
                float(np.mean(accs)), float(np.std(accs)), p)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["leave_one_out", "add_one_in"], default="leave_one_out")
    ap.add_argument("--aux_loss", action="store_true",
                    help="give each module its own target-prediction signal (synchronization)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--T", type=int, default=12, help="trajectory length (timesteps)")
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  T={args.T}  seeds={args.seeds}  aux_loss={args.aux_loss}  mode={args.mode}")
    run = _runner(args, device)

    full_e, full_es, full_a, full_as, full_p = run(dict(TRAJ_DEFAULT_CONFIG))
    print(f"\nFULL stack: err={full_e:.3f}±{full_es:.3f}  acc={full_a:.1%}±{full_as:.1%}  params={full_p:,}\n")

    results = {"task": "path_integration_3d", "T": args.T, "seeds": args.seeds,
               "aux_loss": args.aux_loss, "mode": args.mode, "tol": TOL,
               "full": {"err": full_e, "err_std": full_es, "acc": full_a, "acc_std": full_as,
                        "params": full_p},
               "ablation": []}

    if args.mode == "leave_one_out":
        print(f"{'Disabled module':<20}{'err':>9}{'acc':>9}{'Δacc vs full':>14}")
        print("-" * 52)
        for m in MODULES:
            cfg = dict(TRAJ_DEFAULT_CONFIG); cfg[m] = False
            e, es, a, as_, _ = run(cfg)
            results["ablation"].append({"module": m, "err": e, "acc": a, "delta_acc": a - full_a})
            flag = "  <- load-bearing" if (a - full_a) < -0.02 else ("  (negligible)" if abs(a - full_a) <= 0.02 else "  <- better without")
            print(f"{m:<20}{e:>9.3f}{a:>8.1%}{a - full_a:>+13.1%}{flag}")
    else:
        print(f"{'Only this module':<20}{'err':>9}{'acc':>9}")
        print("-" * 38)
        for m in MODULES:
            cfg = {k: False for k in TRAJ_DEFAULT_CONFIG}; cfg[m] = True
            e, es, a, as_, _ = run(cfg)
            results["ablation"].append({"module": m, "err": e, "acc": a})
            print(f"{m:<20}{e:>9.3f}{a:>8.1%}")

    suffix = "_aux" if args.aux_loss else ""
    fname = f"ablation_trajectory_{args.mode}{suffix}.json"
    json.dump(results, open(fname, "w"), indent=2)
    print(f"\nSaved -> {fname}")


if __name__ == "__main__":
    main()
