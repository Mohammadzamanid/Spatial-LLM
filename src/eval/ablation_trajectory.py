"""
src/eval/ablation_trajectory.py

Ablation + gating on 4D NAVIGATION tasks (the brain's navigation modules earn their
keep only when there is movement AND time).

Tasks (--task):
  pathint : integrate moves (heading, speed, vz) over time -> FINAL (x,y,z).
            Order-INDEPENDENT (commutative sum): needs velocity encoding + integration.
  recall  : "where were you at step k?" -> position at a queried timestep. Order/
            history-DEPENDENT — a sum cannot answer it, so the recurrent integrator
            (running per-step position) becomes load-bearing.

Modes (--mode):
  leave_one_out : disable one module at a time (importance).
  add_one_in    : enable one module at a time.
  gates         : train the FULL model with learned per-module gates (+L1) and report
                  which modules it keeps ON — complexity becomes task-dependent.

Metric: mean Euclidean error (lower better) + within-TOL accuracy.
CPU is fine; uses GPU automatically if present.

    python -m src.eval.ablation_trajectory --task pathint --mode leave_one_out
    python -m src.eval.ablation_trajectory --task recall  --mode leave_one_out
    python -m src.eval.ablation_trajectory --task recall  --mode gates --gate_l1 0.02
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
TOL = 0.15


def _moves(n, T, seed, max_speed=1.0):
    g = torch.Generator().manual_seed(seed)
    heading = torch.rand(n, T, generator=g) * 2 * math.pi
    speed = torch.rand(n, T, generator=g) * max_speed
    vz = (torch.rand(n, T, generator=g) - 0.5) * max_speed
    disp = torch.stack([speed * heading.cos(), speed * heading.sin(), vz], dim=-1)  # (n,T,3)
    return heading, speed, vz, disp


def make_data(task, n, T, seed, max_speed=1.0):
    heading, speed, vz, disp = _moves(n, T, seed, max_speed)
    if task in ("recall", "memrecall"):
        cum = disp.cumsum(dim=1)                                  # (n,T,3) running position
        gk = torch.Generator().manual_seed(seed + 9973)
        k = torch.randint(0, T, (n,), generator=gk)
        target = cum[torch.arange(n), k]                          # position at step k
        return heading, speed, vz, k, target
    return heading, speed, vz, disp.sum(dim=1)                     # final position


def train_eval(task, config, data_tr, data_te, epochs=40, aux_loss=False,
               gated=False, gate_l1=0.0, lr=3e-3, device="cpu"):
    data_tr = [d.to(device) for d in data_tr]
    data_te = [d.to(device) for d in data_te]
    model = TrajectoryCortex(embed_dim=EMBED, config=config, aux_heads=aux_loss,
                             dims=3, task=task, gated=gated).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    def split(batch_idx, data):
        if task in ("recall", "memrecall"):
            h, s, v, k, y = data
            return (h[batch_idx], s[batch_idx], v[batch_idx]), {"k": k[batch_idx]}, y[batch_idx]
        h, s, v, y = data
        return (h[batch_idx], s[batch_idx], v[batch_idx]), {}, y[batch_idx]

    n = data_tr[0].shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            inp, kw, y = split(idx, data_tr)
            opt.zero_grad()
            if aux_loss:
                pred, aux = model(*inp, **kw, return_aux=True)
                loss = mse(pred, y) + sum(0.1 * mse(a, y) for a in aux.values())
            else:
                loss = mse(model(*inp, **kw), y)
            if gated and gate_l1 > 0:
                loss = loss + gate_l1 * model.gate_l1()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        inp, kw, y = split(torch.arange(data_te[0].shape[0], device=device), data_te)
        pred = model(*inp, **kw)
        err = (pred - y).norm(dim=-1)
        mean_err, acc = err.mean().item(), (err < TOL).float().mean().item()
    return mean_err, acc, sum(p.numel() for p in model.parameters()), model.gate_values()


def _runner(args, device):
    def run(config, gated=False):
        errs, accs, gates = [], [], []
        p = 0
        for sd in args.seeds:
            torch.manual_seed(sd); np.random.seed(sd)
            tr = make_data(args.task, args.n_train, args.T, seed=sd * 10 + 1)
            te = make_data(args.task, 4000, args.T, seed=sd * 10 + 2)
            e, a, p, gv = train_eval(args.task, config, tr, te, epochs=args.epochs,
                                     aux_loss=args.aux_loss, gated=gated,
                                     gate_l1=args.gate_l1, device=device)
            errs.append(e); accs.append(a); gates.append(gv)
        mean_gates = {k: round(float(np.mean([g[k] for g in gates])), 3)
                      for k in (gates[0] if gates else {})}
        return (float(np.mean(errs)), float(np.std(errs)),
                float(np.mean(accs)), float(np.std(accs)), p, mean_gates)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["pathint", "recall", "memrecall"], default="pathint")
    ap.add_argument("--mode", choices=["leave_one_out", "add_one_in", "gates"],
                    default="leave_one_out")
    ap.add_argument("--aux_loss", action="store_true")
    ap.add_argument("--gate_l1", type=float, default=0.02,
                    help="L1 penalty on module gates (mode=gates) — pushes unused modules off")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  task={args.task}  mode={args.mode}  T={args.T}  seeds={args.seeds}")
    run = _runner(args, device)
    results = {"task": args.task, "mode": args.mode, "T": args.T, "seeds": args.seeds, "tol": TOL}

    if args.mode == "gates":
        e, es, a, as_, p, gates = run(dict(TRAJ_DEFAULT_CONFIG), gated=True)
        print(f"\nFULL gated model: err={e:.3f}±{es:.3f}  acc={a:.1%}±{as_:.1%}")
        print(f"learned module gates (1=kept on, 0=turned off):")
        for k, v in sorted(gates.items(), key=lambda x: -x[1]):
            print(f"    {k:<20}{v:.3f}{'  (kept)' if v > 0.5 else '  (turned OFF)'}")
        results.update({"full": {"err": e, "acc": a}, "gates": gates, "gate_l1": args.gate_l1})
        fname = f"ablation_trajectory_{args.task}_gates.json"
    else:
        full_e, full_es, full_a, full_as, full_p, _ = run(dict(TRAJ_DEFAULT_CONFIG))
        print(f"\nFULL stack: err={full_e:.3f}±{full_es:.3f}  acc={full_a:.1%}±{full_as:.1%}  params={full_p:,}\n")
        results["full"] = {"err": full_e, "err_std": full_es, "acc": full_a, "acc_std": full_as, "params": full_p}
        results["ablation"] = []
        if args.mode == "leave_one_out":
            print(f"{'Disabled module':<20}{'err':>9}{'acc':>9}{'Δacc':>9}")
            print("-" * 47)
            for m in MODULES:
                cfg = dict(TRAJ_DEFAULT_CONFIG); cfg[m] = False
                e, es, a, as_, _, _ = run(cfg)
                results["ablation"].append({"module": m, "err": e, "acc": a, "delta_acc": a - full_a})
                flag = "  load-bearing" if a - full_a < -0.02 else ("  (negligible)" if abs(a - full_a) <= 0.02 else "  better without")
                print(f"{m:<20}{e:>9.3f}{a:>8.1%}{a - full_a:>+8.1%}{flag}")
        else:
            print(f"{'Only this module':<20}{'err':>9}{'acc':>9}")
            print("-" * 38)
            for m in MODULES:
                cfg = {k: False for k in TRAJ_DEFAULT_CONFIG}; cfg[m] = True
                e, es, a, as_, _, _ = run(cfg)
                results["ablation"].append({"module": m, "err": e, "acc": a})
                print(f"{m:<20}{e:>9.3f}{a:>8.1%}")
        fname = f"ablation_trajectory_{args.task}_{args.mode}{'_aux' if args.aux_loss else ''}.json"

    json.dump(results, open(fname, "w"), indent=2)
    print(f"\nSaved -> {fname}")


if __name__ == "__main__":
    main()
