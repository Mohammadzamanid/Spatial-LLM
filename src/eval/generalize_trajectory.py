"""
src/eval/generalize_trajectory.py

GENERALIZATION STRESS-TEST — does the path integrator learn the OPERATION, or
memorize the training length?

Path integration is "sum the per-step displacements". A model that has learned
that operation should extrapolate: train on short walks (T=8), test on longer,
unseen lengths (T up to 32) with no drop. A model that has merely calibrated to
the training length will UNDER-shoot longer trajectories.

We isolate one suspect: the ``/T`` length-normalisation inside ``_AttractorIntegrator``
(``readout(u / T)``). Three modes, same data, same training:

  - "shipped" : the real TrajectoryCortex(task="pathint") — the Milestone-2 cortex,
                which has BOTH ``/T`` AND an out LayerNorm (two magnitude-destroying ops).
  - "norm"    : standalone integrator faithful to _AttractorIntegrator, readout(u / T).
                Isolates the ``/T`` (no LayerNorm).
  - "free"    : same integrator, readout(u) — scale-free accumulation (the fix).

Metric per (mode, seed, T):
  rel_err   = mean||pred - true|| / mean||true||      (0 = perfect, ~1 = useless)
  mag_ratio = mean||pred|| / mean||true||             (1 = correctly scaled; <1 = under-shoot)

Hypothesis: "shipped" and "norm" hold at the train length (T=8) but mag_ratio
collapses toward train_T/test_T on longer walks; "free" extrapolates flat.

Faithful to: src/models/neuro/trajectory_cortex.py::_AttractorIntegrator and the
ConjunctiveSpatialCells velocity front-end. Self-contained so a container reset
can't strand it. CPU-friendly.

Refs: Banino 2018; Cueva & Wei 2018 (grid codes as a learned path-integration basis).
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neuro.spatial_cells import ConjunctiveSpatialCells
from src.models.neuro.trajectory_cortex import TrajectoryCortex


# ----------------------------------------------------------------------------- data
def gen_walk(B: int, T: int, device="cpu"):
    """Random walk: per-step (heading, speed, vz). Target = summed displacement.
    Directions are random, so |final position| grows ~sqrt(T) — longer walks land
    farther away, which is exactly what an extrapolating integrator must track."""
    heading = torch.rand(B, T, device=device) * 2 * math.pi
    speed = torch.rand(B, T, device=device)
    vz = torch.rand(B, T, device=device) - 0.5
    dx = (speed * heading.cos()).sum(1)
    dy = (speed * heading.sin()).sum(1)
    dz = vz.sum(1)
    target = torch.stack([dx, dy, dz], dim=-1)            # (B, 3)
    return heading, speed, vz, target


# ----------------------------------------------------------- standalone integrator
class StandaloneIntegrator(nn.Module):
    """Faithful copy of _AttractorIntegrator (toroidal Mexican-hat continuous attractor)
    with the ConjunctiveSpatialCells velocity front-end and a ``length_norm`` toggle.
    length_norm=True reproduces the shipped ``readout(u / T)``; False is scale-free."""

    def __init__(self, embed_dim=64, grid_size=16, settle=2, length_norm=True):
        super().__init__()
        self.length_norm = length_norm
        self.conjunctive = ConjunctiveSpatialCells(embed_dim=embed_dim)
        self.vert = nn.Linear(1, embed_dim)
        self.N = grid_size * grid_size
        self.settle = settle
        self.vel_to_sheet = nn.Linear(embed_dim, self.N)
        g = grid_size
        cells = torch.stack(torch.meshgrid(
            torch.arange(g), torch.arange(g), indexing="ij"), dim=-1).reshape(-1, 2).float()
        d = cells.unsqueeze(0) - cells.unsqueeze(1)
        d = torch.minimum(d.abs(), g - d.abs())
        dist_sq = (d ** 2).sum(-1)
        self.register_buffer("W", torch.exp(-dist_sq / 8.0) - 0.6 * torch.exp(-dist_sq / 72.0))
        self.readout = nn.Linear(self.N, 3)

    def forward(self, heading, speed, vz):
        B, T = heading.shape
        step = (self.conjunctive(heading.reshape(B * T), speed.reshape(B * T)).view(B, T, -1)
                + self.vert(vz.reshape(B * T, 1)).view(B, T, -1))          # (B, T, embed)
        u = torch.zeros(B, self.N, device=heading.device, dtype=step.dtype)
        for t in range(T):
            u = u + self.vel_to_sheet(step[:, t])
            for _ in range(self.settle):
                u = u + 0.1 * F.linear(torch.tanh(u), self.W)
        return self.readout(u / T) if self.length_norm else self.readout(u)


def build_model(mode: str):
    if mode == "shipped":
        # the actual Milestone-2 cortex: has /T inside the integrator AND an out LayerNorm
        return TrajectoryCortex(embed_dim=64, task="pathint", dims=3)
    if mode == "norm":
        return StandaloneIntegrator(length_norm=True)
    if mode == "free":
        return StandaloneIntegrator(length_norm=False)
    raise ValueError(mode)


def predict(model, mode, heading, speed, vz):
    if mode == "shipped":
        return model(heading, speed, vz)          # TrajectoryCortex.forward -> (B, 3)
    return model(heading, speed, vz)


# ------------------------------------------------------------------------ train/eval
def run_mode(mode, train_T, seeds, T_sweep, steps, batch, lr, device):
    per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = build_model(mode).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for it in range(steps):
            heading, speed, vz, target = gen_walk(batch, train_T, device)
            pred = predict(model, mode, heading, speed, vz)
            loss = F.mse_loss(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        row = {}
        with torch.no_grad():
            for T in T_sweep:
                heading, speed, vz, target = gen_walk(2048, T, device)
                pred = predict(model, mode, heading, speed, vz)
                err = (pred - target).norm(dim=-1).mean().item()
                tgt_mag = target.norm(dim=-1).mean().item()
                pred_mag = pred.norm(dim=-1).mean().item()
                row[T] = {
                    "rel_err": round(err / tgt_mag, 4),
                    "mag_ratio": round(pred_mag / tgt_mag, 4),
                    "err": round(err, 4),
                    "tgt_mag": round(tgt_mag, 4),
                }
        per_seed.append(row)
        flat = " ".join(f"T{T}:{row[T]['rel_err']:.2f}/{row[T]['mag_ratio']:.2f}" for T in T_sweep)
        print(f"  [{mode}] seed={seed}  (rel_err/mag_ratio)  {flat}", flush=True)

    # aggregate mean±std over seeds
    agg = {}
    for T in T_sweep:
        re = torch.tensor([s[T]["rel_err"] for s in per_seed])
        mr = torch.tensor([s[T]["mag_ratio"] for s in per_seed])
        agg[T] = {
            "rel_err_mean": round(re.mean().item(), 4),
            "rel_err_std": round(re.std(unbiased=False).item(), 4),
            "mag_ratio_mean": round(mr.mean().item(), 4),
            "mag_ratio_std": round(mr.std(unbiased=False).item(), 4),
        }
    return {"per_seed": per_seed, "agg": agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_T", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--T_sweep", type=int, nargs="+", default=[4, 8, 12, 16, 24, 32])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--modes", type=str, nargs="+", default=["shipped", "norm", "free"])
    ap.add_argument("--out", type=str, default="results/generalize_trajectory.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  train_T={args.train_T}  sweep={args.T_sweep}  "
          f"steps={args.steps}  seeds={args.seeds}", flush=True)

    results = {}
    for mode in args.modes:
        print(f"\n===== mode={mode} =====", flush=True)
        results[mode] = run_mode(mode, args.train_T, args.seeds, args.T_sweep,
                                  args.steps, args.batch, args.lr, device)

    out = {
        "config": vars(args),
        "device": device,
        "results": results,
        "note": ("rel_err = mean||pred-true||/mean||true|| (0=perfect); "
                 "mag_ratio = mean||pred||/mean||true|| (1=correct scale, <1=under-shoot). "
                 "Train length is train_T; extrapolation = behaviour at T>train_T."),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # legible summary
    print("\n================ SUMMARY (mag_ratio: 1.00 = extrapolates, <1 = under-shoots) ===")
    header = "mode      " + "".join(f"  T={T:<4}" for T in args.T_sweep)
    print(header, flush=True)
    for mode in args.modes:
        agg = results[mode]["agg"]
        cells = "".join(f"  {agg[T]['mag_ratio_mean']:.2f}  " for T in args.T_sweep)
        print(f"{mode:<10}{cells}", flush=True)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
