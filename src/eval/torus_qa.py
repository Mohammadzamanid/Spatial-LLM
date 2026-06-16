"""
src/eval/torus_qa.py

CPU design-validation for the TORUS-QA LLM headline (before any T4 spend).

The LLM experiment: a frozen cortex lets Qwen answer toroidal-navigation questions that text-only
cannot, on a world with NO faithful Euclidean text description (the leakage rebuttal) — cortex-ON >>
cortex-OFF. Here we check the prerequisite a small readout (the LLM's proxy) can satisfy: does the
cortex encode toroidal position (x mod L, y mod L) so a readout decodes the toroidal cell and
EXTRAPOLATES across many wraps, where a Euclidean code and a text-only proxy cannot?

World: 2-D torus, circumference L. Walk (heading/speed like trajectory_qa); position = (∫velocity)
mod L. Answer = which cell of a G×G toroidal grid (G²-way classification) — the toroidal analog of the
"which sector / how far" questions, and unanswerable from text (the moves never appear). Train on short
paths (few wraps), test on long paths (many wraps).

Encoders compared (readout = same small MLP):
  - grid, toroidal harmonics : Fourier harmonics of the world period L (grid cells adapted to the
    torus; the faithful toroidal code) — should be flat across wraps.
  - grid, Euclidean periods  : the geometric _HexGridModules used on Euclidean tasks (periods not
    matched to L) — tests whether the SHIPPING grid cortex already suffices.
  - place (Euclidean tiling) : bounded, non-toroidal — should fail past the trained range.
  - none (text-only proxy)   : zeros -> chance (1/G²); the cortex-OFF analog.

Picks the cortex config for the Kaggle cell. Writes results/torus_qa.json + .svg.

    python -m src.eval.torus_qa --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neuro.trajectory_cortex import _HexGridModules

L = 1.6                       # torus circumference (matches the grid base period for the Euclidean-grid test)
G = 3                         # G x G toroidal cells -> G*G classes


def make_batch(n, T, gen):
    h = torch.rand(n, T, generator=gen) * 2 * math.pi
    s = torch.rand(n, T, generator=gen) * 0.6 + 0.2
    v = torch.stack([s * h.cos(), s * h.sin()], -1)
    c = v.sum(1)                                          # cumulative displacement (unbounded)
    cell = ((c % L) / L * G).floor().clamp(0, G - 1).long()   # toroidal cell per axis
    label = cell[:, 0] * G + cell[:, 1]                   # G*G classes
    return v, c, label


def mlp(fin, nc):
    return nn.Sequential(nn.Linear(fin, 256), nn.ReLU(), nn.Linear(256, nc))


class ToroidalGrid(nn.Module):                            # harmonics of the world period L (faithful torus code)
    def __init__(self, harmonics=4):
        super().__init__(); self.register_buffer("ks", torch.arange(1, harmonics + 1).float())
        self.head = mlp(4 * harmonics, G * G)

    def forward(self, v, c):
        ph = (TWO_PI := 2 * math.pi) / L * c.unsqueeze(-1) * self.ks.view(1, 1, -1)
        return self.head(torch.cat([ph.cos(), ph.sin()], -1).reshape(c.shape[0], -1))


class EuclidGrid(nn.Module):                              # the shipping geometric grid (periods != L)
    def __init__(self):
        super().__init__(); self.cx = _HexGridModules(64, n_modules=6, base_spacing=L)
        for p in self.cx.parameters():
            p.requires_grad_(False)
        self.head = mlp(self.cx.K * self.cx.M, G * G)

    def forward(self, v, c):
        return self.head(self.cx._grid_code(self.cx.gains.view(-1, 1, 1) * c.unsqueeze(0)))


class EuclidPlace(nn.Module):                             # bounded Euclidean tiling (non-toroidal)
    def __init__(self, cover=3.0, n_side=18):
        super().__init__()
        xs = torch.linspace(-cover, cover, n_side); gx, gy = torch.meshgrid(xs, xs, indexing="ij")
        self.register_buffer("ctr", torch.stack([gx.reshape(-1), gy.reshape(-1)], -1))
        self.sig = 2 * cover / (n_side - 1); self.head = mlp(self.ctr.shape[0], G * G)

    def forward(self, v, c):
        d2 = ((c.unsqueeze(1) - self.ctr.unsqueeze(0)) ** 2).sum(-1)
        return self.head(torch.exp(-d2 / (2 * self.sig ** 2)))


class NoneEnc(nn.Module):                                 # text-only proxy: no position -> chance
    def __init__(self):
        super().__init__(); self.b = nn.Parameter(torch.zeros(G * G))

    def forward(self, v, c):
        return self.b.unsqueeze(0).expand(c.shape[0], -1)


ENCS = {"grid (toroidal harmonics)": ToroidalGrid, "grid (Euclidean periods)": EuclidGrid,
        "place (Euclidean)": EuclidPlace, "none (text-only proxy)": NoneEnc}


def run_seed(seed, train_lengths, test_lengths, steps=700, bs=256, n_eval=4000):
    egen = torch.Generator().manual_seed(90_000 + seed)
    ev = {T: make_batch(n_eval, T, egen) for T in test_lengths}
    out = {}
    for name, Cls in ENCS.items():
        torch.manual_seed(seed); model = Cls()
        ps = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(ps, lr=3e-3) if ps else None
        tgen = torch.Generator().manual_seed(50_000 + seed)
        for _ in range(steps if opt else 0):
            T = train_lengths[_ % len(train_lengths)]
            v, c, y = make_batch(bs, T, tgen)
            opt.zero_grad(); F.cross_entropy(model(v, c), y).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out[name] = {T: (model(ev[T][0], ev[T][1]).argmax(-1) == ev[T][2]).float().mean().item() for T in test_lengths}
    return out


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    return round(t.mean().item(), 4), round(1.96 * (t.std(unbiased=True).item() if n > 1 else 0.0) / math.sqrt(n), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train_lengths", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--test_lengths", type=int, nargs="+", default=[8, 16, 32])
    a = ap.parse_args()
    TL = a.test_lengths
    raw = [run_seed(s, a.train_lengths, TL) for s in range(a.seeds)]
    agg = {nm: {T: dict(zip(("mean", "ci95"), ci95([r[nm][T] for r in raw]))) for T in TL} for nm in ENCS}
    print(f"TORUS-QA design validation (n={a.seeds}; toroidal {G}x{G}-cell accuracy, chance={1/(G*G):.0%})\n"
          f"train {a.train_lengths} (few wraps), test {TL} (many wraps)\n" + "=" * 66, flush=True)
    print("  " + "encoder".ljust(28) + "".join(f"T={T}".rjust(13) for T in TL), flush=True)
    for nm in ENCS:
        print("  " + nm.ljust(28) + "".join(f"{agg[nm][T]['mean']:.0%}±{agg[nm][T]['ci95']:.0%}".rjust(13) for T in TL), flush=True)
    out = {"n_seeds": a.seeds, "L": L, "G": G, "chance": 1 / (G * G),
           "train_lengths": a.train_lengths, "test_lengths": TL, "results": agg}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/torus_qa.json", "w"), indent=2)
    print("\nwrote results/torus_qa.json", flush=True)


if __name__ == "__main__":
    main()
