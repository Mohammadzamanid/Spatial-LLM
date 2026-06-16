"""
src/eval/torus.py

LEAKAGE-KILLER + the tie-breaker: path integration on a TORUS (a world with NO faithful Euclidean
text description, so an LLM's text prior cannot help), where the additive/NoPE-sum integrator that
TIED the grid code on Euclidean paths must FAIL.

Why: on a torus of circumference 2π, true position is θ = (∫velocity) mod 2π. A PERIODIC (grid-cell)
code computes that mod for free — cos(∫v) = cos(θ) for ANY number of wraps — so it path-integrates
toroidal position and extrapolates to arbitrarily long paths. A non-periodic code (raw cumulative
sum, a learned GRU/Transformer sum, or a Euclidean place tiling) sees an UNBOUNDED ∫v and a readout
cannot recover θ = ∫v mod 2π once paths leave the trained range (the wrap count is unseen). So the
periodicity that merely gave the grid code finite range on Euclidean paths is EXACTLY the right
inductive bias for a cyclic world — a regime where it should beat the additive integrator decisively.

Trajectories traverse the torus (per-trajectory heading+speed + small noise), so long paths wrap many
times. Train on short paths (≤1 wrap), test out to many wraps. Decode toroidal position; report mean
angular error (rad) and "within π/4" accuracy. Multi-seed, mean ± 95% CI.
Writes results/torus.json + results/torus.svg.

    python -m src.eval.torus --seeds 8
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

TWO_PI = 2 * math.pi


def make_batch(n, T, gen):
    """Torus traversal: per-trajectory heading & speed + small per-step noise -> cumulative
    displacement grows ~linearly in T (so long paths wrap many times). Returns velocity seq (n,T,2)
    and the toroidal target [cos cx, sin cx, cos cy, sin cy] where c = sum of velocity."""
    h = (torch.rand(n, 1, generator=gen) * TWO_PI)
    s = (torch.rand(n, 1, generator=gen) * 0.5 + 0.3)
    base = torch.stack([s.squeeze(1) * h.squeeze(1).cos(), s.squeeze(1) * h.squeeze(1).sin()], -1)  # (n,2)
    v = base.unsqueeze(1).expand(n, T, 2) + 0.1 * torch.randn(n, T, 2, generator=gen)
    c = v.sum(1)                                                # (n,2) unbounded cumulative displacement
    tgt = torch.stack([c[:, 0].cos(), c[:, 0].sin(), c[:, 1].cos(), c[:, 1].sin()], -1)  # toroidal coord
    return v, c, tgt


def head4(fin):
    return nn.Sequential(nn.Linear(fin, 128), nn.ReLU(), nn.Linear(128, 4))


class GridTorus(nn.Module):
    """PERIODIC (grid-cell) code: harmonics of the cumulative phase — wraps natively on the torus."""
    def __init__(self, harmonics=4):
        super().__init__()
        self.register_buffer("ks", torch.arange(1, harmonics + 1).float())
        self.head = head4(4 * harmonics)

    def code(self, c):
        ph = c.unsqueeze(-1) * self.ks.view(1, 1, -1)           # (B,2,H)
        return torch.cat([ph.cos(), ph.sin()], -1).reshape(c.shape[0], -1)  # (B,4H)

    def forward(self, v, c):
        return self.head(self.code(c))


class AdditiveTorus(nn.Module):
    """Raw cumulative sum (unbounded) -> readout must learn mod 2π; cannot extrapolate the wrap count."""
    def __init__(self):
        super().__init__(); self.head = head4(2)

    def forward(self, v, c):
        return self.head(c)


class SeqSumTorus(nn.Module):
    """NoPE + sum-pool Transformer over the velocity sequence (the baseline that tied on Euclidean)."""
    def __init__(self, d=64, layers=2, heads=4):
        super().__init__()
        self.inp = nn.Linear(2, d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, layers); self.head = head4(d)

    def forward(self, v, c):
        return self.head(self.tr(self.inp(v)).sum(1))


class PlaceTorus(nn.Module):
    """Euclidean Gaussian tiling of the trained cumulative-displacement box (no wrap knowledge)."""
    def __init__(self, cover=4.0, n_side=20):
        super().__init__()
        xs = torch.linspace(-cover, cover, n_side)
        gx, gy = torch.meshgrid(xs, xs, indexing="ij")
        self.register_buffer("centers", torch.stack([gx.reshape(-1), gy.reshape(-1)], -1))
        self.sigma = 2 * cover / (n_side - 1); self.head = head4(self.centers.shape[0])

    def forward(self, v, c):
        d2 = ((c.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2).sum(-1)
        return self.head(torch.exp(-d2 / (2 * self.sigma ** 2)))


class OracleTorus(nn.Module):
    """Fed the true toroidal coordinate (cos/sin of c) — the ceiling (task is solvable at any length)."""
    def __init__(self):
        super().__init__(); self.head = head4(4)

    def forward(self, v, c):
        return self.head(torch.stack([c[:, 0].cos(), c[:, 0].sin(), c[:, 1].cos(), c[:, 1].sin()], -1))


REPS = {"grid (periodic)": GridTorus, "additive (cumsum)": AdditiveTorus,
        "NoPE+sum Transformer": SeqSumTorus, "place (Euclidean)": PlaceTorus, "oracle": OracleTorus}


def decode_angles(pred):
    return torch.stack([torch.atan2(pred[:, 1], pred[:, 0]), torch.atan2(pred[:, 3], pred[:, 2])], -1)


def ang_err(pred, tgt):
    pa = decode_angles(pred); ta = decode_angles(tgt)
    d = torch.atan2((pa - ta).sin(), (pa - ta).cos()).abs()     # wrapped angular error per axis
    return d


def run_seed(seed, train_lengths, test_lengths, steps=700, bs=256, n_eval=4000):
    egen = torch.Generator().manual_seed(90_000 + seed)
    eval_sets = {T: make_batch(n_eval, T, egen) for T in test_lengths}
    out = {}
    for name, Cls in REPS.items():
        torch.manual_seed(seed)
        model = Cls()
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
        tgen = torch.Generator().manual_seed(50_000 + seed)
        for step in range(steps):
            T = train_lengths[step % len(train_lengths)]
            v, c, tgt = make_batch(bs, T, tgen)
            opt.zero_grad(); F.mse_loss(model(v, c), tgt).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            o = {}
            for T in test_lengths:
                v, c, tgt = eval_sets[T]; pred = model(v, c); e = ang_err(pred, tgt)
                o[T] = {"ang_err": e.mean().item(), "within_45deg": (e < math.pi / 4).float().mean().item()}
            out[name] = o
    return out


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float)
    n = len(vals); sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--train_lengths", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--test_lengths", type=int, nargs="+", default=[8, 16, 32, 64])
    a = ap.parse_args()
    seeds = list(range(a.seeds)); TL = a.test_lengths
    print(f"TORUS path integration (n={a.seeds}; mean ± 95% CI) — a world with no Euclidean text prior\n"
          f"train {a.train_lengths} (≤~1 wrap), test {TL} (many wraps)\n" + "=" * 72, flush=True)
    raw = [run_seed(s, a.train_lengths, TL) for s in seeds]
    agg = {nm: {T: {} for T in TL} for nm in REPS}
    for nm in REPS:
        for T in TL:
            for mk in ("ang_err", "within_45deg"):
                m, ci = ci95([r[nm][T][mk] for r in raw]); agg[nm][T][mk] = {"mean": m, "ci95": ci}

    print("[toroidal position error (radians; lower=better)]", flush=True)
    print("  " + "code".ljust(22) + "".join(f"T={T}".rjust(14) for T in TL), flush=True)
    for nm in REPS:
        print("  " + nm.ljust(22) + "".join(f"{agg[nm][T]['ang_err']['mean']:.2f}±{agg[nm][T]['ang_err']['ci95']:.2f}".rjust(14) for T in TL), flush=True)
    print("\n[within 45deg accuracy]", flush=True)
    for nm in REPS:
        print("  " + nm.ljust(22) + "".join(f"{agg[nm][T]['within_45deg']['mean']:.0%}".rjust(14) for T in TL), flush=True)

    out = {"n_seeds": a.seeds, "train_lengths": a.train_lengths, "test_lengths": TL, "results": agg}
    os.makedirs("results", exist_ok=True)
    with open("results/torus.json", "w") as f:
        json.dump(out, f, indent=2)
    svg_torus(agg, TL, "results/torus.svg")
    print("\nwrote results/torus.json and results/torus.svg", flush=True)


COLORS = {"grid (periodic)": "#e6550d", "additive (cumsum)": "#3b528b", "NoPE+sum Transformer": "#21908c",
          "place (Euclidean)": "#9aa5b8", "oracle": "#c9a227"}


def svg_torus(agg, Ts, out):
    pad = 58; pw = 330; ph = 240
    W = pad + pw + 210; H = pad + ph + 60
    ymax = max(agg[nm][T]["ang_err"]["mean"] for nm in agg for T in Ts) * 1.1 + 1e-6
    def X(T): return pad + Ts.index(T) / (len(Ts) - 1) * pw
    def Y(v): return (pad + 22) + ph - min(v, ymax) / ymax * ph
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="28" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Torus path integration: periodicity is necessary where the additive integrator fails</text>')
    e.append('<text x="28" y="44" font-size="10.5" fill="#5b6b8c">toroidal position error vs path length '
             '(many wraps) &#183; a world with no Euclidean text prior &#183; mean &#177; 95% CI</text>')
    oy = pad + 22
    e.append(f'<line x1="{pad}" y1="{oy+ph}" x2="{pad+pw}" y2="{oy+ph}" stroke="#33415c"/>'
             f'<line x1="{pad}" y1="{oy}" x2="{pad}" y2="{oy+ph}" stroke="#33415c"/>')
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        vv = frac * ymax
        e.append(f'<line x1="{pad}" y1="{Y(vv):.1f}" x2="{pad+pw}" y2="{Y(vv):.1f}" stroke="#eef2f8"/>')
        e.append(f'<text x="{pad-7}" y="{Y(vv)+4:.1f}" font-size="9" fill="#5b6b8c" text-anchor="end">{vv:.1f}</text>')
    for T in Ts:
        e.append(f'<text x="{X(T):.1f}" y="{oy+ph+14:.1f}" font-size="9" fill="#5b6b8c" text-anchor="middle">T={T}</text>')
    for nm, col in COLORS.items():
        band_t = " ".join(f"{X(T):.1f},{Y(agg[nm][T]['ang_err']['mean']+agg[nm][T]['ang_err']['ci95']):.1f}" for T in Ts)
        band_b = " ".join(f"{X(T):.1f},{Y(agg[nm][T]['ang_err']['mean']-agg[nm][T]['ang_err']['ci95']):.1f}" for T in reversed(Ts))
        e.append(f'<polygon points="{band_t} {band_b}" fill="{col}" opacity="0.12"/>')
        pts = " ".join(f"{X(T):.1f},{Y(agg[nm][T]['ang_err']['mean']):.1f}" for T in Ts)
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.3"/>')
        for T in Ts:
            e.append(f'<circle cx="{X(T):.1f}" cy="{Y(agg[nm][T]["ang_err"]["mean"]):.1f}" r="3" fill="{col}"/>')
    ly = oy + 6
    for nm, col in COLORS.items():
        e.append(f'<rect x="{pad+pw+12}" y="{ly}" width="14" height="5" fill="{col}"/>')
        e.append(f'<text x="{pad+pw+30}" y="{ly+6}" font-size="10.5" fill="#28324a">{nm}</text>'); ly += 19
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
