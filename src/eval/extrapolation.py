"""
src/eval/extrapolation.py

THE CENTRAL CLAIM, isolated and made honest on CPU (the paper's Figure 1).

Claim (representation-level, language model removed): a self-supervised, velocity-driven GRID-CELL
code lets a position readout generalize to trajectories that travel FARTHER than anything seen in
training, where the bounded "place" population a conventional model would use saturates. We strip
away Qwen and test the REPRESENTATION directly, so any effect is attributable to the code.

Setup (faithful to src/data/trajectory_qa.py, 2-D xy so codes are directly comparable): an agent
random-walks (per-step speed U(0.2,0.8), uniform heading), so displacement grows ~sqrt(T) and longer
paths reach LARGER displacements. We train a position readout on MIXED SHORT lengths {6,8,10,12}
(scale-free, no /T) and test out to 4-6x longer. From the single decoded displacement we derive the
three trajectory-QA tasks (return / distance bucket / bearing) — exactly what the LLM must do.

FAIRNESS — the crux: the place cells tile EXACTLY the region the TRAINING displacements occupy
(data-driven cover), i.e. the model gets place cells everywhere it was trained. Longer test paths
then reach BEYOND that trained box. This is the honest extrapolation question: "you have units where
you've been; what happens past there?" (An over-sized place grid that pre-tiles the test range hides
the effect — and is exactly the trap a careless benchmark falls into.)

Four representations, all with the SAME data, SAME scale-free mixed-length training, SAME-capacity
256-unit readout — only the CODE differs:
  - grid   : velocity-driven hexagonal grid modules (ours; phase = gain*integral(v), periodic →
             unbounded metric range with a fixed cell budget).
  - place  : Gaussian place cells tiling the TRAINED region (fixed centers; no cells past there).
  - gru    : a learned GRU path-integrator over the velocity sequence (the standard deep baseline,
             Banino 2018).
  - oracle : readout fed the EXACT displacement (perfect integration) — the ceiling; shows the task
             is solvable at every length, so any gap is the CODE.

Honest expectation: in-range a dense place code is very precise; OUT of its trained box it cliffs,
while the grid code (and, more gradually, the GRU) keep resolving — grid cells trade a little local
precision for metric RANGE, the grid/place division of labour. Bearing is scale-free so all codes do
well; DISTANCE (magnitude) is the discriminator.

Reports mean +/- 95% CI over n seeds; writes results/extrapolation.json + results/extrapolation.svg.

    python -m src.eval.extrapolation --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neuro.trajectory_cortex import _HexGridModules

RETURN_TOL = 0.5            # matches trajectory_qa
DIST_CAP = 5               # distance bucket saturates at 5+


def make_batch(n, T, gen):
    """Random walk faithful to trajectory_qa._walk (2-D). Returns velocity seq (n,T,2) and final
    displacement (n,2)."""
    h = torch.rand(n, T, generator=gen) * 2 * math.pi
    s = torch.rand(n, T, generator=gen) * 0.6 + 0.2
    v = torch.stack([s * h.cos(), s * h.sin()], -1)         # (n,T,2)
    return v, v.sum(1)


def head(fin):
    return nn.Sequential(nn.Linear(fin, 256), nn.ReLU(), nn.Linear(256, 2))


class GridRep(nn.Module):
    """Velocity-driven hexagonal grid code (fixed gains) -> learned readout. The integrated grid
    phase equals gain*displacement, so we evaluate the grid code analytically at the displacement."""
    def __init__(self, train_cover=None):
        super().__init__()
        self.cx = _HexGridModules(64, n_modules=6, base_spacing=1.6)
        for p in self.cx.parameters():
            p.requires_grad_(False)
        self.head = head(self.cx.K * self.cx.M)

    def forward(self, v, disp):
        phi = self.cx.gains.view(-1, 1, 1) * disp.unsqueeze(0)         # (K,B,2)
        return self.head(self.cx._grid_code(phi))


class PlaceRep(nn.Module):
    """Gaussian place cells tiling EXACTLY the trained displacement region (data-driven cover)."""
    def __init__(self, train_cover, n_side=20):
        super().__init__()
        xs = torch.linspace(-train_cover, train_cover, n_side)
        gx, gy = torch.meshgrid(xs, xs, indexing="ij")
        self.register_buffer("centers", torch.stack([gx.reshape(-1), gy.reshape(-1)], -1))  # (C,2)
        self.sigma = 2 * train_cover / (n_side - 1)        # ~ inter-cell spacing
        self.head = head(self.centers.shape[0])

    def forward(self, v, disp):
        d2 = ((disp.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2).sum(-1)   # (B,C)
        return self.head(torch.exp(-d2 / (2 * self.sigma ** 2)))


class GRURep(nn.Module):
    """Learned GRU path-integrator over the velocity sequence (standard deep-learning baseline)."""
    def __init__(self, train_cover=None, hidden=128):
        super().__init__()
        self.gru = nn.GRU(2, hidden, batch_first=True)
        self.head = head(hidden)

    def forward(self, v, disp):
        return self.head(self.gru(v)[0][:, -1])


class OracleRep(nn.Module):
    """Readout fed the EXACT displacement (perfect integration) — the ceiling."""
    def __init__(self, train_cover=None):
        super().__init__()
        self.head = head(2)

    def forward(self, v, disp):
        return self.head(disp)


REPS = {"grid": GridRep, "place": PlaceRep, "gru": GRURep, "oracle": OracleRep}


# --------------------------------------------------------- task metrics derived from a displacement
def dist_bucket(d):
    return torch.clamp(d.norm(dim=-1).round(), max=DIST_CAP)


def bearing_sector(d):                       # direction from here BACK to start (= -displacement)
    ang = torch.atan2(-d[:, 1], -d[:, 0])
    return torch.remainder((ang / (math.pi / 4)).round(), 8)


def metrics(pred, disp):
    return {
        "pos_decode_error": (pred - disp).norm(dim=-1).mean().item(),
        "distance_exact_acc": (dist_bucket(pred) == dist_bucket(disp)).float().mean().item(),
        "bearing_acc": (bearing_sector(pred) == bearing_sector(disp)).float().mean().item(),
        "return_acc": ((pred.norm(dim=-1) < RETURN_TOL) == (disp.norm(dim=-1) < RETURN_TOL)).float().mean().item(),
    }


def run_seed(seed, train_lengths, test_lengths, steps=600, bs=256, n_eval=4000):
    torch.manual_seed(seed)
    # data-driven place cover: the per-axis 99th-pct of TRAINING displacements (place cells exist
    # exactly where training has been). Estimated once from a large training-length sample.
    cgen = torch.Generator().manual_seed(30_000 + seed)
    tr_disp = torch.cat([make_batch(8000, T, cgen)[1] for T in train_lengths])
    train_cover = round(tr_disp.abs().quantile(0.99).item(), 3)

    egen = torch.Generator().manual_seed(90_000 + seed)
    eval_sets = {T: make_batch(n_eval, T, egen) for T in test_lengths}

    per_rep = {}
    for name, Cls in REPS.items():
        torch.manual_seed(seed)                       # identical init draw point per rep
        model = Cls(train_cover=train_cover)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
        tgen = torch.Generator().manual_seed(50_000 + seed)   # identical training stream per rep
        for step in range(steps):
            T = train_lengths[step % len(train_lengths)]      # cycle through mixed lengths
            v, disp = make_batch(bs, T, tgen)
            opt.zero_grad(); F.mse_loss(model(v, disp), disp).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            per_rep[name] = {T: metrics(model(*eval_sets[T]), eval_sets[T][1]) for T in test_lengths}
    per_rep["_train_cover"] = train_cover
    return per_rep


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float)
    n = len(vals); sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train_lengths", type=int, nargs="+", default=[6, 8, 10, 12])
    ap.add_argument("--test_lengths", type=int, nargs="+", default=[8, 16, 24, 48])
    ap.add_argument("--steps", type=int, default=600)
    a = ap.parse_args()

    print(f"REPRESENTATION-LEVEL LENGTH EXTRAPOLATION  (n={a.seeds} seeds; mean +/- 95% CI)", flush=True)
    print(f"train on lengths {a.train_lengths} (scale-free), test on {a.test_lengths}\n" + "=" * 72, flush=True)
    raw = [run_seed(s, a.train_lengths, a.test_lengths, a.steps) for s in range(a.seeds)]
    cover_m, cover_ci = ci95([r["_train_cover"] for r in raw])
    print(f"place tiling covers the trained displacement range: +/-{cover_m} per axis "
          f"(data-driven 99th pct)\n", flush=True)

    metric_keys = ["pos_decode_error", "distance_exact_acc", "bearing_acc", "return_acc"]
    agg = {name: {T: {} for T in a.test_lengths} for name in REPS}
    for name in REPS:
        for T in a.test_lengths:
            for mk in metric_keys:
                m, ci = ci95([r[name][T][mk] for r in raw])
                agg[name][T][mk] = {"mean": m, "ci95": ci}

    for mk, lab, fmt in [("pos_decode_error", "position-decode error (lower=better)", "{:.3f}"),
                         ("distance_exact_acc", "distance exact-bucket acc", "{:.0%}"),
                         ("bearing_acc", "bearing 8-way acc", "{:.0%}")]:
        print(f"[{lab}]", flush=True)
        print("  " + "rep".ljust(8) + "".join(f"T={T}".rjust(16) for T in a.test_lengths), flush=True)
        for name in REPS:
            cells = []
            for T in a.test_lengths:
                v = agg[name][T][mk]
                cells.append((fmt.format(v["mean"]) + f" +/-{fmt.format(v['ci95'])}").rjust(16))
            print("  " + name.ljust(8) + "".join(cells), flush=True)
        print("", flush=True)

    out = {"n_seeds": a.seeds, "train_lengths": a.train_lengths, "test_lengths": a.test_lengths,
           "max_train_length": max(a.train_lengths),
           "place_train_cover": {"mean": cover_m, "ci95": cover_ci}, "results": agg}
    os.makedirs("results", exist_ok=True)
    with open("results/extrapolation.json", "w") as f:
        json.dump(out, f, indent=2)
    svg_extrapolation(agg, a.test_lengths, max(a.train_lengths), "results/extrapolation.svg")
    print("wrote results/extrapolation.json and results/extrapolation.svg", flush=True)


COLORS = {"grid": "#e6550d", "place": "#3b528b", "gru": "#21908c", "oracle": "#9aa5b8"}
LABELS = {"grid": "grid code (ours)", "place": "place tiling (trained region)",
          "gru": "learned GRU integrator", "oracle": "exact-integration oracle"}


def _panel(e, agg, Ts, max_train, mk, ox, oy, pw, ph, title, ymax, pct, yticks):
    def X(T): return ox + (Ts.index(T)) / (len(Ts) - 1) * pw      # even spacing across test lengths
    def Y(v): return oy + ph - min(v, ymax) / ymax * ph
    e.append(f'<text x="{ox}" y="{oy-10}" font-size="12" font-weight="700" fill="#0b1324">{title}</text>')
    e.append(f'<line x1="{ox}" y1="{oy+ph}" x2="{ox+pw}" y2="{oy+ph}" stroke="#33415c"/>'
             f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+ph}" stroke="#33415c"/>')
    for v in yticks:
        lab = f"{int(v*100)}%" if pct else f"{v:g}"
        e.append(f'<line x1="{ox}" y1="{Y(v):.1f}" x2="{ox+pw}" y2="{Y(v):.1f}" stroke="#eef2f8"/>')
        e.append(f'<text x="{ox-7}" y="{Y(v)+4:.1f}" font-size="9" fill="#5b6b8c" text-anchor="end">{lab}</text>')
    # mark the LLM regime boundary (3x the longest training length) if it falls on a tested length
    for T in Ts:
        if T == 3 * max_train or (T == 24 and max_train == 12):
            e.append(f'<line x1="{X(T):.1f}" y1="{oy}" x2="{X(T):.1f}" y2="{oy+ph}" stroke="#c9341a" '
                     f'stroke-dasharray="4,3" stroke-width="1"/>')
            e.append(f'<text x="{X(T)+4:.1f}" y="{oy+12}" font-size="8" fill="#c9341a">3&#215; (LLM regime)</text>')
    for T in Ts:
        e.append(f'<text x="{X(T):.1f}" y="{oy+ph+14:.1f}" font-size="9" fill="#5b6b8c" text-anchor="middle">T={T}</text>')
    for name in REPS:
        col = COLORS[name]
        band_top = " ".join(f"{X(T):.1f},{Y(agg[name][T][mk]['mean']+agg[name][T][mk]['ci95']):.1f}" for T in Ts)
        band_bot = " ".join(f"{X(T):.1f},{Y(agg[name][T][mk]['mean']-agg[name][T][mk]['ci95']):.1f}" for T in reversed(Ts))
        e.append(f'<polygon points="{band_top} {band_bot}" fill="{col}" opacity="0.13"/>')
        pts = " ".join(f"{X(T):.1f},{Y(agg[name][T][mk]['mean']):.1f}" for T in Ts)
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        for T in Ts:
            e.append(f'<circle cx="{X(T):.1f}" cy="{Y(agg[name][T][mk]["mean"]):.1f}" r="3" fill="{col}"/>')


def svg_extrapolation(agg, Ts, max_train, out):
    pad = 56; pw = 300; ph = 220; gap = 90
    W = pad + pw + gap + pw + pad; H = pad + ph + 90
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="28" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Length extrapolation: the grid code holds where a bounded place code cliffs</text>')
    e.append('<text x="28" y="43" font-size="11" fill="#5b6b8c">trained on mixed short paths (&#8804;12); '
             'place cells tile only the trained region &#183; mean &#177; 95% CI</text>')
    pe_ymax = _ymax(agg, Ts, "pos_decode_error")
    _panel(e, agg, Ts, max_train, "pos_decode_error", pad, pad + 24, pw, ph,
           "position-decode error (lower = better)", pe_ymax, False, _ticks(pe_ymax))
    _panel(e, agg, Ts, max_train, "distance_exact_acc", pad + pw + gap, pad + 24, pw, ph,
           "distance exact-bucket accuracy", 1.0, True, [0.0, 0.25, 0.5, 0.75, 1.0])
    ly = pad + ph + 58; lx = pad
    for name in REPS:
        e.append(f'<rect x="{lx}" y="{ly-9}" width="16" height="5" fill="{COLORS[name]}"/>')
        e.append(f'<text x="{lx+21}" y="{ly-4}" font-size="10.5" fill="#28324a">{LABELS[name]}</text>')
        lx += 30 + int(6.6 * len(LABELS[name]))
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


def _ymax(agg, Ts, mk):
    hi = max(agg[n][T][mk]["mean"] + agg[n][T][mk]["ci95"] for n in REPS for T in Ts)
    return (math.ceil(hi * 1.1 * 2) / 2) or 1.0


def _ticks(ymax):
    step = 0.5 if ymax <= 2 else 1.0 if ymax <= 5 else 2.0
    n = int(ymax / step); return [round(i * step, 2) for i in range(n + 1)]


if __name__ == "__main__":
    main()
