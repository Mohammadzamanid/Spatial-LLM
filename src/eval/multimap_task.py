"""
src/eval/multimap_task.py

DOES THE REMAPPING NECESSITY SURVIVE IN A *LEARNED* MODEL? — CPU validation of the multi-environment
language task before spending GPU on the LLM version.

Fig-3B showed (with a one-shot Hebbian memory) that an additive/metric code collides across
environments and a remapping population code does not. But the LLM learns by gradient, with a room-id
in the *text*. So here we replace the Hebbian memory with a TRAINED classifier and give every code a
learned room embedding (the analog of the text room-id): does remapping the *spatial code* still
matter once the model can train and already knows which room it is in?

Task (the analog of "in room B, what item is at location X?"): M rooms, K recurring locations; each
(room, location) is assigned a RANDOM item label (pure factual recall — no smooth structure to
exploit, so the model must store the bindings). A shared classifier reads [cortex code, room_embed]
and predicts the item. We sweep M and measure recall accuracy.

Codes (cortex fixed; all also receive the learned room embedding, so none is denied room identity):
  - grid + remap     : code(loc + room_shift) — environment-specific (grid realignment).
  - grid, NO remap   : code(loc) — same code in every room (the metric integrator's situation).
  - additive (raw 2-D): loc — the raw displacement an additive integrator outputs.

If grid+remap >> the others as M grows, the remapping necessity survives gradient training + an
explicit room-id — i.e. it will matter for the LLM. Multi-seed, mean ± 95% CI.
Writes results/multimap_task.json + results/multimap_task.svg.

    python -m src.eval.multimap_task --seeds 5
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neuro.trajectory_cortex import _HexGridModules
from src.eval.code_necessity import grid_code, place_centers, place_code


def run_seed(seed, Ms, K=30, V=8, R=3.0, embed_dim=16, steps=400):
    g = torch.Generator().manual_seed(7000 + seed)
    cx = _HexGridModules(64, n_modules=6, base_spacing=1.6)
    gd = cx.K * cx.M
    locs = (torch.rand(K, 2, generator=g) * 2 - 1) * R              # K locations recurring in every room
    Mmax = max(Ms)
    shifts = (torch.rand(Mmax, 2, generator=g) * 2 - 1) * 6.0       # per-room grid realignment

    coders = {                                                      # (room index) -> (K, code_dim)
        "grid + remap": (lambda r: grid_code(cx, locs, shifts[r]), gd),
        "grid, no remap": (lambda r: grid_code(cx, locs, 0.0), gd),
        "additive (raw 2-D)": (lambda r: locs, 2),
    }
    out = {}
    for name, (code_of, cdim) in coders.items():
        accs = {}
        for M in Ms:
            torch.manual_seed(seed)                                 # identical init across M for this coder
            labels = torch.randint(0, V, (M, K), generator=g)       # random (room,loc) -> item
            emb = nn.Embedding(M, embed_dim)
            clf = nn.Sequential(nn.Linear(cdim + embed_dim, 128), nn.ReLU(), nn.Linear(128, V))
            opt = torch.optim.Adam(list(clf.parameters()) + list(emb.parameters()), lr=3e-3)
            codes = torch.stack([code_of(r) for r in range(M)])     # (M, K, cdim)
            rooms = torch.arange(M).unsqueeze(1).expand(M, K)        # (M, K)
            X = codes.reshape(M * K, cdim); ridx = rooms.reshape(-1); y = labels.reshape(-1)
            for _ in range(steps):
                opt.zero_grad()
                logits = clf(torch.cat([X, emb(ridx)], -1))
                F.cross_entropy(logits, y).backward(); opt.step()
            with torch.no_grad():
                pred = clf(torch.cat([X, emb(ridx)], -1)).argmax(-1)
                accs[M] = (pred == y).float().mean().item()
        out[name] = accs
    return out


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float)
    n = len(vals); sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--Ms", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    per = [run_seed(s, a.Ms) for s in seeds]
    names = list(per[0].keys())
    agg = {nm: {M: dict(zip(("mean", "ci95"), ci95([p[nm][M] for p in per]))) for M in a.Ms} for nm in names}

    print(f"MULTI-MAP FACTUAL RECALL through a trained classifier (n={a.seeds}; mean ± 95% CI)\n"
          f"  random (room, location) -> item; shared classifier + learned room-id embedding\n" + "=" * 74, flush=True)
    print("  " + "code".ljust(22) + "".join(f"M={M}".rjust(12) for M in a.Ms) + "   (recall acc)", flush=True)
    for nm in names:
        print("  " + nm.ljust(22) + "".join(f"{agg[nm][M]['mean']:.0%}±{agg[nm][M]['ci95']:.0%}".rjust(12) for M in a.Ms), flush=True)
    chance = 1.0 / 8
    print(f"  (chance = {chance:.0%}; K=30 locations/room, V=8 items)", flush=True)

    out = {"n_seeds": a.seeds, "Ms": a.Ms, "K": 30, "V": 8, "chance": chance, "results": agg}
    os.makedirs("results", exist_ok=True)
    with open("results/multimap_task.json", "w") as f:
        json.dump(out, f, indent=2)
    svg_multimap(agg, a.Ms, chance, "results/multimap_task.svg")
    print("\nwrote results/multimap_task.json and results/multimap_task.svg", flush=True)


PALETTE = {"grid + remap": "#e6550d", "grid, no remap": "#3b528b", "additive (raw 2-D)": "#9aa5b8"}


def svg_multimap(agg, Ms, chance, out):
    pad = 60; pw = 380; ph = 250
    W = pad + pw + pad + 60; H = 64 + ph + 56
    def X(M): lo, hi = math.log(Ms[0]), math.log(Ms[-1]); return pad + (math.log(M) - lo) / (hi - lo) * pw
    def Y(v): return 64 + ph - v * ph
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="28" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Multi-map factual recall: remapping is necessary even for a TRAINED model</text>')
    e.append('<text x="28" y="44" font-size="11" fill="#5b6b8c">random (room, location) &#8594; item, '
             'shared classifier + learned room-id embedding &#183; mean &#177; 95% CI</text>')
    e.append(f'<line x1="{pad}" y1="{Y(0)}" x2="{pad+pw}" y2="{Y(0)}" stroke="#33415c"/>'
             f'<line x1="{pad}" y1="64" x2="{pad}" y2="{Y(0)}" stroke="#33415c"/>')
    for vv in (0.0, 0.25, 0.5, 0.75, 1.0):
        e.append(f'<line x1="{pad}" y1="{Y(vv):.1f}" x2="{pad+pw}" y2="{Y(vv):.1f}" stroke="#eef2f8"/>')
        e.append(f'<text x="{pad-8}" y="{Y(vv)+4:.1f}" font-size="9" fill="#5b6b8c" text-anchor="end">{int(vv*100)}%</text>')
    e.append(f'<line x1="{pad}" y1="{Y(chance):.1f}" x2="{pad+pw}" y2="{Y(chance):.1f}" '
             f'stroke="#c9341a" stroke-dasharray="3,3" opacity="0.5"/>')
    e.append(f'<text x="{pad+pw}" y="{Y(chance)-3:.1f}" font-size="8.5" fill="#c9341a" text-anchor="end">chance</text>')
    for M in Ms:
        e.append(f'<text x="{X(M):.1f}" y="{Y(0)+15:.1f}" font-size="9" fill="#5b6b8c" text-anchor="middle">{M}</text>')
    e.append(f'<text x="{pad+pw/2:.0f}" y="{Y(0)+34:.0f}" font-size="10.5" fill="#28324a" text-anchor="middle"># rooms / maps M</text>')
    for nm, col in PALETTE.items():
        if nm not in agg:
            continue
        band_t = " ".join(f"{X(M):.1f},{Y(agg[nm][M]['mean']+agg[nm][M]['ci95']):.1f}" for M in Ms)
        band_b = " ".join(f"{X(M):.1f},{Y(agg[nm][M]['mean']-agg[nm][M]['ci95']):.1f}" for M in reversed(Ms))
        e.append(f'<polygon points="{band_t} {band_b}" fill="{col}" opacity="0.13"/>')
        pts = " ".join(f"{X(M):.1f},{Y(agg[nm][M]['mean']):.1f}" for M in Ms)
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        for M in Ms:
            e.append(f'<circle cx="{X(M):.1f}" cy="{Y(agg[nm][M]["mean"]):.1f}" r="3" fill="{col}"/>')
    ly = 78
    for nm, col in PALETTE.items():
        e.append(f'<rect x="{pad+pw-150}" y="{ly}" width="13" height="5" fill="{col}"/>')
        e.append(f'<text x="{pad+pw-133}" y="{ly+5}" font-size="10" fill="#28324a">{nm}</text>'); ly += 15
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
