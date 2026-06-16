"""
src/eval/phase_diagram.py

THE SYNTHESIS — "when does each inductive bias win?" Reframes the whole characterization from a
deflation ("grid isn't uniquely necessary") into a predictive map: a regime x code matrix, assembled
from the committed result JSONs, marking for each regime whether the periodic grid code WINS, TIES,
or LOSES against a bounded place tiling and an additive integrator (NoPE+sum / raw cumsum).

The story the matrix tells:
  - grid WINS where periodicity / range / pattern-separation is load-bearing: cyclic (non-Euclidean)
    worlds, one-shot capacity, multi-map storage without an external context label.
  - grid TIES the additive integrator where a plain integration bias suffices: Euclidean magnitude
    extrapolation, multi-map WITH a context label, neural noise.
  - grid LOSES where a low-dimensional code is easier to read: the very-low-data regime.

Reads results/{extrapolation,seq_baselines,code_necessity,multimap_task,torus,frontier_probes}.json
(run those first / `bash reproduce_all.sh`). Writes results/phase_diagram.json + .svg. No new training.

    python -m src.eval.phase_diagram
"""
import json
import os

CODES = ["grid (periodic)", "place (bounded)", "additive integrator"]


def _load(name):
    p = f"results/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def build():
    ex = _load("extrapolation"); sq = _load("seq_baselines"); cn = _load("code_necessity")
    mm = _load("multimap_task"); to = _load("torus"); fr = _load("frontier_probes")
    rows = []

    def row(regime, rule, better, g, p, a, note=""):
        rows.append({"regime": regime, "rule": rule, "better": better,
                     "grid (periodic)": g, "place (bounded)": p, "additive integrator": a, "note": note})

    # Euclidean magnitude extrapolation at 4x (distance exact-acc, higher=better)
    if ex and sq:
        g = ex["results"]["grid"]["48"]["distance_exact_acc"]["mean"]
        p = ex["results"]["place"]["48"]["distance_exact_acc"]["mean"]
        a = sq["results"]["xf-NoPE (sum)"]["48"]["distance_exact_acc"]["mean"]
        row("Euclidean extrapolation (4x length)", "additive prior suffices; grid ties it, beats place",
            "high", g, p, a)
    # Cyclic (torus), many wraps — within-45deg (higher=better)
    if to:
        r = to["results"]; T = str(max(to["test_lengths"]))
        row("Cyclic world (torus, many wraps)", "periodicity NECESSARY; additive/place collapse to chance",
            "high", r["grid (periodic)"][T]["within_45deg"]["mean"],
            r["place (Euclidean)"][T]["within_45deg"]["mean"],
            r["NoPE+sum Transformer"][T]["within_45deg"]["mean"], note="leakage-proof")
    # One-shot capacity at K=200 (recall, higher=better)
    if cn:
        cap = cn["capacity"]
        row("One-shot memory capacity (200 items)", "population code NEEDED; raw 2-D collapses",
            "high", cap["grid (population)"]["200"]["mean"], cap["place (population)"]["200"]["mean"],
            cap["additive (raw 2-D)"]["200"]["mean"])
        mmap = cn["multimap"]; M = "16"
        row("Multi-map, NO context label (16 maps)", "remapping NEEDED; any deterministic metric collides",
            "high", mmap["grid + remap"][M]["mean"], mmap["place + remap"][M]["mean"],
            mmap["additive (raw 2-D)"][M]["mean"])
    # Multi-map WITH a trained context label (recall, higher=better)
    if mm:
        r = mm["results"]; M = "16"
        row("Multi-map, WITH context label (16 rooms)", "label substitutes for remapping; tie",
            "high", r["grid + remap"][M]["mean"], None, r["additive (raw 2-D)"][M]["mean"])
    # Very low data (sample efficiency, N=16) — distance acc (higher=better)
    if fr:
        se = fr["sample_efficiency"]; N = "16"
        row("Very low data (16 train trajectories)", "low-dim code reads more easily; grid LOSES",
            "high", se["grid (fixed prior)"][N]["mean"], se["place (fixed prior)"][N]["mean"],
            se["NoPE+sum Transformer"][N]["mean"])
        nz = fr["noise"]; s = "0.4"
        row("Heavy integration noise (sigma=0.4)", "noise hits all equally; tie",
            "high", nz["grid (fixed prior)"][s]["mean"], nz["place (fixed prior)"][s]["mean"],
            nz["NoPE+sum Transformer"][s]["mean"])
    return rows


def classify(rowvals):
    """Per regime, label each code WIN / TIE / LOSE (all metrics 'higher=better'). If the codes barely
    differ (spread < 0.10) the regime does not discriminate -> everyone TIES (honest: e.g. heavy noise,
    where all codes fail ~equally; or a label that makes remapping unnecessary)."""
    vals = {c: rowvals[c] for c in CODES if rowvals[c] is not None}
    best, worst = max(vals.values()), min(vals.values())
    out = {}
    no_discrim = (best - worst) < 0.10
    for c in CODES:
        v = rowvals[c]
        if v is None:
            out[c] = ("n/a", None)
        elif no_discrim:
            out[c] = ("tie", v)
        elif v >= best - 0.05:
            out[c] = ("win", v)
        elif v >= best - 0.15:
            out[c] = ("tie", v)
        else:
            out[c] = ("lose", v)
    return out


COLOR = {"win": "#2ca25f", "tie": "#d9b400", "lose": "#c9341a", "n/a": "#d7dde6"}


def main():
    rows = build()
    matrix = []
    for r in rows:
        cls = classify(r)
        matrix.append({"regime": r["regime"], "rule": r["rule"], "note": r["note"],
                       "cells": {c: {"outcome": cls[c][0], "value": cls[c][1]} for c in CODES}})
    grid_wins = [m["regime"] for m in matrix if m["cells"]["grid (periodic)"]["outcome"] == "win"]
    print("PHASE DIAGRAM — when does each inductive bias win?\n" + "=" * 70, flush=True)
    for m in matrix:
        cs = "  ".join(f"{c.split()[0]}:{m['cells'][c]['outcome']}" for c in CODES)
        print(f"  {m['regime']:42} {cs}", flush=True)
    print(f"\n  grid's win-regions: {grid_wins}", flush=True)
    out = {"codes": CODES, "matrix": matrix, "grid_win_regions": grid_wins}
    os.makedirs("results", exist_ok=True)
    with open("results/phase_diagram.json", "w") as f:
        json.dump(out, f, indent=2)
    svg(matrix, "results/phase_diagram.svg")
    print("\nwrote results/phase_diagram.json and results/phase_diagram.svg", flush=True)


def svg(matrix, out):
    rh = 46; colw = 150; labelw = 330; pad = 16; top = 92
    W = pad + labelw + len(CODES) * colw + pad
    H = top + rh * len(matrix) + 40
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="16" y="26" font-size="16" font-weight="800" fill="#0b1324">'
             'When does each inductive bias win?</text>')
    e.append('<text x="16" y="45" font-size="11" fill="#5b6b8c">green = wins &#183; yellow = ties &#183; '
             'red = loses &#183; assembled from the committed multi-seed results</text>')
    for j, c in enumerate(CODES):
        cx = pad + labelw + j * colw + colw / 2
        e.append(f'<text x="{cx:.0f}" y="{top-10}" font-size="11.5" font-weight="700" fill="#28324a" '
                 f'text-anchor="middle">{c}</text>')
    for i, m in enumerate(matrix):
        y = top + i * rh
        e.append(f'<text x="{pad}" y="{y+rh/2-2:.0f}" font-size="10.6" font-weight="600" fill="#0b1324">{m["regime"]}</text>')
        e.append(f'<text x="{pad}" y="{y+rh/2+12:.0f}" font-size="8.8" fill="#5b6b8c">{m["rule"]}</text>')
        for j, c in enumerate(CODES):
            cell = m["cells"][c]; col = COLOR[cell["outcome"]]
            x = pad + labelw + j * colw
            e.append(f'<rect x="{x+6:.0f}" y="{y+5:.0f}" width="{colw-12}" height="{rh-10}" rx="6" fill="{col}" opacity="0.82"/>')
            txt = cell["outcome"].upper() if cell["outcome"] != "n/a" else "n/a"
            val = f' {cell["value"]:.0%}' if cell["value"] is not None else ""
            e.append(f'<text x="{x+colw/2:.0f}" y="{y+rh/2+4:.0f}" font-size="11" font-weight="700" '
                     f'fill="#ffffff" text-anchor="middle">{txt}{val}</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
