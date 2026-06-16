"""
src/eval/significance.py

REVIEWER-GRADE STATISTICS — paired significance tests on every headline comparison.

Non-overlapping 95% CIs are informal; reviewers want a test. For each headline comparison we run the
two arms PAIRED on the same seeds and report, on the per-seed differences:
  - mean difference and a 95% BOOTSTRAP CI of the mean (20k resamples),
  - a two-sided SIGN-FLIP PERMUTATION p-value (exact-style, assumption-light, gold standard for paired
    data; no scipy dependency),
  - Cohen's d (paired) and the number of seeds in which arm A beats arm B.

Crucially this also certifies the HONEST NULLS (e.g. grid vs a NoPE+sum Transformer on path
integration should be NOT significant). A forest plot (results/significance.svg) shows every effect
with its CI against zero. Writes results/significance.json.

    python -m src.eval.significance --n_fast 20 --n_slow 8
"""
import argparse
import json
import math
import os

import torch

from src.eval.extrapolation import run_seed as extrap_seed
from src.eval.code_necessity import capacity_seed, remap_seed
from src.eval.stats import continual_metric, goal_metric, relational_metric
from src.eval.ablations import GridRepN, train_eval
from src.eval.seq_baselines import SeqTransformer


# ------------------------------------------------------------------ paired statistics (no scipy)
def paired_stats(a, b, iters=20000, seed=0):
    """a, b: per-seed paired measurements (arm A, arm B). Returns difference stats."""
    g = torch.Generator().manual_seed(seed)
    a = torch.tensor(a, dtype=torch.float64); b = torch.tensor(b, dtype=torch.float64)
    d = a - b; n = d.numel()
    mean = d.mean().item()
    sd = d.std(unbiased=True).item() if n > 1 else 0.0
    # 95% bootstrap CI of the mean difference
    idx = torch.randint(0, n, (iters, n), generator=g)
    boot = d[idx].mean(1)
    lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    # two-sided sign-flip permutation p-value (H0: differences symmetric about 0)
    signs = (torch.randint(0, 2, (iters, n), generator=g, dtype=torch.float64) * 2 - 1)
    perm = (signs * d.abs()).mean(1)
    p = ((perm.abs() >= abs(mean) - 1e-12).float().mean()).item()
    return {"mean_diff": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "p_perm": round(p, 4), "cohen_d": round(mean / (sd + 1e-12), 3),
            "n": int(n), "wins_A": int((d > 0).sum().item()),
            "significant": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_fast", type=int, default=20)     # analytic / light-training comparisons
    ap.add_argument("--n_slow", type=int, default=8)      # heavy: TD goal-nav + Transformer
    ap.add_argument("--raw", default="results/significance_raw.json")  # per-seed checkpoint (resumable)
    a = ap.parse_args()

    print(f"PAIRED SIGNIFICANCE — n_fast={a.n_fast}, n_slow={a.n_slow}\n" + "=" * 78, flush=True)
    arms = {}   # name -> {"labels":(A,B), "a":[...], "b":[...]}

    def add(name, la, lb):
        arms[name] = {"labels": (la, lb), "a": [], "b": []}

    # fast comparisons
    add("extrapolation distance@T24: grid vs place", "grid", "place")
    add("extrapolation distance@T24: grid vs GRU", "grid", "GRU")
    add("extrapolation bearing@T24: grid vs place", "grid", "place")
    add("multi-map@M16: grid+remap vs additive", "grid+remap", "additive")
    add("capacity@K200: population(grid) vs raw-2D", "grid", "raw-2D")
    add("continual: one-shot Hebbian vs gradient", "Hebbian", "gradient")
    add("relational: transitive inference vs chance", "TI", "chance(0.5)")
    # slow comparisons
    add("goal navigation: value vs random walker", "value", "random")
    add("extrapolation distance@T24: grid vs NoPE+sum xf (NULL)", "grid", "NoPE+sum")

    done_fast = done_slow = 0
    if os.path.exists(a.raw):                              # resume after a container reset
        ck = json.load(open(a.raw))
        for nm in arms:
            if nm in ck["arms"]:
                arms[nm]["a"] = ck["arms"][nm]["a"]; arms[nm]["b"] = ck["arms"][nm]["b"]
        done_fast, done_slow = ck["done_fast"], ck["done_slow"]
        print(f"  resumed: done_fast={done_fast}, done_slow={done_slow}", flush=True)

    def save_raw():
        os.makedirs(os.path.dirname(a.raw) or ".", exist_ok=True)
        json.dump({"done_fast": done_fast, "done_slow": done_slow,
                   "arms": {nm: {"a": arms[nm]["a"], "b": arms[nm]["b"]} for nm in arms}},
                  open(a.raw, "w"))

    # ---- fast comparisons (paired on seed), checkpointed per seed ----
    for s in range(done_fast, a.n_fast):
        ex = extrap_seed(s, [6, 8, 10, 12], [8, 16, 24, 48])
        arms["extrapolation distance@T24: grid vs place"]["a"].append(ex["grid"][24]["distance_exact_acc"])
        arms["extrapolation distance@T24: grid vs place"]["b"].append(ex["place"][24]["distance_exact_acc"])
        arms["extrapolation distance@T24: grid vs GRU"]["a"].append(ex["grid"][24]["distance_exact_acc"])
        arms["extrapolation distance@T24: grid vs GRU"]["b"].append(ex["gru"][24]["distance_exact_acc"])
        arms["extrapolation bearing@T24: grid vs place"]["a"].append(ex["grid"][24]["bearing_acc"])
        arms["extrapolation bearing@T24: grid vs place"]["b"].append(ex["place"][24]["bearing_acc"])
        rm = remap_seed(s, [16])
        arms["multi-map@M16: grid+remap vs additive"]["a"].append(rm["grid + remap"][16])
        arms["multi-map@M16: grid+remap vs additive"]["b"].append(rm["additive (raw 2-D)"][16])
        cap = capacity_seed(s, [200])
        arms["capacity@K200: population(grid) vs raw-2D"]["a"].append(cap["grid (population)"][200])
        arms["capacity@K200: population(grid) vs raw-2D"]["b"].append(cap["additive (raw 2-D)"][200])
        cm = continual_metric(s)
        arms["continual: one-shot Hebbian vs gradient"]["a"].append(cm["hebbian_recall"])
        arms["continual: one-shot Hebbian vs gradient"]["b"].append(cm["gradient_recall"])
        rel = relational_metric(s)
        arms["relational: transitive inference vs chance"]["a"].append(rel["transitive_inference_acc"])
        arms["relational: transitive inference vs chance"]["b"].append(0.5)
        done_fast = s + 1; save_raw()
        print(f"  fast seed {s} done", flush=True)

    # ---- slow comparisons (heavy): goal navigation + NoPE+sum Transformer (the honest null) ----
    for s in range(done_slow, a.n_slow):
        gm = goal_metric(s)
        arms["goal navigation: value vs random walker"]["a"].append(gm["value_nav_success"])
        arms["goal navigation: value vs random walker"]["b"].append(gm["random_nav_success"])
        gd = train_eval(lambda: GridRepN(n_modules=6), s, [6, 8, 10, 12], [24])[24]["distance_exact_acc"]
        nd = train_eval(lambda: SeqTransformer(pos="none", pool="sum"), s, [6, 8, 10, 12], [24])[24]["distance_exact_acc"]
        arms["extrapolation distance@T24: grid vs NoPE+sum xf (NULL)"]["a"].append(gd)
        arms["extrapolation distance@T24: grid vs NoPE+sum xf (NULL)"]["b"].append(nd)
        done_slow = s + 1; save_raw()
        print(f"  slow seed {s} done", flush=True)

    out = {}
    print("\n" + "-" * 78, flush=True)
    print(f"{'comparison':52} {'Δ mean [95% CI]':24} {'p':>7}  sig", flush=True)
    for name, arm in arms.items():
        st = paired_stats(arm["a"], arm["b"], seed=hash(name) & 0xffff)
        st["labels"] = list(arm["labels"]); st["mean_A"] = round(sum(arm["a"]) / len(arm["a"]), 4)
        st["mean_B"] = round(sum(arm["b"]) / len(arm["b"]), 4)
        out[name] = st
        ci = f"{st['mean_diff']:+.3f} [{st['ci95'][0]:+.3f},{st['ci95'][1]:+.3f}]"
        flag = "***" if st["significant"] else "ns "
        print(f"{name:52} {ci:24} {st['p_perm']:>7.4f}  {flag}  (d={st['cohen_d']}, {st['wins_A']}/{st['n']})", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/significance.json", "w") as f:
        json.dump({"n_fast": a.n_fast, "n_slow": a.n_slow, "comparisons": out}, f, indent=2)
    svg_forest(out, "results/significance.svg")
    print("\nwrote results/significance.json and results/significance.svg", flush=True)


def svg_forest(out, path):
    names = list(out.keys())
    rowh = 34; pad = 16; top = 64; labelw = 360; plotw = 320
    W = pad + labelw + plotw + pad; H = top + rowh * len(names) + 40
    diffs = [out[n]["mean_diff"] for n in names]
    los = [out[n]["ci95"][0] for n in names]; his = [out[n]["ci95"][1] for n in names]
    lo = min(0.0, min(los)) - 0.05; hi = max(0.0, max(his)) + 0.05
    def X(v): return pad + labelw + (v - lo) / (hi - lo) * plotw
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="16" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'Paired significance — effect size (Δ) with 95% bootstrap CI</text>')
    e.append('<text x="16" y="44" font-size="10.5" fill="#5b6b8c">CI crossing 0 = not significant (the honest '
             'null); to the right of 0 = arm A &gt; arm B</text>')
    e.append(f'<line x1="{X(0):.1f}" y1="{top-6}" x2="{X(0):.1f}" y2="{H-30}" stroke="#c9341a" stroke-dasharray="4,3"/>')
    for i, n in enumerate(names):
        y = top + i * rowh + rowh / 2
        sig = out[n]["significant"]; col = "#2ca25f" if sig else "#9aa5b8"
        short = n if len(n) <= 52 else n[:50] + "…"
        e.append(f'<text x="{pad}" y="{y+4:.1f}" font-size="10" fill="#28324a">{short}</text>')
        e.append(f'<line x1="{X(out[n]["ci95"][0]):.1f}" y1="{y:.1f}" x2="{X(out[n]["ci95"][1]):.1f}" y2="{y:.1f}" '
                 f'stroke="{col}" stroke-width="2.4"/>')
        e.append(f'<circle cx="{X(out[n]["mean_diff"]):.1f}" cy="{y:.1f}" r="3.6" fill="{col}"/>')
        e.append(f'<text x="{pad+labelw+plotw+2}" y="{y+4:.1f}" font-size="8.5" fill="#5b6b8c" text-anchor="end" '
                 f'transform="translate(0,0)">p={out[n]["p_perm"]:.3f}</text>')
    for v in (lo, 0.0, hi):
        e.append(f'<text x="{X(v):.1f}" y="{H-14:.1f}" font-size="9" fill="#5b6b8c" text-anchor="middle">{v:+.2f}</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
