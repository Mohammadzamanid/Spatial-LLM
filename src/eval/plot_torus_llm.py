"""
src/eval/plot_torus_llm.py — render the torus-QA LLM figure from results/torus_llm.json (produced on
Kaggle). cortex-ON vs text-only-OFF per path length, n=3 seeds with 95% CI bands; the leakage-proof
causal headline on a non-Euclidean world. Honest: shows the wide ON band (seed variance) and the large,
consistent ON-OFF gap.

    python -m src.eval.plot_torus_llm
"""
import json
import os

COL = {"on": "#e6550d", "off": "#9aa5b8"}


def main(path="results/torus_llm.json", out="results/torus_llm.svg"):
    d = json.load(open(path))
    rbl = d["results_by_len"]; Ts = sorted(rbl, key=int); n = d["n_seeds"]
    pad = 60; pw = 360; ph = 250; W = pad + pw + 180; H = pad + ph + 70

    def X(i): return pad + (i / (len(Ts) - 1)) * pw
    def Y(v): return pad + 18 + ph - v * ph
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="28" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             f'Torus-QA through a frozen LLM: cortex-ON &#8811; text-only (n={n} seeds, 95% CI)</text>')
    e.append('<text x="28" y="44" font-size="10.5" fill="#5b6b8c">a wrap-around world with no Euclidean '
             'text prior; moves never in the prompt &#183; leakage-proof causal readout</text>')
    oy = pad + 18
    e.append(f'<line x1="{pad}" y1="{Y(0):.1f}" x2="{pad+pw}" y2="{Y(0):.1f}" stroke="#33415c"/>'
             f'<line x1="{pad}" y1="{oy}" x2="{pad}" y2="{Y(0):.1f}" stroke="#33415c"/>')
    for vv in (0.0, 0.25, 0.5, 0.75, 1.0):
        e.append(f'<line x1="{pad}" y1="{Y(vv):.1f}" x2="{pad+pw}" y2="{Y(vv):.1f}" stroke="#eef2f8"/>')
        e.append(f'<text x="{pad-8}" y="{Y(vv)+4:.1f}" font-size="9" fill="#5b6b8c" text-anchor="end">{int(vv*100)}%</text>')
    # chance band (~1/9 to most-common-class ~0.18)
    e.append(f'<line x1="{pad}" y1="{Y(0.13):.1f}" x2="{pad+pw}" y2="{Y(0.13):.1f}" stroke="#c9341a" '
             f'stroke-dasharray="4,3" opacity="0.5"/>')
    e.append(f'<text x="{pad+pw}" y="{Y(0.13)-3:.1f}" font-size="8.5" fill="#c9341a" text-anchor="end">chance</text>')
    for i, T in enumerate(Ts):
        tag = "train" if T == "8" else "extrap."
        e.append(f'<text x="{X(i):.1f}" y="{Y(0)+15:.1f}" font-size="9.5" fill="#5b6b8c" text-anchor="middle">T={T} ({tag})</text>')
    for key, lab in [("on", "cortex-ON"), ("off", "text-only OFF")]:
        col = COL[key]
        pts = " ".join(f"{X(i):.1f},{Y(rbl[T][f'{key}_mean']):.1f}" for i, T in enumerate(Ts))
        top = " ".join(f"{X(i):.1f},{Y(min(1,rbl[T][f'{key}_mean']+rbl[T][f'{key}_ci95'])):.1f}" for i, T in enumerate(Ts))
        bot = " ".join(f"{X(i):.1f},{Y(max(0,rbl[T][f'{key}_mean']-rbl[T][f'{key}_ci95'])):.1f}" for i, T in reversed(list(enumerate(Ts))))
        e.append(f'<polygon points="{top} {bot}" fill="{col}" opacity="0.14"/>')
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.6"/>')
        for i, T in enumerate(Ts):
            e.append(f'<circle cx="{X(i):.1f}" cy="{Y(rbl[T][f"{key}_mean"]):.1f}" r="3.5" fill="{col}"/>')
    ly = oy + 8
    for key, lab in [("on", "cortex-ON (reads the cortex)"), ("off", "text-only (cortex ablated)")]:
        e.append(f'<rect x="{pad+pw+12}" y="{ly}" width="14" height="5" fill="{COL[key]}"/>')
        e.append(f'<text x="{pad+pw+30}" y="{ly+6}" font-size="10" fill="#28324a">{lab}</text>'); ly += 20
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
