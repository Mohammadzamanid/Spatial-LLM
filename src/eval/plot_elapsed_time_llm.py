"""
src/eval/plot_elapsed_time_llm.py — render the elapsed-time LLM-readout figure from
results/elapsed_time_llm.json (produced on Kaggle). cortex-ON vs text-only-OFF, EXACT and WITHIN-1,
n=6 seeds with 95% CI; the temporal analogue of the torus-QA spatial readout. A frozen LLM names how
much time has elapsed purely by reading the emergent time-cell code (elapsed time never in the prompt).

    python -m src.eval.plot_elapsed_time_llm
"""
import json
import os

COL = {"on": "#2ca25f", "off": "#9aa5b8"}


def main(path="results/elapsed_time_llm.json", out="results/elapsed_time_llm.svg"):
    d = json.load(open(path))
    n = d["n_seeds"]; chance = d["chance"]
    groups = [("exact", "EXACT bin"), ("within1", "WITHIN-1 (scalar)")]
    pad = 64; gw = 200; gap = 60; ph = 250
    W = pad + len(groups) * gw + (len(groups) - 1) * gap + 150
    H = pad + ph + 64

    def Y(v): return pad + 18 + ph - v * ph
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="28" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             f'Elapsed-time readout through a frozen LLM: cortex-ON &#8811; text-only (n={n}, 95% CI)</text>')
    e.append('<text x="28" y="44" font-size="10.5" fill="#5b6b8c">the LLM names how much time has elapsed '
             'by reading the emergent time-cell code &#183; elapsed time never in the prompt &#183; leakage-proof</text>')
    oy = pad + 18
    # axes + gridlines
    e.append(f'<line x1="{pad}" y1="{Y(0):.1f}" x2="{W-150}" y2="{Y(0):.1f}" stroke="#33415c"/>'
             f'<line x1="{pad}" y1="{oy}" x2="{pad}" y2="{Y(0):.1f}" stroke="#33415c"/>')
    for vv in (0.0, 0.25, 0.5, 0.75, 1.0):
        e.append(f'<line x1="{pad}" y1="{Y(vv):.1f}" x2="{W-150}" y2="{Y(vv):.1f}" stroke="#eef2f8"/>')
        e.append(f'<text x="{pad-8}" y="{Y(vv)+4:.1f}" font-size="9" fill="#5b6b8c" text-anchor="end">{int(vv*100)}%</text>')
    # chance line
    e.append(f'<line x1="{pad}" y1="{Y(chance):.1f}" x2="{W-150}" y2="{Y(chance):.1f}" stroke="#c9341a" '
             f'stroke-dasharray="4,3" opacity="0.5"/>')
    e.append(f'<text x="{W-152}" y="{Y(chance)-3:.1f}" font-size="8.5" fill="#c9341a" text-anchor="end">chance</text>')
    bw = 56
    for gi, (key, lab) in enumerate(groups):
        g = d[key]; gx = pad + 24 + gi * (gw + gap)
        for bi, side in enumerate(("on", "off")):
            v = g[f"{side}_mean"]; ci = g[f"{side}_ci95"]; x = gx + bi * (bw + 20)
            e.append(f'<rect x="{x}" y="{Y(v):.1f}" width="{bw}" height="{Y(0)-Y(v):.1f}" fill="{COL[side]}" opacity="0.88"/>')
            e.append(f'<line x1="{x+bw/2:.1f}" y1="{Y(min(1,v+ci)):.1f}" x2="{x+bw/2:.1f}" y2="{Y(max(0,v-ci)):.1f}" stroke="#0b1324" stroke-width="1.4"/>')
            e.append(f'<text x="{x+bw/2:.1f}" y="{Y(v)-7:.1f}" font-size="12" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.0%}</text>')
            e.append(f'<text x="{x+bw/2:.1f}" y="{Y(0)+15:.1f}" font-size="10" fill="#28324a" text-anchor="middle">{"cortex-ON" if side=="on" else "text-only"}</text>')
        e.append(f'<text x="{gx+bw+10:.0f}" y="{Y(0)+32:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{lab}</text>')
        e.append(f'<text x="{gx+bw+10:.0f}" y="{oy+12:.0f}" font-size="10" fill="#2ca25f" text-anchor="middle">'
                 f'&#916;{g["delta"]*100:+.0f} &#183; p={g["paired_p"]:.3f}</text>')
    # legend
    lx = W - 132; ly = oy + 8
    for side, txt in [("on", "cortex-ON (reads the cortex)"), ("off", "text-only (cortex ablated)")]:
        e.append(f'<rect x="{lx}" y="{ly}" width="14" height="5" fill="{COL[side]}"/>')
        e.append(f'<text x="{lx+18}" y="{ly+6}" font-size="9.5" fill="#28324a">{txt}</text>'); ly += 18
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
