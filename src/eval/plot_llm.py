"""
src/eval/plot_llm.py — render the multi-seed LLM transfer figure from results/extrapolation_llm.json
(produced on Kaggle by notebooks/m2_extrapolation_multiseed_kaggle.py). Honest: shows the large
seed-variability (95% CI bands) and the cortex-ON vs text-only-OFF gap.

    python -m src.eval.plot_llm
"""
import json
import os

COL = {"grid": "#e6550d", "place": "#3b528b"}


def main(path="results/extrapolation_llm.json", out="results/extrapolation_llm.svg"):
    d = json.load(open(path))
    Ts = d["eval_lengths"]; n = d["n_seeds"]
    tasks = list(d["results"].keys())
    pad = 56; pw = 300; ph = 230; gap = 96
    W = pad + pw + gap + pw + pad; H = 70 + ph + 70

    def panel(e, task, ox):
        r = d["results"][task]
        oy = 70
        def X(T): return ox + Ts.index(T) / (len(Ts) - 1) * pw
        def Y(v): return oy + ph - v * ph
        e.append(f'<text x="{ox}" y="{oy-10}" font-size="12" font-weight="700" fill="#0b1324">{task} (cortex-ON exact)</text>')
        e.append(f'<line x1="{ox}" y1="{oy+ph}" x2="{ox+pw}" y2="{oy+ph}" stroke="#33415c"/>'
                 f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+ph}" stroke="#33415c"/>')
        for vv in (0.0, 0.25, 0.5, 0.75, 1.0):
            e.append(f'<line x1="{ox}" y1="{Y(vv):.1f}" x2="{ox+pw}" y2="{Y(vv):.1f}" stroke="#eef2f8"/>')
            e.append(f'<text x="{ox-7}" y="{Y(vv)+4:.1f}" font-size="9" fill="#5b6b8c" text-anchor="end">{int(vv*100)}%</text>')
        for T in Ts:
            e.append(f'<text x="{X(T):.1f}" y="{oy+ph+14:.1f}" font-size="9" fill="#5b6b8c" text-anchor="middle">T={T}</text>')
        # text-only OFF reference (at T24) + chance
        off = r["grid"].get("off_T24")
        if off is not None:
            e.append(f'<line x1="{ox}" y1="{Y(off):.1f}" x2="{ox+pw}" y2="{Y(off):.1f}" stroke="#c9341a" '
                     f'stroke-dasharray="4,3" opacity="0.6"/>')
            e.append(f'<text x="{ox+pw}" y="{Y(off)-3:.1f}" font-size="8.5" fill="#c9341a" text-anchor="end">text-only (OFF)</text>')
        for code in ("grid", "place"):
            col = COL[code]; m = r[code]
            top = " ".join(f"{X(T):.1f},{Y(min(1,m[f'T{T}']['mean']+m[f'T{T}']['ci95'])):.1f}" for T in Ts)
            bot = " ".join(f"{X(T):.1f},{Y(max(0,m[f'T{T}']['mean']-m[f'T{T}']['ci95'])):.1f}" for T in reversed(Ts))
            e.append(f'<polygon points="{top} {bot}" fill="{col}" opacity="0.13"/>')
            pts = " ".join(f"{X(T):.1f},{Y(m[f'T{T}']['mean']):.1f}" for T in Ts)
            e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>')
            for T in Ts:
                e.append(f'<circle cx="{X(T):.1f}" cy="{Y(m[f"T{T}"]["mean"]):.1f}" r="3" fill="{col}"/>')

    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="28" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             f'Language transfer through the cortex (Qwen2.5-1.5B + LoRA, n={n} seeds, 95% CI)</text>')
    e.append('<text x="28" y="44" font-size="10.5" fill="#5b6b8c">cortex ON &#8811; text-only OFF (the LLM reads '
             'the spatial channel); grid vs place overlaps at n=3 (wide bands) &#8212; bearing trends grid-favorable</text>')
    panel(e, tasks[0], pad)
    panel(e, tasks[1], pad + pw + gap)
    ly = H - 30
    for code in ("grid", "place"):
        e.append(f'<rect x="{pad}" y="{ly}" width="14" height="5" fill="{COL[code]}"/>')
        e.append(f'<text x="{pad+19}" y="{ly+6}" font-size="10.5" fill="#28324a">{code} cortex</text>'); pad += 120
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
