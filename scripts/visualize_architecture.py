"""
scripts/visualize_architecture.py

Render a layer-by-layer diagram of Spatial-LLM / TrajectoryLLM to an SVG, driven by a REAL
forward pass through a (briefly self-supervised) TrajectoryCortex so the activations shown are
genuine: the walked path, the per-step velocity code, the continuous-attractor bump moving
across the 16x16 sheet, and the place- vs grid-cell self-supervised targets.

    python -m scripts.visualize_architecture            # -> results/architecture.svg

No matplotlib/graphviz needed — emits SVG primitives directly.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neuro.trajectory_cortex import TrajectoryCortex
from src.data.trajectory_qa import make_trajectory_qa


# ----------------------------------------------------------------- forward-pass capture
def capture(seed=0, T=8):
    torch.manual_seed(seed)
    cx = TrajectoryCortex(embed_dim=64, task="pathint", length_norm=False)
    # quick self-supervised GRID-code pre-train so the bump/readout are meaningful
    gg = torch.Generator().manual_seed(0); m = 64
    periods = torch.exp(torch.rand(m, generator=gg) * (math.log(40) - math.log(0.8)) + math.log(0.8))
    dirs = torch.randn(m, 3, generator=gg); dirs = dirs / dirs.norm(dim=1, keepdim=True)
    freqs = dirs * (2 * math.pi / periods).unsqueeze(1)
    head = nn.Linear(64, 2 * m)
    opt = torch.optim.Adam(list(cx.parameters()) + list(head.parameters()), lr=3e-3); mse = nn.MSELoss()
    for _ in range(40):
        for Tt in (6, 8, 10, 12):
            H, S, V, _ = make_trajectory_qa(256, T=Tt, seed=Tt)
            pos = torch.stack([(S * H.cos()).sum(1), (S * H.sin()).sum(1), V.sum(1)], -1)
            proj = pos @ freqs.t()
            opt.zero_grad(); h = cx.encode(H, S, V)
            mse(head(h), torch.cat([proj.sin(), proj.cos()], -1)).backward(); opt.step()
    cx.eval()

    # one example path (an out-and-back-ish loop so the bump visibly travels then returns)
    H, S, V, ans = make_trajectory_qa(1, T=T, seed=3, task="distance")
    with torch.no_grad():
        step = (cx.conjunctive(H.reshape(T), S.reshape(T)).view(1, T, -1)
                + cx.vert(V.reshape(T, 1)).view(1, T, -1))          # (1,T,64) velocity code
        # replicate the attractor integrator, capturing the 16x16 sheet u at each step
        u = torch.zeros(1, cx.integrator.N)
        sheets = []
        for t in range(T):
            u = u + cx.integrator.vel_to_sheet(step[:, t])
            for _ in range(cx.integrator.settle):
                u = u + 0.1 * F.linear(torch.tanh(u), cx.integrator.W)
            sheets.append(u.detach().view(16, 16).clone())
        h = cx.encode(H, S, V)                                       # (1,64) spatial summary
        pos = torch.stack([(S * H.cos()).sum(1), (S * H.sin()).sum(1), V.sum(1)], -1)[0]
        # the two self-supervised TARGETS for that endpoint
        cg = torch.Generator().manual_seed(0)
        centers = torch.rand(64, 3, generator=cg) * 8 - 4
        place = torch.exp(-((pos.unsqueeze(0) - centers) ** 2).sum(-1) / (2 * 1.2 ** 2))   # (64,)
        gp = pos @ freqs.t(); grid = torch.cat([gp.sin(), gp.cos()])                        # (128,)

    # 2D trajectory (cumulative x,y) for the input thumbnail
    xs = torch.cumsum(S[0] * H[0].cos(), 0); ys = torch.cumsum(S[0] * H[0].sin(), 0)
    traj = torch.stack([torch.cat([torch.zeros(1), xs]), torch.cat([torch.zeros(1), ys])], -1)  # (T+1,2)
    return dict(T=T, step=step[0], sheets=sheets, place=place, grid=grid[:64], traj=traj, ans=ans[0])


# ------------------------------------------------------------------------- svg helpers
def _cmap(v):
    stops = [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 144, 141)),
             (0.75, (94, 201, 98)), (1.0, (253, 231, 37))]
    v = max(0.0, min(1.0, float(v)))
    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        if v <= b[0]:
            f = (v - a[0]) / (b[0] - a[0] + 1e-9)
            c = [round(a[1][k] + f * (b[1][k] - a[1][k])) for k in range(3)]
            return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
    return "#fde725"


def _norm(t):
    t = t.float(); lo, hi = t.min().item(), t.max().item()
    return (t - lo) / (hi - lo + 1e-9)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, title, lines, fill, stroke="#33415c", title_fill="#0b1324"):
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" '
         f'stroke="{stroke}" stroke-width="1.5"/>')
    s += f'<text x="{x+14}" y="{y+24}" font-size="16" font-weight="700" fill="{title_fill}">{esc(title)}</text>'
    for i, ln in enumerate(lines):
        s += f'<text x="{x+14}" y="{y+46+i*18}" font-size="12.5" fill="#28324a">{esc(ln)}</text>'
    return s


def arrow(x1, y1, x2, y2, label=None, color="#5b6b8c"):
    s = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.2" marker-end="url(#ah)"/>'
    if label:
        s += (f'<text x="{(x1+x2)/2+7}" y="{(y1+y2)/2-3}" font-size="11" fill="#5b6b8c" '
              f'font-style="italic">{esc(label)}</text>')
    return s


def heat(x, y, arr2d, cell, title=None):
    s = ""
    if title:
        s += f'<text x="{x}" y="{y-6}" font-size="11.5" font-weight="600" fill="#28324a">{esc(title)}</text>'
    n = _norm(arr2d)
    rows, cols = arr2d.shape
    for r in range(rows):
        for c in range(cols):
            s += (f'<rect x="{x+c*cell}" y="{y+r*cell}" width="{cell}" height="{cell}" '
                  f'fill="{_cmap(n[r][c].item())}"/>')
    s += (f'<rect x="{x}" y="{y}" width="{cols*cell}" height="{rows*cell}" fill="none" '
          f'stroke="#33415c" stroke-width="1"/>')
    return s


def bars(x, y, vec, bw, maxh, title, color):
    s = f'<text x="{x}" y="{y-6}" font-size="11.5" font-weight="600" fill="#28324a">{esc(title)}</text>'
    n = _norm(vec)
    for i in range(len(vec)):
        bh = 3 + n[i].item() * maxh
        s += (f'<rect x="{x+i*bw}" y="{y+maxh-bh}" width="{max(bw-1,1)}" height="{bh}" fill="{color}"/>')
    s += f'<line x1="{x}" y1="{y+maxh}" x2="{x+len(vec)*bw}" y2="{y+maxh}" stroke="#33415c" stroke-width="1"/>'
    return s


def polyline_path(x, y, w, h, traj, title):
    s = f'<text x="{x}" y="{y-6}" font-size="11.5" font-weight="600" fill="#28324a">{esc(title)}</text>'
    s += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#f4f7fb" stroke="#33415c" stroke-width="1"/>'
    P = traj.clone()
    mn = P.min(0).values; mx = P.max(0).values
    rng = (mx - mn).clamp(min=1e-6)
    pts = []
    for p in P:
        px = x + 10 + (p[0] - mn[0]) / rng[0] * (w - 20)
        py = y + h - 10 - (p[1] - mn[1]) / rng[1] * (h - 20)
        pts.append((px.item(), py.item()))
    s += '<polyline points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + \
         '" fill="none" stroke="#3b528b" stroke-width="2"/>'
    s += f'<circle cx="{pts[0][0]:.1f}" cy="{pts[0][1]:.1f}" r="4" fill="#2ca25f"/>'      # start
    s += f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="4" fill="#de2d26"/>'    # end
    return s


# ------------------------------------------------------------------------- assemble svg
def build(cap, out="results/architecture.svg"):
    W, Hh = 1080, 1180
    bx, bw = 36, 520            # pipeline column
    tx = 596                    # thumbnail column
    el = []
    el.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" '
              f'viewBox="0 0 {W} {Hh}" font-family="Segoe UI, Helvetica, Arial, sans-serif">')
    el.append('<defs><marker id="ah" markerWidth="10" markerHeight="10" refX="8" refY="3" '
              'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,3 L0,6 z" fill="#5b6b8c"/></marker></defs>')
    el.append(f'<rect x="0" y="0" width="{W}" height="{Hh}" fill="#ffffff"/>')
    el.append('<text x="36" y="34" font-size="22" font-weight="800" fill="#0b1324">'
              'Spatial-LLM · TrajectoryLLM — what each layer does</text>')
    el.append('<text x="36" y="56" font-size="13" fill="#5b6b8c">A path of moves is integrated by a '
              'neuroscience cortex into a spatial code, then read by a LoRA-adapted LLM in language. '
              '❄ = frozen, \U0001f525 = trained.</text>')

    CORTEX = "#eaf3ea"; IFACE = "#eef1f8"; LLM = "#fdefea"; IO = "#fff7e6"
    rows = [
        (72,  100, "1 · INPUT — the path (self-motion)", IO,
         ["moves over time t:  heading θₜ , speed sₜ , vertical vₜ",
          "tensor (B, T, 3)   —   the ONLY spatial input; never written in text",
          "neuro: egocentric self-motion (vestibular / proprioceptive)"]),
        (196, 104, "2 · Conjunctive cells  \U0001f525  — velocity code", CORTEX,
         ["HeadDirectionCells (ring attractor, von Mises) ⊗ SpeedCells",
          "+ vert: Linear(1→64);  bind → per-step velocity  (B, T, 64)",
          "neuro: conjunctive head-direction × speed cells (entorhinal)"]),
        (322, 132, "3 · Continuous-attractor integrator  \U0001f525  — path integration", CORTEX,
         ["for each t:  u += vel_to_sheet(vₜ);  settle: u += 0.1·tanh(u)·W",
          "W = toroidal Mexican-hat (16×16 sheet);  readout(u) → (B, 64)",
          "the activity BUMP moves with velocity → its location encodes position",
          "neuro: grid-cell path integrator (medial entorhinal cortex)"]),
        (476, 96, "4 · Refinement modules  \U0001f525  (learned gates)", CORTEX,
         ["+ theta-gamma memory · cortical microcircuit · lateral inhibition",
          "each has a gate+L1: the net switches off what a task doesn't need",
          "neuro: theta-gamma (7±2 WM), L4→L2/3→L5/6 column, surround inhibition"]),
        (596, 92, "5 · LayerNorm → spatial summary  h", CORTEX,
         ["out_norm(·) → h  (B, 64)  — the cortex's integrated rep of the path",
          "stabilises scale across path lengths (magnitude lives in the PATTERN)",
          "cortex is PRE-TRAINED then ❄ FROZEN; the LLM only reads h"]),
        (712, 84, "6 · to_tokens  \U0001f525  — spatial tokens", IFACE,
         ["Linear(64 → 1536×8) → reshape → (B, 8, 1536)",
          "projects the 1 spatial summary into 8 tokens the LLM can attend to"]),
        (812, 96, "7 · MultiScaleSpatialFusion  \U0001f525  — gated cross-attention", IFACE,
         ["×2 Flamingo-style layers: text queries spatial tokens;",
          "tanh-GATED residual (gate_init=2) → fused hidden (B, T_text, 1536)",
          "the moves enter the LLM here — only through this channel"]),
        (928, 92, "8 · Qwen2.5-1.5B + LoRA  ❄\U0001f525", LLM,
         ["frozen 1.5B transformer + LoRA adapters (q,v,o,gate,up,down)",
          "reads the fused spatial context and decodes an answer",
          "prompt holds ONLY the question — no coordinates, no moves"]),
        (1044, 96, "9 · OUTPUT — answer in language", IO,
         [f'e.g.  "Are you back?" → Yes./No.   ·   "How far?" → bucket   ·   "Which way home?" → compass',
          f'this example path → ground-truth distance answer:  "{cap["ans"]}"',
          "cortex-OFF control sits at chance → the answer rides on the spatial code"]),
    ]
    # boxes + vertical arrows
    for i, (y, h, title, fill, lines) in enumerate(rows):
        el.append(box(bx, y, bw, h, title, lines, fill))
        if i > 0:
            py = rows[i - 1][0] + rows[i - 1][1]
            el.append(arrow(bx + bw / 2, py, bx + bw / 2, y))

    # thumbnails (right column), connected with light leader lines
    def leader(y):
        el.append(f'<line x1="{bx+bw}" y1="{y}" x2="{tx-12}" y2="{y}" stroke="#c2cbe0" stroke-width="1" stroke-dasharray="3,3"/>')

    leader(122); el.append(polyline_path(tx, 84, 150, 96, cap["traj"],
              "the walked path (x,y)  — start ●  end ●"))
    leader(248); el.append(heat(tx, 214, cap["step"], 7, "velocity code  (T × 64)"))
    # attractor sheet snapshots (the centrepiece): bump at t=0, mid, mid2, last
    T = cap["T"]; idxs = [0, T // 3, 2 * T // 3, T - 1]
    sx = tx
    el.append(f'<text x="{sx}" y="{330}" font-size="11.5" font-weight="600" fill="#28324a">'
              f'attractor sheet u (16×16): the bump moves with the path</text>')
    for k, ti in enumerate(idxs):
        el.append(heat(sx + k * 116, 340, cap["sheets"][ti], 6))
        el.append(f'<text x="{sx + k*116 + 48}" y="{438}" font-size="10.5" fill="#5b6b8c" '
                  f'text-anchor="middle">t={ti}</text>')
    leader(524)
    el.append(f'<text x="{tx}" y="{500}" font-size="11.5" font-weight="600" fill="#28324a">'
              f'self-supervised TARGET of the endpoint (no coordinate labels):</text>')
    el.append(bars(tx, 520, cap["place"], 5, 46, "place code (bounded → great for direction)", "#3b528b"))
    el.append(bars(tx + 232, 520, cap["grid"], 5, 46, "grid code (periodic → carries magnitude)", "#2ca25f"))
    el.append(f'<text x="{tx}" y="{600}" font-size="11" fill="#5b6b8c" font-style="italic">'
              f'the cortex learns to predict this code; place vs grid is the magnitude-frontier fix.</text>')

    # legend for the heatmap colour scale
    lx, ly = tx, 980
    el.append(f'<text x="{lx}" y="{ly-6}" font-size="11" fill="#5b6b8c">low</text>')
    for i in range(60):
        el.append(f'<rect x="{lx+26+i*3}" y="{ly-16}" width="3" height="12" fill="{_cmap(i/59)}"/>')
    el.append(f'<text x="{lx+26+60*3+6}" y="{ly-6}" font-size="11" fill="#5b6b8c">high activity</text>')

    el.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(el))
    print(f"wrote {out}  ({W}x{Hh})")
    return out


if __name__ == "__main__":
    build(capture())
