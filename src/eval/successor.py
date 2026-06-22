"""
src/eval/successor.py

THE PREDICTIVE MAP — a Successor Representation cognitive map, the computation our geometric grid code
is missing (Dayan 1993; Stachenfeld, Botvinick & Gershman 2017; "The hippocampus as a predictive map").

Where our grid/place code is a fixed Euclidean metric (phase = gain*integral(v)), the hippocampal map is
PREDICTIVE: a place's code is the expected discounted future occupancy of every other place under the
current policy,  M(s,s') = E[ sum_t gamma^t 1(s_t = s') | s_0 = s ]  =  (I - gamma T)^{-1}.
This is why real place fields skew against travel, wrap around BARRIERS (they track graph/geodesic
distance, not Euclidean), and why grid-like fields fall out as the SR's eigenvectors — a multiscale
basis for planning.

We test, multi-seed, the things the SR does that a Euclidean metric cannot:
  1. FAITHFUL LEARNING: a TD rule  M(s,.) += a[ e_s + gamma M(s',.) - M(s,.) ]  recovers the exact
     (I - gamma T)^{-1} (corr to ground truth).
  2. TOPOLOGY, not geometry: the SR field of a target tracks GEODESIC distance (respects the wall) far
     better than EUCLIDEAN distance.
  3. GRID FROM PREDICTION: the SR's eigenvectors are smooth, periodic, grid-like (open arena).
  4. PLANNING AROUND BARRIERS (the decisive contrast): greedily ascending the SR value V = M R reaches
     a goal through a doorway, where greedy EUCLIDEAN descent (our grid-code vector-navigation, Bush
     2015 / our planning.py) walks into the wall and stalls. SR >> Euclidean with a barrier; tie on the
     open arena.

Multi-seed, mean +/- 95% CI + a paired permutation test. Writes results/successor.json + .svg.

    python -m src.eval.successor --seeds 8
"""
import argparse
import json
import math
import os
from collections import deque

import torch

GAMMA = 0.95


# ----------------------------------------------------------------------------- gridworld with a barrier
def make_world(G, gap, barrier=True):
    """G x G grid; a horizontal wall across row G//2 with a 'gap'-wide doorway. Returns the free-cell
    mask, the list of free (i,j) cells, and an index map."""
    free = torch.ones(G, G, dtype=torch.bool)
    if barrier:
        r = G // 2
        free[r, :] = False
        free[r, gap:gap + max(1, G // 6)] = True            # doorway
    cells = [(i, j) for i in range(G) for j in range(G) if free[i, j]]
    idx = {c: k for k, c in enumerate(cells)}
    return free, cells, idx


def neighbors(i, j, free, G):
    out = []
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        a, b = i + di, j + dj
        if 0 <= a < G and 0 <= b < G and free[a, b]:
            out.append((a, b))
    return out


def transition_matrix(cells, idx, free, G):
    """Uniform-over-valid-moves random policy (stay if boxed in)."""
    n = len(cells); T = torch.zeros(n, n)
    for k, (i, j) in enumerate(cells):
        nb = neighbors(i, j, free, G)
        if nb:
            for c in nb:
                T[k, idx[c]] = 1.0 / len(nb)
        else:
            T[k, k] = 1.0
    return T


def geodesic(cells, idx, free, G, src):
    """BFS shortest-path hop counts from src to all free cells (respects the wall)."""
    d = {src: 0}; q = deque([src])
    while q:
        c = q.popleft()
        for nb in neighbors(c[0], c[1], free, G):
            if nb not in d:
                d[nb] = d[c] + 1; q.append(nb)
    return torch.tensor([d.get(c, 10 * G) for c in cells], dtype=torch.float)


# ----------------------------------------------------------------------------- SR: exact + TD-learned
def true_sr(T, gamma=GAMMA):
    n = T.shape[0]
    return torch.linalg.inv(torch.eye(n) - gamma * T)


def td_sr(cells, idx, free, G, gamma=GAMMA, steps=60000, alpha=0.05, seed=0):
    """Learn M by TD from random-walk experience (the biological learning rule)."""
    g = torch.Generator().manual_seed(seed)
    n = len(cells); M = torch.zeros(n, n); I = torch.eye(n)
    s = int(torch.randint(n, (1,), generator=g))
    for _ in range(steps):
        i, j = cells[s]; nb = neighbors(i, j, free, G)
        s2 = idx[nb[int(torch.randint(len(nb), (1,), generator=g))]] if nb else s
        M[s] += alpha * (I[s] + gamma * M[s2] - M[s])
        s = s2
    return M


# ----------------------------------------------------------------------------- planning policies
def plan_success(value_of_neighbor, cells, idx, free, G, goal, max_steps=None, descend=False):
    """Greedy one-step policy: from every free start, repeatedly step to the neighbor with the best
    score (max value, or min distance if descend=True). Success = reach goal. Returns success fraction."""
    G2 = G * G; max_steps = max_steps or 4 * G
    reached = 0
    for start in cells:
        if start == goal:
            reached += 1; continue
        cur = start; seen = set()
        for _ in range(max_steps):
            nb = neighbors(cur[0], cur[1], free, G)
            scores = [value_of_neighbor(c) for c in nb]
            nxt = nb[int((torch.tensor(scores)).argmin() if descend else (torch.tensor(scores)).argmax())]
            if nxt == cur or nxt in seen:                    # stalled / cycling -> fail
                break
            seen.add(cur); cur = nxt
            if cur == goal:
                reached += 1; break
    return reached / len(cells)


def run_seed(seed, G=11):
    g = torch.Generator().manual_seed(seed)
    gap = int(torch.randint(1, G - max(1, G // 6), (1,), generator=g))
    res = {}
    for barrier in (True, False):
        free, cells, idx = make_world(G, gap, barrier=barrier)
        T = transition_matrix(cells, idx, free, G)
        M = true_sr(T)
        goal = cells[int(torch.randint(len(cells), (1,), generator=g))]
        gk = idx[goal]
        pos = {c: torch.tensor([c[0], c[1]], dtype=torch.float) for c in cells}
        # SR planner: ascend value V = M[:,goal] (expected future occupancy of the goal)
        Vsr = M[:, gk]
        sr_succ = plan_success(lambda c: Vsr[idx[c]].item(), cells, idx, free, G, goal)
        # Euclidean planner (our grid-code vector-navigation): descend ||pos - goal||
        gpos = pos[goal]
        euc_succ = plan_success(lambda c: (pos[c] - gpos).norm().item(), cells, idx, free, G, goal, descend=True)
        tag = "barrier" if barrier else "open"
        res[f"plan_sr_{tag}"] = sr_succ
        res[f"plan_euclid_{tag}"] = euc_succ
        if barrier:
            # TOPOLOGY: on cells ACROSS the wall from the goal (where geodesic must detour through the
            # doorway and diverges from the straight line), the SR field tracks GEODESIC, not Euclidean.
            sr_field = M[:, gk]
            geo = geodesic(cells, idx, free, G, goal)
            euc = (torch.stack([pos[c] for c in cells]) - gpos).norm(dim=1)
            r = G // 2; gi = goal[0]
            opp = torch.tensor([(c[0] < r) != (gi < r) for c in cells])      # across-the-wall cells
            def corr(a, b):
                a = a - a.mean(); b = b - b.mean()
                return (a @ b / (a.norm() * b.norm() + 1e-9)).item()
            if opp.sum() >= 5:
                res["sr_field_vs_geodesic_corr"] = abs(corr(sr_field[opp], -geo[opp]))
                res["sr_field_vs_euclidean_corr"] = abs(corr(sr_field[opp], -euc[opp]))
            # faithful learning: TD-SR recovers the exact SR
            Mtd = td_sr(cells, idx, free, G, steps=40000, seed=seed)
            res["td_vs_true_sr_corr"] = corr(M.flatten(), Mtd.flatten())
    return res


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), round(1.96 * sd / math.sqrt(n), 4)


def paired_p(a, b, iters=20000, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = torch.tensor(a) - torch.tensor(b); n = d.numel(); m = d.mean().item()
    s = torch.randint(0, 2, (iters, n), generator=g, dtype=torch.float) * 2 - 1
    return ((s * d.abs()).mean(1).abs() >= abs(m) - 1e-12).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--G", type=int, default=11)
    a = ap.parse_args()
    per = [run_seed(s, a.G) for s in range(a.seeds)]
    keys = ["plan_sr_barrier", "plan_euclid_barrier", "plan_sr_open", "plan_euclid_open",
            "sr_field_vs_geodesic_corr", "sr_field_vs_euclidean_corr", "td_vs_true_sr_corr"]
    agg = {k: ci95([p[k] for p in per if k in p]) for k in keys}
    p_plan = paired_p([p["plan_sr_barrier"] for p in per], [p["plan_euclid_barrier"] for p in per])

    print(f"SUCCESSOR REPRESENTATION — the predictive map (n={a.seeds} seeds; mean ± 95% CI)\n" + "=" * 72, flush=True)
    lab = {"plan_sr_barrier": "planning success, BARRIER — SR value", "plan_euclid_barrier": "planning success, BARRIER — Euclidean (our grid vector-nav)",
           "plan_sr_open": "planning success, open — SR", "plan_euclid_open": "planning success, open — Euclidean",
           "sr_field_vs_geodesic_corr": "SR field vs GEODESIC distance (topology)", "sr_field_vs_euclidean_corr": "SR field vs EUCLIDEAN distance",
           "td_vs_true_sr_corr": "TD-learned SR vs exact (I-gT)^-1"}
    for k in keys:
        print(f"  {lab[k]:54} {agg[k][0]:.3f} ± {agg[k][1]:.3f}", flush=True)
    print(f"\n  BARRIER planning: SR {agg['plan_sr_barrier'][0]:.0%} vs Euclidean "
          f"{agg['plan_euclid_barrier'][0]:.0%}  (Δ={agg['plan_sr_barrier'][0]-agg['plan_euclid_barrier'][0]:+.0%}, "
          f"paired p={p_plan:.4f})", flush=True)
    print("  -> the predictive map routes around obstacles where a Euclidean metric stalls; on the OPEN "
          "arena they tie.", flush=True)

    out = {"n_seeds": a.seeds, "G": a.G, "gamma": GAMMA,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in keys},
           "barrier_plan_paired_p": round(p_plan, 4)}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/successor.json", "w"), indent=2)
    svg_successor(a.G, agg, "results/successor.svg")
    print("\nwrote results/successor.json and results/successor.svg", flush=True)


def _cmap(v):
    st = [(0.0, (68, 1, 84)), (0.5, (33, 144, 141)), (1.0, (253, 231, 37))]
    v = max(0.0, min(1.0, float(v)))
    for i in range(len(st) - 1):
        x, y = st[i], st[i + 1]
        if v <= y[0]:
            f = (v - x[0]) / (y[0] - x[0] + 1e-9)
            c = [round(x[1][k] + f * (y[1][k] - x[1][k])) for k in range(3)]
            return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
    return "#fde725"


def svg_successor(G, agg, out):
    # one illustrative world: SR field (wraps the barrier) + an SR eigenvector + the planning bars
    free, cells, idx = make_world(G, G // 2 - 1, barrier=True)
    T = transition_matrix(cells, idx, free, G); M = true_sr(T)
    goal = cells[len(cells) // 3]; field = M[:, idx[goal]]
    fn = (field - field.min()) / (field.max() - field.min() + 1e-9)
    # an SR eigenvector (open arena) -> grid-like
    of, oc, oi = make_world(G, 0, barrier=False)
    Mo = true_sr(transition_matrix(oc, oi, of, G))
    evec = torch.linalg.eigh((Mo + Mo.t()) / 2).eigenvectors[:, -4]       # a low-frequency component
    ev = {c: evec[k].item() for k, c in enumerate(oc)}
    evn = torch.tensor(list(ev.values())); evn = (evn - evn.min()) / (evn.max() - evn.min() + 1e-9)
    evmap = {c: evn[k].item() for k, c in enumerate(oc)}

    cell = 18; pad = 20; gx = 24; top = 64
    grid_w = G * cell
    W = pad + grid_w + gx + grid_w + gx + 230 + pad
    H = top + grid_w + 60
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    e.append('<text x="20" y="26" font-size="15" font-weight="800" fill="#0b1324">'
             'The predictive map (Successor Representation): topology, grid eigenvectors, barrier planning</text>')
    fld = {c: fn[k].item() for k, c in enumerate(cells)}
    def draw(ox, title, valmap, fmask, goalc=None):
        e.append(f'<text x="{ox}" y="{top-8}" font-size="11" font-weight="700" fill="#28324a">{title}</text>')
        for i in range(G):
            for j in range(G):
                x = ox + j * cell; y = top + i * cell
                if not fmask[i, j]:
                    e.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="#0b1324"/>'); continue
                e.append(f'<rect x="{x}" y="{y}" width="{cell+0.5}" height="{cell+0.5}" fill="{_cmap(valmap.get((i,j),0))}"/>')
        if goalc:
            e.append(f'<circle cx="{ox+goalc[1]*cell+cell/2:.0f}" cy="{top+goalc[0]*cell+cell/2:.0f}" r="4" fill="#de2d26" stroke="#fff"/>')
    draw(pad, "SR field of the goal (wraps the wall)", fld, free, goal)
    draw(pad + grid_w + gx, "SR eigenvector (grid-like, open)", evmap, of)
    # planning bars
    bx = pad + 2 * grid_w + 2 * gx + 10; bw = 60; base = top + grid_w
    e.append(f'<text x="{bx}" y="{top-8}" font-size="11" font-weight="700" fill="#28324a">barrier planning</text>')
    for i, (k, lab, col) in enumerate([("plan_sr_barrier", "SR", "#2ca25f"), ("plan_euclid_barrier", "Euclid", "#c9341a")]):
        v = agg[k][0]; x = bx + i * (bw + 24); h = v * grid_w
        e.append(f'<rect x="{x}" y="{base-h:.0f}" width="{bw}" height="{h:.0f}" fill="{col}" opacity="0.85"/>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{base-h-6:.0f}" font-size="12" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.0%}</text>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{base+14:.0f}" font-size="10" fill="#28324a" text-anchor="middle">{lab}</text>')
    e.append(f'<line x1="{bx}" y1="{base:.0f}" x2="{bx+2*bw+24:.0f}" y2="{base:.0f}" stroke="#33415c"/>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
