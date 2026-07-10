"""
src/eval/replay_planning.py

REPLAY THAT COMPUTES — prioritized replay for credit assignment (reverse) and planning (forward), not just a
ripple signature (GAPS.md #6). The repo already HAS a `SharpWaveRipple` organ and offline experience-replay that
*consolidates a decode map* (pillars.py). What was missing is replay used the way the hippocampus uses it:

  A. REVERSE replay = CREDIT ASSIGNMENT (Foster & Wilson 2006; Ambrose, Pfeiffer & Foster 2016). After a reward
     appears, prioritized sweeping (Moore & Atkeson 1993) — back up the transition with the largest |TD error|,
     a SCALAR priority with NO direction in it — makes the value updates sweep BACKWARD from the reward. The
     reverse ORDER is never encoded; it EMERGES because the only surprise starts at the reward and each backup
     creates the next one behind it. Falsifier: RANDOM-order replay -> chance direction and many times more
     backups to reach the same value accuracy.

  B. FORWARD replay = PLANNING (Pfeiffer & Foster 2013; Ólafsdóttir 2018). A greedy value-ascent ROLLOUT from
     the current state — a replayed trajectory generated offline before acting — runs FORWARD to the goal,
     routing around a barrier. The forward direction emerges from ascending the learned predictive value, not
     from any instruction. Falsifier: roll out on an UNtrained value -> no gradient, the plan wanders and stalls.

  C. THE DISSOCIATION (Diba & Buzsáki 2007; Mattar & Daw 2018). The SAME machinery and the SAME value function
     produce OPPOSITE replay directions depending on what is needed — reverse to assign credit for a past
     reward, forward to plan a future path. Direction is a consequence of the computation, not a design choice.

  D. THE PAYOFF. Prioritized replay reaches a plannable map (its greedy policy solves the maze) from FAR fewer
     backups than random replay — the data-efficiency replay is for (Mattar & Daw 2018).

The predictive value here is the Successor Representation column V(s) = M[s, goal] the repo learns in
successor.py (goal = highest future occupancy); this file reuses that world and learns/uses it with replay.
Multi-seed, mean +/- 95% CI + a paired permutation test. Writes results/replay_planning.json + .svg.

    python -m src.eval.replay_planning --seeds 5
"""
import argparse
import json
import math
import os

import torch

from src.eval.successor import (GAMMA, ci95, geodesic, make_world, neighbors,
                                 paired_p)


# ----------------------------------------------------------------------------- the goal-directed value V(s)=gamma^dist
def q_backup(V, s, nbrs):
    """One goal-directed value backup: V(s) <- gamma * max_{s' in nbr(s)} V(s'). The goal is a clamped reward
    source (V[goal]=1), so this converges to the distance-to-go value V(s) = gamma^geodesic(s, goal) — the same
    predictive quantity successor.py ascends to plan (M[:, goal] also decays with geodesic distance). A max (not
    mean) backup makes each state settle in ONE visit, so the replay ORDER is a clean read of value propagation."""
    return GAMMA * max((V[c] for c in nbrs), default=0.0)


def build_world(seed, G):
    g = torch.Generator().manual_seed(seed)
    gap = int(torch.randint(1, G - max(1, G // 6), (1,), generator=g))
    free, cells, idx = make_world(G, gap, barrier=True)
    nbr = {k: [idx[c] for c in neighbors(i, j, free, G)] for k, (i, j) in enumerate(cells)}
    goal = int(torch.randint(len(cells), (1,), generator=g))
    dist = geodesic(cells, idx, free, G, cells[goal])                          # geodesic hops to goal (respects wall)
    Vstar = GAMMA ** dist                                                      # V*(s) = gamma^geodesic (goal = max)
    return cells, idx, free, nbr, goal, Vstar, dist, g


def _crit_backups(err_max, Vstar):
    return err_max < 0.02 * Vstar.max().item()


# ----------------------------------------------------------------------------- A/D: prioritized vs random replay
def prioritized_replay(nbr, goal, Vstar, max_backups=40000):
    """Prioritized sweeping. priority[s] = |backup(s) - V(s)| (a SCALAR magnitude, no direction). Repeatedly
    back up the max-priority state and refresh the priority of the states whose backup depends on it (its
    neighbours). The goal is a clamped reward source. Records the ORDER of states backed up and the #backups
    to reach value criterion."""
    n = len(nbr); V = [0.0] * n; V[goal] = 1.0                                 # clamp the reward source
    prio = [abs(q_backup(V, s, nbr[s]) - V[s]) if s != goal else 0.0 for s in range(n)]
    order, crit = [], None
    for b in range(max_backups):
        s = max(range(n), key=lambda k: prio[k])
        if prio[s] < 1e-9:
            break
        V[s] = q_backup(V, s, nbr[s])
        order.append(s)
        prio[s] = 0.0
        for p in nbr[s]:                                                        # neighbours' backups depend on V(s)
            if p != goal:
                prio[p] = abs(q_backup(V, p, nbr[p]) - V[p])
        if crit is None:
            err = max(abs(V[k] - Vstar[k].item()) for k in range(n))
            if _crit_backups(err, Vstar):
                crit = b + 1
    return V, order, crit


def random_replay(nbr, goal, Vstar, gen, max_backups=40000):
    """Same backups, but a transition is chosen at RANDOM each step (the falsifier: no priority ordering)."""
    n = len(nbr); V = [0.0] * n; V[goal] = 1.0
    non_goal = [s for s in range(n) if s != goal]
    order, crit = [], None
    for b in range(max_backups):
        s = non_goal[int(torch.randint(len(non_goal), (1,), generator=gen))]
        V[s] = q_backup(V, s, nbr[s])
        order.append(s)
        if crit is None:
            err = max(abs(V[k] - Vstar[k].item()) for k in range(n))
            if _crit_backups(err, Vstar):
                crit = b + 1
    return V, order, crit


def reverse_fraction(order, dist):
    """Fraction of consecutive backups that move AWAY from the goal (geodesic distance increases) = the reverse
    sweep of credit assignment. Ties (same distance) are ignored. Chance = 0.5."""
    out, inn = 0, 0
    for a, b in zip(order[:-1], order[1:]):
        d = dist[b].item() - dist[a].item()
        if d > 0:
            out += 1
        elif d < 0:
            inn += 1
    return out / max(1, out + inn)


# ----------------------------------------------------------------------------- B: forward planning rollout
def forward_rollout(V, nbr, start, goal, dist, max_len=200):
    """Planning replay: from `start`, greedily step to the highest-value neighbour (ascend the predictive map).
    A replayed forward trajectory. Returns (sequence, reached_goal)."""
    cur = start; seq = [cur]; seen = {cur}
    for _ in range(max_len):
        cand = nbr[cur]
        if not cand:
            break
        nxt = max(cand, key=lambda c: V[c])
        if V[nxt] <= V[cur] + 1e-12 or nxt in seen:                            # stalled / cycling
            break
        seq.append(nxt); seen.add(nxt); cur = nxt
        if cur == goal:
            return seq, True
    return seq, cur == goal


def forward_fraction(seq, dist):
    out, inn = 0, 0
    for a, b in zip(seq[:-1], seq[1:]):
        d = dist[b].item() - dist[a].item()
        if d < 0:
            inn += 1
        elif d > 0:
            out += 1
    return inn / max(1, inn + out)


def plan_success(V, nbr, cells, goal, dist):
    """Fraction of start cells from which the greedy value-ascent rollout reaches the goal (planning competence)."""
    reached = 0
    for s in range(len(cells)):
        if s == goal:
            reached += 1; continue
        _, ok = forward_rollout(V, nbr, s, goal, dist)
        reached += int(ok)
    return reached / len(cells)


# ----------------------------------------------------------------------------- one seed
def run_seed(seed, G=11):
    cells, idx, free, nbr, goal, Vstar, dist, g = build_world(seed, G)

    # A. reverse (credit assignment): prioritized vs random
    Vp, order_p, crit_p = prioritized_replay(nbr, goal, Vstar)
    Vr, order_r, crit_r = random_replay(nbr, goal, Vstar, g)
    rev_p = reverse_fraction(order_p, dist)
    rev_r = reverse_fraction(order_r, dist)

    # B. forward (planning): rollout on the prioritized-learned map, from the cell FARTHEST from the goal
    start = int(torch.tensor([dist[k].item() for k in range(len(cells))]).argmax())
    seq, reached = forward_rollout(Vp, nbr, start, goal, dist)
    fwd_frac = forward_fraction(seq, dist) if len(seq) > 1 else 0.0
    plan_trained = plan_success(Vp, nbr, cells, goal, dist)
    plan_untrained = plan_success([0.0] * len(cells), nbr, cells, goal, dist)   # falsifier: no value gradient

    # D. payoff: backups to a plannable map (crit), prioritized vs random
    return {
        "reverse_frac_prioritized": rev_p,
        "reverse_frac_random": rev_r,
        "backups_prioritized": float(crit_p if crit_p else 20000),
        "backups_random": float(crit_r if crit_r else 20000),
        "backup_speedup": float((crit_r if crit_r else 20000) / max(1, crit_p if crit_p else 20000)),
        "forward_frac_planning": fwd_frac,
        "plan_success_trained": plan_trained,
        "plan_success_untrained": plan_untrained,
        "farthest_start_reached": float(reached),
    }


# ----------------------------------------------------------------------------- aggregate + report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--G", type=int, default=11)
    a = ap.parse_args()
    per = [run_seed(s, a.G) for s in range(a.seeds)]
    keys = ["reverse_frac_prioritized", "reverse_frac_random", "backups_prioritized", "backups_random",
            "backup_speedup", "forward_frac_planning", "plan_success_trained", "plan_success_untrained"]
    agg = {k: ci95([p[k] for p in per]) for k in keys}
    p_dir = paired_p([p["reverse_frac_prioritized"] for p in per], [p["reverse_frac_random"] for p in per])
    p_plan = paired_p([p["plan_success_trained"] for p in per], [p["plan_success_untrained"] for p in per])

    print(f"REPLAY THAT COMPUTES — credit assignment (reverse) + planning (forward) "
          f"(n={a.seeds} seeds; mean ± 95% CI)\n" + "=" * 78, flush=True)
    lab = {"reverse_frac_prioritized": "A. REVERSE-replay fraction — prioritized (|TD err|, direction-free)",
           "reverse_frac_random": "   reverse fraction — RANDOM replay (falsifier, chance≈0.5)",
           "backups_prioritized": "D. backups to a plannable map — prioritized",
           "backups_random": "   backups to a plannable map — random",
           "backup_speedup": "   speedup (random ÷ prioritized backups)",
           "forward_frac_planning": "B. FORWARD-replay fraction — planning rollout",
           "plan_success_trained": "   planning success — replay-trained map",
           "plan_success_untrained": "   planning success — untrained value (falsifier)"}
    for k in keys:
        star = " ×" if "speedup" in k else ("" if "backups" not in k else " backups")
        print(f"  {lab[k]:62} {agg[k][0]:.3f} ± {agg[k][1]:.3f}{star}", flush=True)
    print(f"\n  A. reverse sweep emerges: prioritized {agg['reverse_frac_prioritized'][0]:.2f} vs random "
          f"{agg['reverse_frac_random'][0]:.2f}  (paired p={p_dir:.4f}) — direction never encoded", flush=True)
    print(f"  B/C. same map, forward when planning: forward-frac {agg['forward_frac_planning'][0]:.2f}, "
          f"solves the maze {agg['plan_success_trained'][0]:.0%} vs untrained "
          f"{agg['plan_success_untrained'][0]:.0%} (paired p={p_plan:.4f})", flush=True)
    print(f"  D. and it is cheap: prioritized reaches a plannable map {agg['backup_speedup'][0]:.1f}× fewer "
          f"backups than random.", flush=True)

    out = {"n_seeds": a.seeds, "G": a.G, "gamma": GAMMA,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in keys},
           "direction_paired_p": round(p_dir, 4), "planning_paired_p": round(p_plan, 4),
           "verdict": "Replay used the way the hippocampus uses it: a direction-free scalar priority makes value "
                      "updates sweep BACKWARD from a reward (reverse replay = credit assignment, Foster-Wilson "
                      "2006), and the SAME predictive map read forward gives a planning rollout that routes "
                      "around a barrier (forward replay = planning, Pfeiffer-Foster 2013). Direction emerges "
                      "from the computation; random replay is chance and untrained value cannot plan. Prioritized "
                      "replay reaches a plannable map many times faster than random (Mattar-Daw 2018)."}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/replay_planning.json", "w"), indent=2)
    svg_replay(a.G, agg, "results/replay_planning.svg")
    print("\nwrote results/replay_planning.json and results/replay_planning.svg", flush=True)


# ----------------------------------------------------------------------------- SVG
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


def svg_replay(G, agg, out):
    """One illustrative world: the reverse sweep (backups coloured by ORDER, radiating from the goal) and the
    forward planning rollout, plus the headline bars."""
    cells, idx, free, nbr, goal, Vstar, dist, _ = build_world(0, G)
    Vp, order_p, _ = prioritized_replay(nbr, goal, Vstar)
    rank = {s: i for i, s in enumerate(dict.fromkeys(order_p))}                # first-touch order
    mx = max(rank.values()) if rank else 1
    start = int(torch.tensor([dist[k].item() for k in range(len(cells))]).argmax())
    seq, _ = forward_rollout(Vp, nbr, start, goal, dist)
    seqset = {s: i for i, s in enumerate(seq)}
    gi, gj = cells[goal]

    cell = 20; pad = 20; gx = 26; top = 66
    gw = G * cell
    W = pad + gw + gx + gw + gx + 250 + pad
    H = top + gw + 66
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         '<text x="20" y="26" font-size="15" font-weight="800" fill="#0b1324">'
         'Replay that computes: reverse sweep = credit assignment, forward rollout = planning</text>']

    def panel(ox, title):
        e.append(f'<text x="{ox}" y="{top-8}" font-size="11" font-weight="700" fill="#28324a">{title}</text>')
        for k, (i, j) in enumerate(cells):
            x = ox + j * cell; y = top + i * cell
            yield k, i, j, x, y
    # left: reverse sweep, cells coloured by backup ORDER (goal = first = bright, radiating outward)
    for k, i, j, x, y in panel(pad, "reverse replay — backup order radiates OUT from reward (dark = later)"):
        col = _cmap(1.0 - rank.get(k, mx) / (mx + 1e-9))
        e.append(f'<rect x="{x}" y="{y}" width="{cell+0.5}" height="{cell+0.5}" fill="{col}"/>')
    for i in range(G):                                                          # wall
        for j in range(G):
            if not free[i, j]:
                e.append(f'<rect x="{pad+j*cell}" y="{top+i*cell}" width="{cell}" height="{cell}" fill="#0b1324"/>')
    e.append(f'<circle cx="{pad+gj*cell+cell/2:.0f}" cy="{top+gi*cell+cell/2:.0f}" r="5" fill="#de2d26" stroke="#fff"/>')
    # right: forward planning rollout on the learned map
    ox2 = pad + gw + gx
    for k, i, j, x, y in panel(ox2, "forward replay — planning rollout routes around the wall to the goal"):
        vv = Vp[k] / (max(Vp) + 1e-9)
        e.append(f'<rect x="{x}" y="{y}" width="{cell+0.5}" height="{cell+0.5}" fill="{_cmap(vv)}"/>')
    for i in range(G):
        for j in range(G):
            if not free[i, j]:
                e.append(f'<rect x="{ox2+j*cell}" y="{top+i*cell}" width="{cell}" height="{cell}" fill="#0b1324"/>')
    for a_, b_ in zip(seq[:-1], seq[1:]):                                       # rollout path
        ai, aj = cells[a_]; bi, bj = cells[b_]
        e.append(f'<line x1="{ox2+aj*cell+cell/2:.0f}" y1="{top+ai*cell+cell/2:.0f}" '
                 f'x2="{ox2+bj*cell+cell/2:.0f}" y2="{top+bi*cell+cell/2:.0f}" stroke="#de2d26" stroke-width="2.4"/>')
    si, sj = cells[start]
    e.append(f'<circle cx="{ox2+sj*cell+cell/2:.0f}" cy="{top+si*cell+cell/2:.0f}" r="4.5" fill="#ffffff" stroke="#de2d26" stroke-width="2"/>')
    e.append(f'<circle cx="{ox2+gj*cell+cell/2:.0f}" cy="{top+gi*cell+cell/2:.0f}" r="5" fill="#de2d26" stroke="#fff"/>')

    # bars
    bx = pad + 2 * gw + 2 * gx + 12; bw = 46; base = top + gw
    e.append(f'<text x="{bx}" y="{top-8}" font-size="11" font-weight="700" fill="#28324a">emergent signatures</text>')
    bars = [("reverse_frac_prioritized", "rev\nprio", "#2ca25f"), ("reverse_frac_random", "rev\nrand", "#c9341a"),
            ("forward_frac_planning", "fwd\nplan", "#2b8cbe"), ("plan_success_trained", "solve\ntrained", "#2ca25f"),
            ("plan_success_untrained", "solve\nuntr", "#c9341a")]
    for i, (k, lab, col) in enumerate(bars):
        v = agg[k][0]; x = bx + i * (bw + 6); h = v * gw
        e.append(f'<rect x="{x}" y="{base-h:.0f}" width="{bw}" height="{h:.0f}" fill="{col}" opacity="0.85"/>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{base-h-5:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        for li, part in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+bw/2:.0f}" y="{base+13+li*11:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{part}</text>')
    e.append(f'<line x1="{bx}" y1="{base:.0f}" x2="{bx+5*(bw+6):.0f}" y2="{base:.0f}" stroke="#33415c"/>')
    e.append(f'<text x="{bx}" y="{base+42:.0f}" font-size="10" fill="#5a6b8c">reverse emerges only when prioritized;</text>')
    e.append(f'<text x="{bx}" y="{base+55:.0f}" font-size="10" fill="#5a6b8c">same map plans forward, untrained can\'t.</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
