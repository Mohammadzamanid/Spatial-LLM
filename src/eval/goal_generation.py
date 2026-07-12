"""
src/eval/goal_generation.py

GOAL GENERATION — an agent proposes its OWN goals at the frontier of its competence, and a developmental curriculum
emerges without ever being scheduled (GAPS.md: agency / autonomy frontier, organ 2 — the agent decides what to want).

Intrinsic motivation (organ 1) gave the agent a drive; goal generation turns that drive into self-proposed GOALS,
so the agent is no longer handed a goal vector — it chooses what to pursue. The faithful, non-circular form is the
autotelic agent (Colas, Karch, Sigaud & Oudeyer 2022; developmental robotics): it samples goals to practise by
LEARNING PROGRESS over a goal space, preferring goals at the frontier of its ability — the zone of proximal
development. Per the standing rule nothing is scheduled: the agent is never told a goal, never told a difficulty
order, and the goal space contains IMPOSSIBLE goals (ceiling competence 0 — the goal-space "noisy TV") it must learn
to avoid. A developmental trajectory then emerges and is measured:

  (A) A CURRICULUM EMERGES. The mean difficulty of the goals the autotelic agent proposes for itself RISES over its
      lifetime — easy goals first, harder goals as it masters them — a developmental ordering that was never
      scheduled. A random-goal agent's proposed difficulty stays flat.
  (B) GOAL-SPACE MASTERY. The autotelic agent masters essentially ALL the learnable goals, where a random-goal agent
      masters fewer and the fixed strategies fail.
  (C) IT THREADS THE ZONE OF PROXIMAL DEVELOPMENT — between two failure modes. An "always hardest" agent wastes
      ~100% of its practice on IMPOSSIBLE goals and masters NOTHING (the goal-space noisy TV); an "always easiest"
      agent stalls on already-trivial goals and masters almost nothing; the autotelic agent self-organises onto the
      productive frontier (learnable, not-yet-mastered) and so masters the space.

Honest note: a random-goal agent also masters many goals (novelty carries it part way) — the autotelic advantage is
the emergent curriculum, completeness, and avoiding BOTH failure modes, not raw activity.

Multi-seed, mean ± 95% CI. Writes results/goal_generation.json + .svg.

    python -m src.eval.goal_generation --seeds 5
"""
import argparse
import json
import math
import os

import torch

N_GOALS = 60           # goals in the space
IMPOSSIBLE_ABOVE = 0.8  # goals with difficulty >= this are unlearnable (ceiling competence 0)
T = 1600               # practice steps of life
EMA = 0.4
MASTER = 0.9           # competence above which a goal is "mastered"
STRATS = ["autotelic", "random", "hardest", "easiest"]


def run(strategy, seed):
    """One life. The agent picks a goal to practise each step by `strategy` (autotelic = its own learning progress),
    with no external goal and no difficulty schedule. Returns per-life behavioural measures."""
    g = torch.Generator().manual_seed(seed)
    diff = torch.rand(N_GOALS, generator=g)                        # each goal's difficulty in [0,1]
    learnable = diff < IMPOSSIBLE_ABOVE                            # goals above the threshold are IMPOSSIBLE
    ceiling = learnable.float()
    comp = torch.zeros(N_GOALS)                                    # current competence per goal
    last_comp = torch.zeros(N_GOALS)                              # competence at the previous practice of that goal
    lp = torch.zeros(N_GOALS)                                     # learning progress = across-practice competence gain
    practiced = torch.zeros(N_GOALS)
    proposed_diff = []
    impossible_props = 0
    frontier_props = 0

    for t in range(T):
        novelty = 1.0 / (1.0 + practiced)
        if strategy == "autotelic":
            score = lp.clamp(min=0) + 0.2 * novelty                # the frontier: goals where competence is GAINING
        elif strategy == "random":
            score = torch.rand(N_GOALS, generator=g)
        elif strategy == "hardest":
            score = diff + 1e-3 * novelty                          # always chase the hardest goals
        else:  # easiest
            score = comp + 1e-3 * novelty                          # stick to what is already easy
        goal = int(score.argmax())

        rate = 0.25 * (1.0 - diff[goal].item())                    # harder goals improve slower
        if practiced[goal] > 0:                                    # across-practice learning progress
            lp[goal] = (1 - EMA) * lp[goal] + EMA * (comp[goal].item() - last_comp[goal].item())
        last_comp[goal] = comp[goal].item()
        comp[goal] += rate * (ceiling[goal].item() - comp[goal].item())
        practiced[goal] += 1

        proposed_diff.append(diff[goal].item())
        if not learnable[goal]:
            impossible_props += 1
        elif comp[goal].item() < MASTER:
            frontier_props += 1

    pd = torch.tensor(proposed_diff)
    return {
        "mastered": int(((comp > MASTER) & learnable).sum()),
        "n_learn": int(learnable.sum()),
        "diff_early": pd[:T // 4].mean().item(),
        "diff_late": pd[3 * T // 4:].mean().item(),
        "impossible": impossible_props / T,
        "frontier": frontier_props / T,
    }


def run_seed(seed):
    out = {}
    for s in STRATS:
        r = run(s, seed)
        out[f"mastered_{s}"] = r["mastered"]
        out[f"curric_{s}"] = r["diff_late"] - r["diff_early"]       # curriculum: rise in proposed difficulty
        out[f"impossible_{s}"] = r["impossible"]
        out[f"frontier_{s}"] = r["frontier"]
        if s == "autotelic":
            out["diff_early_auto"] = r["diff_early"]
            out["diff_late_auto"] = r["diff_late"]
            out["n_learn"] = r["n_learn"]
    return out


KEYS = ([f"mastered_{s}" for s in STRATS] + [f"curric_{s}" for s in STRATS]
        + [f"impossible_{s}" for s in STRATS] + [f"frontier_{s}" for s in STRATS]
        + ["diff_early_auto", "diff_late_auto", "n_learn"])


def ci95(vals):
    t = torch.tensor(vals, dtype=torch.float); n = len(vals)
    sd = t.std(unbiased=True).item() if n > 1 else 0.0
    return round(t.mean().item(), 4), (round(1.96 * sd / math.sqrt(n), 4) if n > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    per = [run_seed(s) for s in range(a.seeds)]
    agg = {k: ci95([p[k] for p in per]) for k in KEYS}
    nl = agg["n_learn"][0]

    print(f"GOAL GENERATION — the agent proposes its own goals; a curriculum emerges "
          f"(n={a.seeds}; mean ± 95% CI)\n" + "=" * 82, flush=True)
    print(f"  (A) A CURRICULUM EMERGES (no schedule): autotelic proposed difficulty rises "
          f"{agg['diff_early_auto'][0]:.2f} -> {agg['diff_late_auto'][0]:.2f} (Δ {agg['curric_autotelic'][0]:+.2f}) "
          f"vs random flat (Δ {agg['curric_random'][0]:+.2f})", flush=True)
    print(f"  (B) GOAL-SPACE MASTERY ({nl:.0f} learnable): autotelic {agg['mastered_autotelic'][0]:.0f} vs random "
          f"{agg['mastered_random'][0]:.0f} vs hardest {agg['mastered_hardest'][0]:.0f} vs easiest "
          f"{agg['mastered_easiest'][0]:.0f}", flush=True)
    print(f"  (C) IT THREADS THE ZONE OF PROXIMAL DEVELOPMENT — between two failure modes:", flush=True)
    print(f"      'always hardest' wastes {agg['impossible_hardest'][0]*100:.0f}% of practice on IMPOSSIBLE goals "
          f"(masters {agg['mastered_hardest'][0]:.0f} — the goal-space noisy TV); 'always easiest' stalls on trivial "
          f"goals (frontier {agg['frontier_easiest'][0]*100:.0f}%, masters {agg['mastered_easiest'][0]:.0f})", flush=True)
    print(f"      autotelic self-organises onto the productive frontier ({agg['frontier_autotelic'][0]*100:.0f}% of "
          f"proposals) and avoids impossible goals ({agg['impossible_autotelic'][0]*100:.0f}%)", flush=True)
    print(f"  (D) HONEST: random-goal also masters many ({agg['mastered_random'][0]:.0f}) — novelty carries it part "
          f"way; the autotelic edge is the emergent curriculum, completeness, and avoiding BOTH failure modes.",
          flush=True)
    print(f"\n  The agent decides what to want: it proposes its own goals at the frontier of its ability, a "
          f"developmental curriculum emerges with no schedule, and it threads between impossible goals (where "
          f"'always hardest' masters nothing) and trivial ones. Autonomy over goals, not just drive.", flush=True)

    out = {"n_seeds": a.seeds, "n_goals": N_GOALS, "impossible_above": IMPOSSIBLE_ABOVE, "steps": T,
           "results": {k: {"mean": agg[k][0], "ci95": agg[k][1]} for k in KEYS},
           "verdict": "An autotelic agent proposes its OWN goals by learning progress over a goal space, with no "
                      "handed goal and no difficulty schedule. A developmental curriculum EMERGES -- the difficulty "
                      "of self-proposed goals rises over the lifetime (easy first, hard later), never scheduled -- "
                      "and the agent masters essentially all learnable goals. It threads the zone of proximal "
                      "development between two failure modes: an 'always hardest' agent wastes ~100% of practice on "
                      "impossible goals and masters nothing (the goal-space noisy TV), while an 'always easiest' "
                      "agent stalls on trivial goals; the autotelic agent self-organises onto the productive "
                      "frontier. Honest note: a random-goal agent also masters many goals (novelty carries it part "
                      "way); the autotelic advantage is the emergent curriculum, completeness, and avoiding both "
                      "failure modes, not raw activity. Builds on organ 1 (intrinsic motivation) -- the drive now "
                      "generates the goals."}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/goal_generation.json", "w"), indent=2)
    svg_goalgen(agg, "results/goal_generation.svg")
    print("\nwrote results/goal_generation.json and results/goal_generation.svg", flush=True)


def svg_goalgen(agg, out):
    W_, H = 770, 320
    nl = agg["n_learn"][0]
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W_}" height="{H}" fill="#ffffff"/>',
         '<text x="20" y="26" font-size="15" font-weight="800" fill="#0b1324">'
         'Goal generation: the agent proposes its own goals &#8212; a curriculum emerges, unscheduled</text>',
         '<text x="20" y="45" font-size="10.5" fill="#5b6b8c">no handed goal, no difficulty schedule; the agent '
         'picks goals at its own competence frontier (learning progress)</text>']
    col = {"autotelic": "#2ca25f", "random": "#8c8c8c", "hardest": "#c9341a", "easiest": "#e6842a"}
    # left: curriculum (early vs late difficulty, autotelic)
    bx, by, bh, bw = 44, 100, 150, 46
    e.append(f'<text x="{bx}" y="{by-8}" font-size="11" font-weight="700" fill="#28324a">autotelic goal difficulty</text>')
    for i, (k, lab) in enumerate([("diff_early_auto", "early\nlife"), ("diff_late_auto", "late\nlife")]):
        v = agg[k][0]; x = bx + i * (bw + 20); h = v * bh
        e.append(f'<rect x="{x}" y="{by+bh-h:.0f}" width="{bw}" height="{h:.0f}" fill="#2ca25f" opacity="{0.55+0.3*i}"/>')
        e.append(f'<text x="{x+bw/2:.0f}" y="{by+bh-h-5:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.2f}</text>')
        for li, part in enumerate(lab.split("\n")):
            e.append(f'<text x="{x+bw/2:.0f}" y="{by+bh+13+li*11:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{part}</text>')
    e.append(f'<line x1="{bx-4}" y1="{by+bh}" x2="{bx+2*(bw+20):.0f}" y2="{by+bh}" stroke="#33415c"/>')
    e.append(f'<path d="M{bx+bw/2:.0f} {by+bh-agg["diff_early_auto"][0]*bh-16:.0f} L{bx+bw+20+bw/2:.0f} {by+bh-agg["diff_late_auto"][0]*bh-16:.0f}" stroke="#2ca25f" stroke-width="2" marker-end="url(#a)" fill="none"/>')
    e.append('<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#2ca25f"/></marker></defs>')
    e.append(f'<text x="{bx}" y="{by+bh+38:.0f}" font-size="8.5" fill="#5b6b8c">curriculum: easy &#8594; hard</text>')
    # middle: mastery (4 strategies)
    m0 = 300; mw = 40
    e.append(f'<text x="{m0}" y="{by-8}" font-size="11" font-weight="700" fill="#28324a">goals mastered (of {nl:.0f})</text>')
    for i, s in enumerate(STRATS):
        v = agg[f"mastered_{s}"][0]; x = m0 + i * (mw + 8); h = v / nl * bh
        e.append(f'<rect x="{x}" y="{by+bh-h:.0f}" width="{mw}" height="{h:.0f}" fill="{col[s]}" opacity="0.85"/>')
        e.append(f'<text x="{x+mw/2:.0f}" y="{by+bh-h-5:.0f}" font-size="10" font-weight="700" fill="#0b1324" text-anchor="middle">{v:.0f}</text>')
        e.append(f'<text x="{x+mw/2:.0f}" y="{by+bh+13:.0f}" font-size="8" fill="#28324a" text-anchor="middle">{s[:6]}</text>')
    e.append(f'<line x1="{m0-4}" y1="{by+bh}" x2="{m0+4*(mw+8):.0f}" y2="{by+bh}" stroke="#33415c"/>')
    e.append(f'<text x="{m0}" y="{by+bh+34:.0f}" font-size="8.5" fill="#5b6b8c">hardest &amp; easiest both fail</text>')
    # right: impossible-goal waste (noisy-TV parallel)
    rx = 560; rw = 60
    e.append(f'<text x="{rx}" y="{by-8}" font-size="11" font-weight="700" fill="#28324a">practice on IMPOSSIBLE goals</text>')
    for i, s in enumerate(["autotelic", "hardest"]):
        v = agg[f"impossible_{s}"][0]; x = rx + i * (rw + 18); h = v * bh
        e.append(f'<rect x="{x}" y="{by+bh-h:.0f}" width="{rw}" height="{h:.0f}" fill="{col[s]}" opacity="0.85"/>')
        e.append(f'<text x="{x+rw/2:.0f}" y="{by+bh-h-5:.0f}" font-size="11" font-weight="700" fill="#0b1324" text-anchor="middle">{v*100:.0f}%</text>')
        e.append(f'<text x="{x+rw/2:.0f}" y="{by+bh+13:.0f}" font-size="9" fill="#28324a" text-anchor="middle">{s}</text>')
    e.append(f'<line x1="{rx-4}" y1="{by+bh}" x2="{rx+2*(rw+18):.0f}" y2="{by+bh}" stroke="#33415c"/>')
    e.append(f'<text x="{rx}" y="{by+bh+34:.0f}" font-size="8.5" fill="#5b6b8c">the goal-space noisy TV</text>')
    e.append(f'<text x="20" y="{H-12}" font-size="9.5" fill="#5b6b8c">Autonomy over goals: self-proposed at the '
             f'competence frontier &#8212; a developmental curriculum, no schedule. Builds on intrinsic motivation.</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
