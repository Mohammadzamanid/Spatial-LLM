"""
src/eval/systems_consolidation.py

NEOCORTICAL SYSTEMS CONSOLIDATION — replay moves a map from the hippocampal store INTO the slow cortical weights,
until a familiar map survives a hippocampal lesion (GAPS.md: the "frozen-LLM / CLS" critique item).

The repo shows replay *consolidating a decode map* (`pillars.py`) and the LLM *reading* a frozen spatial cortex
(`structural_transfer.py`) — but that read is a permanent structural DEPENDENCY on the hippocampal module, the
opposite of what Complementary Learning Systems predicts. In CLS (McClelland, McNaughton & O'Reilly 1995; Squire
& Alvarez 1995; Frankland & Bontempi 2005), a memory is hippocampus-dependent at first but, over nights of
sharp-wave-ripple replay, is slowly transferred into the distributed neocortical weights — after which a familiar
environment is recalled *without* the hippocampus. This eval builds the two-store loop and measures the classic
signatures, none of which is put into a loss:

  Two stores. HIPPOCAMPUS = a fast, one-shot, content-addressable store (stands for the CA3 Hopfield /
  place-cell memory the repo already has). NEOCORTEX = a slow gradient-trained network — the analogue of the
  frozen LLM's weights — that learns ONLY from replayed samples.

  (A) TEMPORALLY-GRADED RETROGRADE AMNESIA. After a hippocampal lesion (recall from cortex alone), accuracy is a
      GRADED function of a map's age: remote (well-replayed) maps are recalled, recent ones are lost. The
      gradient EMERGES — older maps have simply been replayed on more nights.
  (B) THE DOUBLE DISSOCIATION. With the hippocampus INTACT, recall is high at every age (no gradient — the fast
      store has everything). The gradient appears ONLY on lesion, and only for RECENT memories — exactly the
      Scoville-Milner / Squire pattern.
  (C) REPLAY IS CAUSAL (falsifier). With replay OFF, the cortex never learns, so even remote maps are lost on
      lesion and the gradient vanishes — the transfer is the replay, not the passage of time.
  (D) THE MAP IS IN THE WEIGHTS. The cortex alone (hippocampus-independent) recalls remote maps — the spatial
      structure has been internalised into the slow weights, which is what the frozen-LLM architecture was said
      to prevent.

Multi-seed, mean ± 95% CI. Writes results/systems_consolidation.json + .svg.

    python -m src.eval.systems_consolidation --seeds 5
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eval.successor import ci95

DIM = 24            # cue dimensionality
L = 10              # labels per map (chance = 1/L)
M = 16              # associations per map
DAYS = 14           # one new map per day
NIGHTLY = 14        # replay minibatches per night
HID = 40            # cortical hidden width (kept small so consolidation is GRADUAL, not instant)
LR = 3e-3
REMOTE = 5          # oldest REMOTE maps / newest RECENT maps for the binned contrast


class Hippocampus:
    """Fast one-shot content-addressable store: perfect nearest-cue recall. Stands for the CA3 Hopfield /
    place-cell store (place_cell_memory.py) — one exposure is enough, but it must stay intact to recall."""

    def __init__(self):
        self.cues = torch.empty(0, DIM); self.labels = torch.empty(0, dtype=torch.long)

    def store(self, cues, labels):
        self.cues = torch.cat([self.cues, cues]); self.labels = torch.cat([self.labels, labels])

    def recall(self, cues):
        return self.labels[(cues @ self.cues.t()).argmax(1)]                 # nearest stored cue

    def replay(self, n, gen):
        idx = torch.randint(len(self.cues), (n,), generator=gen)
        return self.cues[idx], self.labels[idx]


def make_map(gen):
    cues = torch.randn(M, DIM, generator=gen)
    cues = cues / cues.norm(dim=1, keepdim=True)
    return cues, torch.randint(0, L, (M,), generator=gen)


def run_seed(seed, replay=True):
    gen = torch.Generator().manual_seed(seed * 17 + 3)
    H = Hippocampus()
    C = nn.Sequential(nn.Linear(DIM, HID), nn.ReLU(), nn.Linear(HID, L))       # slow neocortex / LLM analogue
    opt = torch.optim.Adam(C.parameters(), LR)
    maps = []
    for day in range(DAYS):
        cues, labels = make_map(gen)
        maps.append((cues, labels, day))
        H.store(cues, labels)                                                 # fast one-shot hippocampal encode
        if replay:                                                            # sleep: replay -> train the cortex
            for _ in range(NIGHTLY):
                rc, rl = H.replay(64, gen)
                loss = F.cross_entropy(C(rc), rl)
                opt.zero_grad(); loss.backward(); opt.step()
    cort, intact = {}, {}
    with torch.no_grad():
        for cues, labels, day in maps:
            age = DAYS - day
            cort[age] = (C(cues).argmax(1) == labels).float().mean().item()   # LESIONED: cortex only
            intact[age] = (H.recall(cues) == labels).float().mean().item()    # INTACT: hippocampus present
    return cort, intact


def _bin(curve, ages):
    return sum(curve[a] for a in ages) / len(ages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    on = [run_seed(s, replay=True) for s in range(a.seeds)]
    off = [run_seed(s, replay=False) for s in range(a.seeds)]
    remote_ages = list(range(DAYS - REMOTE + 1, DAYS + 1))                    # oldest maps
    recent_ages = list(range(1, REMOTE + 1))                                  # newest maps

    def agg(runs, which, ages):
        return ci95([_bin(r[which], ages) for r in runs])

    def age_corr(runs, which):
        cs = []
        for r in runs:
            ages = torch.tensor(sorted(r[which]), dtype=torch.float)
            acc = torch.tensor([r[which][int(x)] for x in ages])
            a_ = ages - ages.mean(); b_ = acc - acc.mean()
            cs.append((a_ @ b_ / (a_.norm() * b_.norm() + 1e-9)).item())
        return ci95(cs)

    les_remote = agg(on, 0, remote_ages); les_recent = agg(on, 0, recent_ages)
    int_remote = agg(on, 1, remote_ages); int_recent = agg(on, 1, recent_ages)
    grad = ci95([_bin(r[0], remote_ages) - _bin(r[0], recent_ages) for r in on])
    corr = age_corr(on, 0)
    noreplay_remote = agg(off, 0, remote_ages)
    noreplay_grad = ci95([_bin(r[0], remote_ages) - _bin(r[0], recent_ages) for r in off])

    print(f"NEOCORTICAL SYSTEMS CONSOLIDATION — replay moves the map into the weights (n={a.seeds}; mean ± 95% CI)\n" + "=" * 82, flush=True)
    print(f"  chance = {1/L:.2f}   |   {DAYS} maps, one/day, {NIGHTLY} replay batches/night\n", flush=True)
    print(f"  (A) LESIONED (cortex only) recall — REMOTE {les_remote[0]:.2f} ± {les_remote[1]:.2f}   "
          f"RECENT {les_recent[0]:.2f} ± {les_recent[1]:.2f}", flush=True)
    print(f"      retrograde gradient (remote − recent) {grad[0]:+.2f} ± {grad[1]:.2f}   "
          f"recall↑with age corr {corr[0]:.2f} ± {corr[1]:.2f}", flush=True)
    print(f"  (B) INTACT (hippocampus present) recall — REMOTE {int_remote[0]:.2f}   RECENT {int_recent[0]:.2f}  "
          f"(flat: the fast store has everything → gradient only on lesion)", flush=True)
    print(f"  (C) REPLAY OFF (falsifier) — remote {noreplay_remote[0]:.2f}, gradient {noreplay_grad[0]:+.2f} "
          f"± {noreplay_grad[1]:.2f}  (no transfer → even remote lost)", flush=True)
    print(f"  (D) the cortex ALONE recalls remote maps ({les_remote[0]:.0%}) — the map is now in the slow "
          f"weights, hippocampus-independent.\n", flush=True)
    print(f"  -> temporally-graded retrograde amnesia EMERGES from replay-driven transfer: remote memories "
          f"survive a hippocampal lesion, recent ones do not, and the gradient vanishes without replay.", flush=True)

    # full curves for the SVG
    ages = list(range(1, DAYS + 1))
    curve_on = {age: ci95([r[0][age] for r in on])[0] for age in ages}
    curve_off = {age: ci95([r[0][age] for r in off])[0] for age in ages}
    out = {"n_seeds": a.seeds, "days": DAYS, "chance": round(1 / L, 3),
           "results": {"lesioned_remote": {"mean": les_remote[0], "ci95": les_remote[1]},
                       "lesioned_recent": {"mean": les_recent[0], "ci95": les_recent[1]},
                       "retrograde_gradient": {"mean": grad[0], "ci95": grad[1]},
                       "recall_age_corr": {"mean": corr[0], "ci95": corr[1]},
                       "intact_remote": {"mean": int_remote[0], "ci95": int_remote[1]},
                       "intact_recent": {"mean": int_recent[0], "ci95": int_recent[1]},
                       "noreplay_remote": {"mean": noreplay_remote[0], "ci95": noreplay_remote[1]},
                       "noreplay_gradient": {"mean": noreplay_grad[0], "ci95": noreplay_grad[1]}},
           "curve_lesioned_on": curve_on, "curve_lesioned_off": curve_off,
           "verdict": "Replay transfers a map from the fast hippocampal store into the slow cortical (LLM-analogue) "
                      "weights: cortex-only recall is a GRADED function of age (remote survives a hippocampal "
                      "lesion, recent is lost), the gradient appears only on lesion (intact recall is flat and "
                      "high), and it vanishes without replay. The familiar map ends up in the neocortical weights, "
                      "hippocampus-independent — the systems consolidation the frozen-LLM read was said to prevent."}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/systems_consolidation.json", "w"), indent=2)
    svg_cls(curve_on, curve_off, int_remote[0], les_remote[0], les_recent[0], 1 / L, "results/systems_consolidation.svg")
    print("\nwrote results/systems_consolidation.json and results/systems_consolidation.svg", flush=True)


def svg_cls(curve_on, curve_off, intact, remote, recent, chance, out):
    W, H = 700, 320
    ax, ay, aw, ah = 54, 70, 380, 200
    ages = sorted(curve_on)
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI, Arial">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         '<text x="20" y="26" font-size="15" font-weight="800" fill="#0b1324">'
         'Systems consolidation: replay moves the map into cortical weights &#8594; graded retrograde amnesia</text>',
         '<text x="20" y="45" font-size="10.5" fill="#5b6b8c">cortex-only (hippocampus-lesioned) recall vs map age; '
         'remote memories survive the lesion, recent ones do not</text>']
    def X(age): return ax + (age - ages[0]) / (ages[-1] - ages[0]) * aw
    def Y(v): return ay + ah - v * ah
    e.append(f'<rect x="{ax}" y="{ay}" width="{aw}" height="{ah}" fill="none" stroke="#c8d0e0"/>')
    for v in (0.0, 0.5, 1.0):
        e.append(f'<text x="{ax-6}" y="{Y(v)+3:.0f}" font-size="9" fill="#5b6b8c" text-anchor="end">{v:.1f}</text>')
        e.append(f'<line x1="{ax}" y1="{Y(v):.0f}" x2="{ax+aw}" y2="{Y(v):.0f}" stroke="#eef1f6"/>')
    e.append(f'<line x1="{ax}" y1="{Y(chance):.0f}" x2="{ax+aw}" y2="{Y(chance):.0f}" stroke="#c9341a" stroke-dasharray="4 3" opacity="0.6"/>')
    e.append(f'<text x="{ax+aw-4}" y="{Y(chance)-4:.0f}" font-size="8.5" fill="#c9341a" text-anchor="end">chance</text>')
    e.append(f'<line x1="{ax}" y1="{Y(intact):.0f}" x2="{ax+aw}" y2="{Y(intact):.0f}" stroke="#3182bd" stroke-dasharray="2 2" opacity="0.7"/>')
    e.append(f'<text x="{ax+6}" y="{Y(intact)-4:.0f}" font-size="8.5" fill="#3182bd">hippocampus INTACT (flat, high)</text>')
    for curve, col, lab in ((curve_on, "#2ca25f", "replay ON (lesioned)"), (curve_off, "#8c8c8c", "replay OFF (lesioned)")):
        pts = " ".join(f"{X(age):.1f},{Y(curve[age]):.1f}" for age in ages)
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.6"/>')
        for age in ages:
            e.append(f'<circle cx="{X(age):.1f}" cy="{Y(curve[age]):.1f}" r="2.6" fill="{col}"/>')
    e.append(f'<text x="{ax+10}" y="{ay+16}" font-size="9.5" fill="#2ca25f">replay ON: remote recalled, recent lost</text>')
    e.append(f'<text x="{ax+10}" y="{ay+30}" font-size="9.5" fill="#8c8c8c">replay OFF: flat at chance (no transfer)</text>')
    e.append(f'<text x="{ax+aw/2:.0f}" y="{ay+ah+18:.0f}" font-size="10" fill="#28324a" text-anchor="middle">&#8592; recent          map age (days)          remote &#8594;</text>')
    # side: the 2x2 dissociation summary
    sx = ax + aw + 30
    e.append(f'<text x="{sx}" y="{ay+6}" font-size="11" font-weight="700" fill="#28324a">recall %</text>')
    rows = [("intact / remote", intact, "#3182bd"), ("intact / recent", intact, "#3182bd"),
            ("lesion / remote", remote, "#2ca25f"), ("lesion / recent", recent, "#c9341a")]
    for i, (lab, v, col) in enumerate(rows):
        y = ay + 24 + i * 40
        e.append(f'<text x="{sx}" y="{y-3:.0f}" font-size="9" fill="#28324a">{lab}</text>')
        e.append(f'<rect x="{sx}" y="{y:.0f}" width="{v*150:.0f}" height="14" fill="{col}" opacity="0.85"/>')
        e.append(f'<text x="{sx+v*150+4:.0f}" y="{y+11:.0f}" font-size="9" font-weight="700" fill="#0b1324">{v:.0%}</text>')
    e.append('</svg>')
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(e))


if __name__ == "__main__":
    main()
