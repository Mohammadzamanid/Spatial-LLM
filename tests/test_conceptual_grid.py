"""Tests for the #8 conceptual-grid CPU de-risk (GAPS.md Tier 3, #8).

The FROZEN space-pretrained cortex should expose a genuine 2-D metric that a 1-D (rank) code cannot produce:
OFF-AXIS "closer" (where the 1-D x-projection ordering disagrees with the true 2-D answer) beats chance and
collapses under shuffled positions; held-out linear decode generalizes and beats the shuffled refit. All
readout-free or held-out (un-memorizable). Averaged over 3 seeds (the 2-D signal is modest on CPU).
"""
from src.eval.conceptual_grid_cortex import run_seed


def _mean(key, seeds=(0, 1, 2)):
    return sum(run_seed(s)[key] for s in seeds) / len(seeds)


def test_conceptual_grid_exposes_2d_metric():
    per = [run_seed(s) for s in (0, 1, 2)]

    def m(k):
        return sum(p[k] for p in per) / len(per)

    # (A) READOUT-FREE off-axis "closer" beats chance (a 1-D code is <=0.5 here by construction)...
    assert m("offaxis_closer_free") > 0.55, "the frozen code should carry genuine 2-D beyond a 1-D projection"
    # ...and collapses under shuffled positions (parameter-free falsifier)
    assert m("offaxis_gap") > 0.05, "off-axis 'closer' should need the true concept<->position map"
    assert m("offaxis_closer_free_shuffled") < 0.55, "shuffled positions should read at chance"
    assert abs(m("metric_spearman_shuffled")) < 0.15, "shuffled distance-correlation should collapse to ~0"

    # (B) HELD-OUT linear decode generalizes (never-seen concepts) and beats the shuffled refit by >0.5 spacing
    assert m("decode_gap") > 0.5, "held-out decode should beat the shuffled refit"
    assert m("heldout_offaxis_closer") > 0.55, "off-axis 'closer' in the held-out decoded space beats chance"
