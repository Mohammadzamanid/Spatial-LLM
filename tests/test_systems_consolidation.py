"""Tests for neocortical systems consolidation — CLS replay transfer (GAPS.md frozen-LLM/CLS item).

Replay should transfer a map from the fast hippocampal store into the slow cortical (LLM-analogue) weights,
producing a temporally-graded retrograde amnesia: cortex-only (lesioned) recall rises with map age (remote
survives, recent lost), the gradient appears only on lesion (intact recall is flat and high), and it vanishes
without replay. Measured, never in a loss.
"""
from src.eval.systems_consolidation import run_seed, DAYS, REMOTE

REMOTE_AGES = list(range(DAYS - REMOTE + 1, DAYS + 1))
RECENT_AGES = list(range(1, REMOTE + 1))


def _bin(curve, ages):
    return sum(curve[a] for a in ages) / len(ages)


def test_replay_transfers_map_and_grades_retrograde_amnesia():
    cort_on, intact_on = run_seed(0, replay=True)
    cort_off, _ = run_seed(0, replay=False)

    remote_on = _bin(cort_on, REMOTE_AGES); recent_on = _bin(cort_on, RECENT_AGES)

    # (A) temporally-graded retrograde amnesia: remote survives the lesion, recent is lost
    assert remote_on > 0.35, "remote maps should be recalled from cortex alone (in the weights)"
    assert remote_on - recent_on > 0.15, "cortex-only recall should be graded by age (retrograde gradient)"

    # (B) the gradient appears only on lesion: with the hippocampus intact, recall is flat and high
    assert _bin(intact_on, REMOTE_AGES) > 0.9 and _bin(intact_on, RECENT_AGES) > 0.9, \
        "the intact fast store recalls every age"

    # (C) replay is causal: with no replay the cortex never learns -> even remote lost, no gradient
    remote_off = _bin(cort_off, REMOTE_AGES); recent_off = _bin(cort_off, RECENT_AGES)
    assert remote_off < 0.3, "without replay, remote maps are not in the cortex (near chance)"
    assert abs(remote_off - recent_off) < 0.2, "without replay there is no retrograde gradient (falsifier)"
