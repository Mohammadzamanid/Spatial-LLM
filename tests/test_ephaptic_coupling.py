"""Tests for ephaptic coupling — a non-synaptic field that shapes spike TIMING (GAPS.md Tier 2).

A self-generated ZERO-MEAN field should synchronize spike timing at MATCHED firing rate (so it is timing, not
rate), where a matched-budget sparse-synaptic network cannot; zeroing the field abolishes the synchrony; and the
field-made synchrony drives a downstream coincidence detector that the asynchronous state does not. Measured.
"""
from src.eval.ephaptic_coupling import run_seed


def test_ephaptic_field_synchronizes_at_matched_rate():
    o = run_seed(0)

    # (A) SYNCHRONY AT MATCHED RATE: field ON synchronizes, field OFF does not, with the firing rate matched
    assert o["chi_field"] > 0.8 and o["chi_off"] < 0.2, "the field should synchronize; without it, incoherent"
    assert o["sync_gap"] > 0.3
    assert o["rate_mismatch"] < 0.15, "the effect must be at MATCHED rate (timing, not drive)"
    # dose-response: half the field already lifts synchrony above the uncoupled baseline
    assert o["chi_half_field"] > o["chi_off"]

    # (B) a GLOBAL field beats matched-budget SPARSE synapses (a coherent non-synaptic channel)
    assert o["field_vs_sparse"] > 0.2 and o["chi_sparse_syn"] < 0.4

    # (D) COMPUTATIONAL WORK: the synchrony drives a downstream coincidence detector the async state does not
    assert o["coin_gap"] > 0.05 and o["coin_off"] < 0.1, "synchrony should be readable where rate alone is not"
