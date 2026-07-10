"""Tests for the astrocyte syncytium — spatial-density-gated plasticity + heterosynaptic binding (GAPS.md Tier 2).

The gap-junction-coupled astrocyte network should bind a silent-but-surrounded synapse heterosynaptically (fill
in) and gate plasticity SELECTIVELY by the spatial density of co-activity (clustered co-active synapses'
core potentiates where the same number scattered does not) — where a point-wise astrocyte does nothing at this
sub-threshold drive and a FULLY regenerative wave floods indiscriminately (the honest finding). Measured.
"""
from src.eval.astrocyte_syncytium import run_seed


def test_syncytium_binds_and_density_gates():
    o = run_seed(0)

    # (B) HETEROSYNAPTIC BINDING (headline): a silent surrounded gap is bound by pooled neighbour Ca2+
    assert o["gate_fillin_syncytium"] > 0.5 and o["fillin_gap"] > 0.5, "the syncytium should bind a silent gap"

    # (A) SPATIAL-DENSITY GATE: clustered co-activity's core potentiates where scattered (matched count) does not
    assert o["density_selectivity_syncytium"] > 0.15, "clustered should potentiate more than scattered"

    # (C) FALSIFIERS: uncoupled does nothing at this sub-threshold drive; a regenerative wave floods
    assert o["gate_clustered_uncoupled"] < 0.15, "point-wise (uncoupled) can't pool sub-threshold activity"
    assert abs(o["density_selectivity_uncoupled"]) < 0.1, "no spread -> no density selectivity"
    assert o["gate_scattered_regenwave"] > 0.6, "a regenerative wave should FLOOD (no spatial selectivity)"
