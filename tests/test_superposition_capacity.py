"""Tests for polysemantic superposition (GAPS.md monosemantic-readout item).

A localized one-cell-per-place code is monosemantic: N cells store at most N fields. But an N-cell bottleneck
trained to reconstruct a SPARSE field vector packs far more fields than it has cells by high-dimensional
superposition (Elhage 2022), each cell emerging polysemantic — and the compression collapses when activity is
dense. Nothing about the coding is imposed; only the mechanism (a tied autoencoder) and the task (reconstruct
sparse fields) are built.
"""
from src.eval.superposition_capacity import run_seed, F, N


def test_superposition_beats_the_monosemantic_ceiling():
    o = run_seed(0)

    # (A) N cells recall far MORE than N fields when activity is sparse — above the one-cell-per-place ceiling
    assert o["monosemantic_ceiling"] == N / F, "the monosemantic ceiling is N/F fields per cell"
    assert o["recall_superposition"] > 0.9, "sparse activity lets N cells recall ~all F fields"
    assert o["fields_recalled"] > 2 * N, "more fields stored than a monosemantic code (N) could hold"

    # (B) polysemanticity emerges: each cell participates in many fields
    assert o["polysemanticity"] > 1.5, "each cell encodes several fields (>1 = polysemantic)"

    # (C) sparsity is load-bearing: dense training collapses superposition toward the ceiling
    assert o["recall_dense"] < o["recall_superposition"] - 0.3, "densifying activity destroys the compression"
    assert o["dose_p04"] > o["dose_p30"] + 0.2, "recall degrades monotonically as activity densifies"
