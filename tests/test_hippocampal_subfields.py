"""Tests for the hippocampal subfield triad — DG pattern separation + CA1 comparator (GAPS.md Tier 2, #2).

Sparse DG separation should let CA3 recall SIMILAR environments where a matched-size DENSE expansion interferes
(and a dense expansion is even worse than not expanding — sparsity is necessary, not the dimensionality); and a
CA1 comparator should detect novelty only with the CA3 memory stream. All measured, never in a loss.
"""
from src.eval.hippocampal_subfields import run_seed


def test_dg_separation_and_ca1_comparator():
    o = run_seed(0)

    # (A) SEPARATION -> INTERFERENCE-FREE RECALL: DG (sparse) recalls far better than a matched-size dense expansion
    assert o["recall_dg"] > o["recall_dense"] + 0.2, "sparse DG separation should beat a matched-size dense expansion"
    assert o["recall_gap"] > 0.2
    # the dense expansion is even WORSE than not expanding -> the sparsity is load-bearing, not the size
    assert o["recall_dense"] < o["recall_direct"], "a dense expansion of the same size should be actively harmful"
    # mechanism: DG orthogonalizes (out overlap << in overlap), dense preserves the overlap
    assert o["sep_index_dg"] < 0.7 and o["sep_index_dense"] > 0.9

    # (C) CA1 COMPARATOR: detects novelty, and NEEDS the CA3 memory stream (ablation -> chance)
    assert o["ca1_auc"] > 0.8, "the comparator should discriminate novel from familiar"
    assert o["ca1_gap"] > 0.3 and o["ca1_auc_ablate"] < 0.65, "novelty detection should need the stored memory"
