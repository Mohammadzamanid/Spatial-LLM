"""Tests for small-world searchability (GAPS.md purely-local-lattice item).

A navigable shortcut structure EMERGES from use-dependent plasticity — nothing about the shortcut distribution is
imposed. Greedy decentralised delivery is an interior optimum in the shortcut exponent (too-local is catastrophic);
at the flat prior short paths exist but are unfindable (worst stretch); use-based selection from that flat prior
grows the exponent into the navigable band and beats the flat prior, the best fixed-exponent graph, and a
random-prune control at matched budget+pool.
"""
from src.eval.small_world_search import run_seed


def test_navigable_exponent_emerges_and_beats_controls():
    o = run_seed(0)

    # (A) navigability is an INTERIOR optimum: a too-local exponent (α=3) is far worse than the navigable band
    assert o["deliver_a3"] > o["deliver_a1"] + 5, "too-local shortcuts (α=3) route catastrophically"
    assert o["deliver_a1"] < o["deliver_a0"], "an interior exponent beats the flat random-shortcut prior"
    # and α=3 scales worse than the navigable band
    assert o["grow_a3"] > o["grow_a1"], "too-local delivery grows faster with grid size (does not scale)"

    # (B) findability, not existence: the flat prior has the SHORTEST true paths but the WORST greedy stretch,
    # and the emergent graph improves findability
    assert o["stretch_flat"] > 1.5, "greedy cannot find the short paths under the flat prior (high stretch)"
    assert o["stretch_emergent"] < o["stretch_flat"], "the emergent graph is more findable (lower stretch)"

    # (C) the navigable exponent EMERGES from the flat (α=0) prior and beats every fixed-exponent graph
    assert o["alpha_emergent"] > 0.8, "use-based selection grows the exponent into the navigable band (from 0)"
    assert o["deliver_emergent"] < o["best_fixed"], "the emergent graph routes in fewer hops than any fixed α"
    assert o["deliver_emergent"] < o["deliver_a0"] - 1, "and clearly beats the flat prior it started from"

    # (D) falsifier: random pruning (same budget & candidate pool) keeps the exponent flat and gains nothing
    assert o["alpha_random"] < 0.5, "random pruning does not grow a navigable exponent"
    assert o["deliver_random"] > o["deliver_emergent"] + 1, "random pruning gives no navigability gain"
