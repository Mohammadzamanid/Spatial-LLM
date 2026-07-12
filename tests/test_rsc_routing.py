"""Tests for bifurcated RSC routing (GAPS.md unified-gate item).

Splitting the spatial read-out into an action pathway (M2) and a memory pathway (AD), as the retrosplenial cortex
does (Molecular Psychiatry 2024), makes the two reference frames dissociate emergently (action=egocentric,
memory=allocentric), routes each target its own signal, and ENABLES the double dissociation a unified code cannot
(a unified lesion hits both tasks). A no-conflict falsifier blurs the specialization. Honest: the split does not
lower total loss; the benefit is clean segregation.
"""
from src.eval.rsc_routing import run_seed


def test_pathways_dissociate_and_enable_double_dissociation():
    o = run_seed(0)

    # (A) reference frames dissociate emergently: heading lives in the action head, not the memory head
    assert o["act_head_heading"] > 0.5, "the action pathway is egocentric (heading decodable)"
    assert o["mem_head_heading"] < 0.25, "the memory pathway is allocentric (heading not decodable)"
    assert o["act_head_heading"] > o["mem_head_heading"] + 0.4, "clean egocentric/allocentric split"

    # (B) selective routing: the memory pathway carries location, not the action; a unified code carries both
    assert o["mem_pathway_selectivity"] > 0.5, "memory pathway carries WHERE, not the turn (selective)"
    assert o["unified_entanglement"] > 0.5, "the unified code is entangled — it carries BOTH signals"

    # (C) the split enables the double dissociation; a unified lesion cannot dissociate
    assert o["split_les_action_on_action"] > 1.5, "lesioning the action pathway impairs the action task"
    assert o["split_les_action_on_memory"] < 1.2, "...but not the memory task"
    assert o["split_les_memory_on_memory"] > 1.5, "lesioning the memory pathway impairs the memory task"
    assert o["split_les_memory_on_action"] < 1.2, "...but not the action task"
    assert o["unified_lesion_action_deficit"] > 0.2 and o["unified_lesion_memory_deficit"] > 0.2, \
        "a unified lesion degrades BOTH tasks — no dissociation possible"

    # (D) falsifier: with no reference-frame conflict the memory pathway stops excluding the action signal
    assert o["mem_pathway_action_aligned"] > o["mem_pathway_action_conflict"] + 0.15, \
        "the specialization emerges from the conflicting frames, not the wiring"
