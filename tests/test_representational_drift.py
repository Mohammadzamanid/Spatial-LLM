"""Tests for representational drift and the conserved population geometry (GAPS.md Tier 5, #C6).

At MATCHED single-cell drift, a label-free geometry read-out (and a held-out supervised decoder) should recover
position under geometry-PRESERVING drift (field relocation, even a full remap) but FAIL under geometry-DESTROYING
drift (independent high-D noise); a fixed decoder fails under any drift. So stable read-out rides on the conserved
GEOMETRY, not on single-cell stability. All measured, never imposed.

(This replaces an earlier, circular version — RSA over a Gaussian tiling is blind to remapping — that an
adversarial red-team rejected; here the reader is label-free, the falsifier is a matched valid spatial code, and
a full remap is honestly shown to SURVIVE because the read-out reads geometry, not cell identity.)
"""
from src.eval.representational_drift import run_seed


def test_conserved_geometry_survives_drift():
    o = run_seed(0)

    # single-cell drift is MATCHED across the geometry-preserving and geometry-destroying conditions
    assert o["drift_match"] < 0.12, "single-cell drift must be matched between relocate and noise"

    # (A) LABEL-FREE geometry read-out: survives geometry-preserving drift, fails geometry-destroying drift
    assert o["manifold_relocate"] < 0.05, "geometry read-out should recover position under geometry-preserving drift"
    assert o["manifold_noise"] > 0.15, "geometry read-out should fail once the geometry is destroyed"
    assert o["geometry_gap"] > 0.15, "the difference is the drift STRUCTURE, not the (matched) single-cell drift"

    # (A') SUPERVISED confirmation (held-out) — not an overfit artifact: even with labels, position does not
    # generalise once the geometry is gone
    assert o["heldout_relocate"] < 0.10 and o["heldout_noise"] > 0.20, "held-out decode confirms geometry is destroyed by noise"

    # (B) a FIXED decoder degrades under drift while the geometry read-out survives
    assert o["fixed_relocate"] > 0.10 and o["reader_vs_fixed"] > 0.08, "geometry read-out should beat the fixed decoder"

    # (C) ROBUST TO REMAPPING: the read-out survives even a FULL remap (0% cells conserved) — geometry, not cells
    assert o["manifold_remap"] < 0.05, "the geometry read-out reads the environment's geometry, not cell identity"
