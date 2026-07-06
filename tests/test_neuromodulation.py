"""Tests for neuromodulation modules."""
import pytest
import torch
from src.models.neuromodulation import (
    SpatialNeuromodulator, AdaptiveGain, PredictionErrorGate,
    AcetylcholineGate, LocusCoeruleusReset,
)
from src.models.neuro import HopfieldAssociativeMemory


def test_neuromodulator_2d():
    mod = SpatialNeuromodulator(hidden_dim=64)
    x = torch.randn(2, 64)
    ctx = torch.randn(2, 64)
    out = mod(x, ctx)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()


def test_neuromodulator_3d():
    """Works on (B, T, D) sequence inputs."""
    mod = SpatialNeuromodulator(hidden_dim=64)
    x = torch.randn(2, 10, 64)
    ctx = torch.randn(2, 64)
    out = mod(x, ctx)
    assert out.shape == (2, 10, 64)


def test_adaptive_gain_output_shape():
    ag = AdaptiveGain(hidden_dim=64)
    x = torch.randn(2, 64)
    out, uncertainty = ag(x)
    assert out.shape == (2, 64)
    assert uncertainty.shape == (2,)
    assert (uncertainty >= 0).all(), "Uncertainty should be non-negative"


def test_adaptive_gain_no_nan():
    ag = AdaptiveGain(hidden_dim=64)
    x = torch.randn(4, 64)
    out, unc = ag(x)
    assert not torch.isnan(out).any()
    assert not torch.isnan(unc).any()


def test_prediction_error_gate():
    gate = PredictionErrorGate(hidden_dim=64)
    x = torch.randn(2, 64)
    err = torch.tensor([0.1, 2.5])   # low vs high prediction error
    out = gate(x, err)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()


# ── Acetylcholine encode/retrieve gate + noradrenaline surprise (GAPS.md #5) ────────

def test_acetylcholine_gate_directionality():
    """High ACh (encoding) suppresses recurrent recall and raises plasticity; low ACh is retrieval."""
    gate = AcetylcholineGate(recurrent_gain_max=1.0)
    rec_hi, plast_hi, ff_hi = gate(1.0)   # encoding
    rec_lo, plast_lo, ff_lo = gate(0.0)   # retrieval
    assert rec_hi == 0.0 and rec_lo == 1.0, "recurrent recall must be suppressed at high ACh"
    assert plast_hi == 1.0 and plast_lo == 0.0, "plasticity must be enhanced at high ACh"


def test_locus_coeruleus_surprise():
    """Surprise ~0 for a matched prediction, large for a mismatched one; reset fires only above threshold."""
    lc = LocusCoeruleusReset(threshold=0.5)
    x = torch.randn(3, 16)
    s_match, reset_match = lc(x, x.clone())
    s_mis, reset_mis = lc(x, torch.randn(3, 16))
    assert s_match.max() < 0.05, "matched prediction should be unsurprising"
    assert s_mis.mean() > 0.5, "mismatched input should be surprising"
    assert not reset_match.any() and reset_mis.all()


def test_hopfield_store_and_complete():
    """A degraded cue is pattern-completed toward the stored pattern only WITH the recurrent weights."""
    N = 64
    def bump(c):
        i = torch.arange(N).float(); d = (i - c).abs(); d = torch.minimum(d, N - d)
        b = torch.exp(-d ** 2 / (2 * 3.0 ** 2)); return b / b.sum() * N * 0.1
    mem = HopfieldAssociativeMemory(N, steps=12); mem.reset()
    dW = mem.store(bump(20.0), rate=1.0)
    assert dW > 0, "store should induce a synaptic change"
    gen = torch.Generator().manual_seed(0)
    cue = (bump(20.0) * (torch.rand(N, generator=gen) > 0.5).float()).unsqueeze(0)   # 50% dropout
    cos = lambda a: torch.nn.functional.cosine_similarity(a.flatten(), bump(20.0).flatten(), dim=0).item()
    with_w = cos(mem.settle(cue, recurrent_gain=1.0, drive_decay=0.35).squeeze(0))
    no_w = cos(mem.settle(cue, recurrent_gain=0.0, drive_decay=0.35).squeeze(0))
    assert with_w > cos(cue.squeeze(0)) + 0.15, "recurrent weights should complete the cue"
    assert with_w > no_w + 0.15, "completion must require W_rec"


def test_signature_A_ach_encode_retrieve():
    """Emergent ACh signatures (measured, not trained): overlap-specific intrusion, recurrent (not
    non-storage) contamination at matched write energy, and W_rec-dependent completion."""
    from src.eval.neuromodulation import run_seed
    o = run_seed(0, reps=6)
    # (A1) intrusion is OVERLAP-SPECIFIC: a near field intrudes on the old memory, a far one does not
    assert o["intr_near_hi"] > o["intr_far_hi"] + 0.3
    assert o["A_specificity"] > 0.4
    # (A2) it is RECURRENT contamination, not non-storage: intrusion grows with encode gain at MATCHED ||ΔW||
    assert o["A_encode_effect"] > 0.15
    assert abs(o["dW_near_hi"] - o["dW_near_lo"]) < 0.3, "storage energy must be matched across the sweep"
    # (A3) retrieval completion REQUIRES the recurrent weights
    assert o["A_completion"] > 0.12
    assert abs(o["comp_noW"] - o["comp_cue"]) < 0.05, "no completion without W_rec"


def test_signature_B_ne_surprise_remap():
    """Emergent NE signatures: surprise tracks NOVELTY not change magnitude, and a surprise-triggered
    remap is adaptive on BOTH sides vs a matched no-reset+re-encode control."""
    from src.eval.neuromodulation import run_seed
    o = run_seed(0, reps=6)
    # (B1) novelty, not change: a big EXPECTED jump stays at the familiar floor; only novelty is surprising
    assert o["surp_novel"] > o["surp_familiar"] + 0.4
    assert o["surp_expbig"] < o["surp_familiar"] + 0.15
    assert o["B_auc_novel"] > 0.9
    # (B2) two-sided adaptive benefit vs the matched no-reset control
    assert o["B_benefit_new"] > 0.1, "remap should learn the new environment better"
    assert o["B_benefit_old"] > 0.1, "remap should protect the old map from overwrite"


def test_cortex_ach_modulates_recall():
    """The ACh gate is wired into BrainSpatialCortex: it changes the grid attractor's recall, and the
    default (ach=None) is identical to retrieval mode (ach=0.0)."""
    from src.models.neuro.brain_spatial_cortex import BrainSpatialCortex
    torch.manual_seed(0)
    cortex = BrainSpatialCortex(embed_dim=32, num_tokens=2)
    coords = torch.tensor([[35.69, 139.69], [51.5, -0.13]])
    base = cortex(coords)
    encode = cortex(coords, ach=1.0)     # encoding suppresses recurrent recall
    assert (base - encode).abs().max() > 1e-4, "ACh should modulate the cortex output"
    assert torch.allclose(base, cortex(coords, ach=0.0), atol=1e-6), "default == retrieval mode"
