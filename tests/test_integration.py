"""
tests/test_integration.py
Integration tests — wire multiple components together without the full LLM.
Tests the spatial encoding → fusion → output pipeline end-to-end.
"""
import pytest
import torch
import torch.nn as nn

from src.models.coord_embedder import CoordinateEmbedderWithTokens
from src.models.grid_cell_encoder import GridCellEncoderWithTokens
from src.models.place_cell_memory import HippocampalMemory
from src.models.fusion import MultiScaleSpatialFusion
from src.models.neuromodulation import SpatialNeuromodulator, AdaptiveGain, PredictionErrorGate
from src.models.predictive_coding import SpatialPredictiveCoding
from src.eval.metrics import mean_haversine_error


HIDDEN = 64
BATCH  = 2
SEQ    = 8


@pytest.fixture
def coords():
    return torch.tensor([[35.69, 139.69], [51.51, -0.13]], dtype=torch.float32)


@pytest.fixture
def text_hidden():
    return torch.randn(BATCH, SEQ, HIDDEN)


# ── Grid + Fourier → Fusion ────────────────────────────────────────────────────

def test_grid_plus_fourier_fusion(coords, text_hidden):
    """Grid cell + Fourier coord tokens both feed into cross-attention fusion."""
    grid_enc  = GridCellEncoderWithTokens(embed_dim=HIDDEN, num_modules=3)
    coord_enc = CoordinateEmbedderWithTokens(embed_dim=HIDDEN, num_tokens=4)
    fusion    = MultiScaleSpatialFusion(HIDDEN, num_heads=4, num_layers=2)

    grid_tokens  = grid_enc(coords)   # (B, 3, D)
    coord_tokens = coord_enc(coords)  # (B, 4, D)
    spatial = torch.cat([grid_tokens, coord_tokens], dim=1)  # (B, 7, D)

    out = fusion(text_hidden, spatial)
    assert out.shape == text_hidden.shape
    assert not torch.isnan(out).any()


# ── Hippocampal memory pipeline ────────────────────────────────────────────────

def test_hippocampal_store_and_retrieve_pipeline(coords):
    """Store context then retrieve it — retrieved embedding is non-zero."""
    mem = HippocampalMemory(embed_dim=HIDDEN, num_cells=64, buffer_size=16)
    context = torch.randn(BATCH, HIDDEN)

    mem.store(coords, context)
    retrieved = mem.retrieve(coords, top_k=2)

    assert retrieved.shape == (BATCH, HIDDEN)
    assert retrieved.abs().sum() > 0, "Retrieved memory should be non-zero after storing"


# ── Neuromodulation pipeline ───────────────────────────────────────────────────

def test_full_neuromodulation_pipeline(text_hidden):
    """AdaptiveGain → PredictionErrorGate → SpatialNeuromodulator chain."""
    gain_ctrl  = AdaptiveGain(HIDDEN)
    pe_gate    = PredictionErrorGate(HIDDEN)
    neuromod   = SpatialNeuromodulator(HIDDEN)

    # Adaptive gain
    modulated, uncertainty = gain_ctrl(text_hidden)
    assert modulated.shape == text_hidden.shape
    assert uncertainty.shape == (BATCH,)

    # Prediction error gate (operates on pooled spatial repr)
    spatial_repr = modulated.mean(dim=1)   # (B, D)
    gated = pe_gate(spatial_repr, uncertainty)
    assert gated.shape == (BATCH, HIDDEN)

    # Neuromodulator
    context = torch.randn(BATCH, HIDDEN)
    neuro_out = neuromod(text_hidden, context)
    assert neuro_out.shape == text_hidden.shape
    assert not torch.isnan(neuro_out).any()


# ── Predictive coding pipeline ─────────────────────────────────────────────────

def test_predictive_coding_produces_training_loss(coords):
    """Predictive coding auxiliary loss is a scalar and differentiable."""
    pc = SpatialPredictiveCoding(spatial_dim=HIDDEN, llm_dim=HIDDEN, num_levels=3)
    x = torch.randn(BATCH, HIDDEN)
    _, pc_loss = pc(x)

    assert pc_loss.shape == ()
    pc_loss.backward()   # Must not raise


# ── Full spatial encoding pipeline (no LLM) ────────────────────────────────────

def test_full_spatial_pipeline_no_llm(coords, text_hidden):
    """
    Grid cells + Fourier + HippocampalMemory + Neuromodulation + Fusion.
    This is the complete spatial encoding stack without the LLM backbone.
    """
    grid_enc  = GridCellEncoderWithTokens(embed_dim=HIDDEN, num_modules=3)
    coord_enc = CoordinateEmbedderWithTokens(embed_dim=HIDDEN, num_tokens=4)
    hippo     = HippocampalMemory(embed_dim=HIDDEN, num_cells=64, buffer_size=16)
    gain_ctrl = AdaptiveGain(HIDDEN)
    fusion    = MultiScaleSpatialFusion(HIDDEN, num_heads=4, num_layers=2)

    # 1. Encode coords spatially
    grid_tokens  = grid_enc(coords)
    coord_tokens = coord_enc(coords)
    spatial      = torch.cat([grid_tokens, coord_tokens], dim=1)  # (B, 7, D)

    # 2. Hippocampal memory augmentation
    mem_context  = text_hidden.mean(dim=1)               # (B, D) — proxy
    hippo_out    = hippo(coords, context=mem_context)     # (B, D)
    hippo_tokens = hippo_out.unsqueeze(1)                 # (B, 1, D)
    spatial      = torch.cat([spatial, hippo_tokens], dim=1)  # (B, 8, D)

    # 3. Adaptive gain on text
    text_mod, _ = gain_ctrl(text_hidden)

    # 4. Fuse into text hidden
    fused = fusion(text_mod, spatial)

    assert fused.shape == text_hidden.shape
    assert not torch.isnan(fused).any()
    assert not torch.isinf(fused).any()


# ── Metrics sanity ─────────────────────────────────────────────────────────────

def test_metrics_pipeline():
    """Verify the eval metrics work on realistic-scale outputs."""
    preds = [(35.7, 139.7), (51.5, -0.1)]
    trues = [(35.69, 139.69), (48.86, 2.35)]  # Tokyo→Tokyo OK, London→Paris wrong
    m = mean_haversine_error(preds, trues)

    assert m["mean_km"] > 0
    assert m["within_10km"] == 0.5   # Tokyo jitter (~1.4km) within 10km; London→Paris not
    assert m["within_100km"] == 0.5  # London→Paris is ~340km, not within 100km
