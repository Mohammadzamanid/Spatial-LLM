"""
tests/test_real_data.py
Tests against real generated spatial QA data (data/processed/train.jsonl).
Validates the full data pipeline end-to-end with actual records.
"""
import json
import math
import pytest
import torch
from pathlib import Path

# ── Fixtures ───────────────────────────────────────────────────────────────────

TRAIN_PATH = Path("data/processed/train.jsonl")
VAL_PATH   = Path("data/processed/val.jsonl")


@pytest.fixture(scope="module")
def train_records():
    if not TRAIN_PATH.exists():
        pytest.skip("train.jsonl not found — run: python -m src.data.synthetic")
    records = []
    with open(TRAIN_PATH) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


@pytest.fixture(scope="module")
def val_records():
    if not VAL_PATH.exists():
        pytest.skip("val.jsonl not found — run: python -m src.data.synthetic")
    records = []
    with open(VAL_PATH) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


# ── Data integrity tests ───────────────────────────────────────────────────────

def test_train_jsonl_has_records(train_records):
    assert len(train_records) > 0, "Train set should not be empty"


def test_val_jsonl_has_records(val_records):
    assert len(val_records) > 0, "Val set should not be empty"


def test_all_records_have_required_fields(train_records):
    for i, rec in enumerate(train_records):
        assert "question" in rec, f"Record {i} missing 'question'"
        assert "answer"   in rec, f"Record {i} missing 'answer'"
        assert "lat"      in rec, f"Record {i} missing 'lat'"
        assert "lon"      in rec, f"Record {i} missing 'lon'"


def test_coordinates_in_valid_range(train_records):
    for i, rec in enumerate(train_records):
        assert -90.0 <= rec["lat"] <= 90.0,   f"Record {i} lat out of range: {rec['lat']}"
        assert -180.0 <= rec["lon"] <= 180.0, f"Record {i} lon out of range: {rec['lon']}"


def test_questions_non_empty(train_records):
    for i, rec in enumerate(train_records):
        assert len(rec["question"].strip()) > 0, f"Record {i} has empty question"
        assert len(rec["answer"].strip()) > 0,   f"Record {i} has empty answer"


def test_no_duplicate_questions(train_records):
    questions = [r["question"] for r in train_records]
    # Allow some duplicates from template variation but not 100%
    unique_ratio = len(set(questions)) / len(questions)
    assert unique_ratio > 0.5, f"Too many duplicate questions: {unique_ratio:.2%} unique"


# ── Coordinate embedding on real data ─────────────────────────────────────────

def test_coord_embedder_on_real_data(train_records):
    from src.models.coord_embedder import CoordinateEmbedder
    model = CoordinateEmbedder(embed_dim=128)
    model.eval()

    # Take first 16 records
    batch = train_records[:16]
    coords = torch.tensor([[r["lat"], r["lon"]] for r in batch], dtype=torch.float32)
    with torch.no_grad():
        emb = model(coords)

    assert emb.shape == (16, 128)
    assert not torch.isnan(emb).any(), "NaN in embeddings from real coordinates"
    assert not torch.isinf(emb).any(), "Inf in embeddings from real coordinates"


def test_grid_cell_encoder_on_real_data(train_records):
    from src.models.grid_cell_encoder import GridCellEncoderWithTokens
    model = GridCellEncoderWithTokens(embed_dim=64, num_modules=4)
    model.eval()

    batch = train_records[:8]
    coords = torch.tensor([[r["lat"], r["lon"]] for r in batch], dtype=torch.float32)
    with torch.no_grad():
        tokens = model(coords)

    assert tokens.shape == (8, 4, 64)
    assert not torch.isnan(tokens).any()


def test_place_cell_memory_on_real_data(train_records):
    from src.models.place_cell_memory import HippocampalMemory
    mem = HippocampalMemory(embed_dim=64, num_cells=128, buffer_size=32)

    batch = train_records[:4]
    coords = torch.tensor([[r["lat"], r["lon"]] for r in batch], dtype=torch.float32)
    context = torch.randn(4, 64)

    mem.store(coords, context)
    retrieved = mem.retrieve(coords)

    assert retrieved.shape == (4, 64)
    assert retrieved.abs().sum() > 0


def test_different_cities_have_different_embeddings(train_records):
    """Real-world test: embeddings for different cities must differ."""
    from src.models.grid_cell_encoder import GridCellEncoder
    model = GridCellEncoder(embed_dim=128, num_modules=4)
    model.eval()

    # Find records from at least 2 different cities
    seen = {}
    for r in train_records:
        city = r["answer"].split()[0] if r["answer"] else "?"
        if city not in seen:
            seen[city] = (r["lat"], r["lon"])
        if len(seen) >= 2:
            break

    cities = list(seen.values())
    coords = torch.tensor(cities[:2], dtype=torch.float32)

    with torch.no_grad():
        embs = model(coords)

    cos_sim = torch.nn.functional.cosine_similarity(
        embs[0].unsqueeze(0), embs[1].unsqueeze(0)
    ).item()
    assert cos_sim < 0.99, f"Different cities should have distinct embeddings, got cos_sim={cos_sim:.4f}"


# ── Haversine sanity on real data ──────────────────────────────────────────────

def test_haversine_on_real_coord_pairs(train_records):
    """Self-distance on all real records must be zero."""
    from src.eval.metrics import haversine_km
    for rec in train_records[:20]:
        d = haversine_km(rec["lat"], rec["lon"], rec["lat"], rec["lon"])
        assert d == pytest.approx(0.0, abs=1e-5), f"Self-distance non-zero for {rec}"


def test_loss_on_real_coord_batch(train_records):
    """HaversineLoss backpropagates through real coordinate batches."""
    from src.training.loss import HaversineLoss
    loss_fn = HaversineLoss()

    batch = train_records[:8]
    pred  = torch.tensor([[r["lat"] + 0.1, r["lon"] + 0.1] for r in batch],
                         requires_grad=True, dtype=torch.float32)
    true  = torch.tensor([[r["lat"], r["lon"]] for r in batch], dtype=torch.float32)

    loss = loss_fn(pred, true)
    assert loss.item() > 0
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
