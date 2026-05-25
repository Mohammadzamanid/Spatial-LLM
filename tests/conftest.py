"""
tests/conftest.py
Shared pytest fixtures for Spatial-LLM test suite.
"""
import pytest
import torch


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_coords():
    """Batch of 4 real-world coordinates: Tokyo, London, NYC, Tehran."""
    return torch.tensor([
        [35.6895,  139.6917],   # Tokyo
        [51.5074,   -0.1278],   # London
        [40.7128,  -74.0060],   # New York
        [35.6892,   51.3890],   # Tehran
    ], dtype=torch.float32)


@pytest.fixture
def single_coord():
    return torch.tensor([[35.6895, 139.6917]], dtype=torch.float32)


@pytest.fixture
def batch_text_hidden():
    """Fake LLM hidden states: batch=2, seq_len=10, hidden=64."""
    return torch.randn(2, 10, 64)


@pytest.fixture
def batch_spatial_tokens():
    """Fake spatial tokens: batch=2, num_tokens=8, hidden=64."""
    return torch.randn(2, 8, 64)
