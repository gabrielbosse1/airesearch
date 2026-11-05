"""Shared fixtures: one tiny architecture, identical classes to production."""
import pytest
import torch

from models.foundation import CausalByteTransformer, FoundationConfig


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(24)
    yield


@pytest.fixture
def tiny_config():
    return FoundationConfig(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                            d_ff=128, max_seq_len=128, dropout=0.0,
                            memory_dim=128, refinement_steps=4, refinement_rank=32)


@pytest.fixture
def tiny_model(tiny_config):
    return CausalByteTransformer(tiny_config)
