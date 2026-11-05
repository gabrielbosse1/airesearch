"""Shared fixtures: one tiny architecture, identical classes to production.

Phase 24.2 offline contract: a session-wide guard replaces the REAL
``datasets.load_dataset`` with a loud failure (the sandbox cannot stream from
the Hub reliably, and a real load would write to disk). Tests that exercise
the data pipeline use ``mock_streaming_hub``, which serves tiny in-memory
``IterableDataset`` objects -- so the production streaming-enforcement and
buffered-shuffle code paths run for real, with zero network and zero disk.
"""
import pytest
import torch

from models.foundation import CausalByteTransformer, FoundationConfig


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(24)
    yield


@pytest.fixture(autouse=True, scope="session")
def _offline_hub_guard():
    """Fail loudly if any test reaches the REAL ``datasets.load_dataset``.

    Function-scoped mocks (``mock_streaming_hub``) patch on top of this guard
    and are restored back to it on teardown, so the whole suite stays offline.
    """
    import datasets

    def _forbidden(*args, **kwargs):
        raise RuntimeError(
            "real datasets.load_dataset was called during tests; use the "
            "mock_streaming_hub fixture (Phase 24.2 offline testing contract)")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(datasets, "load_dataset", _forbidden)
        yield


def _hub_rows(kwargs):
    """Tiny deterministic rows per dataset id -- in memory, never on disk."""
    path = kwargs.get("path", "")
    if path in ("TucanoBR/GigaVerbo", "HuggingFaceFW/fineweb-2"):
        return [{"text": f"O Brasil é um país da América do Sul. documento {i}."}
                for i in range(120)]
    if path in ("bigcode/the-stack", "bigcode/the-stack-smol"):
        return [{"content": f"def funcao_{i}(a, b):\n    return a + b + {i}\n"}
                for i in range(120)]
    if path == "kakaobrain/coyo-700m":
        # Dead URL chain: exercises the embedded-image fallback fully offline.
        return []
    if path == "nlphuji/mscoco_2014_5k_test_image_text_retrieval":
        from PIL import Image
        image = Image.new("RGB", (16, 16), color=(120, 30, 200))
        return [{"caption": [f"um gato laranja número {i}"], "image": image.copy()}
                for i in range(120)]
    if path in ("openslr/librispeech_asr", "hf-internal-testing/librispeech_asr_dummy"):
        return [{"audio": {"array": [0.05] * 3200, "sampling_rate": 16000},
                 "text": f"fala curta em português {i}"} for i in range(120)]
    if path == "HuggingFaceFW/fineweb":
        return [{"text": f"The quick brown fox jumps over the lazy dog. doc {i}."}
                for i in range(120)]
    return [{"text": f"generic row {i}"} for i in range(8)]


@pytest.fixture
def mock_streaming_hub(monkeypatch):
    """Patch ``datasets.load_dataset`` -> tiny in-memory ``IterableDataset``s.

    Returns the recorded call list so tests can assert ``streaming=True`` on
    EVERY call. The returned objects are REAL HF ``IterableDataset`` instances
    (``from_list``), so ``assert_streaming`` and the buffered ``.shuffle`` run
    for real. Mirrors the Hub API: without ``streaming=True`` it would return
    a map-style ``Dataset`` -- which the production guard must reject.
    """
    import datasets

    calls: list[dict] = []

    def fake_load_dataset(**kwargs):
        calls.append(dict(kwargs))
        rows = _hub_rows(kwargs)
        if kwargs.get("streaming") is not True:
            return datasets.Dataset.from_list(rows)
        return datasets.IterableDataset.from_list(rows)

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return calls


@pytest.fixture
def tiny_config():
    return FoundationConfig(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                            d_ff=128, max_seq_len=128, dropout=0.0,
                            memory_dim=128, refinement_steps=4, refinement_rank=32)


@pytest.fixture
def tiny_model(tiny_config):
    return CausalByteTransformer(tiny_config)
