"""Phase 24.2: STRICT streaming enforcement -- mocked Hub, zero network/disk.

Every test here runs against the ``mock_streaming_hub`` fixture (tiny
in-memory ``IterableDataset`` objects), so the production enforcement path
(``open_stream`` -> ``assert_streaming`` -> buffered ``.shuffle``) executes
for real without touching the network or the disk.
"""
import ast
from pathlib import Path

import pytest
import torch
import yaml

import data.sources as sources
from data.sources import (DEFAULT_SOURCES, SHUFFLE_BUFFER_SIZE, MixtureSampler,
                          SourceSpec, build_mixture, open_stream, source_samples)

REPO = Path(__file__).resolve().parent.parent
GROUPS = ("ptbr", "code", "multimodal", "english")


# ---------------------------------------------------------------------------
# 24.2.1 -- streaming=True everywhere + IterableDataset verification
# ---------------------------------------------------------------------------
def test_every_default_spec_requests_streaming():
    for specs in DEFAULT_SOURCES.values():
        for spec in specs:
            kwargs = spec.load_kwargs()
            assert kwargs["streaming"] is True, spec.id
            assert isinstance(kwargs["split"], str) and kwargs["split"], spec.id


def test_open_stream_returns_a_verified_iterable_dataset(mock_streaming_hub):
    from datasets import IterableDataset
    spec = DEFAULT_SOURCES["ptbr"][0]
    stream = open_stream(spec, seed=24, shuffle_buffer=64)
    assert isinstance(stream, IterableDataset)
    assert not hasattr(stream, "__len__")          # streams are never sized
    first = next(iter(stream))
    assert isinstance(first["text"], str)
    assert mock_streaming_hub and mock_streaming_hub[0]["streaming"] is True


def test_build_mixture_calls_load_dataset_only_with_streaming_true(mock_streaming_hub):
    mix = build_mixture(seed=24)
    assert isinstance(mix, MixtureSampler)
    # One recorded call per opened source (coyo is empty -> mscoco fallback).
    assert len(mock_streaming_hub) >= len(GROUPS)
    for call in mock_streaming_hub:
        assert call["streaming"] is True, call
    opened = {call["path"] for call in mock_streaming_hub}
    assert "TucanoBR/GigaVerbo" in opened
    assert "bigcode/the-stack" in opened
    assert "HuggingFaceFW/fineweb" in opened


def test_disk_backed_dataset_is_rejected(mock_streaming_hub, monkeypatch):
    import datasets
    # A loader that ignores streaming and returns a map-style Dataset: the
    # exact failure mode that would fill the Colab disk. Must raise TypeError.
    monkeypatch.setattr(datasets, "load_dataset",
                        lambda **kw: datasets.Dataset.from_list([{"text": "oi"}]))
    with pytest.raises(TypeError, match="IterableDataset"):
        open_stream(SourceSpec("fake/disk"))
    # An unresolved IterableDatasetDict (split missing) is also rejected.
    monkeypatch.setattr(datasets, "load_dataset", lambda **kw: datasets.IterableDatasetDict(
        {"train": datasets.IterableDataset.from_list([{"text": "oi"}])}))
    with pytest.raises(TypeError, match="IterableDataset"):
        open_stream(SourceSpec("fake/dict"))


def test_tampered_non_streaming_spec_never_reaches_load_dataset(mock_streaming_hub, monkeypatch):
    monkeypatch.setattr(SourceSpec, "load_kwargs",
                        lambda self: {"path": self.id, "split": "train", "streaming": False})
    with pytest.raises(ValueError, match="streaming=True is mandatory"):
        open_stream(SourceSpec("fake/bad"))
    assert mock_streaming_hub == []                # rejected BEFORE any load
    # A missing split would return an IterableDatasetDict: also rejected early.
    monkeypatch.setattr(SourceSpec, "load_kwargs",
                        lambda self: {"path": self.id, "streaming": True})
    with pytest.raises(ValueError, match="concrete split"):
        open_stream(SourceSpec("fake/nosplit"))
    assert mock_streaming_hub == []


def test_build_mixture_rejects_a_non_streaming_config(mock_streaming_hub):
    with pytest.raises(ValueError, match="streaming"):
        build_mixture({"streaming": False})
    assert mock_streaming_hub == []                # nothing was even opened


# ---------------------------------------------------------------------------
# 24.2.2 -- iterable-safe shuffling (seed + buffer_size) and step-based loop
# ---------------------------------------------------------------------------
def test_shuffle_is_called_with_seed_and_buffer_size(mock_streaming_hub, monkeypatch):
    from datasets import IterableDataset
    seen = []
    original = IterableDataset.shuffle

    def spy(self, *args, **kwargs):
        seen.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(IterableDataset, "shuffle", spy)
    open_stream(DEFAULT_SOURCES["ptbr"][0], seed=7, shuffle_buffer=123)
    assert seen == [{"seed": 7, "buffer_size": 123}]


def test_shuffle_is_seeded_deterministic_and_order_breaking(mock_streaming_hub):
    spec = DEFAULT_SOURCES["ptbr"][0]

    def first_texts(seed):
        stream = open_stream(spec, seed=seed, shuffle_buffer=16)
        return [row["text"] for _, row in zip(range(12), stream)]

    assert first_texts(24) == first_texts(24)      # deterministic given seed
    assert first_texts(24) != first_texts(7)       # seed actually changes order
    ordered = [f"O Brasil é um país da América do Sul. documento {i}."
               for i in range(12)]
    assert first_texts(24) != ordered              # source order is broken


def test_default_buffer_is_10000_and_config_drives_it(mock_streaming_hub, monkeypatch):
    from datasets import IterableDataset
    assert SHUFFLE_BUFFER_SIZE == 10_000
    config = yaml.safe_load((REPO / "configs" / "t4_scaled.yaml").read_text(encoding="utf8"))
    assert config["data"]["streaming"] is True
    assert config["data"]["shuffle_buffer_size"] == SHUFFLE_BUFFER_SIZE

    seen = []
    original = IterableDataset.shuffle

    def spy(self, *args, **kwargs):
        seen.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(IterableDataset, "shuffle", spy)
    build_mixture({"shuffle_buffer_size": 8}, seed=3)
    assert seen and all(call["buffer_size"] == 8 and call["seed"] == 3 for call in seen)
    with pytest.raises(ValueError):
        build_mixture({"shuffle_buffer_size": -1})


def test_zero_buffer_disables_shuffle_for_debugging(mock_streaming_hub, monkeypatch):
    from datasets import IterableDataset

    def forbidden(self, *args, **kwargs):
        raise AssertionError("shuffle must not run with buffer_size=0")

    monkeypatch.setattr(IterableDataset, "shuffle", forbidden)
    stream = open_stream(DEFAULT_SOURCES["ptbr"][0], shuffle_buffer=0)
    assert next(iter(stream))["text"].endswith("documento 0.")  # raw source order


def test_build_mixture_streams_the_exact_quota_end_to_end_offline(mock_streaming_hub, monkeypatch):
    # Belt and braces: even a stray image URL fetch stays offline.
    monkeypatch.setattr(sources, "_fetch_url", lambda *a, **k: None)
    mix = build_mixture(seed=24)
    assert not hasattr(mix, "__len__")
    batch = mix.next_batch(100)
    counts = {name: 0 for name in GROUPS}
    for group, sample in batch:
        counts[group] += 1
        assert sample["kind"] in ("text", "multimodal")
    assert counts == {"ptbr": 50, "code": 25, "multimodal": 15, "english": 10}
    assert not mix.failures                        # coyo fell back to embedded MSCOCO
    # The mock rows are finite: exhaustion is reported, never an epoch restart.
    drained = 0
    try:
        while True:
            mix.next_batch(100)
            drained += 1
            assert drained < 100, "finite mock streams must exhaust, not loop"
    except StopIteration:
        pass
    assert mix.failures                            # exhausted groups are logged


def test_source_samples_threads_the_shuffle_seed(mock_streaming_hub, monkeypatch):
    seen = {}
    original = sources._open_rows

    def spy(spec, **kwargs):
        seen.update(kwargs)
        return original(spec, **kwargs)

    monkeypatch.setattr(sources, "_open_rows", spy)
    next(source_samples(DEFAULT_SOURCES["english"][0], seed=11, shuffle_buffer=32))
    assert seen == {"seed": 11, "shuffle_buffer": 32}


def test_trainer_is_step_based_on_a_streaming_mixture(mock_streaming_hub, tmp_path):
    from training.train import Trainer, smoke_config
    config = smoke_config()
    config["training"]["tool_trajectory_rate"] = 0.0
    config["checkpoints"]["local_dir"] = str(tmp_path / "ckpt")
    config["output"]["metrics_jsonl"] = str(tmp_path / "metrics.jsonl")
    mix = build_mixture(seed=24)
    trainer = Trainer(config, device="cpu", samples=mix, resume=False)
    # max_steps=2 on a stream with no len(): the loop must stop on STEPS.
    summary = trainer.run(max_steps=2)
    assert summary["steps"] == 2
    assert summary["stop_reason"] == "max_steps"
    assert not hasattr(trainer.samples, "__len__")


def test_trainer_rejects_sized_epoch_style_samples(tmp_path):
    from training.train import Trainer, smoke_config
    config = smoke_config()
    config["checkpoints"]["local_dir"] = str(tmp_path / "ckpt")
    config["output"]["metrics_jsonl"] = str(tmp_path / "metrics.jsonl")
    sized = [{"kind": "text", "data": b"x" * 300}] * 10   # a list HAS __len__
    trainer = Trainer(config, device="cpu", samples=sized, resume=False)
    with pytest.raises(TypeError, match="streaming iterator"):
        trainer.run(max_steps=1)


# ---------------------------------------------------------------------------
# Static enforcement: the contract cannot regress by adding a new call site
# ---------------------------------------------------------------------------
def _production_python_files():
    files = []
    for folder in ("data", "training", "models"):
        files.extend(sorted((REPO / folder).glob("*.py")))
    return files


def _notebook_code_sources():
    import json
    nb = json.loads((REPO / "pretrain_4h.ipynb").read_text(encoding="utf8"))
    return ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]


def test_load_dataset_has_exactly_one_call_site_inside_open_stream():
    calls = []          # (file, lineno, inside_open_stream)
    for path in _production_python_files():
        tree = ast.parse(path.read_text(encoding="utf8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "open_stream":
                for inner in ast.walk(node):
                    if (isinstance(inner, ast.Call)
                            and getattr(inner.func, "id", getattr(inner.func, "attr", None)) == "load_dataset"):
                        calls.append((path.name, inner.lineno, True))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", getattr(node.func, "attr", None)) == "load_dataset"):
                if not any(c[0] == path.name and c[1] == node.lineno for c in calls):
                    calls.append((path.name, node.lineno, False))
    assert calls, "open_stream must call load_dataset"
    assert all(inside for _, _, inside in calls), f"stray load_dataset call sites: {calls}"
    assert {name for name, _, _ in calls} == {"sources.py"}
    # The notebook never calls load_dataset directly: it goes through open_stream.
    for source in _notebook_code_sources():
        assert "load_dataset(" not in source


def test_every_seeded_shuffle_passes_buffer_size():
    sources_to_check = [(str(path), path.read_text(encoding="utf8"))
                        for path in _production_python_files()]
    sources_to_check += [("pretrain_4h.ipynb", src) for src in _notebook_code_sources()]
    for name, text in sources_to_check:
        for node in ast.walk(ast.parse(text)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "shuffle"):
                keywords = {kw.arg for kw in node.keywords}
                if "seed" in keywords:
                    assert "buffer_size" in keywords, (
                        f"{name}:{node.lineno}: seeded .shuffle() on a stream must "
                        "pass buffer_size (iterable-safe reservoir)")


def test_training_loop_has_no_epoch_or_len_over_data():
    text = (REPO / "training" / "train.py").read_text(encoding="utf8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        # No len(dataset)/len(dataloader)/len(samples) anywhere in the trainer.
        if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "len"):
            if node.args and isinstance(node.args[0], ast.Attribute):
                assert node.args[0].attr not in ("dataset", "dataloader", "samples", "mixture")
            if node.args and isinstance(node.args[0], ast.Name):
                assert node.args[0].id not in ("dataset", "dataloader", "samples", "mixture")
    # The loop bound is max_steps, and no `for epoch` loop exists.
    assert "while self.step < max_steps:" in text
    assert "for epoch" not in text
