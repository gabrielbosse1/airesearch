"""Colab system-RAM safety: device-first allocation, kill diagnostics, bounded media."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

import training.train as train
from models.encoders import MultimodalIngestor, TextSkillProjector
from models.foundation import CausalByteTransformer, FoundationConfig
from training.train import BatchBuilder, Trainer, load_config, smoke_config


def test_cgroup_headroom_subtracts_reclaimable_file_cache(monkeypatch):
    files = {
        "/proc/self/cgroup": "0::/user\n",
        "/sys/fs/cgroup/user/memory.current": "3000",
        "/sys/fs/cgroup/user/memory.max": "4000",
        "/sys/fs/cgroup/user/memory.peak": "3500",
        "/sys/fs/cgroup/user/memory.stat": "anon 1500\ninactive_file 1000\n",
        "/sys/fs/cgroup/user/memory.events": "oom 1\noom_kill 2\n",
    }
    monkeypatch.setattr(train, "_read_text", lambda path: files.get(str(path)))
    cgroup = train._cgroup_memory()
    assert cgroup["version"] == 2
    assert cgroup["current_bytes"] == 3000
    assert cgroup["anon_kernel_estimate_bytes"] == 2000
    assert cgroup["events"]["oom_kill"] == 2


def test_host_guard_fails_closed_when_known_headroom_is_low(monkeypatch):
    monkeypatch.setattr(train, "host_memory_snapshot", lambda: {
        "measured": True, "available_bytes": 500_000_000, "cgroup": {}})
    with pytest.raises(MemoryError, match="host RAM guard.*model allocation"):
        train.require_host_memory(2.0, stage="model allocation")


def test_sigkill_report_does_not_invent_an_oom_confirmation():
    before = {"cgroup": {"events": {"oom_kill": 4}}}
    same = {"cgroup": {"events": {"oom_kill": 4}}}
    report = train.explain_child_exit(-9, before=before, after=same)
    assert report["signal_name"] == "SIGKILL"
    assert report["confirmed_oom_kill"] is False
    assert "not proven" in report["probable_cause"]
    increased = {"cgroup": {"events": {"oom_kill": 5}}}
    confirmed = train.explain_child_exit(-9, before=before, after=increased)
    assert confirmed["confirmed_oom_kill"] is True
    assert confirmed["cgroup_oom_kill_delta"] == 1


def test_production_trainable_state_can_initialize_directly_on_meta(tmp_path):
    """Exercises the same device-first constructor without claiming a T4 run."""
    config = load_config()
    config["runtime"]["min_host_available_gb"] = 0
    config["output"].update(run_dir=str(tmp_path / "run"),
                            metrics_jsonl=str(tmp_path / "run" / "metrics.jsonl"))
    config["checkpoints"]["local_dir"] = str(tmp_path / "run" / "checkpoints")
    trainer = Trainer(config, device="meta", samples=iter(()), resume=False)
    assert trainer.model.parameters_count == 366_316_833
    for module in (trainer.model, trainer.text_projector, trainer.memory):
        assert all(p.device.type == "meta" for p in module.parameters())
        assert all(b.device.type == "meta" for b in module.buffers())
    stage = json.loads((tmp_path / "run" / "startup_stage.json").read_text())
    assert stage["stage"] == "allocate_trainable_state" and stage["status"] == "completed"


def test_media_queue_has_no_graph_until_pop_and_survives_adam_boundary(tmp_path):
    cfg = FoundationConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128,
                           max_seq_len=256, dropout=0.0, memory_dim=128,
                           refinement_steps=4, refinement_rank=32)
    model = CausalByteTransformer(cfg)
    ingestor = MultimodalIngestor(None, memory_dim=128, key_dim=64, encoder_dim=32)
    builder = BatchBuilder(model, {"image": ingestor}, TextSkillProjector(64, 128, 64), 128)
    builder.feed({"kind": "multimodal", "modality": "image", "caption": b"um gato",
                  "features": torch.randn(32)})
    assert len(builder._rows) == 2
    assert all(row.key is None and not row.features.requires_grad for row in builder._rows)
    first = builder.pop_batch(1)
    assert first["memory"].keys.requires_grad
    assert builder._rows[0].key is None  # leftover row still graph-free

    # Each media document produces two rows. With accum=1 an Adam update lands
    # between them; queued pre-built graphs used to become stale here.
    config = smoke_config()
    config["training"].update(micro_batch_size=1, grad_accum=1, tool_trajectory_rate=0.0)
    config["output"].update(run_dir=str(tmp_path / "run"),
                            metrics_jsonl=str(tmp_path / "run" / "metrics.jsonl"))
    config["checkpoints"]["local_dir"] = str(tmp_path / "run" / "checkpoints")

    def media():
        while True:
            yield "multimodal", {"kind": "multimodal", "modality": "image",
                                 "caption": b"um gato laranja", "features": torch.randn(768)}
    trainer = Trainer(config, device="cpu", samples=media(), resume=False)
    summary = trainer.run(max_steps=2)
    assert summary["steps"] == 2
    assert summary["batch_stats"]["media_to_text"] >= 1
    assert summary["batch_stats"]["text_to_media"] >= 1


def test_audio_encoder_input_is_bounded():
    model = CausalByteTransformer(FoundationConfig(d_model=64, n_layers=1, n_heads=4,
                                                   d_ff=128, max_seq_len=128, memory_dim=128))
    ingestor = MultimodalIngestor(None, memory_dim=128, key_dim=64, encoder_dim=32)
    seen = []

    class Encoder:
        def __call__(self, waveform):
            seen.append(tuple(waveform.shape))
            return torch.zeros(1, 32)
    ingestor.encoder = Encoder()
    builder = BatchBuilder(model, {"audio": ingestor}, TextSkillProjector(64, 128, 64),
                           128, max_audio_seconds=30)
    features = builder._features({"kind": "multimodal", "modality": "audio",
                                  "waveform": torch.zeros(60 * 16000), "sample_rate": 16000})
    assert seen == [(1, 30 * 16000)] and features.shape == (1, 32)


def test_cuda_oom_in_probe_halves_batch_then_accepts_measured_fit(monkeypatch):
    config = smoke_config()
    config["training"]["micro_batch_size"] = 2
    trainer = Trainer(config, device="cpu", samples=iter(()), resume=False)
    results = iter([torch.cuda.OutOfMemoryError("mock probe OOM"),
                    {"measured": True, "peak_gb": 9.0, "micro_batch": 1,
                     "seq_len": trainer.seq_len, "window_size": 64}])

    def probe():
        value = next(results)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(trainer, "probe_vram", probe)
    report = trainer.autotune_batch(13.5)
    assert trainer.micro_batch == 1 and report["peak_gb"] == 9.0


def test_cli_constructor_failure_retains_artifacts(tmp_path, monkeypatch):
    config = smoke_config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf8")
    run = tmp_path / "run"

    def broken(*args, **kwargs):
        raise RuntimeError("mock initialization failure")
    monkeypatch.setattr(train, "Trainer", broken)
    with pytest.raises(RuntimeError, match="mock initialization failure"):
        train.main(["--config", str(config_path), "--run-dir", str(run), "--local-only"])
    assert (run / "failure.json").exists()
    assert (run / "startup_events.jsonl").exists()
    assert (run / "debug_bundle.zip").exists()


def test_probe_moment_allocation_preserves_resumed_optimizer_state():
    first = torch.nn.Parameter(torch.ones(3))
    second = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([first, second], lr=1e-3)
    optimizer.state[first]["exp_avg"] = torch.full((3,), 7.0)
    optimizer.state[first]["exp_avg_sq"] = torch.full((3,), 11.0)
    optimizer.state[first]["step"] = torch.tensor(4.0)
    before = {key: value.clone() for key, value in optimizer.state[first].items()}
    added = train._allocate_adam_probe_moments(optimizer)
    assert torch.equal(optimizer.state[first]["exp_avg"], before["exp_avg"])
    assert second in optimizer.state and optimizer.state[second]["exp_avg"].shape == (2,)
    train._rollback_adam_probe_moments(optimizer, added)
    assert second not in optimizer.state
    assert all(torch.equal(optimizer.state[first][key], value) for key, value in before.items())
