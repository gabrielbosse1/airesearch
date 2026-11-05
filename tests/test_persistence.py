"""Durable Colab persistence: secrets, retries, artifacts, crash/resume contracts."""
from __future__ import annotations

import json
import os
import types
import zipfile

import pytest
import torch

import training.train as train
from training.train import (Trainer, error_record, file_sha256, hf_download,
                            hf_upload, load_config, redact_secrets, smoke_config,
                            smoke_samples, source_snapshot, sync_run_folder)


class _FakeApi:
    instances = []
    fail_uploads = 0

    def __init__(self, token=None):
        self.token = token
        self.calls = []
        _FakeApi.instances.append(self)

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))
        return types.SimpleNamespace(url=f"https://huggingface.co/{repo_id}")

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs))
        if _FakeApi.fail_uploads > 0:
            _FakeApi.fail_uploads -= 1
            raise ConnectionError("temporary network failure")
        return types.SimpleNamespace(oid="commit123", commit_url="https://example/commit")

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        return types.SimpleNamespace(oid="folder123", commit_url="https://example/folder")


@pytest.fixture
def fake_hub(monkeypatch):
    import huggingface_hub
    _FakeApi.instances = []
    _FakeApi.fail_uploads = 0
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setattr(train.time, "sleep", lambda _seconds: None)
    return _FakeApi


def test_default_repo_and_cadences_are_colab_safe():
    config = load_config()
    assert config["checkpoints"]["hf_repo"] == "gabrielbosse9/airesearch2"
    assert config["training"]["checkpoint_minutes"] == 10
    assert config["training"]["checkpoint_every"] == 100
    assert config["checkpoints"]["upload_minutes"] == 5
    assert config["checkpoints"]["fail_on_upload_error"] is True
    assert config["output"]["run_dir"] == "runs/colab_latest"


def test_redaction_removes_env_and_explicit_tokens(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_env_secret_value")
    text = "Authorization: Bearer abc.def token=query-secret hf_env_secret_value hf_explicit_secret"
    clean = redact_secrets(text, "hf_explicit_secret")
    assert "hf_env_secret_value" not in clean
    assert "hf_explicit_secret" not in clean
    assert "abc.def" not in clean and "query-secret" not in clean
    assert clean.count("[REDACTED]") >= 4


def test_hf_upload_retries_then_succeeds_without_leaking_token(fake_hub, tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("{}", encoding="utf8")
    fake_hub.fail_uploads = 2
    result = hf_upload(path, "user/repo", "runs/a.json", token="hf_private_token", retries=3)
    assert result["uploaded"] is True and result["attempts"] == 3
    assert result["commit"] == "commit123"
    assert all(api.token == "hf_private_token" for api in fake_hub.instances)
    assert "hf_private_token" not in json.dumps(result)


def test_hf_upload_failure_is_loud_by_default(fake_hub, tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("{}", encoding="utf8")
    fake_hub.fail_uploads = 3
    with pytest.raises(RuntimeError, match="HF upload failed"):
        hf_upload(path, "user/repo", "x.json", token="hf_private_token", retries=3)
    fake_hub.fail_uploads = 1
    quiet = hf_upload(path, "user/repo", "x.json", token="hf_private_token", retries=1, strict=False)
    assert quiet["uploaded"] is False
    assert "hf_private_token" not in json.dumps(quiet)


def test_folder_sync_excludes_checkpoints(fake_hub, tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "huge.pt").write_bytes(b"not uploaded")
    (tmp_path / "metrics.jsonl").write_text("{}\n", encoding="utf8")
    result = sync_run_folder(tmp_path, "user/repo", "runs/run/session", token="secret")
    assert result["uploaded"] is True
    kwargs = fake_hub.instances[-1].calls[-1][1]
    assert "*.pt" in kwargs["ignore_patterns"]
    assert "checkpoints/**" in kwargs["ignore_patterns"]
    assert "*.jsonl" in kwargs["allow_patterns"]


def test_hf_download_missing_is_distinct_from_auth_failure(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    def missing(**kwargs):
        raise EntryNotFoundError("missing")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    result = hf_download("user/repo", "checkpoints/checkpoint_latest.pt", tmp_path / "c.pt")
    assert result["downloaded"] is False and result["not_found"] is True

    def denied(**kwargs):
        raise PermissionError("denied hf_private_token")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", denied)
    with pytest.raises(RuntimeError) as exc:
        hf_download("user/repo", "x.pt", tmp_path / "x.pt", token="hf_private_token")
    assert "hf_private_token" not in str(exc.value)


def test_source_snapshot_is_code_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf8")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "data.pt").write_bytes(b"data")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret", encoding="utf8")
    (tmp_path / "credentials.env").write_text("secret", encoding="utf8")
    bundle = source_snapshot(tmp_path / "source.zip")
    with zipfile.ZipFile(bundle) as archive:
        assert archive.namelist() == ["app.py"]


def test_trainer_writes_complete_local_debug_artifacts(tmp_path, monkeypatch):
    secret = "hf_" + os.urandom(12).hex()
    monkeypatch.setenv("HF_TOKEN", secret)
    config = smoke_config()
    config["output"]["run_dir"] = str(tmp_path / "run")
    config["output"]["metrics_jsonl"] = str(tmp_path / "run" / "metrics.jsonl")
    config["checkpoints"]["local_dir"] = str(tmp_path / "run" / "checkpoints")
    config["training"]["checkpoint_every"] = 1

    def noisy_samples():
        print("token accidentally printed:", os.environ["HF_TOKEN"])
        yield from smoke_samples(128)

    trainer = Trainer(config, device="cpu", samples=noisy_samples(), resume=False)
    summary = trainer.run(max_steps=2)
    run = tmp_path / "run"
    for name in ("console.log", "events.jsonl", "metrics.jsonl", "manifest.json",
                 "diagnostics.json", "config.resolved.yaml", "environment.json",
                 "provenance.json", "source_snapshot.zip", "summary.json", "debug_bundle.zip"):
        assert (run / name).exists(), name
    assert secret not in (run / "console.log").read_text(encoding="utf8")
    assert summary["tokens"] > 0 and summary["steps"] == 2
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert sum(event.get("event") == "optimizer_step" for event in events) == 2
    with zipfile.ZipFile(run / "debug_bundle.zip") as archive:
        names = archive.namelist()
        assert not any(name.endswith(".pt") or "checkpoints" in name for name in names)
        assert "console.log" in names and "source_snapshot.zip" in names
    checkpoint = run / "checkpoints" / "checkpoint_latest.pt"
    manifest = json.loads((run / "checkpoint_manifest.json").read_text())
    assert manifest["sha256"] == file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert secret not in json.dumps(payload["training_state"], default=str)


def test_remote_startup_failure_fails_closed_and_keeps_crash_bundle(tmp_path, monkeypatch):
    config = smoke_config()
    config["output"]["run_dir"] = str(tmp_path / "run")
    config["output"]["metrics_jsonl"] = str(tmp_path / "run" / "metrics.jsonl")
    config["checkpoints"]["local_dir"] = str(tmp_path / "run" / "checkpoints")
    config["checkpoints"].update(hf_repo="user/repo", fail_on_upload_error=True)

    def denied(*args, **kwargs):
        raise PermissionError("write denied")
    monkeypatch.setattr(train, "verify_hf_access", denied)
    monkeypatch.setattr(train, "hf_upload", lambda *a, **k: {"uploaded": False, "reason": "mock"})
    monkeypatch.setattr(train, "sync_run_folder", lambda *a, **k: {"uploaded": False, "reason": "mock"})
    trainer = Trainer(config, device="cpu", samples=smoke_samples(128), resume=False)
    with pytest.raises(PermissionError, match="write denied"):
        trainer.run(max_steps=1)
    run = tmp_path / "run"
    assert trainer.step == 0
    assert (run / "failure.json").exists()
    assert (run / "debug_bundle.zip").exists()
    assert (run / "checkpoints" / "checkpoint_latest.pt").exists()


def test_telemetry_sync_is_independent_of_checkpoint_cadence(tmp_path, monkeypatch):
    config = smoke_config()
    config["output"]["run_dir"] = str(tmp_path / "run")
    config["training"].update(checkpoint_minutes=0, checkpoint_every=0)
    config["checkpoints"]["upload_minutes"] = 1e-9
    trainer = Trainer(config, device="cpu", samples=smoke_samples(128), resume=False)
    reasons = []
    original = trainer._sync_artifacts

    def record(reason, **kwargs):
        reasons.append(reason)
        return original(reason, **kwargs)
    monkeypatch.setattr(trainer, "_sync_artifacts", record)
    trainer.run(max_steps=2)
    assert "telemetry" in reasons
    assert "final" in reasons


def test_resume_restores_optimizer_tokens_and_declares_stream_limit(tmp_path):
    config = smoke_config()
    config["output"]["run_dir"] = str(tmp_path / "run")
    first = Trainer(config, device="cpu", samples=smoke_samples(128), resume=False)
    first.run(max_steps=2)
    weight = first.model.head.weight.detach().clone()
    moment = first.optimizer.state[first.model.head.weight]["exp_avg"].clone()

    resumed = Trainer(config, device="cpu", samples=smoke_samples(128), resume=True)
    resumed.optimizer = resumed._build_optimizer()
    resumed._maybe_resume()
    assert resumed.step == 2
    assert resumed._total_tokens == first._total_tokens
    assert torch.equal(resumed.model.head.weight, weight)
    assert torch.equal(resumed.optimizer.state[resumed.model.head.weight]["exp_avg"], moment)
    assert resumed._resume_info["exact_stream_cursor"] is False
    summary = resumed.run(max_steps=3)
    assert summary["steps"] == 3 and summary["tokens"] > first._total_tokens


def test_incompatible_checkpoint_is_rejected(tmp_path):
    config = smoke_config()
    trainer = Trainer(config, device="cpu", samples=smoke_samples(8), resume=False)
    path = tmp_path / "bad.pt"
    torch.save({"schema_version": train.CHECKPOINT_SCHEMA, "config": {}, "step": 9}, path)
    with pytest.raises(ValueError, match="incomplete unified checkpoint"):
        train.load_checkpoint(path, model=trainer.model, ingestors=trainer.ingestors,
                              text_projector=trainer.text_projector, memory=trainer.memory)
