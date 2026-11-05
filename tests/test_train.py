"""Unified trainer: batches, precision, accumulation, checkpoints, resume."""
import pytest
import torch

from models.encoders import MultimodalIngestor, TextSkillProjector
from models.foundation import PAD, FoundationConfig
from training.train import (BatchBuilder, Trainer, load_config,
                            make_grad_scaler, resolve_precision, run_smoke,
                            smoke_config, smoke_samples)


def test_production_config_loads():
    config = load_config("configs/t4_scaled.yaml")
    cfg = FoundationConfig.from_mapping(config["model"])
    assert cfg.n_layers == 8 and cfg.max_seq_len == 4096 and cfg.memory_dim == 8192
    assert cfg.refinement_steps == 4 and cfg.confidence_threshold == 0.9
    assert config["training"]["micro_batch_size"] == 4
    assert config["training"]["grad_accum"] == 8
    weights = config["data"]["weights"]
    assert weights == {"code": 0.4, "language": 0.3, "multimodal": 0.2, "logic": 0.1}


def test_precision_resolution_cpu():
    plan = resolve_precision("cpu", "bfloat16", "auto")
    assert plan.resolved == "float32" and not plan.autocast and not plan.scaler_required
    scaler = make_grad_scaler(plan)
    assert not scaler.is_enabled()
    with pytest.raises(ValueError):
        resolve_precision("cpu", "tf32", "auto")


def _builder(seq_len=128):
    cfg = FoundationConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128,
                           max_seq_len=256, dropout=0.0, memory_dim=128,
                           refinement_steps=4, refinement_rank=32)
    from models.foundation import CausalByteTransformer
    model = CausalByteTransformer(cfg)
    ingestors = {
        "image": MultimodalIngestor(None, memory_dim=128, key_dim=64, encoder_dim=32),
        "audio": MultimodalIngestor(None, memory_dim=128, key_dim=64, encoder_dim=32),
    }
    text_projector = TextSkillProjector(64, 128, 64)
    return model, BatchBuilder(model, ingestors, text_projector, seq_len)


def test_bidirectional_multimodal_batch():
    model, builder = _builder()
    sample = {"kind": "multimodal", "modality": "image",
              "caption": b"a red cube", "features": torch.randn(32)}
    builder.feed(sample)
    assert builder.stats["media_to_text"] == 1
    assert builder.stats["text_to_media"] == 1  # BOTH directions built
    batch = builder.pop_batch(2)
    assert batch["ids"].shape == (2, 128)
    memory = batch["memory"]
    assert memory.keys.shape == (2, 1, 64) and memory.values.shape == (2, 1, 128)
    assert bool(memory.mask.all())
    # Direction A learns caption bytes (prompt masked); B learns codec bytes.
    assert (batch["targets"][0] == PAD).any()
    assert (batch["targets"][1] == PAD).any()
    # Gradients reach both projector families through the batch view.
    # Open the zero-init bridge first (ReZero: closed at init by design).
    for block in model.blocks:
        block.skill_adapter.scale.data.fill_(1.0)
    loss = model.loss(batch["ids"], batch["targets"], memory, confidence_weight=0.0)
    loss.backward()
    grads = builder.ingestors["image"].value_projector.projection.weight.grad
    assert grads is not None and grads.abs().sum().item() > 0
    assert builder.text_projector.value.projection.weight.grad.abs().sum().item() > 0


def test_text_rows_get_masked_memory_slots():
    model, builder = _builder()
    builder.feed({"kind": "text", "data": b"x" * 150})  # exactly one packed row
    builder.feed({"kind": "multimodal", "modality": "audio",
                  "caption": b"speech", "features": torch.randn(32)})
    batch = builder.pop_batch(2)  # first row is packed text, second is media->text
    mask = batch["memory"].mask
    assert not bool(mask[0, 0]) and bool(mask[1, 0])


def test_ids_samples_train_control_tokens():
    model, builder = _builder()
    from models.foundation import TOOL_SEARCH, TOOL_SEARCH_END
    ids = [256, ord("q"), TOOL_SEARCH] + list(b"abc") + [TOOL_SEARCH_END, 257]
    builder.feed({"kind": "ids", "ids": ids})
    batch = builder.pop_batch(1)
    assert TOOL_SEARCH in batch["targets"][0].tolist()
    loss = model.loss(batch["ids"], batch["targets"], confidence_weight=0.0)
    loss.backward()
    assert model.head.weight.grad[TOOL_SEARCH].abs().sum().item() > 0


def test_gradient_accumulation_equivalence():
    cfg = FoundationConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128,
                           max_seq_len=128, dropout=0.0, memory_dim=128,
                           refinement_steps=2, refinement_rank=32)
    from models.foundation import CausalByteTransformer
    torch.manual_seed(0)
    model_a = CausalByteTransformer(cfg)
    torch.manual_seed(0)
    model_b = CausalByteTransformer(cfg)
    ids = torch.randint(0, 256, (4, 16))
    # One full batch...
    model_a.loss(ids, confidence_weight=0.0).backward()
    # ...vs two accumulated halves.
    for half in ids.chunk(2):
        (model_b.loss(half, confidence_weight=0.0) / 2).backward()
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        if pa.grad is not None:
            assert torch.allclose(pa.grad, pb.grad, atol=1e-5)


def test_trainer_runs_and_checkpoints(tmp_path):
    config = smoke_config()
    config["checkpoints"]["local_dir"] = str(tmp_path / "ckpt")
    config["output"]["metrics_jsonl"] = str(tmp_path / "metrics.jsonl")
    trainer = Trainer(config, device="cpu", samples=smoke_samples(128), resume=False)
    summary = trainer.run(max_steps=2)
    assert summary["steps"] == 2
    assert (tmp_path / "ckpt" / "checkpoint_latest.pt").exists()
    assert (tmp_path / "metrics.jsonl").exists()
    assert summary["parameters"] > 0
    assert summary["precision"]["resolved"] == "float32"


def test_time_based_checkpoint_cadence(tmp_path):
    config = smoke_config()
    config["checkpoints"]["local_dir"] = str(tmp_path / "ckpt")
    config["output"]["metrics_jsonl"] = str(tmp_path / "metrics.jsonl")
    config["training"]["checkpoint_minutes"] = 0.0  # due every optimizer step
    trainer = Trainer(config, device="cpu", samples=smoke_samples(128), resume=False)
    trainer.run(max_steps=2)
    events = [h for h in trainer.history if "checkpoint_event" in h]
    assert len(events) >= 2


def test_resume_restores_step_and_weights(tmp_path):
    config = smoke_config()
    config["checkpoints"]["local_dir"] = str(tmp_path / "ckpt")
    config["output"]["metrics_jsonl"] = str(tmp_path / "metrics.jsonl")
    first = Trainer(config, device="cpu", samples=smoke_samples(128), resume=False)
    first.run(max_steps=2)
    weight = first.model.head.weight.detach().clone()

    resumed = Trainer(config, device="cpu", samples=smoke_samples(128), resume=True)
    resumed.optimizer = resumed._build_optimizer()
    resumed._maybe_resume()
    assert resumed.step == 2
    assert torch.equal(resumed.model.head.weight, weight)
    assert len(resumed.memory) == len(first.memory)  # bank contents restored


def test_run_smoke_full_pipeline(tmp_path):
    summary = run_smoke(steps=3, workdir=tmp_path)
    assert summary["steps"] == 3
    assert summary["resume_step"] == 3
    assert summary["batch_stats"]["media_to_text"] > 0
    assert summary["batch_stats"]["text_to_media"] > 0
    assert summary["batch_stats"]["ingested"] > 0
    assert "tool_demo" in summary
