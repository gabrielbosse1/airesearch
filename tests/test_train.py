"""Unified trainer: batches, precision, masking, mix, checkpoints, resume."""
import pytest
import torch

from data.sources import MixtureSampler
from models.encoders import MultimodalIngestor, TextSkillProjector
from models.foundation import PAD, FoundationConfig, encode_bytes
from training.train import (BatchBuilder, Trainer, load_config,
                            make_grad_scaler, resolve_precision, run_smoke,
                            smoke_config, smoke_samples)

PTBR_TARGETS = {"ptbr": 0.50, "code": 0.25, "multimodal": 0.15, "english": 0.10}


def test_production_config_loads():
    config = load_config("configs/t4_scaled.yaml")
    cfg = FoundationConfig.from_mapping(config["model"])
    assert cfg.n_layers == 8 and cfg.max_seq_len == 16384 and cfg.memory_dim == 8192
    assert cfg.refinement_steps == 4 and cfg.confidence_threshold == 0.9
    assert config["training"]["micro_batch_size"] == 1
    assert config["training"]["grad_accum"] == 32
    weights = config["data"]["weights"]
    # PT-BR-first: 50% Portuguese documents, then code, multimodal, English.
    assert weights == PTBR_TARGETS
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert config["training"]["mix_audit_batches"] == 100
    assert config["training"]["mix_audit_tolerance"] == 0.02


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


def _masked_rows(lines, length=64):
    """Right-pad per-row byte sequences; targets are PAD wherever there is no
    next byte. This is exactly what BatchBuilder does via pad_batch."""
    ids = torch.full((len(lines), length), PAD, dtype=torch.long)
    targets = torch.full((len(lines), length), PAD, dtype=torch.long)
    for index, line in enumerate(lines):
        row = encode_bytes(line)[:length]
        ids[index, : row.numel()] = row
        targets[index, : row.numel() - 1] = row[1:]
    return ids, targets


def _drain_rows(builder):
    """Pop every complete row currently buffered (feed-then-drain pattern)."""
    rows = []
    while builder.ready(1):
        rows.append(builder.pop_batch(1))
    return rows


def test_pad_masking_holds_for_all_four_ptbr_mix_sources():
    """ignore_index=PAD is source-agnostic: PT-BR text, code and English rows are
    packed full (no padding), while multimodal rows mask their prompt prefix."""
    model, builder = _builder()
    documents = {
        "ptbr": "O Brasil é um país da América do Sul, e o português é a língua oficial. ".encode("utf8") * 6,
        "code": b"def add(a, b):\n    return a + b\n" * 8,
        "english": b"The quick brown fox jumps over the lazy dog. " * 6,
    }
    for name, data in documents.items():
        builder.feed({"kind": "text", "data": data})
        batches = _drain_rows(builder)
        assert batches, name
        for batch in batches:
            assert batch["targets"].min().item() >= 0
            assert batch["targets"].max().item() <= PAD
            # Text rows are packed full: any mask is a right-hand pad tail.
            masked = batch["targets"][0] == PAD
            if bool(masked.any()):
                assert bool(masked[int(masked.nonzero()[0]):].all())

    # Multimodal: prompt targets are masked (prefix), the caption is learned.
    builder.feed({"kind": "multimodal", "modality": "image",
                  "caption": b"um gato laranja", "features": torch.randn(32)})
    media_to_text, text_to_media = _drain_rows(builder)[:2]
    assert bool(media_to_text["memory"].mask.all()) and bool(text_to_media["memory"].mask.all())
    prompt = len(b"caption: ")
    assert bool((media_to_text["targets"][0, :prompt] == PAD).all())
    assert int((media_to_text["targets"][0] != PAD).sum()) == len(b"um gato laranja") + 1
    assert bool((text_to_media["targets"][0, : len(b"matrix: um gato laranja\n")] == PAD).all())
    loss = model.loss(media_to_text["ids"], media_to_text["targets"],
                      media_to_text["memory"], confidence_weight=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    grads = builder.ingestors["image"].value_projector.projection.weight.grad
    assert grads is not None and torch.isfinite(grads).all()


def test_pad_masking_excludes_padding_and_prompt_prefixes():
    """PAD targets are ignored exactly: CE equals the mean over the *unmasked*
    positions, truncating the pad tail changes nothing, and a fully masked
    batch is a zero loss instead of CE's 0/0 NaN."""
    import torch.nn.functional as F
    model, builder = _builder()
    builder.feed({"kind": "multimodal", "modality": "audio",
                  "caption": b"fala curta em portugues", "features": torch.randn(32)})
    caption, codec = _drain_rows(builder)[:2]
    ids, targets = caption["ids"], caption["targets"]
    valid = targets != PAD
    assert int(valid.sum()) == len(b"fala curta em portugues") + 1

    with torch.no_grad():
        logits = model(ids)
    masked_mean = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                  targets.reshape(-1), ignore_index=PAD)
    only_valid = F.cross_entropy(logits[valid], targets[valid])
    assert torch.allclose(masked_mean, only_valid, atol=1e-6)

    ce = model.loss(ids, targets, confidence_weight=0.0)
    assert torch.allclose(ce, only_valid, atol=1e-5)

    # Dropping the PAD tail shortens the sequence without changing the loss.
    keep = int(valid.nonzero().max().item()) + 1
    ce_truncated = model.loss(ids[:, :keep], targets[:, :keep], confidence_weight=0.0)
    assert torch.allclose(ce, ce_truncated, atol=1e-5)

    def truncation_is_inert(batch):
        live = batch["targets"] != PAD
        cut = int(live.nonzero().max().item()) + 1
        return torch.allclose(
            model.loss(batch["ids"], batch["targets"], confidence_weight=0.0),
            model.loss(batch["ids"][:, :cut], batch["targets"][:, :cut],
                       confidence_weight=0.0), atol=1e-5)

    assert truncation_is_inert(caption) and truncation_is_inert(codec)

    # All-PAD filler rows and fully masked batches contribute zero tokens.
    pad_row = torch.full((1, ids.shape[1]), PAD, dtype=ids.dtype)
    padded, _ = model.loss(torch.cat([ids, pad_row]), torch.cat([targets, pad_row]),
                           confidence_weight=0.1, return_metrics=True)
    assert torch.allclose(padded, model.loss(ids, targets, confidence_weight=0.1), atol=1e-6)
    empty_loss, empty_metrics = model.loss(pad_row, pad_row, confidence_weight=0.1,
                                           return_metrics=True)
    assert torch.isfinite(empty_loss) and empty_loss.item() == 0.0
    assert empty_metrics["ce"] == 0.0 and empty_metrics["confidence_bce"] == 0.0
    empty_loss.backward()   # zero gradient, never NaN
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_loss_is_token_weighted_so_50pct_ptbr_does_not_skew_it():
    """CE and the auxiliary confidence BCE are means over non-PAD tokens, so a
    batch that is 50% PT-BR is exactly the token-weighted blend of its parts."""
    cfg = FoundationConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128,
                           max_seq_len=128, dropout=0.0, memory_dim=128,
                           refinement_steps=4, refinement_rank=32)
    from models.foundation import CausalByteTransformer
    torch.manual_seed(0)
    model = CausalByteTransformer(cfg)
    model.train()
    ptbr = "O Brasil é um país da América do Sul, e o português é a língua oficial. ".encode("utf8")[:64]
    caption = b"um gato laranja"                      # short, heavily padded row
    half = [ptbr, ptbr, caption, caption]
    half_ids, half_targets = _masked_rows(half)
    ptbr_ids, ptbr_targets = _masked_rows([ptbr, ptbr])
    cap_ids, cap_targets = _masked_rows([caption, caption])

    _, mixed = model.loss(half_ids, half_targets, confidence_weight=0.1,
                          return_metrics=True)
    _, only_ptbr = model.loss(ptbr_ids, ptbr_targets, confidence_weight=0.1,
                              return_metrics=True)
    _, only_cap = model.loss(cap_ids, cap_targets, confidence_weight=0.1,
                             return_metrics=True)
    n_ptbr = int((ptbr_targets != PAD).sum())
    n_cap = int((cap_targets != PAD).sum())
    assert n_ptbr > n_cap
    for key in ("ce", "confidence_bce", "total"):
        expected = (n_ptbr * only_ptbr[key] + n_cap * only_cap[key]) / (n_ptbr + n_cap)
        assert mixed[key] == pytest.approx(expected, rel=1e-4, abs=1e-6), key
    assert 0.0 <= mixed["confidence_bce"] <= 0.6932


def test_trainer_reports_the_effective_mix(tmp_path):
    config = smoke_config()
    config["training"]["tool_trajectory_rate"] = 0.0
    config["checkpoints"]["local_dir"] = str(tmp_path / "ckpt")
    config["output"]["metrics_jsonl"] = str(tmp_path / "metrics.jsonl")

    def group(name):
        while True:
            yield {"kind": "text", "data": f"amostra {name} do mix pt-br".encode()}

    sampler = MixtureSampler({name: group(name) for name in PTBR_TARGETS}, seed=24)
    trainer = Trainer(config, device="cpu", samples=sampler, resume=False)
    summary = trainer.run(max_steps=2)
    mix = summary["data_mix"]
    assert mix["targets"] == PTBR_TARGETS
    assert mix["window"] == 100
    assert sum(mix["counts"].values()) > 0
    assert 0.2 <= mix["observed_share"]["ptbr"] <= 0.8   # few draws, quota window
    assert sum(mix["observed_share"].values()) == pytest.approx(1.0, abs=1e-12)


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


def test_smoke_coherence_script_reports_thought_norms(tmp_path, capsys):
    """Phase 24.4.5: the coherence smoke prints the scratchpad activation
    norm (proof of usage) and passes its own probes, fully offline."""
    import smoke_test_coherence
    result = smoke_test_coherence.main(["--steps", "2", "--workdir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "latent thought buffer" in out
    assert "activation_norm" in out and "slot_norms" in out
    assert result["probe"]["activation_norm"] > 0.0
    assert result["probe"]["buffer_shape"][1] == 4
    assert result["smoke"]["thought"]["buffer_shape"][1] == 4
    assert "sample_continuation" in result["smoke"]
