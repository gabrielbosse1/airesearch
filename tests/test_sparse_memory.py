"""Sliding-window causality, memory teleportation, decay policy, T4 preflight.

Offline contract: no CUDA device is assumed. Exact SWA is checked against a
dense reference on small tensors; T4 VRAM is exercised through the explicitly
labelled simulator and mocked autotune decisions, never a fake measurement.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
import yaml

from models.foundation import (CausalByteTransformer, FoundationConfig,
                               GatedSkillAdapter, TransformerBlock,
                               sliding_window_attention, sliding_window_chunk_mask)
from models.memory_model import (DualMemory, EvictionEvent, MEMORY_SCHEMA,
                                 ProceduralSkillBank, SkillBankView)
from data.sources import BytePacker
from training.train import (BatchBuilder, CHECKPOINT_SCHEMA, Trainer,
                            simulate_t4_vram, smoke_config, smoke_samples)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _softplus_inv(x: float) -> float:
    return math.log(math.expm1(x))


def _small_swa_config(**overrides) -> FoundationConfig:
    base = dict(d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64,
                max_seq_len=64, dropout=0.0, attention_type="sliding_window",
                window_size=8, sliding_chunk_size=4,
                memory_dim=64, refinement_steps=2, refinement_rank=16,
                gradient_checkpointing=True)
    base.update(overrides)
    return FoundationConfig(**base)


def test_production_preset_is_16k_sliding_window_with_checkpointing():
    with open("configs/t4_scaled.yaml", "r", encoding="utf8") as handle:
        config = yaml.safe_load(handle)
    cfg = FoundationConfig.from_mapping(config["model"])
    assert cfg.max_seq_len == 16384 and cfg.attention_type == "sliding_window"
    assert cfg.window_size == 1024 and cfg.sliding_chunk_size == 512
    assert cfg.gradient_checkpointing and cfg.memory_cross_attention
    assert config["training"]["seq_len"] == 16384
    assert config["training"]["micro_batch_size"] == 1
    assert config["training"]["grad_accum"] == 32
    assert config["vram"]["target_max_gb"] == 13.5
    assert config["vram"]["frozen_encoder_dtype"] == "float16"
    assert config["encoders"]["video"] is None  # no dead video backbone on T4


def test_sliding_chunk_mask_is_exact_and_strictly_causal():
    window = 5
    for start in range(0, 20, 4):
        stop = min(start + 4, 20)
        key_start = max(0, start - window)
        mask = sliding_window_chunk_mask(start, stop - start, key_start,
                                         stop - key_start, window)
        assert mask.dtype == torch.bool
        assert mask.numel() <= 4 * (window + 4)
        for r, i in enumerate(range(start, stop)):
            for c, j in enumerate(range(key_start, stop)):
                assert bool(mask[r, c]) == bool(i - window <= j <= i)


def test_sliding_window_attention_matches_dense_reference_and_gqa():
    torch.manual_seed(0)
    length, heads, kv_heads, width, window, chunk = 24, 4, 2, 8, 5, 7
    q = torch.randn(1, heads, length, width)
    k = torch.randn(1, kv_heads, length, width)
    v = torch.randn(1, kv_heads, length, width)
    out = sliding_window_attention(q, k, v, window_size=window,
                                   chunk_size=chunk, repeats=heads // kv_heads)
    rows = torch.arange(length)[:, None]
    cols = torch.arange(length)[None, :]
    reference_mask = (cols <= rows) & (cols >= rows - window)
    kr = k.repeat_interleave(heads // kv_heads, dim=1)
    vr = v.repeat_interleave(heads // kv_heads, dim=1)
    reference = F.scaled_dot_product_attention(q, kr, vr, attn_mask=reference_mask[None, None])
    assert torch.allclose(out, reference, atol=1e-5, rtol=1e-5)
    # A key/value outside the local window cannot change a late query row.
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, :, 0] += 100.0
    changed_v[:, :, 0] += 100.0
    far = sliding_window_attention(q, changed_k, changed_v, window_size=window,
                                   chunk_size=chunk, repeats=heads // kv_heads)
    assert torch.equal(out[:, :, window + 2:], far[:, :, window + 2:])


def test_production_model_allocates_no_full_length_mask(monkeypatch):
    with open("configs/t4_scaled.yaml", "r", encoding="utf8") as handle:
        cfg = FoundationConfig.from_mapping(yaml.safe_load(handle)["model"])
    with torch.device("meta"):
        model = CausalByteTransformer(cfg)
    assert model.attention_mask is None and model.causal_mask is None
    assert not any(name.endswith("causal_mask") for name, _ in model.named_buffers())
    # Runtime chunk masks stay local even at L=16384.
    import models.foundation as foundation
    seen = []
    original = foundation.sliding_window_chunk_mask

    def record(*args, **kwargs):
        mask = original(*args, **kwargs)
        seen.append(mask.numel())
        return mask

    monkeypatch.setattr(foundation, "sliding_window_chunk_mask", record)
    q = torch.randn(1, 1, 16384, 8)
    k = torch.randn(1, 1, 16384, 8)
    v = torch.randn(1, 1, 16384, 8)
    with torch.no_grad():
        sliding_window_attention(q, k, v, window_size=1024, chunk_size=512)
    assert seen and max(seen) <= 512 * (1024 + 512)
    assert max(seen) < 16384 * 16384 // 100


def test_swa_model_blocks_future_gradients_and_checkpoints_all_blocks(monkeypatch):
    cfg = _small_swa_config(n_layers=8, max_seq_len=32)
    model = CausalByteTransformer(cfg)
    calls = []
    import models.foundation as foundation
    original = foundation.checkpoint

    def record(fn, *args, **kwargs):
        if isinstance(fn, TransformerBlock):
            calls.append(fn)
        return original(fn, *args, **kwargs)

    monkeypatch.setattr(foundation, "checkpoint", record)
    ids = torch.randint(0, 256, (1, 12))
    loss = model.loss(ids, confidence_weight=0.1)
    loss.backward()
    assert len(calls) == 8 and len(set(id(c) for c in calls)) == 8
    emb = model.embed(ids).detach().requires_grad_()
    hidden = emb
    for block in model.blocks:
        hidden = block(hidden, None, None)
    hidden[0, 4].sum().backward()
    assert emb.grad is not None and float(emb.grad[0, 5:].abs().max()) == 0.0


def test_memory_cross_attention_is_the_long_range_handoff():
    # Self-attention is strictly local: changing a far past byte cannot affect
    # a late row. The adapter cross-attention CAN read that far past when it is
    # committed to the memory bank, and it is enabled in every production layer.
    torch.manual_seed(0)
    window = 4
    q = torch.randn(1, 2, 12, 8)
    k = torch.randn(1, 2, 12, 8)
    v = torch.randn(1, 2, 12, 8)
    base = sliding_window_attention(q, k, v, window_size=window, chunk_size=3)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 0] += 50.0
    v2[:, :, 0] += 50.0
    local = sliding_window_attention(q, k2, v2, window_size=window, chunk_size=3)
    assert torch.equal(base[:, :, 8:], local[:, :, 8:])
    adapter = GatedSkillAdapter(16, 32, n_heads=2)
    adapter.cross_scale.data.fill_(1.0)
    hidden = torch.randn(1, 12, 16)
    keys = torch.randn(1, 16)
    values = torch.randn(1, 32)
    first = adapter.cross_delta(hidden, SkillBankView(keys, values, torch.ones(1, dtype=torch.bool)))
    values2 = values + 10.0
    second = adapter.cross_delta(hidden, SkillBankView(keys, values2, torch.ones(1, dtype=torch.bool)))
    assert not torch.allclose(first, second)
    with open("configs/t4_scaled.yaml", "r", encoding="utf8") as handle:
        cfg = FoundationConfig.from_mapping(yaml.safe_load(handle)["model"])
    assert cfg.memory_cross_attention
    with torch.device("meta"):
        model = CausalByteTransformer(cfg)
    assert all(block.skill_adapter.cross_attention_enabled for block in model.blocks)


def test_cross_attention_shapes_gates_gradients_and_empty_rows():
    torch.manual_seed(0)
    adapter = GatedSkillAdapter(64, 128, n_heads=4)
    hidden = torch.randn(2, 5, 64, requires_grad=True)
    keys = torch.randn(2, 3, 64, requires_grad=True)
    values = torch.randn(2, 3, 128, requires_grad=True)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    assert float(adapter.cross_scale.detach()) == 0.0
    assert float(adapter.cross_attend(hidden, keys, values, mask).detach().abs().max()) == 0.0
    adapter.cross_scale.data.fill_(1.0)
    out = adapter.cross_attend(hidden, keys, values, mask)
    assert out.shape == hidden.shape and torch.isfinite(out).all()
    assert float(out[1].detach().abs().max()) == 0.0
    out[0].square().mean().backward()
    assert hidden.grad is not None and keys.grad is not None and values.grad is not None
    for module in (adapter.cross_q, adapter.cross_k, adapter.cross_v, adapter.cross_o):
        assert module.weight.grad is not None and float(module.weight.grad.abs().sum()) > 0
    with torch.device("meta"):
        prod = GatedSkillAdapter(1024, 8192, n_heads=8)
    assert prod.cross_q.in_features == 1024 and prod.cross_k.in_features == 1024
    assert prod.cross_v.in_features == 8192 and prod.cross_o.out_features == 1024


def test_cross_attention_is_wired_in_every_layer():
    cfg = _small_swa_config(attention_type="dense", max_seq_len=32, n_layers=3)
    model = CausalByteTransformer(cfg)
    view = SkillBankView(torch.randn(4, 32), torch.randn(4, 64), torch.ones(4, dtype=torch.bool))
    for block in model.blocks:
        block.skill_adapter.scale.data.fill_(1.0)
        block.skill_adapter.cross_scale.data.fill_(1.0)
    model.loss(torch.randint(0, 256, (1, 8)), memory=view, confidence_weight=0.0).backward()
    for block in model.blocks:
        adapter = block.skill_adapter
        assert adapter.cross_attention_enabled
        assert adapter.cross_q.weight.grad is not None
        assert float(adapter.cross_q.weight.grad.abs().sum()) > 0
        assert float(adapter.cross_v.weight.grad.abs().sum()) > 0


def test_unified_dual_memory_is_one_physical_bank():
    memory = DualMemory(procedural_capacity=2, episodic_capacity=3, key_dim=4, value_dim=8)
    assert memory.episodic is memory.procedural and memory.procedural.capacity == 5
    memory.write_event(torch.ones(4), torch.ones(8), metadata={"note": "event"})
    memory.write_skill(torch.ones(4), torch.ones(8), modality="audio")
    assert len(memory) == 2 and memory.view().keys.shape == (2, 4)
    assert "unified" in memory.manifest()
    explicit = DualMemory(capacity=7, key_dim=4, value_dim=8)
    assert explicit.procedural.capacity == 7


def test_decay_score_math_and_forever_bounds():
    bank = ProceduralSkillBank(capacity=4, key_dim=2, value_dim=2)
    for _ in range(3):
        bank.write(torch.zeros(2), torch.zeros(2))
    bank.mark_forever(0)
    bank._tick = 12
    bank.written_at.copy_(torch.tensor([0, 4, 8, 0]))
    with torch.no_grad():
        bank.importance_raw.copy_(torch.tensor([_logit(0.9), _logit(0.2), _logit(0.8), 0.0]))
        bank.decay_raw.copy_(torch.tensor([100.0, _softplus_inv(0.5), _softplus_inv(2.0), 0.0]))
    ages = torch.tensor([3.0, 2.0, 1.0, 3.0])
    assert torch.allclose(bank.ages(), ages)
    assert float((bank.decay_rate[0]).detach()) == 0.0
    assert float((bank.importance[0]).detach()) >= 2.0
    assert bool((bank.importance[1:3] < 1.0).all())
    expected = bank.importance - bank.decay_rate * ages
    assert torch.allclose(bank.scores(), expected)
    assert float((bank.scores()[0]).detach()) > float((bank.scores()[1:3].max()).detach())


def test_eviction_uses_score_not_fifo_and_preserves_learned_policy():
    bank = ProceduralSkillBank(capacity=2, key_dim=2, value_dim=2)
    bank.write(torch.zeros(2), torch.full((2,), 1.0))  # older, high importance/tiny decay
    bank.write(torch.zeros(2), torch.full((2,), 2.0))  # newer, low importance
    bank._tick = 10
    bank.written_at.copy_(torch.tensor([0, 10]))
    with torch.no_grad():
        bank.importance_raw.copy_(torch.tensor([_logit(0.99), _logit(0.01)]))
        bank.decay_raw.copy_(torch.tensor([_softplus_inv(0.01), _softplus_inv(1.0)]))
    before = bank.decay_raw.detach().clone()
    slot = bank.write(torch.zeros(2), torch.full((2,), 3.0))
    assert slot == 1  # newer slot has lower score; FIFO would have chosen 0
    assert torch.equal(bank.decay_raw, before)  # replacement does not reset learning
    assert len(bank.eviction_events) == 1


def test_forever_slots_cannot_be_overwritten_and_attempt_is_penalized():
    bank = ProceduralSkillBank(capacity=2, key_dim=2, value_dim=2)
    bank.write(torch.zeros(2), torch.ones(2), forever=True)
    bank.write(torch.zeros(2), torch.full((2,), 2.0))
    bank._tick = 100
    slot = bank.write(torch.zeros(2), torch.full((2,), 3.0))
    assert slot == 1 and bool(bank.forever[0]) and float(bank.values[0, 0]) == 1.0
    bank.mark_forever(1)
    with pytest.raises(MemoryError):
        bank.write(torch.zeros(2), torch.full((2,), 4.0))
    assert bank.forever_overwrite_attempts == 1
    assert bank.eviction_events[-1].victim_forever
    loss = bank.policy_loss()
    loss.backward()
    assert bank.decay_raw.grad is not None and torch.isfinite(bank.decay_raw.grad).all()
    assert float(bank.decay_raw.grad.abs().sum()) > 0


def test_policy_loss_rewards_stale_eviction_and_has_stable_extreme_gradients():
    bank = ProceduralSkillBank(capacity=3, key_dim=2, value_dim=2)
    for _ in range(3):
        bank.write(torch.zeros(2), torch.zeros(2))
    bank.eviction_events.append(EvictionEvent(
        0, torch.tensor([1, 2]), torch.tensor([1.0, 0.0, 0.0]), 1.0, False))
    with torch.no_grad():
        bank.decay_raw[0] = _softplus_inv(0.1)
    low = float((bank.policy_loss()).detach())
    with torch.no_grad():
        bank.decay_raw[0] = _softplus_inv(4.0)
    high = float((bank.policy_loss()).detach())
    assert high < low  # stale victim: larger decay is rewarded
    with torch.no_grad():
        bank.decay_raw.fill_(100.0)
        bank.importance_raw.fill_(-100.0)
        bank._tick = 10_000
    bank.zero_grad()
    loss = bank.policy_loss()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(bank.decay_raw.grad).all() and torch.isfinite(bank.importance_raw.grad).all()


def test_reinforce_refreshes_only_used_slots_and_roundtrips_policy(tmp_path):
    bank = ProceduralSkillBank(capacity=3, key_dim=2, value_dim=2)
    bank.write(torch.zeros(2), torch.zeros(2))
    bank._tick = 9
    bank.reinforce([0])
    assert float(bank.ages()[0]) == 0.0 and int(bank.read_count[0]) == 1
    with pytest.raises(ValueError):
        bank.reinforce([1])  # unused
    with torch.no_grad():
        bank.decay_raw[0] = 1.25
    payload = bank.state_payload()
    restored = ProceduralSkillBank(capacity=3, key_dim=2, value_dim=2)
    restored.load_payload(payload)
    assert torch.equal(restored.decay_raw, bank.decay_raw)
    assert restored._tick == 9 and MEMORY_SCHEMA == 2
    bad = dict(payload, schema_version=1)
    with pytest.raises(ValueError):
        restored.load_payload(bad)


def test_delayed_memory_commit_prevents_same_batch_leakage():
    cfg = FoundationConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32,
                           max_seq_len=128, memory_dim=8, dropout=0.0)
    model = CausalByteTransformer(cfg)
    from models.encoders import MultimodalIngestor, TextSkillProjector
    ingestor = MultimodalIngestor(None, memory_dim=8, key_dim=16, encoder_dim=4)
    memory = DualMemory(capacity=4, key_dim=16, value_dim=8)
    builder = BatchBuilder(model, {"image": ingestor}, TextSkillProjector(16, 8, 16),
                           128, memory=memory, ingest_to_memory=True)
    builder.feed({"kind": "multimodal", "modality": "image", "caption": b"cat",
                  "features": torch.randn(4)})
    first = builder.pop_batch(1)
    # Only the row-local slot is visible; the persistent write commits AFTER pop.
    assert first["memory"].keys.shape[-2] == 1 and len(memory) == 1
    second = builder.pop_batch(1)
    assert second["memory"].keys.shape[-2] == 2  # local + prior committed slot


def test_trainer_optimizer_includes_policy_and_policy_loss_trains_it(tmp_path):
    config = smoke_config()
    config["checkpoints"]["local_dir"] = str(tmp_path / "ckpt")
    config["output"]["metrics_jsonl"] = str(tmp_path / "metrics.jsonl")
    trainer = Trainer(config, device="cpu", samples=smoke_samples(128), resume=False)
    optimizer = trainer._build_optimizer()
    ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(trainer.memory.procedural.decay_raw) in ids
    assert id(trainer.memory.procedural.importance_raw) in ids
    before = trainer.memory.procedural.decay_raw.detach().clone()
    trainer.run(max_steps=2)
    assert not torch.equal(before, trainer.memory.procedural.decay_raw)
    assert trainer.history[-1].get("policy_loss") is not None or any(
        "policy_loss" in h for h in trainer.history)


def test_probe_memory_view_is_full_bank_plus_differentiable_local_slots():
    trainer = Trainer(smoke_config(), device="cpu", samples=smoke_samples(8), resume=False)
    trainer.ensure_ingestor("image", feature_dim=32)
    view = trainer._probe_memory_view(2)
    assert view.batched and view.keys.requires_grad and view.values.requires_grad
    assert view.keys.shape[-2] == trainer.memory.procedural.capacity + 2  # text + image
    report = trainer.probe_vram()
    assert report["measured"] is False and report["simulated"] is False


def test_autotune_halves_batch_then_fails_closed_without_shrinking_window(monkeypatch):
    config = smoke_config()
    config["model"].update(attention_type="sliding_window", max_seq_len=32,
                           window_size=4, sliding_chunk_size=2)
    config["training"]["micro_batch_size"] = 2
    config["vram"] = {"target_max_gb": 13.5}
    trainer = Trainer(config, device="cpu", samples=smoke_samples(8), resume=False)
    peaks = iter([14.0, 14.0])
    monkeypatch.setattr(trainer, "probe_vram", lambda: {
        "measured": True, "peak_gb": next(peaks), "micro_batch": trainer.micro_batch,
        "seq_len": trainer.seq_len})
    with pytest.raises(MemoryError, match="refusing to start"):
        trainer.autotune_batch(13.5)
    assert trainer.micro_batch == 1
    assert trainer.model_config.window_size == 4  # architecture contract preserved


def test_sliding_window_preset_requires_gradient_checkpointing():
    config = smoke_config()
    config["model"].update(attention_type="sliding_window", max_seq_len=32,
                           window_size=4, sliding_chunk_size=2, gradient_checkpointing=False)
    with pytest.raises(ValueError, match="gradient_checkpointing=True"):
        Trainer(config, device="cpu", samples=smoke_samples(8), resume=False)


def test_simulated_t4_production_preset_fits_and_is_labelled_estimate():
    with open("configs/t4_scaled.yaml", "r", encoding="utf8") as handle:
        config = yaml.safe_load(handle)
    report = simulate_t4_vram(config)
    assert report["measured"] is False and report["simulated"] is True
    assert "simulated" in report["device"].lower()
    assert report["seq_len"] == 16384 and report["micro_batch"] == 1
    assert report["window_size"] == 1024 and report["sliding_chunk_size"] == 512
    assert report["attention_type"] == "sliding_window"
    assert report["fits"] and report["peak_gb"] <= 13.5
    assert report["meets_design_target"] and report["peak_gb"] < 10.0
    assert report["frozen_encoder_bytes_per_parameter"] == 2
    assert sum(report["components_gb"].values()) == pytest.approx(report["peak_gb"], abs=0.01)
    larger = simulate_t4_vram(config, micro_batch=2)
    assert larger["peak_gb"] > report["peak_gb"]
    bad = dict(config, model=dict(config["model"], gradient_checkpointing=False))
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        simulate_t4_vram(bad)


def test_streaming_packer_emits_actual_16384_byte_chunks():
    packer = BytePacker(16384)
    rows = list(packer.feed(b"x" * (16384 * 2 + 17)))
    assert len(rows) >= 2
    assert all(ids.shape[0] == 16384 and targets.shape[0] == 16384 for ids, targets in rows[:-1])
    assert CHECKPOINT_SCHEMA == 7


def test_inline_confidence_loss_matches_collected_state_reference():
    cfg = _small_swa_config(max_seq_len=32, refinement_steps=3)
    torch.manual_seed(0)
    model = CausalByteTransformer(cfg)
    ids = torch.randint(0, 256, (2, 12))
    targets = ids[:, 1:].clone()
    inputs = ids[:, :-1].clone()
    loss, metrics = model.loss(inputs, targets, confidence_weight=0.1, return_metrics=True)
    logits, info = model.forward(inputs, collect_states=True, return_info=True)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    valid = targets != 258
    terms = []
    thought_states = info.get("thought_states") or [None] * len(info["states"])
    for state, thought in zip(info["states"], thought_states):
        with torch.no_grad():
            step_logits = model.head(model.norm_final(state)).float()
            logp = torch.log_softmax(step_logits, dim=-1)
            entropy = -(logp.exp() * logp).sum(-1) / math.log(cfg.vocab_size)
            target_t = (entropy < 0.35).float()
        conf_logits = model.refinement.confidence_logits(state, thought)
        with torch.autocast(device_type=state.device.type, enabled=False):
            terms.append(F.binary_cross_entropy_with_logits(
                conf_logits[valid].float(), target_t[valid].float()))
    reference = ce + 0.1 * torch.stack(terms).mean()
    assert torch.allclose(loss, reference, atol=1e-5, rtol=1e-5)
    assert metrics["confidence_bce"] == pytest.approx(float(torch.stack(terms).detach().mean()), abs=1e-5)


def test_sliding_window_checkpointing_matches_plain_gradients():
    cfg = _small_swa_config(max_seq_len=32, refinement_steps=2)
    torch.manual_seed(0)
    plain = CausalByteTransformer(cfg.scaled(gradient_checkpointing=False))
    torch.manual_seed(0)
    checked = CausalByteTransformer(cfg.scaled(gradient_checkpointing=True))
    ids = torch.randint(0, 256, (1, 16))
    l1 = plain.loss(ids, confidence_weight=0.1)
    l2 = checked.loss(ids, confidence_weight=0.1)
    assert torch.allclose(l1, l2, atol=1e-5, rtol=1e-5)
    l1.backward()
    l2.backward()
    compared = 0
    for p1, p2 in zip(plain.parameters(), checked.parameters()):
        if p1.grad is not None:
            assert p2.grad is not None
            assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-5)
            compared += 1
    assert compared > 0
