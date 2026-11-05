"""BigBird causality, memory teleportation, decay policy, and T4 preflight.

Offline contract: no CUDA device is assumed. The production-size mask is
checked on CPU in chunks; T4 VRAM is exercised through the explicitly labelled
simulator and mocked autotune decisions, never presented as a measurement.
"""
from __future__ import annotations

import gc
import math
import weakref

import pytest
import torch
import yaml

from models.foundation import (BigBirdSparseMask, CausalByteTransformer,
                               FoundationConfig, GatedSkillAdapter,
                               TransformerBlock, save_foundation, load_foundation)
from models.memory_model import (DualMemory, EvictionEvent, MEMORY_SCHEMA,
                                 ProceduralSkillBank, SkillBankView)
from data.sources import BytePacker
from training.train import (BatchBuilder, CHECKPOINT_SCHEMA, Trainer,
                            simulate_t4_vram, smoke_config, smoke_samples)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _softplus_inv(x: float) -> float:
    return math.log(math.expm1(x))


def _small_bigbird_config(**overrides) -> FoundationConfig:
    base = dict(d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64,
                max_seq_len=64, dropout=0.0, attention_type="bigbird",
                bigbird_window=8, bigbird_random_tokens=3, bigbird_global_tokens=2,
                memory_dim=64, refinement_steps=2, refinement_rank=16,
                gradient_checkpointing=True)
    base.update(overrides)
    return FoundationConfig(**base)


def test_production_preset_is_16k_bigbird_with_checkpointing():
    with open("configs/t4_scaled.yaml", "r", encoding="utf8") as handle:
        config = yaml.safe_load(handle)
    cfg = FoundationConfig.from_mapping(config["model"])
    assert cfg.max_seq_len == 16384 and cfg.attention_type == "bigbird"
    assert (cfg.bigbird_window, cfg.bigbird_random_tokens, cfg.bigbird_global_tokens) == (256, 32, 16)
    assert cfg.gradient_checkpointing and cfg.memory_cross_attention
    assert config["training"]["seq_len"] == 16384
    assert config["training"]["micro_batch_size"] == 1
    assert config["training"]["grad_accum"] == 32
    assert config["vram"]["target_max_gb"] == 13.5


def test_bigbird_mask_matches_causal_local_random_global_reference():
    length, window, random_tokens, global_tokens, heads = 40, 5, 3, 2, 2
    mask = BigBirdSparseMask(length, heads, window=window, random_tokens=random_tokens,
                             global_tokens=global_tokens, seed=7)
    assert mask.base.shape == (1, 1, length, length) and mask.base.dtype == torch.bool
    for i in range(length):
        for j in range(length):
            local = i - window + 1 <= j <= i
            glob = (j < global_tokens or i < global_tokens) and j <= i
            assert bool(mask.base[0, 0, i, j]) == bool(j <= i and (local or glob))
    overlays = []
    for head in range(heads):
        overlay = mask.head_mask(head)
        assert overlay.shape == (1, 1, length, length)
        assert bool((overlay | mask.base == overlay).all())  # base subset
        keys = mask.random_keys[head]
        eligible = torch.tensor([max(0, i - window - global_tokens + 1) for i in range(length)])
        counts = (keys >= 0).sum(-1)
        assert torch.equal(counts, torch.minimum(eligible, torch.full_like(eligible, random_tokens)))
        for i in range(length):
            picks = keys[i][keys[i] >= 0]
            assert picks.numel() == len(set(picks.tolist()))  # without replacement
            assert bool((picks <= i).all())
            assert bool((picks >= global_tokens).all())
            assert bool((picks < i - window + 1).all())       # nonlocal, nonglobal
            assert bool(overlay[0, 0, i, picks].all())
        overlays.append(overlay)
    assert not torch.equal(overlays[0], overlays[1])          # per-head random sets


def test_production_16384_mask_is_causal_and_exact_density():
    length, heads = 16384, 8
    mask = BigBirdSparseMask(length, heads, window=256, random_tokens=32,
                             global_tokens=16, seed=24)
    assert mask.base.shape == (1, 1, length, length)
    assert mask.base.numel() == 16384 * 16384
    # Chunked audit avoids a second full-size temporary.
    for start in range(0, length, 1024):
        chunk = mask.base[0, 0, start:start + 1024]
        rows = torch.arange(start, min(start + 1024, length))[:, None]
        cols = torch.arange(length)[None, :]
        assert not bool((chunk & (cols > rows)).any())
    overlay = mask.head_mask(3, length)
    row = 10000
    assert int(mask.base[0, 0, row].sum()) == 256 + 16
    assert int(overlay[0, 0, row].sum()) == 256 + 16 + 32
    assert bool(overlay[0, 0, row, :16].all())
    assert bool(overlay[0, 0, row, row - 255:row + 1].all())
    assert not bool(overlay[0, 0, row, row + 1:].any())
    keys = mask.random_keys[:, row]
    assert keys.shape == (heads, 32)
    for head in range(heads):
        picks = keys[head]
        assert len(set(picks.tolist())) == 32
        assert bool((picks <= row).all())
        assert not bool(mask.base[0, 0, row, picks].any())
    # Early prefixes cannot supply 32 eligible nonlocal/nonglobal past keys.
    assert int((mask.random_keys[0, 100] >= 0).sum()) == 0
    assert int((mask.random_keys[0, 272] >= 0).sum()) == 1


def test_lazy_head_overlay_is_not_cached_or_retained():
    mask = BigBirdSparseMask(32, 2, window=4, random_tokens=2, global_tokens=2)
    overlay = mask.head_mask(0)
    ref = weakref.ref(overlay)
    del overlay
    gc.collect()
    assert ref() is None
    assert not hasattr(mask, "_head_cache")


def test_bigbird_sdpa_receives_shared_shape_per_head_and_matches_manual_attention():
    torch.manual_seed(0)
    length, heads, head_dim = 24, 3, 8
    mask = BigBirdSparseMask(length, heads, window=5, random_tokens=2, global_tokens=2)
    q, k, v = (torch.randn(1, heads, length, head_dim) for _ in range(3))
    seen = []
    original = torch.nn.functional.scaled_dot_product_attention

    def record(qq, kk, vv, *args, **kwargs):
        seen.append(kwargs.get("attn_mask"))
        return original(qq, kk, vv, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = record
    try:
        outs = []
        for head in range(heads):
            attn_mask = mask.head_mask(head)
            outs.append(torch.nn.functional.scaled_dot_product_attention(
                q[:, head:head + 1], k[:, head:head + 1], v[:, head:head + 1],
                attn_mask=attn_mask, is_causal=False))
            scores = (q[:, head:head + 1] @ k[:, head:head + 1].transpose(-1, -2)) / math.sqrt(head_dim)
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
            manual = torch.softmax(scores, dim=-1) @ v[:, head:head + 1]
            assert torch.allclose(outs[-1], manual, atol=1e-5)
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
    assert all(m.shape == (1, 1, length, length) and m.dtype == torch.bool for m in seen)


def test_bigbird_model_blocks_future_gradients_and_checkpoints_all_blocks(monkeypatch):
    cfg = _small_bigbird_config(n_layers=8, max_seq_len=32)
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
        hidden = block(hidden, model.attention_mask, None)
    hidden[0, 4].sum().backward()
    assert emb.grad is not None and float(emb.grad[0, 5:].abs().max()) == 0.0


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
    cfg = _small_bigbird_config(attention_type="dense", max_seq_len=32, n_layers=3)
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


def test_autotune_reduces_batch_then_sparse_counts_and_fails_closed(monkeypatch):
    config = smoke_config()
    config["model"].update(attention_type="bigbird", max_seq_len=32,
                           bigbird_window=4, bigbird_random_tokens=4, bigbird_global_tokens=2)
    config["training"]["micro_batch_size"] = 2
    config["vram"] = {"target_max_gb": 13.5, "auto_reduce_sparse": True,
                      "min_random_tokens": 0, "min_global_tokens": 1}
    trainer = Trainer(config, device="cpu", samples=smoke_samples(8), resume=False)
    peaks = iter([14.0, 14.0, 14.0, 14.0, 14.0, 14.0])
    monkeypatch.setattr(trainer, "probe_vram", lambda: {
        "measured": True, "peak_gb": next(peaks), "micro_batch": trainer.micro_batch,
        "seq_len": trainer.seq_len})
    with pytest.raises(MemoryError, match="refusing to start"):
        trainer.autotune_batch(13.5)
    assert trainer.micro_batch == 1
    assert trainer.model_config.bigbird_random_tokens == 0
    assert trainer.model_config.bigbird_global_tokens == 1


def test_bigbird_preset_requires_gradient_checkpointing():
    config = smoke_config()
    config["model"].update(attention_type="bigbird", max_seq_len=32,
                           bigbird_window=4, bigbird_random_tokens=2,
                           bigbird_global_tokens=2, gradient_checkpointing=False)
    with pytest.raises(ValueError, match="gradient_checkpointing=True"):
        Trainer(config, device="cpu", samples=smoke_samples(8), resume=False)


def test_simulated_t4_production_preset_fits_and_is_labelled_estimate():
    with open("configs/t4_scaled.yaml", "r", encoding="utf8") as handle:
        config = yaml.safe_load(handle)
    report = simulate_t4_vram(config)
    assert report["measured"] is False and report["simulated"] is True
    assert "simulated" in report["device"].lower()
    assert report["seq_len"] == 16384 and report["micro_batch"] == 1
    assert report["bigbird_random_tokens"] == 32 and report["bigbird_global_tokens"] == 16
    assert report["fits"] and report["peak_gb"] <= 13.5
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
    assert CHECKPOINT_SCHEMA == 6
