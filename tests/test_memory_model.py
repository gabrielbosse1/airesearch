"""Regression tests for :mod:`models.memory_model` (Level 3 of the audit).

The hierarchical store is a *tensor* structure, not part of the byte stream:
episodic ``(capacity, d)``, semantic ``(capacity, n_exemplars, d)`` and
procedural ``(capacity, d)`` plus ``(capacity, max_trace_steps, d)`` traces.
These tests pin the retrieval algebra (scaled dot product + decay + importance,
mixture LogSumExp for semantic slots, top-K softmax), the write/eviction rules
and the provenance guard. They assert nothing about answer quality.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from models.holdout import HoldoutGuard, HoldoutViolation, content_hash
from models.memory_model import (
    ALL_TIERS, TIER_EPISODIC, TIER_PROCEDURAL, TIER_SEMANTIC,
    ImportanceScorer, MemoryModel, SemanticExemplarBank,
)


def small_model(**overrides) -> MemoryModel:
    values = dict(d_model=16, capacity=12, n_heads=4, n_exemplars=2,
                  max_trace_steps=3, importance_weight=0.35,
                  promotion_threshold=0.62)
    values.update(overrides)
    return MemoryModel(**values)


def write(model: MemoryModel, *, tier: str, seed: int, **metadata) -> int:
    generator = torch.Generator().manual_seed(seed)
    embedding = torch.randn(model.d_model, generator=generator)
    metadata.setdefault("tier", tier)
    return model.write(embedding, metadata)


# ---------------------------------------------------------------------------
# Shapes and tier layout
# ---------------------------------------------------------------------------
def test_capacity_is_split_into_the_three_tiers():
    model = small_model(capacity=12)
    assert (model.epi_cap, model.sem_cap, model.proc_cap) == (6, 3, 3)
    assert model.capacity == 12
    assert model.episodic.keys.shape == (6, 16)
    assert model.semantic.keys.shape == (3, 2, 16)      # (slots, exemplars, d)
    assert model.procedural.keys.shape == (3, 16)
    assert model.procedural.trace_steps.shape == (3, 3, 16)
    explicit = small_model(tier_capacities=(4, 2, 1))
    assert (explicit.epi_cap, explicit.sem_cap, explicit.proc_cap) == (4, 2, 1)


def test_global_and_local_slot_indices_are_consistent():
    model = small_model(capacity=12)
    assert model._to_global_slot(TIER_EPISODIC, 0) == 0
    assert model._to_global_slot(TIER_SEMANTIC, 0) == 6
    assert model._to_global_slot(TIER_PROCEDURAL, 0) == 9
    for tier in ALL_TIERS:
        for local in range(2):
            assert model._from_global_slot(model._to_global_slot(tier, local)) == (tier, local)
    with pytest.raises(IndexError):
        model._from_global_slot(99)


def test_writes_route_to_the_tier_named_by_metadata():
    model = small_model()
    episodic = model.write(torch.randn(16), {"type": "event"})
    semantic = model.write(torch.randn(16), {"type": "fact", "concept": "c"})
    procedural = model.write(torch.randn(16), {"type": "skill", "utility": 1.2})
    assert model._from_global_slot(episodic)[0] == TIER_EPISODIC
    assert model._from_global_slot(semantic)[0] == TIER_SEMANTIC
    assert model._from_global_slot(procedural)[0] == TIER_PROCEDURAL
    stats = model.tier_stats()
    assert stats["episodic_used"] == stats["semantic_slots_used"] == stats["procedural_used"] == 1


# ---------------------------------------------------------------------------
# Retrieval algebra
# ---------------------------------------------------------------------------
def test_empty_memory_returns_empty_tokens_not_an_error():
    model = small_model()
    entries, tokens = model.retrieve(torch.zeros(16), top_k=3)
    assert entries == []
    assert tokens.shape == (0, 16)
    _, pooled = model.retrieve(torch.zeros(16), top_k=3, pool=True)
    assert pooled.shape == (16,)


def test_episodic_logits_are_scaled_dot_product_minus_decay_plus_importance():
    torch.manual_seed(0)
    model = small_model(capacity=12)
    model.session_tick = 40
    for index in range(3):
        write(model, tier=TIER_EPISODIC, seed=index, person=f"p{index}")
    bank = model.episodic
    query = torch.randn(1, model.d_model)
    _, logits, values, importance = bank.score_active(
        query, model.session_tick, model.importance_scorer, model.importance_weight)
    used = bank.used.nonzero().squeeze(-1)
    similarity = (query @ bank.keys[used].T) / math.sqrt(model.d_model)
    decay = bank.decay_rate * (
        (float(model.session_tick) - bank.written_at[used].float()) / bank.time_scale)
    expected = similarity - decay.unsqueeze(0) + model.importance_weight * importance.unsqueeze(0)
    assert logits.shape == (1, 3)
    assert values.shape == (1, 3, model.d_model)
    assert torch.allclose(logits, expected, atol=1e-6)


def test_decay_prefers_the_recent_entry_when_keys_are_identical():
    model = small_model(capacity=12)
    key = torch.ones(model.d_model)
    model.episodic.write_slot(key, torch.ones(model.d_model), {"tier": TIER_EPISODIC},
                              current_tick=0, prediction_error=0.0,
                              importance_scorer=model.importance_scorer)
    model.episodic.write_slot(key, torch.ones(model.d_model), {"tier": TIER_EPISODIC},
                              current_tick=90, prediction_error=0.0,
                              importance_scorer=model.importance_scorer)
    _, logits, _, _ = model.episodic.score_active(
        key.unsqueeze(0), 100, model.importance_scorer, model.importance_weight)
    assert int(logits.argmax(dim=-1)) == 1  # the newer write wins


def test_importance_scorer_is_bounded_and_increasing_in_surprise():
    lens = small_model().importance_scorer if False else ImportanceScorer(d_model=16)
    values = torch.randn(4, 16)
    low = lens(torch.zeros(4), torch.zeros(4, dtype=torch.long),
               torch.zeros(4), values)
    high = lens(torch.full((4,), 5.0), torch.zeros(4, dtype=torch.long),
                torch.zeros(4), values)
    assert torch.all(low > 0) and torch.all(low < 1)
    assert torch.all(high > low)
    assert lens(torch.empty(0), torch.empty(0, dtype=torch.long),
                torch.empty(0), torch.empty(0, 16)).numel() == 0


def test_retrieve_returns_top_k_multivectors_with_normalised_weights():
    torch.manual_seed(0)
    model = small_model(capacity=12)
    for index in range(5):
        write(model, tier=TIER_EPISODIC, seed=index, person="alice")
    entries, tokens = model.retrieve(torch.randn(16), top_k=3)
    assert len(entries) == 3
    assert tokens.shape == (3, 16)
    assert math.isclose(sum(entry["weight"] for entry in entries), 1.0, rel_tol=1e-5)
    assert [entry["weight"] for entry in entries] == sorted(
        (entry["weight"] for entry in entries), reverse=True)
    # Access statistics are updated on a real (non read-only) read.
    total_accesses = int(model.episodic.access_counts.sum())
    assert total_accesses == 3


def test_metadata_and_person_filters_narrow_the_candidate_set():
    model = small_model(capacity=12)
    write(model, tier=TIER_EPISODIC, seed=1, person="alice", modality="text")
    write(model, tier=TIER_EPISODIC, seed=2, person="bob", modality="voice")
    entries, _ = model.retrieve(torch.randn(16), top_k=5, person_filter="alice")
    assert [entry["metadata"]["person"] for entry in entries] == ["alice"]
    entries, _ = model.retrieve(torch.randn(16), top_k=5, metadata_filter={"modality": "voice"})
    assert [entry["metadata"]["person"] for entry in entries] == ["bob"]
    with pytest.raises(ValueError):
        model.retrieve(torch.randn(16), metadata_filter={1: "x"})


def test_semantic_mixture_uses_logsumexp_and_posterior_readout():
    torch.manual_seed(0)
    model = small_model()
    bank: SemanticExemplarBank = model.semantic
    with torch.no_grad():
        bank.used[0] = True
        bank.exemplar_used[0, 0] = True
        bank.exemplar_used[0, 1] = True
        bank.keys[0, 0] = torch.tensor([1.0] + [0.0] * 15)
        bank.keys[0, 1] = torch.tensor([0.0, 1.0] + [0.0] * 14)
        bank.values[0, 0] = torch.full((16,), 2.0)
        bank.values[0, 1] = torch.full((16,), -2.0)
        bank.exemplar_counts[0, 0] = 3.0
        bank.exemplar_counts[0, 1] = 1.0
        bank._renormalize_slot(0)
        assert torch.allclose(bank.mixture_weights[0], torch.tensor([0.75, 0.25]))
        query = torch.tensor([1.0] + [0.0] * 15).unsqueeze(0)
        _, logits, values, importance = bank.score_active(
            query, 0, model.importance_scorer, model.importance_weight)
        sim = (query @ bank.keys[0].T) / math.sqrt(model.d_model)
        comp = sim + torch.log(bank.mixture_weights[0])
        expected_logit = (torch.logsumexp(comp, dim=-1)
                          + model.importance_weight * importance[0])
        assert torch.allclose(logits[0, 0], expected_logit, atol=1e-6)
        posterior = torch.softmax(comp.squeeze(0), dim=-1)
        expected_value = (posterior.unsqueeze(-1) * bank.values[0]).sum(dim=0)
        assert torch.allclose(values[0, 0], expected_value, atol=1e-6)


def test_semantic_write_or_merge_keeps_modes_distinct_and_merges_duplicates():
    model = small_model()
    bank = model.semantic
    first = torch.zeros(16)
    first[0] = 1.0
    write(model, tier=TIER_SEMANTIC, seed=0, concept_cluster="c")
    model.write(first, {"tier": TIER_SEMANTIC, "concept_cluster": "c"})
    second = torch.zeros(16)
    second[1] = 1.0
    model.write(second, {"tier": TIER_SEMANTIC, "concept_cluster": "c"})
    active = int(bank.exemplar_used[0].sum())
    assert active == 2, "an orthogonal exemplar opens a new mode"
    # A near-duplicate of an existing exemplar updates it rather than opening one.
    model.write(first * (1.0 + 1e-4), {"tier": TIER_SEMANTIC, "concept_cluster": "c"})
    assert int(bank.exemplar_used[0].sum()) == 2
    assert torch.allclose(bank.mixture_weights[0].sum(), torch.tensor(1.0), atol=1e-6)


def test_procedural_trace_is_a_weighted_sum_of_steps():
    torch.manual_seed(0)
    model = small_model()
    steps = torch.randn(2, 16, generator=torch.Generator().manual_seed(3))
    index = model.write(torch.randn(16), {"tier": TIER_PROCEDURAL, "type": "skill"},
                        trace_steps=steps)
    tier, local = model._from_global_slot(index)
    assert tier == TIER_PROCEDURAL
    bank = model.procedural
    assert bank.trace_lens[local] == 2
    assert torch.count_nonzero(bank.trace_steps[local, 2]) == 0  # unused steps stay zero
    with pytest.raises(ValueError):
        model.write(torch.randn(16), {"tier": TIER_PROCEDURAL},
                    trace_steps=torch.randn(9, 16))  # exceeds max_trace_steps


# ---------------------------------------------------------------------------
# Differentiable read (STE) used by co-training
# ---------------------------------------------------------------------------
def test_differentiable_read_is_exact_top_k_in_the_forward_pass():
    torch.manual_seed(0)
    model = small_model(capacity=12)
    for index in range(4):
        write(model, tier=TIER_EPISODIC, seed=index, person="alice")
    query = torch.randn(16, requires_grad=True)
    tokens, weights, mask = model.retrieve_differentiable(query, top_k=2)
    candidates = model.episodic.values[model.episodic.used]
    picked = {tuple(value.tolist()) for value in candidates}
    for token in tokens[0]:
        assert tuple(token.tolist()) in picked  # forward value is an untouched slot value
    assert tokens.shape == (1, 2, 16)
    assert weights.shape == (1, 2) and torch.allclose(weights.sum(), torch.tensor(1.0))
    assert mask.dtype == torch.bool and bool(mask.all())
    tokens.sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert float(query.grad.abs().sum()) > 0.0, "STE must carry a gradient to the query"


# ---------------------------------------------------------------------------
# Guard, read-only mode, activation
# ---------------------------------------------------------------------------
def test_read_only_mode_blocks_writes_but_allows_reads():
    model = small_model()
    write(model, tier=TIER_EPISODIC, seed=0, person="alice")
    with model.read_only():
        entries, _ = model.retrieve(torch.randn(16), top_k=1)
        assert len(entries) == 1
        with pytest.raises(HoldoutViolation):
            model.write(torch.randn(16), {"tier": TIER_EPISODIC})
    model.write(torch.randn(16), {"tier": TIER_EPISODIC})  # writable again


def test_holdout_guard_rejects_protected_bytes_and_requires_provenance():
    guard = HoldoutGuard.from_records([{"input": b"secret question", "output": b"secret answer"}])
    model = small_model(capacity=12)  # empty, so the audit passes
    model.set_holdout_guard(guard)
    with pytest.raises(HoldoutViolation):
        model.write(torch.randn(16), {"tier": TIER_EPISODIC}, raw_input=b"secret question")
    with pytest.raises(HoldoutViolation):  # guarded writes need raw provenance at all
        model.write(torch.randn(16), {"tier": TIER_EPISODIC})
    index = model.write(torch.randn(16), {"tier": TIER_EPISODIC}, raw_input=b"allowed input")
    provenance = model.episodic.metadata[index][HoldoutGuard.PROVENANCE_KEY]
    assert provenance["input_hash"] == content_hash(b"allowed input")
    with pytest.raises(HoldoutViolation):  # low-level writes bypass the guard
        model.episodic.write_slot(torch.zeros(16), torch.zeros(16), {}, 0, 0.0,
                                  model.importance_scorer)


def test_checkpoint_roundtrip_preserves_slots_and_metadata(tmp_path):
    model = small_model()
    write(model, tier=TIER_EPISODIC, seed=0, person="alice")
    write(model, tier=TIER_SEMANTIC, seed=1, concept_cluster="c")
    write(model, tier=TIER_PROCEDURAL, seed=2, type="skill")
    model.session_tick = 7
    path = tmp_path / "memory.pt"
    model.save(str(path))
    restored = small_model()
    restored.load(str(path))
    assert torch.equal(restored.episodic.keys, model.episodic.keys)
    assert restored.semantic.metadata == model.semantic.metadata
    assert restored.session_tick == 7
    assert restored.tier_stats() == model.tier_stats()


def test_consolidation_promotes_frequent_surprising_episodes():
    torch.manual_seed(0)
    model = small_model(promotion_threshold=0.0)
    index = write(model, tier=TIER_EPISODIC, seed=0, person="alice", concept="c")
    local = index
    model.episodic.access_counts[local] = 5
    model.episodic.pred_errors[local] = 5.0
    report = model.consolidate(lr=0.01)
    assert report["promoted_to_semantic"] == 1
    assert model.episodic.metadata[local]["promoted_to_semantic"] is True
    assert model.tier_stats()["semantic_slots_used"] >= 1
    second = model.consolidate(lr=0.0)
    assert second["promoted_to_semantic"] == 0  # already promoted, not repeated


def test_episodic_eviction_replaces_the_lowest_retention_slot():
    model = small_model(tier_capacities=(2, 1, 1))
    for tick in (0, 1):
        model.episodic.write_slot(torch.randn(16), torch.randn(16), {"tier": TIER_EPISODIC},
                                  current_tick=tick, prediction_error=0.0,
                                  importance_scorer=model.importance_scorer)
    model.episodic.access_counts[1] = 10  # slot 1 is valuable
    victim = model.episodic.allocate_slot(2, model.importance_scorer)
    assert victim == 0


def test_retrieve_filters_out_unknown_tier_names():
    model = small_model()
    write(model, tier=TIER_EPISODIC, seed=0, person="alice")
    entries, tokens = model.retrieve(torch.randn(16), top_k=3, tier_filter="semantic")
    assert entries == [] and tokens.shape == (0, 16)
