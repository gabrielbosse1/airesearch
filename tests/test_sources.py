"""Data mix: PT-BR-first weights, quota exactness, audit, packing, protocol."""
import pytest
import torch

import data.sources as sources
from data.sources import (BytePacker, MixtureSampler, SourceSpec, SourceUnavailable,
                          _group_iterator, audit_batches, audit_mixture,
                          build_mixture, format_mixture_report, pack_documents,
                          quota_allocation, sample_bytes, source_samples, strict_utf8)
from models.foundation import BOS

GROUPS = ("ptbr", "code", "multimodal", "english")
TARGETS = {"ptbr": 0.50, "code": 0.25, "multimodal": 0.15, "english": 0.10}


def _infinite(value):
    while True:
        yield value


def test_quota_allocation_is_exact_and_integral():
    counts = quota_allocation(TARGETS, 100)
    assert counts == {"ptbr": 50, "code": 25, "multimodal": 15, "english": 10}
    # Largest remainder still sums to the window for ugly weights.
    odd = quota_allocation({"a": 0.5, "b": 0.3, "c": 0.2}, 7)
    assert sum(odd.values()) == 7 and odd["a"] == 4
    assert quota_allocation({}, 10) == {}      # no groups -> nothing to plan
    with pytest.raises(ValueError):
        quota_allocation({"a": 0.0}, 10)


def test_mixture_is_strictly_50_25_15_10_per_window():
    groups = {name: _infinite({"kind": "text", "data": name.encode()}) for name in GROUPS}
    sampler = MixtureSampler(groups, seed=24)
    assert sampler.weights == TARGETS
    # Every 100-draw window must be EXACTLY on target (not merely on average).
    for window in range(4):
        for _ in range(sampler.window):
            next(sampler)
        observed = {name: sampler.counts[name] / ((window + 1) * sampler.window)
                    for name in GROUPS}
        for name, target in TARGETS.items():
            assert observed[name] == pytest.approx(target, abs=1e-12), (window, name)


def test_mixture_batches_are_quota_exact_and_seeded():
    def make(seed):
        groups = {name: _infinite({"kind": "text", "data": name.encode()})
                  for name in GROUPS}
        sampler = MixtureSampler(groups, seed=seed)
        return [group for batch in (sampler.next_batch(4) for _ in range(25))
                for group, _ in batch]

    drawn = make(24)
    assert len(drawn) == 100
    assert {name: drawn.count(name) for name in GROUPS} == {"ptbr": 50, "code": 25,
                                                           "multimodal": 15, "english": 10}
    assert drawn == make(24)        # deterministic
    assert drawn != make(7)         # order is seeded, not fixed


def test_next_batch_rejects_empty_size():
    sampler = MixtureSampler({"ptbr": _infinite({})}, {"ptbr": 1.0})
    with pytest.raises(ValueError):
        sampler.next_batch(0)


def test_exhausted_group_renormalises():
    groups = {"code": iter([{"kind": "text", "data": b"a"}]),
              "logic": _infinite({"kind": "text", "data": b"b"})}
    sampler = MixtureSampler(groups, {"code": 0.5, "logic": 0.5}, seed=1)
    for _ in range(10):
        next(sampler)
    assert "code" in sampler.failures
    assert sampler.weights == {"logic": 1.0}
    with pytest.raises(StopIteration):
        finite = MixtureSampler({"x": iter([1])}, {"x": 1.0})
        next(finite)
        next(finite)


def test_packer_produces_shifted_pairs_and_carries():
    packer = BytePacker(8)
    pairs = packer.feed(b"abcdefghij")  # BOS + 10 + EOS = 12 ids
    assert len(pairs) == 1
    ids, targets = pairs[0]
    assert ids.shape == (8,) and targets.shape == (8,)
    assert int(ids[0]) == BOS
    assert torch.equal(ids[1:], targets[:-1])
    assert packer.pending == 4  # 12 - 8 carried, nothing dropped
    pairs2 = packer.feed(b"klmnopqrst")
    assert len(pairs2) >= 1


def test_pack_documents_stream():
    pairs = list(pack_documents([b"hello world"] * 10, 16))
    assert all(ids.shape == (16,) for ids, _ in pairs)


def test_strict_utf8_drops_surrogates():
    assert strict_utf8("ação") == "ação".encode("utf8")
    assert strict_utf8("\ud800abc") == b"abc"
    assert strict_utf8("\ud800") is None


def test_text_source_protocol(monkeypatch):
    spec = SourceSpec("fake/text", text_field="text")
    monkeypatch.setattr(sources, "_open_rows",
                        lambda s: iter([{"text": "olá"}, {"text": ""}, {"text": 5},
                                        {"text": "world"}]))
    samples = list(source_samples(spec))
    assert samples == [{"kind": "text", "data": b"ol\xc3\xa1"},
                       {"kind": "text", "data": b"world"}]


def test_image_embedded_protocol(monkeypatch):
    from PIL import Image
    image = Image.new("RGB", (16, 16), color=(120, 30, 200))
    spec = SourceSpec("fake/img", kind="image_embedded", caption_field="caption")
    monkeypatch.setattr(sources, "_open_rows",
                        lambda s: iter([{"caption": ["a purple square"], "image": image}]))
    samples = list(source_samples(spec))
    assert len(samples) == 1
    sample = samples[0]
    assert sample["kind"] == "multimodal" and sample["modality"] == "image"
    assert sample["caption"] == b"a purple square"
    assert isinstance(sample["image"], bytes) and len(sample["image"]) > 0


def test_audio_protocol(monkeypatch):
    spec = SourceSpec("fake/audio", kind="audio", text_field="text")
    waveform = [0.1] * 3200
    monkeypatch.setattr(sources, "_open_rows",
                        lambda s: iter([{"audio": {"array": waveform, "sampling_rate": 16000},
                                         "text": "hello speech"}]))
    samples = list(source_samples(spec))
    assert len(samples) == 1
    assert samples[0]["modality"] == "audio"
    assert samples[0]["waveform"].shape == (3200,)
    assert samples[0]["sample_rate"] == 16000


def test_fallback_chain_skips_dead_sources(monkeypatch):
    good = SourceSpec("fake/good", text_field="text")
    bad = SourceSpec("fake/bad", text_field="text")

    def fake_open(spec):
        if spec.id == "fake/bad":
            raise ConnectionError("hub down")
        return iter([{"text": "alive"}])

    monkeypatch.setattr(sources, "_open_rows", fake_open)
    iterator = _group_iterator("code", [bad, good])
    assert next(iterator) == {"kind": "text", "data": b"alive"}


def test_all_sources_dead_raises(monkeypatch):
    def fake_open(spec):
        raise ConnectionError("hub down")
    monkeypatch.setattr(sources, "_open_rows", fake_open)
    with pytest.raises(SourceUnavailable):
        _group_iterator("code", [SourceSpec("fake/a"), SourceSpec("fake/b")])


def test_default_mix_matches_spec():
    # PT-BR-first: exactly four groups, exactly these weights, no logic bucket.
    assert sources.GROUP_WEIGHTS == {"ptbr": 0.50, "code": 0.25,
                                     "multimodal": 0.15, "english": 0.10}
    assert set(sources.DEFAULT_SOURCES) == set(sources.GROUP_WEIGHTS)
    assert abs(sum(sources.GROUP_WEIGHTS.values()) - 1.0) < 1e-12
    assert "TucanoBR/GigaVerbo" in {s.id for s in sources.DEFAULT_SOURCES["ptbr"]}
    assert "HuggingFaceFW/fineweb" in {s.id for s in sources.DEFAULT_SOURCES["english"]}
    ids = {spec.id for specs in sources.DEFAULT_SOURCES.values() for spec in specs}
    assert "bigcode/the-stack" in ids
    assert any("coyo" in i for i in ids)
    assert any("librispeech" in i for i in ids)
    assert "open-web-math/open-web-math" not in ids   # dropped: no fifth bucket


def test_build_mixture_rejects_unknown_group_and_uses_config_weights(monkeypatch):
    with pytest.raises(ValueError):
        build_mixture({"weights": {"ptbr": 0.5, "logic": 0.5}})

    def ptbr_is_dead(group, specs, *, limit=0):
        if group == "ptbr":
            raise SourceUnavailable(f"group {group!r}: all sources failed")
        return _infinite({"kind": "text", "data": group.encode()})

    monkeypatch.setattr(sources, "_group_iterator", ptbr_is_dead)
    survivors = build_mixture()
    assert "ptbr" in survivors.failures
    # 0.50 PT-BR removed: the other three renormalise to their relative shares.
    assert survivors.weights == {"code": 0.5, "multimodal": 0.3, "english": 0.2}
    assert len(survivors.next_batch(10)) == 10          # still streams

    monkeypatch.setattr(sources, "_group_iterator",
                        lambda group, specs, *, limit=0: _infinite(
                            {"kind": "text", "data": b"x"}))
    sampler = build_mixture({"weights": {"ptbr": 0.6, "english": 0.4}})
    assert sampler.weights == {"ptbr": 0.6, "english": 0.4}
    assert set(sampler.counts) == {"ptbr", "english"}   # only configured groups open


def test_sample_bytes_counts_each_protocol():
    assert sample_bytes({"kind": "text", "data": b"abc"}) == 3
    assert sample_bytes({"kind": "ids", "ids": [1, 2]}) == 2
    image = {"kind": "multimodal", "modality": "image", "caption": b"abcd",
             "image": b"\xff" * 10, "waveform": None}
    assert sample_bytes(image) == 14
    audio = {"kind": "multimodal", "modality": "audio", "caption": b"ab",
             "image": None, "waveform": torch.zeros(8)}
    assert sample_bytes(audio) == 18
    assert sample_bytes({"kind": "unknown"}) == 0


def test_audit_mixture_proves_the_distribution():
    groups = {name: _infinite({"kind": "text", "data": name.encode() * 8})
              for name in GROUPS}
    sampler = MixtureSampler(groups, seed=24)
    report = audit_mixture(sampler, batches=100, batch_size=4)
    assert report["samples"] == 400 and report["batches"] == 100
    assert report["sample_share"] == pytest.approx(TARGETS)
    assert report["quotas_ok"] and report["complete"] and not report["truncated"]
    # Byte tally: 8x-repeated group name, per-group document counts.
    assert report["bytes_by_group"] == {"ptbr": 200 * 32, "code": 100 * 32,
                                        "multimodal": 60 * 80, "english": 40 * 56}
    assert report["bytes"] == 16640
    assert report["estimated_rows"] == 460     # 60 multimodal docs -> 2 rows each
    assert "expand into 2 training rows" in format_mixture_report(report)
    table = format_mixture_report(report)
    for label in ("PT-BR", "Code", "Multimodal", "English", "verdict: PASS"):
        assert label in table


def test_audit_flags_a_missing_group_and_truncation():
    groups = {name: _infinite({"kind": "text", "data": b"x"}) for name in GROUPS}
    sampler = MixtureSampler(groups, seed=5, window=10)
    sampler._groups.pop("english")          # simulate a dead source chain
    report = audit_mixture(sampler, batches=10, batch_size=3)
    assert not report["complete"] and not report["quotas_ok"]
    assert report["status"]["english"] == "missing"
    assert "WARNING: a source chain is dead" in format_mixture_report(report)

    finite = MixtureSampler({name: iter([{"kind": "text", "data": b"x"}])
                             for name in GROUPS}, seed=1)
    truncated = audit_mixture(finite, batches=5, batch_size=4)
    assert truncated["truncated"] and truncated["batches"] < 5


def test_audit_batches_of_a_renormalised_mix_stays_honest():
    batches = [[("ptbr", {"kind": "text", "data": b"a"})] * 3
               + [("code", {"kind": "text", "data": b"b"})]]
    report = audit_batches(batches, targets={"ptbr": 0.75, "code": 0.25})
    assert report["sample_share"] == {"ptbr": 0.75, "code": 0.25}
    assert report["quotas_ok"]
