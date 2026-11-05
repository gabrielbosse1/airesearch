"""Data mix: weights, fallbacks, packing, sample protocol."""
import pytest
import torch

import data.sources as sources
from data.sources import (BytePacker, MixtureSampler, SourceSpec, SourceUnavailable,
                          _group_iterator, pack_documents, source_samples,
                          strict_utf8)
from models.foundation import BOS


def _infinite(value):
    while True:
        yield value


def test_mixture_respects_weights():
    groups = {name: _infinite({"kind": "text", "data": name.encode()})
              for name in ("code", "language", "multimodal", "logic")}
    sampler = MixtureSampler(groups, seed=24)
    draws = 8000
    for _ in range(draws):
        next(sampler)
    for name, target in (("code", 0.40), ("language", 0.30),
                         ("multimodal", 0.20), ("logic", 0.10)):
        observed = sampler.counts[name] / draws
        assert abs(observed - target) < 0.03, (name, observed)


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
    assert sources.GROUP_WEIGHTS == {"code": 0.40, "language": 0.30,
                                     "multimodal": 0.20, "logic": 0.10}
    ids = {spec.id for specs in sources.DEFAULT_SOURCES.values() for spec in specs}
    assert "bigcode/the-stack" in ids
    assert "TucanoBR/GigaVerbo" in ids
    assert "HuggingFaceFW/fineweb" in ids
    assert "open-web-math/open-web-math" in ids
    assert any("coyo" in i for i in ids)
