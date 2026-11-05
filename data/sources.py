"""Streaming data mix for the 4-hour T4 pre-training run (Phase 24).

Weights (of the token budget):
    40% code      -- bigcode/the-stack (Python + JavaScript)
    30% language  -- TucanoBR/GigaVerbo (PT-BR) + HuggingFaceFW/fineweb (EN)
    20% multimodal-- COYO image/caption pairs + LibriSpeech audio/transcripts
    10% logic     -- open-web-math

Everything streams (``load_dataset(..., streaming=True)``): nothing is
downloaded wholesale. Each source has a fallback chain; a group whose every
source fails is dropped and its weight renormalised over the survivors
(logged, never silent). The mixture is deterministic given a seed.

Sample protocol yielded by the mixture:
    text:       {"kind": "text", "data": bytes}
    multimodal: {"kind": "multimodal", "modality": "image"|"audio",
                 "caption": bytes, "image": bytes|None,
                 "waveform": Tensor|None, "sample_rate": int|None}

The trainer turns these into byte sequences and (for multimodal) bidirectional
training pairs. This module never imports the model.
"""
from __future__ import annotations

import io
import random
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping

import torch

from models.foundation import BOS, EOS

GROUP_WEIGHTS: dict[str, float] = {"code": 0.40, "language": 0.30,
                                   "multimodal": 0.20, "logic": 0.10}


class SourceUnavailable(RuntimeError):
    """Every source in a chain failed to open or produce rows."""


@dataclass(frozen=True)
class SourceSpec:
    id: str
    name: str | None = None
    data_dir: str | None = None
    split: str = "train"
    text_field: str = "text"
    kind: str = "text"            # "text" | "image_url" | "image_embedded" | "audio"
    caption_field: str = "text"

    def load_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"path": self.id, "split": self.split, "streaming": True}
        if self.name is not None:
            kwargs["name"] = self.name
        if self.data_dir is not None:
            kwargs["data_dir"] = self.data_dir
        return kwargs


# Default chains: first that opens AND yields wins; the rest are fallbacks.
DEFAULT_SOURCES: dict[str, tuple[SourceSpec, ...]] = {
    "code": (
        SourceSpec("bigcode/the-stack", data_dir="data/python", text_field="content"),
        SourceSpec("bigcode/the-stack", data_dir="data/javascript", text_field="content"),
        SourceSpec("bigcode/the-stack-smol", data_dir="data/python", text_field="content"),
    ),
    "language": (
        SourceSpec("TucanoBR/GigaVerbo", text_field="text"),
        SourceSpec("HuggingFaceFW/fineweb", name="sample-10BT", text_field="text"),
    ),
    "multimodal": (
        SourceSpec("kakaobrain/coyo-700m", kind="image_url",
                   caption_field="text", split="train"),
        SourceSpec("nlphuji/mscoco_2014_5k_test_image_text_retrieval",
                   kind="image_embedded", caption_field="caption", split="test"),
        SourceSpec("openslr/librispeech_asr", name="clean", split="train.100",
                   kind="audio", text_field="text"),
        SourceSpec("hf-internal-testing/librispeech_asr_dummy", name="clean",
                   split="validation", kind="audio", text_field="text"),
    ),
    "logic": (
        SourceSpec("open-web-math/open-web-math", text_field="text"),
    ),
}


def strict_utf8(text: str) -> bytes | None:
    """Encode text; drop lone surrogates rather than crashing the stream."""
    try:
        return text.encode("utf8", errors="strict")
    except UnicodeEncodeError:
        cleaned = text.encode("utf8", errors="ignore")
        return cleaned if cleaned else None


def _open_rows(spec: SourceSpec) -> Iterator[Mapping[str, Any]]:
    from datasets import load_dataset  # lazy: CPU tests never need it
    dataset = load_dataset(**spec.load_kwargs())
    return iter(dataset)


def _fetch_url(url: str, *, timeout: float = 8.0, max_bytes: int = 4_000_000) -> bytes | None:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "airesearch-pretrain/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read(max_bytes + 1)
        if len(data) > max_bytes or not data:
            return None
        return data
    except Exception:
        return None


def _image_payload(raw: bytes) -> bytes | None:
    """Validate that fetched bytes really decode as an image."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        return raw
    except Exception:
        return None


def source_samples(spec: SourceSpec, *, limit: int = 0) -> Iterator[dict[str, Any]]:
    """Turn one source into the sample protocol. Skips bad rows, never raises
    mid-stream (a dead URL is data loss, not a crash)."""
    emitted = 0
    for row in _open_rows(spec):
        if limit and emitted >= limit:
            return
        try:
            if spec.kind == "text":
                text = row.get(spec.text_field)
                if not isinstance(text, str) or not text.strip():
                    continue
                data = strict_utf8(text)
                if data:
                    emitted += 1
                    yield {"kind": "text", "data": data}
            elif spec.kind == "image_url":
                caption = row.get(spec.caption_field)
                url = row.get("url")
                if not isinstance(caption, str) or not isinstance(url, str):
                    continue
                raw = _fetch_url(url)
                if raw is None:
                    continue
                payload = _image_payload(raw)
                caption_bytes = strict_utf8(caption)
                if payload is None or not caption_bytes:
                    continue
                emitted += 1
                yield {"kind": "multimodal", "modality": "image",
                       "caption": caption_bytes, "image": payload,
                       "waveform": None, "sample_rate": None}
            elif spec.kind == "image_embedded":
                caption = row.get(spec.caption_field)
                image = row.get("image")
                if isinstance(caption, list):
                    caption = caption[0] if caption else None
                if not isinstance(caption, str) or image is None:
                    continue
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="JPEG", quality=90)
                caption_bytes = strict_utf8(caption)
                if not caption_bytes:
                    continue
                emitted += 1
                yield {"kind": "multimodal", "modality": "image",
                       "caption": caption_bytes, "image": buffer.getvalue(),
                       "waveform": None, "sample_rate": None}
            elif spec.kind == "audio":
                audio = row.get("audio")
                text = row.get(spec.text_field)
                if not isinstance(audio, Mapping) or not isinstance(text, str):
                    continue
                array = audio.get("array")
                rate = audio.get("sampling_rate")
                if array is None or rate is None:
                    continue
                waveform = torch.as_tensor(array, dtype=torch.float32)
                if waveform.ndim > 1:
                    waveform = waveform.mean(dim=0)
                caption_bytes = strict_utf8(text)
                if not caption_bytes or waveform.numel() < 1600:  # <0.1s @16k
                    continue
                emitted += 1
                yield {"kind": "multimodal", "modality": "audio",
                       "caption": caption_bytes, "image": None,
                       "waveform": waveform, "sample_rate": int(rate)}
            else:
                raise ValueError(f"unknown source kind {spec.kind!r}")
        except Exception:
            continue  # a malformed row is skipped, the stream survives


def _group_iterator(group: str, specs: Iterable[SourceSpec], *, limit: int = 0
                    ) -> Iterator[dict[str, Any]]:
    """First source that opens and yields at least one sample wins."""
    errors: list[str] = []
    for spec in specs:
        try:
            stream = source_samples(spec, limit=limit)
            first = next(stream)
        except StopIteration:
            errors.append(f"{spec.id}: empty")
            continue
        except Exception as exc:
            errors.append(f"{spec.id}: {type(exc).__name__}: {exc}")
            continue

        def chain(first_sample: dict[str, Any], rest: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
            yield first_sample
            yield from rest

        return chain(first, stream)
    raise SourceUnavailable(f"group {group!r}: all sources failed: {errors}")


class MixtureSampler:
    """Deterministic weighted sampling over group iterators.

    Exhausted (or failed) groups are dropped and weights renormalised over the
    survivors. ``failures`` records what was dropped and why.
    """

    def __init__(self, groups: Mapping[str, Iterator[dict[str, Any]]],
                 weights: Mapping[str, float] | None = None, *, seed: int = 24) -> None:
        base = dict(weights or GROUP_WEIGHTS)
        self._groups: dict[str, Iterator[dict[str, Any]]] = {}
        self._weights: dict[str, float] = {}
        self.failures: dict[str, str] = {}
        for name, iterator in groups.items():
            weight = float(base.get(name, 0.0))
            if weight <= 0:
                continue
            self._groups[name] = iterator
            self._weights[name] = weight
        self._renormalise()
        self._rng = random.Random(seed)
        self.counts: dict[str, int] = {name: 0 for name in self._groups}

    def _renormalise(self) -> None:
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {name: weight / total for name, weight in self._weights.items()}

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def __iter__(self) -> "MixtureSampler":
        return self

    def __next__(self) -> tuple[str, dict[str, Any]]:
        while self._groups:
            names = sorted(self._groups)
            draw = self._rng.random()
            cumulative = 0.0
            chosen = names[-1]
            for name in names:
                cumulative += self._weights[name]
                if draw <= cumulative:
                    chosen = name
                    break
            try:
                sample = next(self._groups[chosen])
            except StopIteration:
                del self._groups[chosen]
                del self._weights[chosen]
                self.failures[chosen] = "exhausted"
                self._renormalise()
                continue
            self.counts[chosen] += 1
            return chosen, sample
        raise StopIteration


def build_mixture(config: Mapping[str, Any] | None = None, *, seed: int = 24,
                  limit_per_source: int = 0) -> MixtureSampler:
    """Open the default streaming mix. Groups whose chains fail are dropped."""
    weights = dict(GROUP_WEIGHTS)
    if config and "weights" in config:
        weights = {name: float(value) for name, value in config["weights"].items()}
    groups: dict[str, Iterator[dict[str, Any]]] = {}
    failures: dict[str, str] = {}
    for group, specs in DEFAULT_SOURCES.items():
        try:
            groups[group] = _group_iterator(group, specs, limit=limit_per_source)
        except SourceUnavailable as exc:
            failures[group] = str(exc)
    if not groups:
        raise SourceUnavailable(f"no data group could be opened: {failures}")
    sampler = MixtureSampler(groups, weights, seed=seed)
    sampler.failures.update(failures)
    return sampler


# ---------------------------------------------------------------------------
# Byte packing (text groups)
# ---------------------------------------------------------------------------
class BytePacker:
    """Pack framed documents (BOS doc EOS) into fixed-length training pairs.

    A pair needs ``seq_len + 1`` ids: ``ids = chunk[:-1]``, ``targets =
    chunk[1:]``. Leftover bytes carry across documents; nothing is dropped.
    """

    def __init__(self, seq_len: int) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        self.seq_len = int(seq_len)
        self._buffer: list[int] = []

    def feed(self, document: bytes) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if not isinstance(document, (bytes, bytearray)):
            raise TypeError("document must be bytes")
        self._buffer.extend([BOS] + list(bytes(document)) + [EOS])
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        while len(self._buffer) >= self.seq_len + 1:
            chunk = self._buffer[: self.seq_len + 1]
            self._buffer = self._buffer[self.seq_len:]
            ids = torch.tensor(chunk[:-1], dtype=torch.long)
            targets = torch.tensor(chunk[1:], dtype=torch.long)
            pairs.append((ids, targets))
        return pairs

    @property
    def pending(self) -> int:
        return len(self._buffer)


def pack_documents(documents: Iterable[bytes], seq_len: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    packer = BytePacker(seq_len)
    for document in documents:
        yield from packer.feed(document)
