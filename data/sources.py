"""Streaming data mix for the 4-hour T4 pre-training run (Phase 24.1).

PT-BR-FIRST distribution (fractions of *drawn documents*, not bytes):

    50% ptbr       -- TucanoBR/GigaVerbo, fallback HuggingFaceFW/fineweb-2 (por_Latn)
    25% code       -- bigcode/the-stack (Python + JavaScript)
    15% multimodal -- COYO image/caption pairs + LibriSpeech audio/transcripts
    10% english    -- HuggingFaceFW/fineweb (EN)

The quota is **strict, not just in expectation**. ``MixtureSampler`` allocates
each window of ``MIX_WINDOW`` draws with the largest-remainder method
(100 draws -> exactly 50/25/15/10) and shuffles that window with the seeded
RNG, so any multiple of 100 draws carries the exact ratio while the document
order stays stochastic. A group that dies is dropped, the survivors are
renormalised and the current window is rebuilt (logged in ``failures``).

Honest unit note: the weights govern *documents*, which is what a sampler can
control. A code file is far longer than an image caption, so the **byte/token**
share of the stream is reported alongside the document share by
``audit_mixture`` and will not equal 50/25/15/10. Both numbers are printed;
neither is hidden. Multimodal documents additionally expand into two training
rows (caption and media codec).

Everything streams (``load_dataset(..., streaming=True)``): nothing is
downloaded wholesale. Each source has a fallback chain; a group whose every
source fails is dropped and its weight renormalised over the survivors
(logged, never silent). The mixture is deterministic given a seed.

Sample protocol yielded by the mixture:
    text:       {"kind": "text", "data": bytes}
    ids:        {"kind": "ids", "ids": list[int]}          (tool/edit rows)
    multimodal: {"kind": "multimodal", "modality": "image"|"audio",
                 "caption": bytes, "image": bytes|None,
                 "waveform": Tensor|None, "sample_rate": int|None}

The trainer turns these into byte sequences and (for multimodal) bidirectional
training pairs. This module never imports the model.
"""
from __future__ import annotations

import io
import math
import random
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

from models.foundation import BOS, EOS

#: The Phase 24.1 PT-BR-first mix. Keys are group ids, values are fractions of
#: drawn documents and must sum to 1.0.
GROUP_WEIGHTS: dict[str, float] = {"ptbr": 0.50, "code": 0.25,
                                   "multimodal": 0.15, "english": 0.10}

#: Draws per quota window. 100 keeps 50/25/15/10 integral, so a full window is
#: exactly on target instead of merely on average.
MIX_WINDOW: int = 100

#: Human-readable labels for the audit table (and the trainer's summary).
GROUP_LABELS: dict[str, str] = {"ptbr": "PT-BR", "code": "Code",
                                "multimodal": "Multimodal", "english": "English",
                                "logic": "Logic", "language": "Language",
                                "tools": "Tools"}


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
    # 50% -- Brazilian Portuguese is the model's primary language.
    "ptbr": (
        SourceSpec("TucanoBR/GigaVerbo", text_field="text"),
        SourceSpec("HuggingFaceFW/fineweb-2", name="por_Latn", text_field="text"),
    ),
    "code": (
        SourceSpec("bigcode/the-stack", data_dir="data/python", text_field="content"),
        SourceSpec("bigcode/the-stack", data_dir="data/javascript", text_field="content"),
        SourceSpec("bigcode/the-stack-smol", data_dir="data/python", text_field="content"),
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
    # 10% -- English is the secondary language; it keeps cross-lingual transfer
    # alive (code/technical vocabulary) without crowding PT-BR out.
    "english": (
        SourceSpec("HuggingFaceFW/fineweb", name="sample-10BT", text_field="text"),
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


def quota_allocation(weights: Mapping[str, float], window: int = MIX_WINDOW) -> dict[str, int]:
    """Integer draw counts per group summing to ``window`` (largest remainder).

    ``window=100`` with 0.50/0.25/0.15/0.10 gives exactly 50/25/15/10, so a
    window is on target instead of merely on target in expectation.
    """
    if window < 1:
        raise ValueError("window must be positive")
    if not weights:
        return {}
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    exact = {name: float(weight) / total * window for name, weight in weights.items()}
    counts = {name: int(math.floor(value)) for name, value in exact.items()}
    leftover = window - sum(counts.values())
    # Ties broken by name so the allocation is deterministic across platforms.
    ranked = sorted(exact, key=lambda name: (-(exact[name] - counts[name]), name))
    for name in ranked[: max(leftover, 0)]:
        counts[name] += 1
    return counts


class MixtureSampler:
    """Deterministic, quota-exact weighted sampling over group iterators.

    Each window of ``window`` draws contains exactly
    ``quota_allocation(weights, window)`` samples per group, in a seeded
    shuffled order -- so any multiple of the window carries the target mix
    exactly, while document order stays stochastic. Exhausted (or failed)
    groups are dropped, the survivors renormalised and the window rebuilt;
    ``failures`` records what was dropped and why.
    """

    def __init__(self, groups: Mapping[str, Iterator[dict[str, Any]]],
                 weights: Mapping[str, float] | None = None, *, seed: int = 24,
                 window: int = MIX_WINDOW) -> None:
        base = dict(weights or GROUP_WEIGHTS)
        self._groups: dict[str, Iterator[dict[str, Any]]] = {}
        self._weights: dict[str, float] = {}
        self.failures: dict[str, str] = {}
        self.window = int(window)
        for name, iterator in groups.items():
            weight = float(base.get(name, 0.0))
            if not math.isfinite(weight):
                raise ValueError(f"group {name!r}: weight must be finite")
            if weight < 0:
                raise ValueError(f"group {name!r}: weight must be >= 0")
            if weight == 0.0:
                continue
            self._groups[name] = iterator
            self._weights[name] = weight
        self._renormalise()
        self._rng = random.Random(seed)
        self._queue: deque[str] = deque()
        self.counts: dict[str, int] = {name: 0 for name in self._groups}

    def _renormalise(self) -> None:
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {name: weight / total for name, weight in self._weights.items()}

    def _refill(self) -> None:
        """Lay out the next quota window and shuffle it deterministically."""
        tickets: list[str] = []
        for name, count in quota_allocation(self._weights, self.window).items():
            tickets.extend([name] * count)
        self._rng.shuffle(tickets)
        self._queue = deque(tickets)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    @property
    def targets(self) -> dict[str, float]:
        """Alias kept explicit for the audit/reporting code."""
        return dict(self._weights)

    def __iter__(self) -> "MixtureSampler":
        return self

    def __next__(self) -> tuple[str, dict[str, Any]]:
        while self._groups:
            if not self._queue:
                self._refill()
            chosen = self._queue.popleft()
            if chosen not in self._groups:  # dropped since the window was built
                continue
            try:
                sample = next(self._groups[chosen])
            except StopIteration:
                del self._groups[chosen]
                del self._weights[chosen]
                self.failures[chosen] = "exhausted"
                self._renormalise()
                self._queue.clear()  # the surviving weights changed: re-plan
                continue
            self.counts[chosen] += 1
            return chosen, sample
        raise StopIteration

    def next_batch(self, size: int) -> list[tuple[str, dict[str, Any]]]:
        """One batch of ``size`` draws, mixed by quota (the DataLoader unit)."""
        if size < 1:
            raise ValueError("size must be positive")
        return [next(self) for _ in range(size)]


def build_mixture(config: Mapping[str, Any] | None = None, *, seed: int = 24,
                  limit_per_source: int = 0,
                  window: int = MIX_WINDOW) -> MixtureSampler:
    """Open the default streaming mix. Groups whose chains fail are dropped."""
    weights = dict(GROUP_WEIGHTS)
    if config and "weights" in config:
        requested = {name: float(value) for name, value in config["weights"].items()}
        unknown = sorted(set(requested) - set(DEFAULT_SOURCES))
        if unknown:
            raise ValueError(f"unknown data group(s) {unknown}; "
                             f"known groups are {sorted(DEFAULT_SOURCES)}")
        if sum(requested.values()) <= 0:
            raise ValueError("data weights must sum to a positive number")
        weights = requested
    groups: dict[str, Iterator[dict[str, Any]]] = {}
    failures: dict[str, str] = {}
    for group, specs in DEFAULT_SOURCES.items():
        if group not in weights or weights[group] <= 0:
            continue
        try:
            groups[group] = _group_iterator(group, specs, limit=limit_per_source)
        except SourceUnavailable as exc:
            failures[group] = str(exc)
    if not groups:
        raise SourceUnavailable(f"no data group could be opened: {failures}")
    sampler = MixtureSampler(groups, weights, seed=seed, window=window)
    sampler.failures.update(failures)
    return sampler


# ---------------------------------------------------------------------------
# Mix audit: prove the distribution BEFORE burning a 4-hour GPU budget
# ---------------------------------------------------------------------------
def sample_bytes(sample: Mapping[str, Any]) -> int:
    """Payload size of one sample: the token/byte proxy for the audit."""
    kind = sample.get("kind")
    if kind == "text":
        return len(sample.get("data") or b"")
    if kind == "ids":
        return len(sample.get("ids") or ())
    if kind == "multimodal":
        size = len(sample.get("caption") or b"")
        image = sample.get("image")
        if image:
            size += len(image)
        waveform = sample.get("waveform")
        if waveform is not None:
            size += int(getattr(waveform, "numel", lambda: 0)()) * 2  # fp16 payload
        return size
    return 0


def _share(counts: Mapping[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {name: 0.0 for name in counts}
    return {name: value / total for name, value in counts.items()}


def audit_batches(batches: Sequence[Sequence[tuple[str, dict[str, Any]]]], *,
                  targets: Mapping[str, float], tolerance: float = 0.02,
                  window: int | None = None,
                  failures: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Measure the realised mix of already-drawn batches.

    Returns document share (the contract), byte share (informational, it is a
    different quantity) and a pass/fail verdict per group.
    """
    sample_counts: dict[str, int] = {name: 0 for name in targets}
    byte_counts: dict[str, int] = {name: 0 for name in targets}
    observed_batches = 0
    for batch in batches:
        if not batch:
            continue
        observed_batches += 1
        for group, sample in batch:
            sample_counts[group] = sample_counts.get(group, 0) + 1
            byte_counts[group] = byte_counts.get(group, 0) + sample_bytes(sample)
    drawn = sum(sample_counts.values())
    payload = sum(byte_counts.values())
    share = _share(sample_counts, drawn)
    byte_share = _share(byte_counts, payload)
    missing = sorted(name for name, weight in targets.items()
                     if weight > 0 and sample_counts.get(name, 0) == 0)
    status = {}
    for name, weight in targets.items():
        if name in missing:
            status[name] = "missing"
        elif abs(share[name] - weight) > tolerance + 1e-9:
            status[name] = "off-target"
        else:
            status[name] = "ok"
    # Each multimodal document expands into TWO training rows (caption and
    # media codec), so the row mix is not the document mix. Report the bound.
    estimated_rows = drawn + sample_counts.get("multimodal", 0)
    return {
        "window": window,
        "batches": observed_batches,
        "estimated_rows": estimated_rows,
        "samples": drawn,
        "bytes": payload,
        "targets": dict(targets),
        "samples_by_group": sample_counts,
        "bytes_by_group": byte_counts,
        "sample_share": share,
        "byte_share": byte_share,
        "tolerance": float(tolerance),
        "status": status,
        "quotas_ok": bool(targets) and all(state == "ok" for state in status.values()),
        "complete": bool(targets) and not missing,
        "failures": dict(failures or {}),
    }


def audit_mixture(sampler: MixtureSampler, *, batches: int = 100,
                  batch_size: int = 4, tolerance: float = 0.02,
                  max_seconds: float | None = None
                  ) -> dict[str, Any]:
    """Draw ``batches`` real quota batches from ``sampler`` and audit them.

    ``batches=100, batch_size=4`` = 400 documents = four full quota windows,
    so a healthy mix lands exactly on 50/25/15/10. ``max_seconds`` bounds the
    wall clock on a slow streaming chain; a truncated audit is reported as
    such (``truncated=True``), never silently rounded.
    """
    import time

    started = time.monotonic()
    collected: list[list[tuple[str, dict[str, Any]]]] = []
    truncated = False
    for _ in range(int(batches)):
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            truncated = True
            break
        try:
            collected.append(sampler.next_batch(int(batch_size)))
        except StopIteration:
            truncated = True
            break
    report = audit_batches(collected, targets=sampler.targets, tolerance=tolerance,
                           window=getattr(sampler, "window", None),
                           failures=getattr(sampler, "failures", {}))
    report["batches_requested"] = int(batches)
    report["batch_size"] = int(batch_size)
    report["truncated"] = truncated
    report["elapsed_s"] = round(time.monotonic() - started, 2)
    return report


def format_mixture_report(report: Mapping[str, Any]) -> str:
    """Render the audit as a clean table the user can read at a glance."""
    targets = report.get("targets", {})
    samples = report.get("samples_by_group", {})
    share = report.get("sample_share", {})
    byte_share = report.get("byte_share", {})
    status = report.get("status", {})
    order = sorted(targets, key=lambda name: (-float(targets[name]), name))
    header = (f"PT-BR-first data mix audit  |  {report.get('batches', 0)} batches x "
              f"{report.get('batch_size', 0)} = {report.get('samples', 0)} documents"
              f"  ({report.get('elapsed_s', 0.0)} s)")
    lines = [header]
    lines.append(f"{'source':<12}{'target':>8}{'docs':>8}{'doc %':>9}"
                 f"{'bytes':>12}{'byte %':>9}   status")
    lines.append("-" * len(header))
    for name in order:
        label = GROUP_LABELS.get(name, name)
        lines.append(f"{label:<12}{targets[name] * 100:>7.1f}%{samples.get(name, 0):>8}"
                     f"{share.get(name, 0.0) * 100:>8.1f}%{report.get('bytes_by_group', {}).get(name, 0):>12,}"
                     f"{byte_share.get(name, 0.0) * 100:>8.1f}%   {status.get(name, '?')}")
    lines.append("-" * len(header))
    lines.append(f"{'total':<12}{sum(targets.values()) * 100:>7.1f}%{report.get('samples', 0):>8}"
                 f"{100.0:>8.1f}%{report.get('bytes', 0):>12,}{100.0:>8.1f}%")
    verdict = "PASS" if report.get("quotas_ok") else "FAIL"
    lines.append(f"verdict: {verdict} -- document share within "
                 f"+/-{report.get('tolerance', 0.02) * 100:.1f} pp of target"
                 f" (quota window = {report.get('window')})")
    if not report.get("complete", True):
        lines.append("WARNING: a source chain is dead -- its weight was renormalised "
                     "over the survivors; see failures below.")
    for name, reason in (report.get("failures") or {}).items():
        lines.append(f"  dropped {name}: {reason[:160]}")
    if report.get("truncated"):
        lines.append("NOTE: audit truncated (stream exhausted or max_seconds hit) "
                     "-- shares cover fewer documents than requested.")
    lines.append("note: the weights govern DOCUMENTS; byte share differs because a "
                 "code file is far longer than an image caption.")
    if report.get("estimated_rows"):
        lines.append(f"note: multimodal documents expand into 2 training rows each "
                     f"-> ~{report['estimated_rows']} rows for {report.get('samples', 0)} "
                     f"documents (row mix != document mix).")
    return "\n".join(lines)


def mixture_snapshot(sampler: Any) -> dict[str, Any]:
    """Effective weights + observed counts, for the trainer's run summary."""
    weights = dict(getattr(sampler, "targets", {}) or {})
    counts = dict(getattr(sampler, "counts", {}) or {})
    total = sum(counts.values())
    return {"targets": weights,
            "counts": counts,
            "observed_share": _share(counts, total) if counts else {},
            "failures": dict(getattr(sampler, "failures", {}) or {}),
            "window": getattr(sampler, "window", None)}


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
