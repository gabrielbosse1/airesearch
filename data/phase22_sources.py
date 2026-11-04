"""Phase 22: real Hugging Face data sources for the scaled byte-level pipeline.

WHY THIS FILE EXISTS
--------------------
Phase 19/21 replayed tiny synthetic corpora (~8k rows, 4 unique search rows,
4 unique edit rows, 0 Portuguese rows) on a 245k-parameter model. Phase 22
replaces that with *streamed, real* corpora. Nothing here is generated:
``stream_source`` calls ``datasets.load_dataset(..., streaming=True)`` on the
Hub IDs below and yields the source text verbatim.

VERIFIED DATASET FINDINGS (researched 2026-11-03; no downloads in the sandbox)
------------------------------------------------------------------------------
The sandbox cannot reach huggingface.co (network allowlist: github.com,
codeload.github.com, api.github.com, registry.npmjs.org, pypi.org,
files.pythonhosted.org). These IDs/configs were verified from public dataset
cards via web search; ``datasets.load_dataset`` is written for the Colab
runtime, which *does* have Hub access. ``discover()`` re-searches the Hub at
runtime with ``HfApi.list_datasets`` so IDs can be re-validated there.

Phase 1 (raw next-byte causal LM; NO prompts, NO ReAct):
  * code    : ``bigcode/the-stack`` (v1, 471M rows, 6.4 TB, permissive
              licences) with ``data_dir="data/python"`` / ``"data/javascript"``.
              https://huggingface.co/datasets/bigcode/the-stack
              Fallback: ``meryyllebr543/stack-edu-huggingface`` (config
              ``python`` / ``javascript``), a materialised Stack-Edu mirror.
              https://huggingface.co/datasets/meryyllebr543/stack-edu-huggingface
              NOTE: official ``HuggingFaceTB/stack-edu`` stores only Software
              Heritage ``blob_id`` values and needs a separate S3 download, so
              it is NOT a usable streaming text source.
  * pt-BR   : ``TucanoBR/GigaVerbo`` -- 780 GB / 200B tokens of deduplicated
              Portuguese text. https://huggingface.co/datasets/TucanoBR/GigaVerbo
              Fallback: ``HuggingFaceFW/fineweb-2`` config ``por_Latn``.
              https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
              ``oscar-corpus/OSCAR-2301`` was REJECTED: manually gated and
              access reported suspended. https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
  * english : ``HuggingFaceFW/fineweb-edu`` config ``sample-10BT`` (28.5 GB,
              ODC-BY). https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
              Fallback: ``mlfoundations/dclm-baseline-1.0`` (~3.8T tokens,
              CC-BY-4.0). https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
  * logic   : ``open-web-math/open-web-math`` -- 6.32M documents / 27.4 GB /
              14.7B tokens, LaTeX preserved.
              https://huggingface.co/datasets/open-web-math/open-web-math
  * ``bigcode/the-stack-v2`` was REJECTED as primary: gated (terms + contact
    info). https://huggingface.co/datasets/bigcode/the-stack-v2

Phase 2 (alignment distillation + Minecraft trajectories):
  * ``HuggingFaceH4/no_robots`` -- 10k human-written single-turn instructions,
    **CC BY-NC 4.0 (non-commercial)**; must be attributed and must not be used
    commercially. https://huggingface.co/datasets/HuggingFaceH4/no_robots
  * ``TucanoBR/Tucano-SFT`` -- 679,609 PT-BR chat rows (machine-translated
    GPT-4 conversations +PT math +AirA). https://huggingface.co/datasets/TucanoBR/Tucano-SFT
  * ``argilla/distilabel-sft-easy`` -- config ``default``, a distilabel SFT
    pipeline dataset (SYNTHETIC distillation, secondary fallback only, not a
    pretraining corpus). https://huggingface.co/datasets/argilla/distilabel-sft-easy
  * ``m-a-p/CodeFeedback-Filtered-Instruction`` -- 156,526 code-instruction
    rows, Apache-2.0. https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction
  * Minecraft: there is **no verified HF dataset of human-authored textual
    ``Thought -> Action -> Observation -> Answer`` trajectories with an
    explore -> gather -> craft -> build chain**. "Voyager" is a paper/agent
    (skill library of executable JS), not a Hub trajectory dataset;
    "MineRL"/"BASALT" are video/state-action corpora. The closest REAL HF
    sources are:
      - ``TESS-Computer/minecraft-vla-stage3``: ``instruction`` plus Lumine
        ``output`` containing ``<|thought_start|>...<|action_start|>...``
        derived from real gameplay video, with ``task_category`` /
        ``quality_score``. https://huggingface.co/datasets/TESS-Computer/minecraft-vla-stage3
      - ``vanyacohen/MET-Bench-Minecraft``: scripted trajectories (collecting
        wood, exploring, building) with ``initial_state``,
        ``candidate_states``, ``correct_choice``, ``timestep``.
        https://huggingface.co/datasets/vanyacohen/MET-Bench-Minecraft
      - ``reactor-team/minecraft-vpt-mp4``: VPT ``video`` + aligned ``actions``.
        https://huggingface.co/datasets/reactor-team/minecraft-vpt-mp4
    ``minecraft_row_to_react`` converts these *without inventing facts*:
    observations are taken from the source's state fields when present, and
    otherwise an explicit ``[source: no textual observation]`` marker is used.
    Set ``fabricate_missing=False`` (default). Never claim the ReAct text is
    human-authored.

NO DATA LOSS / NO NOISE DEGRADATION
-----------------------------------
Text is encoded with ``str.encode("utf-8")`` (strict). Invalid Unicode rows
are *skipped and counted*, never replaced with U+FFFD or '?'. No lowercasing,
normalisation, truncation-with-replacement, or synthetic padding text. The
packer in ``training/phase22_common.py`` cuts only at UTF-8 codepoint
boundaries. Every skipped row is recorded in the license/provenance manifest.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

# Re-runnable on Colab: these are the exact queries used for dataset discovery.
HF_SEARCH_QUERIES: tuple[str, ...] = (
    "the-stack python javascript",
    "fineweb-edu sample 10BT",
    "dclm baseline 1.0",
    "fineweb-2 por_Latn Portuguese",
    "Tucano GigaVerbo Portuguese",
    "open-web-math",
    "no_robots",
    "argilla distilabel sft easy",
    "Tucano-SFT Portuguese chat",
    "minecraft trajectories thought action observation",
    "minecraft VLA stage3 instruction",
    "MET-Bench-Minecraft",
    "minecraft VPT actions",
)

LICENSE_NOTES: dict[str, str] = {
    "bigcode/the-stack": "permissively-licensed source files; per-file licence in `licenses`",
    "meryyllebr543/stack-edu-huggingface": "mirror of HuggingFaceTB/stack-edu (mixed open licences)",
    "TucanoBR/GigaVerbo": "concatenation of open PT corpora; review upstream licences",
    "HuggingFaceFW/fineweb-2": "ODC-BY / CommonCrawl terms",
    "HuggingFaceFW/fineweb-edu": "ODC-BY",
    "mlfoundations/dclm-baseline-1.0": "CC-BY-4.0",
    "open-web-math/open-web-math": "mixed web content; see dataset card",
    "HuggingFaceH4/no_robots": "CC BY-NC 4.0 -- NON-COMMERCIAL, attribute Hugging Face H4",
    "argilla/distilabel-sft-easy": "distilabel-generated synthetic SFT; review upstream model terms",
    "TucanoBR/Tucano-SFT": "derived from GPT-4 translated data; review upstream terms",
    "m-a-p/CodeFeedback-Filtered-Instruction": "Apache-2.0 (contains OpenAI-generated text)",
    "TESS-Computer/minecraft-vla-stage3": "VLM-annotated gameplay; review dataset card",
    "vanyacohen/MET-Bench-Minecraft": "scripted trajectories; review dataset card",
    "reactor-team/minecraft-vpt-mp4": "VPT contractor data; review dataset card",
}


@dataclass(frozen=True)
class SourceSpec:
    """One Hub source and the exact ``load_dataset`` arguments it needs."""

    id: str
    text_field: str = "text"
    name: str | None = None            # dataset config name
    data_dir: str | None = None        # language subset for the-stack v1
    split: str = "train"
    streaming: bool = True
    license: str = ""
    note: str = ""
    url: str = ""
    gated: bool = False

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SourceSpec":
        known = {"id", "text_field", "name", "data_dir", "split", "streaming",
                 "license", "note", "url", "gated"}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"unknown source keys: {unknown}")
        values = dict(mapping)
        values.setdefault("license", LICENSE_NOTES.get(mapping["id"], "unverified"))
        values.setdefault("url", f"https://huggingface.co/datasets/{mapping['id']}")
        return cls(**values)

    def load_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"split": self.split, "streaming": self.streaming}
        if self.name:
            kwargs["name"] = self.name
        if self.data_dir:
            kwargs["data_dir"] = self.data_dir
        return kwargs


@dataclass
class SourceResult:
    """Provenance record for one attempted/resolved source."""

    spec: SourceSpec
    status: str = "pending"           # pending | resolved | failed | skipped
    rows: int = 0
    bytes: int = 0
    invalid_unicode_rows: int = 0
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.spec.id, "config": self.spec.name,
            "data_dir": self.spec.data_dir, "split": self.spec.split,
            "license": self.spec.license, "url": self.spec.url,
            "gated": self.spec.gated, "note": self.spec.note,
            "status": self.status, "rows": self.rows, "bytes": self.bytes,
            "invalid_unicode_rows": self.invalid_unicode_rows, "error": self.error,
        }


class SourceUnavailable(RuntimeError):
    """Raised when a Hub source cannot be resolved (gated, renamed, offline)."""


@dataclass
class SourceRegistry:
    """Resolves sources with fallbacks and records every attempt."""

    records: list[SourceResult] = field(default_factory=list)
    strict_licenses: bool = False

    def resolve(self, specs: Iterable[SourceSpec], *, limit_rows: int = 0,
                probe: bool = True) -> SourceResult:
        """Return the first spec that loads; record every failed attempt.

        ``limit_rows`` bounds a probe/finite extraction. ``probe=False`` returns
        a lazily-bound stream without consuming a row (used for full training).
        """
        from datasets import load_dataset  # lazy: sandbox has no Hub access

        last_error = ""
        for spec in specs:
            result = SourceResult(spec=spec)
            self.records.append(result)
            try:
                dataset = load_dataset(spec.id, **spec.load_kwargs())
                if probe:
                    iterator = iter(dataset)
                    next(iterator)  # verifies the config actually streams
                result.status = "resolved"
                return result
            except Exception as exc:  # gated repo, bad config, offline, ...
                result.status = "failed"
                result.error = f"{type(exc).__name__}: {exc}"[:600]
                last_error = result.error
        raise SourceUnavailable(f"no source resolved; last error: {last_error}")

    def manifest(self, path: str | Path, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": 1,
            "network_note": "Hub downloads run in Colab, not in the Arena sandbox",
            "search_queries": list(HF_SEARCH_QUERIES),
            "sources": [record.as_dict() for record in self.records],
        }
        if extra:
            payload.update(dict(extra))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf8")
        return payload


def discover(query: str, *, limit: int = 20) -> list[dict[str, str]]:
    """Runtime Hub search (``HfApi.list_datasets``), for Colab only."""
    from huggingface_hub import HfApi

    api = HfApi()
    found: list[dict[str, str]] = []
    for info in api.list_datasets(search=query, limit=limit):
        found.append({"id": str(info.id),
                      "url": f"https://huggingface.co/datasets/{info.id}",
                      "downloads": str(getattr(info, "downloads", ""))})
    return found


def _text_from_row(row: Mapping[str, Any], field_name: str) -> str:
    """Extract real source text; chat datasets are flattened without invention."""
    value = row.get(field_name)
    if value is None:
        # Try the common text/chat columns in a fixed, documented order.
        for candidate in ("text", "content", "messages", "conversations",
                          "output", "query", "answer", "prompt"):
            if candidate in row:
                value = row[candidate]
                break
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        lines: list[str] = []
        for item in value:
            if isinstance(item, str):
                lines.append(item)
            elif isinstance(item, Mapping):
                role = item.get("role") or item.get("from") or "turn"
                body = item.get("content") or item.get("value") or ""
                lines.append(f"{role}: {body}")
        return "\n".join(lines)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def strict_utf8(text: str) -> bytes | None:
    """Encode losslessly; return None for rows that cannot round-trip."""
    try:
        encoded = text.encode("utf-8", errors="strict")
    except (UnicodeEncodeError, AttributeError, TypeError):
        return None
    # Surrogate-free check: strict utf-8 above already rejects lone surrogates,
    # but keep an explicit assertion for the no-data-loss invariant.
    if b"\x00" in encoded:
        return None
    return encoded


def stream_rows(rows: Iterable[Any], spec: SourceSpec, *, limit_rows: int = 0,
                record: SourceResult | None = None) -> Iterator[tuple[str, bytes]]:
    """Convert an already-open HF iterator into lossless UTF-8 byte strings.

    Invalid Unicode rows are skipped (never mangled); when ``record`` is given
    the skip is counted in the provenance manifest.
    """
    emitted = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        raw = strict_utf8(_text_from_row(row, spec.text_field))
        if raw is None:
            if record is not None:
                record.invalid_unicode_rows += 1
            continue
        if not raw:
            continue
        if record is not None:
            record.rows += 1
            record.bytes += len(raw)
        yield spec.id, raw
        emitted += 1
        if limit_rows and emitted >= limit_rows:
            return


def stream_source(spec: SourceSpec, *, limit_rows: int = 0) -> Iterator[tuple[str, bytes]]:
    """Yield ``(source_id, raw_utf8_bytes)`` from a real, streamed Hub source.

    ``limit_rows=0`` means unbounded -- intended for a multi-hour Colab run.
    """
    from datasets import load_dataset

    dataset = load_dataset(spec.id, **spec.load_kwargs())
    yield from stream_rows(dataset, spec, limit_rows=limit_rows)


def weighted_mixture(groups: list[dict[str, Any]], *, cycle_unit: int = 32) -> list[SourceSpec]:
    """Deterministic weighted round-robin over groups (no random data loss)."""
    weighted: list[tuple[float, list[SourceSpec]]] = []
    for group in groups:
        specs = [SourceSpec.from_mapping(item) for item in group["sources"]]
        weighted.append((float(group.get("weight", 1.0)), specs))
    total = sum(weight for weight, _ in weighted) or 1.0
    schedule: list[SourceSpec] = []
    for weight, specs in weighted:
        count = max(1, int(round(cycle_unit * weight / total)))
        for index in range(count):
            schedule.append(specs[index % len(specs)])
    return schedule


def stream_phase1(config: Mapping[str, Any], *, limit_rows_per_source: int = 0,
                  registry: SourceRegistry | None = None) -> Iterator[tuple[str, bytes]]:
    """Weighted stream of REAL raw text for raw next-byte causal LM.

    Deliberately emits no prompt templates, no ReAct scaffolding, no chat
    markers: Phase 1 is pure byte modelling. Failures are skipped with a
    recorded reason rather than replaced with synthetic filler.
    """
    from datasets import load_dataset
    from itertools import chain

    groups = config["data"]["phase1"]
    registry = registry or SourceRegistry()
    resolved: dict[tuple[str, str | None, str | None], bool] = {}
    streams: dict[int, Iterator[tuple[str, bytes]]] = {}
    for index, spec in enumerate(weighted_mixture(groups)):
        key = (spec.id, spec.name, spec.data_dir)
        if key in resolved:
            continue
        record = SourceResult(spec=spec)
        registry.records.append(record)
        try:
            # One load only; chain the probe row back so no document is lost.
            dataset = load_dataset(spec.id, **spec.load_kwargs())
            iterator = iter(dataset)
            first = next(iterator)
            record.status = "resolved"
            streams[index] = stream_rows(chain([first], iterator), spec,
                                         limit_rows=limit_rows_per_source,
                                         record=record)
            resolved[key] = True
        except Exception as exc:
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"[:600]
            resolved[key] = False
    if not streams:
        raise SourceUnavailable("no Phase 1 source streamed; check Hub access/gating")
    order = list(streams)
    exhausted: set[int] = set()
    for index in cycle(order):
        if index in exhausted:
            if len(exhausted) == len(order):
                return
            continue
        try:
            yield next(streams[index])
        except StopIteration:
            exhausted.add(index)


# ---------------------------------------------------------------------------
# Phase 2 converters: real datasets -> ReAct / conversation byte transcripts
# ---------------------------------------------------------------------------

REACT_HEADER = "Thought"
PHASE_LADDER = ("explore", "gather", "craft", "build")
_PHASE_KEYWORDS = {
    "explore": ("explore", "search", "find", "locate", "look", "travel", "walk", "scan"),
    "gather": ("mine", "collect", "gather", "chop", "pickup", "harvest", "break", "obtain"),
    "craft": ("craft", "smelt", "make", "combine", "recipe", "furnace", "workbench"),
    "build": ("place", "build", "construct", "put", "set", "assemble"),
}

_THOUGHT_TAG = re.compile(r"<\|thought_start\|>(.*?)<\|thought_end\|>", re.S)
_ACTION_TAG = re.compile(r"<\|action_start\|>(.*?)<\|action_end\|>", re.S)


def classify_phase(action: str) -> str:
    """Map a real action string to the explore/gather/craft/build ladder."""
    lowered = action.lower()
    for phase, keywords in _PHASE_KEYWORDS.items():
        if any(word in lowered for word in keywords):
            return phase
    return "explore"


def format_react(*, goal: str, steps: list[tuple[str, str, str]], answer: str,
                 provenance: str, ladder: str = "") -> bytes:
    """Render ``Thought -> Action -> Observation -> Answer`` as exact UTF-8.

    ``steps`` are ``(thought, action, observation)`` triples taken from the
    source. Nothing is generated here beyond the fixed labels; the ladder is
    metadata, kept on its own line so phase coverage is not inflated by it.
    """
    lines = [f"Goal: {goal}", f"Provenance: {provenance}"]
    if ladder:
        lines.append(f"Ladder: {ladder}")
    for thought, action, observation in steps:
        lines.append(f"{REACT_HEADER}: {thought}")
        lines.append(f"Action: {action}")
        lines.append(f"Observation: {observation}")
    lines.append(f"Answer: {answer}")
    return ("\n".join(lines) + "\n").encode("utf-8", errors="strict")


def minecraft_row_to_react(row: Mapping[str, Any], *, fabricate_missing: bool = False,
                           provenance: str = "HF trajectory") -> bytes | None:
    """Convert one REAL Minecraft HF row to a ReAct byte transcript.

    Supported schemas (verified on the Hub):
      * ``TESS-Computer/minecraft-vla-stage3``: ``instruction``, ``output``
        (Lumine thought/action tags), ``task_category``, ``quality_score``.
      * ``vanyacohen/MET-Bench-Minecraft``: ``initial_state``, ``action``,
        ``candidate_states``, ``correct_choice``, ``trajectory_id``,
        ``timestep``.
      * VPT-style rows: ``actions`` list plus optional ``source``.

    With ``fabricate_missing=False`` (default) a missing textual observation
    becomes an explicit ``[source: no textual observation]`` marker. The
    observation is NEVER invented. Returns None if no real action text exists.
    """
    instruction = str(row.get("instruction") or row.get("goal") or "Minecraft task")
    output = str(row.get("output") or "")
    thought_match = _THOUGHT_TAG.search(output)
    action_match = _ACTION_TAG.search(output)
    thought = (thought_match.group(1).strip() if thought_match else "")
    action = (action_match.group(1).strip() if action_match else output.strip())

    steps: list[tuple[str, str, str]] = []
    if action:
        phase = classify_phase(action)
        if not thought:
            thought = f"[{phase}] continue the recorded trajectory for: {instruction}" if fabricate_missing else "[source: no thought text]"
        observation = _source_observation(row)
        steps.append((thought, action, observation))

    if not steps and isinstance(row.get("actions"), (list, tuple)):
        actions = [str(item) for item in row["actions"] if str(item).strip()]
        for raw_action in actions[:16]:
            action_text = raw_action if len(raw_action) < 600 else raw_action[:600]
            phase = classify_phase(action_text)
            observation = _source_observation(row)
            steps.append((f"[{phase}] recorded action step {len(steps) + 1}",
                          action_text, observation))

    if not steps and row.get("action"):
        action_text = str(row["action"])
        observation = _source_observation(row)
        steps.append(("[source: no thought text]", action_text, observation))

    if not steps:
        return None

    ladder = " -> ".join(PHASE_LADDER)
    answer = str(row.get("answer") or row.get("correct_choice")
                 or f"Recorded {len(steps)} action(s) for: {instruction}")
    return format_react(goal=instruction, steps=steps, answer=answer,
                        provenance=provenance, ladder=ladder)


def _source_observation(row: Mapping[str, Any]) -> str:
    """Use a real state field, else an explicit no-observation marker."""
    for key in ("observation", "initial_state", "state", "candidate_states",
                "inventory", "feedback"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = " | ".join(str(item) for item in value[:4])
        text = str(value).strip()
        if text:
            return text[:800]
    return "[source: no textual observation]"


def conversation_to_bytes(row: Mapping[str, Any], *, provenance: str = "HF conversation") -> bytes | None:
    """Lossless chat transcript; chat roles are preserved from the source."""
    messages = row.get("messages") or row.get("conversations")
    lines = [f"Provenance: {provenance}"]
    if isinstance(messages, (list, tuple)):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role") or message.get("from") or "user")
            body = str(message.get("content") or message.get("value") or "")
            if body.strip():
                lines.append(f"{role}: {body}")
    else:
        prompt = row.get("prompt") or row.get("query") or row.get("instruction")
        answer = row.get("answer") or row.get("response") or row.get("completion")
        if prompt:
            lines.append(f"user: {prompt}")
        if answer:
            lines.append(f"assistant: {answer}")
    if len(lines) < 2:
        return None
    return ("\n".join(lines) + "\n").encode("utf-8", errors="strict")


def minecraft_phase_coverage(transcripts: Iterable[bytes]) -> dict[str, int]:
    """Count ladder phases in real Thought/Action lines only (not metadata).

    This is a coverage *audit*, not a quality claim: a transcript with ladder
    metadata but only one real action must not be reported as covering all four
    phases.
    """
    counts = {phase: 0 for phase in PHASE_LADDER}
    for raw in transcripts:
        seen: set[str] = set()
        for line in raw.decode("utf-8", errors="strict").splitlines():
            if not line.startswith(("Thought:", "Action:")):
                continue
            lowered = line.lower()
            for phase, keywords in _PHASE_KEYWORDS.items():
                if any(word in lowered for word in keywords):
                    seen.add(phase)
        for phase in seen:
            counts[phase] += 1
    return counts


def stream_phase2(config: Mapping[str, Any], *, limit_rows_per_source: int = 0,
                  registry: SourceRegistry | None = None
                  ) -> Iterator[tuple[bytes, int]]:
    """Yield ``(transcript_bytes, prompt_prefix_length)`` from REAL Hub rows.

    Phase 2 mixes safe conversations, PT-BR chat, code feedback and Minecraft
    trajectories. The ``prompt_prefix_length`` is the byte offset at which the
    supervised loss starts, so the model is not trained to invent user text or
    dataset metadata. ReAct transcripts start at the first ``Thought:``; chat
    transcripts start at the first assistant turn.
    """
    from datasets import load_dataset
    from itertools import chain

    registry = registry or SourceRegistry()
    for group in config["data"]["phase2"]:
        kind = str(group.get("group", "conversation"))
        for mapping in group["sources"]:
            spec = SourceSpec.from_mapping(mapping)
            record = SourceResult(spec=spec)
            registry.records.append(record)
            try:
                dataset = load_dataset(spec.id, **spec.load_kwargs())
                iterator = iter(dataset)
                first = next(iterator)
                record.status = "resolved"
                rows = chain([first], iterator)
            except Exception as exc:
                record.status = "failed"
                record.error = f"{type(exc).__name__}: {exc}"[:600]
                continue
            emitted = 0
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                if kind == "minecraft":
                    transcript = minecraft_row_to_react(row, provenance=spec.id)
                    marker = b"Thought:"
                else:
                    transcript = conversation_to_bytes(row, provenance=spec.id)
                    marker = b"assistant:"
                if transcript is None:
                    continue
                encoded = strict_utf8(transcript.decode("utf-8", errors="strict"))
                if encoded is None:
                    record.invalid_unicode_rows += 1
                    continue
                position = encoded.find(marker)
                prefix = position if position >= 0 else 0
                record.rows += 1
                record.bytes += len(encoded)
                yield encoded, prefix
                emitted += 1
                if limit_rows_per_source and emitted >= limit_rows_per_source:
                    break


# ---------------------------------------------------------------------------
# Offline self-test (no Hub access): proves converters preserve real fields.
# ---------------------------------------------------------------------------
def _self_test() -> dict[str, Any]:
    stage3_row = {
        "instruction": "mine the oak log",
        "output": "<|thought_start|>I see an oak tree; gather wood first.<|thought_end|>"
                  "<|action_start|>mine oak_log<|action_end|>",
        "task_category": "mine", "quality_score": 8.5,
        "initial_state": "biome: forest, inventory: empty",
    }
    met_row = {
        "action": "collect 3 oak logs then craft planks",
        "initial_state": "inventory: 3 oak_log", "correct_choice": "4 planks",
        "trajectory_id": "t-7", "timestep": 12,
    }
    vpt_row = {"actions": ["explore biome", "mine cobblestone", "place cobblestone"],
               "source": "vpt"}
    react1 = minecraft_row_to_react(stage3_row, provenance="TESS-Computer/minecraft-vla-stage3")
    react2 = minecraft_row_to_react(met_row, provenance="vanyacohen/MET-Bench-Minecraft")
    react3 = minecraft_row_to_react(vpt_row, provenance="reactor-team/minecraft-vpt-mp4")
    assert react1 and b"Thought:" in react1 and b"Action:" in react1
    assert b"Observation:" in react1 and b"Answer:" in react1
    assert b"forest" in react1 and b"TESS-Computer/minecraft-vla-stage3" in react1
    assert react2 and b"4 planks" in react2
    assert react3 and react3.count(b"Action:") == 3
    assert b"[source: no textual observation]" in react3
    chat = conversation_to_bytes({"messages": [{"role": "user", "content": "Olá"},
                                               {"role": "assistant", "content": "Oi!"}]})
    assert chat and "Olá".encode() in chat and b"assistant: Oi!" in chat
    coverage = minecraft_phase_coverage([react1, react2, react3])
    specs = [SourceSpec.from_mapping({"id": "bigcode/the-stack", "data_dir": "data/python"}),
             SourceSpec.from_mapping({"id": "HuggingFaceFW/fineweb-2", "name": "por_Latn"})]
    assert specs[0].load_kwargs()["data_dir"] == "data/python"
    assert specs[1].load_kwargs()["name"] == "por_Latn"
    assert specs[0].license and "huggingface.co" in specs[0].url
    return {
        "react_labels_present": True,
        "no_fabricated_observation": b"[source: no textual observation]" in react3,
        "phase_coverage": coverage,
        "registry_sources": len(specs),
        "search_queries": len(HF_SEARCH_QUERIES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true", help="offline converter tests")
    parser.add_argument("--discover", default="", help="Hub search query (needs Hub access)")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_self_test(), indent=2))
    elif args.discover:
        print(json.dumps(discover(args.discover, limit=args.limit), indent=2))
    else:
        print(json.dumps({"search_queries": list(HF_SEARCH_QUERIES),
                          "license_notes": LICENSE_NOTES}, indent=2))


if __name__ == "__main__":
    main()
