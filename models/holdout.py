"""Exact raw-byte holdout guard for accidental memory contamination.

The guard protects registered byte inputs, outputs and length-framed pairs. It
can also recognize a registered byte string embedded in a larger raw field.
Digests are stable across processes, but cannot identify paraphrases, shifted
excerpts, unknown sources, or data concealed inside opaque embeddings. All
provenance is supplied by the caller; this is an audit aid, not a security
boundary or a proof that model weights have never seen held-out data.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any


class HoldoutViolation(ValueError):
    """A write or checkpoint cannot be proved clean under the current guard."""


def _frame(data: bytes) -> bytes:
    return len(data).to_bytes(8, "big") + data


def content_hash(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("holdout content must be bytes")
    return hashlib.sha256(b"airesearch.content.v1\0" + _frame(data)).hexdigest()


def pair_hash(source: bytes, target: bytes) -> str:
    if not isinstance(source, bytes) or not isinstance(target, bytes):
        raise TypeError("holdout pair components must be bytes")
    return hashlib.sha256(b"airesearch.pair.v1\0" + _frame(source) + _frame(target)).hexdigest()


def reject_heldout_split(value: Any, *, depth: int = 0) -> None:
    """Never persist explicitly labelled held-out records, even unguarded."""
    if depth > 32:
        raise HoldoutViolation("metadata nesting is too deep to audit")
    if isinstance(value, Mapping):
        split = value.get("split")
        if isinstance(split, str) and split.lower() in {"test", "validation", "val", "heldout", "hold-out", "holdout"}:
            raise HoldoutViolation("metadata identifies a held-out split")
        for child in value.values():
            reject_heldout_split(child, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for child in value:
            reject_heldout_split(child, depth=depth + 1)


class HoldoutGuard:
    """Immutable-style registry of held-out content and pair hashes.

    The full input and output are checked in *either* role, conservatively;
    common identical targets can therefore reject valid training examples.
    Protected substrings of at least ``substring_min_bytes`` are also checked
    in supplied raw fields. Empty byte strings are exact-match-only.
    """
    PROVENANCE_KEY = "_holdout_provenance"

    def __init__(self, contents: Iterable[tuple[int, str]], pairs: Iterable[str], *,
                 substring_min_bytes: int = 16) -> None:
        if substring_min_bytes < 1:
            raise ValueError("substring_min_bytes must be positive")
        self.substring_min_bytes = int(substring_min_bytes)
        self.contents = frozenset((int(n), str(h)) for n, h in contents)
        self.held_out_hashes = frozenset(str(h) for h in pairs)
        if any(n < 0 or not self._valid_hash(h) for n, h in self.contents) or any(
            not self._valid_hash(h) for h in self.held_out_hashes
        ):
            raise ValueError("invalid holdout hash registry")
        by_length: dict[int, set[str]] = {}
        for n, digest in self.contents:
            by_length.setdefault(n, set()).add(digest)
        self._by_length = by_length
        self.fingerprint = hashlib.sha256(json.dumps(self.to_state(), sort_keys=True,
                                                     separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _valid_hash(value: str) -> bool:
        return len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]], *, substring_min_bytes: int = 16) -> HoldoutGuard:
        contents: set[tuple[int, str]] = set()
        pairs: set[str] = set()
        for record in records:
            source, target = record["input"], record.get("output")
            if not isinstance(source, bytes) or (target is not None and not isinstance(target, bytes)):
                raise TypeError("holdout inputs and outputs must be bytes")
            contents.add((len(source), content_hash(source)))
            if target is not None:
                contents.add((len(target), content_hash(target)))
                pairs.add(pair_hash(source, target))
        return cls(contents, pairs, substring_min_bytes=substring_min_bytes)

    def to_state(self) -> dict[str, Any]:
        return {"version": 1, "contents": [[n, h] for n, h in sorted(self.contents)],
                "pairs": sorted(self.held_out_hashes), "substring_min_bytes": self.substring_min_bytes}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> HoldoutGuard:
        if state.get("version") != 1:
            raise ValueError("unsupported holdout guard format")
        return cls(((int(x[0]), str(x[1])) for x in state["contents"]), state["pairs"],
                   substring_min_bytes=int(state["substring_min_bytes"]))

    def assert_compatible(self, other: HoldoutGuard) -> None:
        if self.fingerprint != other.fingerprint:
            raise HoldoutViolation("checkpoint and active holdout registries differ; rebuild from trusted source")

    def _check_bytes(self, data: bytes) -> None:
        if len(data) > 1_000_000:
            raise HoldoutViolation("raw field exceeds the guard scan limit; split it into auditable records")
        hashes = self._by_length.get(len(data), ())
        if hashes and content_hash(data) in hashes:
            raise HoldoutViolation("protected held-out bytes cannot be written to memory")
        for length, digests in self._by_length.items():
            if length < self.substring_min_bytes or length >= len(data):
                continue
            for start in range(len(data) - length + 1):
                if content_hash(data[start:start + length]) in digests:
                    raise HoldoutViolation("raw field contains a protected held-out byte sequence")

    def _check_metadata(self, value: Any, *, depth: int = 0) -> None:
        if depth > 32:
            raise HoldoutViolation("metadata nesting is too deep to audit")
        if isinstance(value, bytes):
            if value:
                self._check_bytes(value)
        elif isinstance(value, bytearray):
            if value:
                self._check_bytes(bytes(value))
        elif isinstance(value, str):
            if value:
                self._check_bytes(value.encode("utf-8"))
        elif isinstance(value, Mapping):
            split = value.get("split")
            if isinstance(split, str) and split.lower() in {"test", "validation", "val", "heldout", "hold-out", "holdout"}:
                raise HoldoutViolation("metadata identifies a held-out split")
            for key, child in value.items():
                if key == self.PROVENANCE_KEY:
                    continue
                if not isinstance(key, str):
                    raise HoldoutViolation("guarded metadata keys must be strings")
                self._check_metadata(child, depth=depth + 1)
                if key == "trace_actions" and isinstance(child, (list, tuple)):
                    # One-byte supervised edits may assemble a protected value
                    # across successive action records. Audit each payload run.
                    payload = bytearray()
                    for action in child:
                        encoded = action.get("data_hex") if isinstance(action, Mapping) else None
                        if isinstance(encoded, str) and encoded:
                            try:
                                payload.extend(bytes.fromhex(encoded))
                            except ValueError:
                                raise HoldoutViolation("invalid trace action hex") from None
                        elif payload:
                            self._check_bytes(bytes(payload))
                            payload.clear()
                    if payload:
                        self._check_bytes(bytes(payload))
                if key.endswith("_hex") and isinstance(child, str):
                    try:
                        decoded = bytes.fromhex(child)
                    except ValueError:
                        continue
                    if decoded:
                        self._check_bytes(decoded)
        elif isinstance(value, (list, tuple)):
            if value and all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 255 for x in value):
                self._check_bytes(bytes(value))
            for child in value:
                self._check_metadata(child, depth=depth + 1)
        elif value is not None and not isinstance(value, (bool, int, float)):
            raise HoldoutViolation("guarded metadata contains an opaque unsupported value")

    def validate(self, source: bytes | None, target: bytes | None = None, *,
                 metadata: Mapping[str, Any] | None = None, extra: Iterable[bytes] = ()) -> None:
        if source is None:
            raise HoldoutViolation("guarded writes require raw input provenance; opaque embeddings are not auditable")
        if not isinstance(source, bytes) or (target is not None and not isinstance(target, bytes)):
            raise TypeError("guarded source and target must be bytes")
        self._check_bytes(source)
        if target is not None:
            self._check_bytes(target)
            if pair_hash(source, target) in self.held_out_hashes:
                raise HoldoutViolation("protected held-out pair cannot be written to memory")
        for value in extra:
            if not isinstance(value, bytes):
                raise TypeError("guarded additional content must be bytes")
            if value:
                self._check_bytes(value)
        if metadata is not None:
            self._check_metadata(metadata)

    def authorize(self, source: bytes | None, target: bytes | None = None, *,
                  metadata: Mapping[str, Any] | None = None, extra: Iterable[bytes] = ()) -> dict[str, Any]:
        self.validate(source, target, metadata=metadata, extra=extra)
        assert source is not None
        result = dict(metadata or {})
        result[self.PROVENANCE_KEY] = {
            "policy": self.fingerprint, "input_hash": content_hash(source),
            "input_length": len(source), "output_hash": None if target is None else content_hash(target),
            "output_length": None if target is None else len(target),
            "pair_hash": None if target is None else pair_hash(source, target),
        }
        return result

    def validate_provenance(self, metadata: Mapping[str, Any]) -> None:
        """Fail closed for old/unverifiable provenance or a changed registry."""
        self._check_metadata(metadata)
        p = metadata.get(self.PROVENANCE_KEY)
        if not isinstance(p, Mapping) or p.get("policy") != self.fingerprint:
            raise HoldoutViolation("missing or incompatible write provenance")
        ih, il = p.get("input_hash"), p.get("input_length")
        oh, ol, ph = p.get("output_hash"), p.get("output_length"), p.get("pair_hash")
        if not isinstance(il, int) or il < 0 or not isinstance(ih, str) or not self._valid_hash(ih):
            raise HoldoutViolation("invalid input provenance")
        if (il, ih) in self.contents:
            raise HoldoutViolation("input provenance matches held-out content")
        if oh is not None:
            if not isinstance(ol, int) or ol < 0 or not isinstance(oh, str) or not self._valid_hash(oh):
                raise HoldoutViolation("invalid output provenance")
            if (ol, oh) in self.contents or ph in self.held_out_hashes:
                raise HoldoutViolation("output provenance matches held-out content")
            if not isinstance(ph, str) or not self._valid_hash(ph):
                raise HoldoutViolation("invalid pair provenance")
        elif ol is not None or ph is not None:
            raise HoldoutViolation("incomplete output provenance")
        for exemplar in metadata.get("exemplars", []):
            if not isinstance(exemplar, Mapping):
                raise HoldoutViolation("opaque semantic exemplar provenance")
            self.validate_provenance(exemplar)

    def audit_checkpoint(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Audit supported checkpoint payloads without mutating a live model."""
        tensors = state.get("module_state")
        if not isinstance(tensors, Mapping):
            raise HoldoutViolation("legacy/opaque checkpoints cannot be audited")
        checked = 0
        failures: list[str] = []
        for tier in ("episodic", "semantic", "procedural"):
            used = tensors.get(f"{tier}.used")
            metadata = state.get(f"{tier}_metadata")
            if used is None or not isinstance(metadata, (list, tuple)):
                raise HoldoutViolation(f"checkpoint lacks {tier} audit metadata")
            for index in used.nonzero(as_tuple=False).flatten().tolist():
                checked += 1
                try:
                    md = metadata[index]
                    if not isinstance(md, Mapping):
                        raise HoldoutViolation("invalid slot metadata")
                    self.validate_provenance(md)
                    if tier == "procedural" and bool(tensors["procedural.structured"][index]):
                        def unpack(name: str) -> bytes:
                            length = int(tensors[f"procedural.{name}_lens"][index])
                            raw = tensors[f"procedural.{name}_bytes"][index]
                            if length < 0 or length > len(raw):
                                raise HoldoutViolation("invalid stored byte length")
                            return bytes(raw[:length].cpu().tolist())
                        source, target, explanation = unpack("input"), unpack("output"), unpack("explanation")
                        self.validate(source, target, metadata=md, extra=(explanation,))
                        p = md[self.PROVENANCE_KEY]
                        if p["input_hash"] != content_hash(source) or p["output_hash"] != content_hash(target) or p["pair_hash"] != pair_hash(source, target):
                            raise HoldoutViolation("stored bytes differ from provenance")
                except (HoldoutViolation, IndexError, KeyError, TypeError, ValueError) as exc:
                    failures.append(f"{tier}:{index}: {exc}")
        return {"checked_slots": checked, "failed_slots": len(failures), "failures": failures,
                "policy_fingerprint": self.fingerprint}
