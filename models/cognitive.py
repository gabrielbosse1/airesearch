"""Typed, raw-byte canvas operations for the cognitive generation loop.

Edits are staged in a loaded span and are invisible on the canvas until SAVE.
No control token, latent state, or function call is ever serialized as a byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import torch

EditKind = Literal["insert", "delete", "replace", "move"]


@dataclass(frozen=True)
class CognitiveAction:
    """One host-supplied or model-selected control action.

    LOAD uses absolute half-open canvas indices. EDIT uses half-open workspace
    indices; a MOVE destination refers to the workspace *before* removal. SAVE,
    THINK, and DONE take no byte indices. ``data`` is always literal raw bytes.
    """

    token: int
    start: int | None = None
    end: int | None = None
    kind: EditKind | None = None
    data: bytes = b""
    destination: int | None = None
    think_steps: int = 1


@dataclass(frozen=True)
class CognitiveObservation:
    step: int
    canvas: bytes
    workspace: bytes | None
    span: tuple[int, int] | None
    dirty: bool
    scratchpad: torch.Tensor
    confidence: float | None


def action_to_record(action: CognitiveAction) -> dict[str, Any]:
    """Serialize an internal action as primitive metadata, never executable code."""
    return {"token": action.token, "start": action.start, "end": action.end,
            "kind": action.kind, "data_hex": action.data.hex(),
            "destination": action.destination, "think_steps": action.think_steps}


def action_from_record(record: Mapping[str, Any]) -> CognitiveAction:
    """Decode a stored control record; loop validation still applies at replay."""
    if not isinstance(record, Mapping):
        raise TypeError("action record must be a mapping")
    try:
        return CognitiveAction(token=record["token"], start=record.get("start"), end=record.get("end"),
                               kind=record.get("kind"), data=bytes.fromhex(record.get("data_hex", "")),
                               destination=record.get("destination"), think_steps=record.get("think_steps", 1))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid cognitive action record") from exc


CognitiveController = Callable[[CognitiveObservation], CognitiveAction]


@dataclass(frozen=True)
class CognitiveEvent:
    action: CognitiveAction
    canvas_length: int
    workspace_length: int | None
    thought_steps: int = 0
    confidence: float | None = None
    # Raw serialized EDIT arguments, not control IDs or output canvas bytes.
    argument_bytes: bytes | None = None

    @property
    def control_and_arguments(self) -> tuple[int, ...]:
        """Internal control ID followed by literal argument bytes, if emitted.

        This is an event stream, not a raw-byte canvas: control IDs exceed 255.
        Host-supplied typed actions have no generated argument frame by default.
        """
        return (self.action.token, *(() if self.argument_bytes is None else self.argument_bytes))


@dataclass(frozen=True)
class CognitiveResult:
    canvas: bytes
    done: bool
    halt_reason: Literal["done", "max_steps"]
    events: tuple[CognitiveEvent, ...]
    scratchpad: torch.Tensor
    pending_workspace: bytes | None
    pending_span: tuple[int, int] | None


def _index(value: int | None, name: str, limit: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= limit:
        raise ValueError(f"{name} must be an integer in [0, {limit}]")
    return value


def apply_edit(workspace: bytes, action: CognitiveAction, *, max_length: int) -> bytes:
    """Apply a single validated, non-monotonic edit to a byte workspace."""
    if not isinstance(workspace, bytes) or not isinstance(action.data, bytes):
        raise TypeError("workspace and edit data must be bytes")
    n = len(workspace)
    start = _index(action.start, "start", n)
    kind = action.kind
    if kind == "insert":
        edited = workspace[:start] + action.data + workspace[start:]
    elif kind in ("delete", "replace", "move"):
        end = _index(action.end, "end", n)
        if end < start:
            raise ValueError("end must not precede start")
        if kind == "delete":
            edited = workspace[:start] + workspace[end:]
        elif kind == "replace":
            edited = workspace[:start] + action.data + workspace[end:]
        else:
            dest = _index(action.destination, "destination", n)
            if start < dest < end:
                raise ValueError("move destination cannot lie inside the moved span")
            piece = workspace[start:end]
            rest = workspace[:start] + workspace[end:]
            # Destination is in original coordinates; adjust after removal.
            new_dest = dest if dest <= start else dest - (end - start)
            edited = rest[:new_dest] + piece + rest[new_dest:]
    else:
        raise ValueError(f"unknown edit kind: {kind!r}")
    if len(edited) > max_length:
        raise ValueError("edit exceeds the configured maximum canvas length")
    return edited
