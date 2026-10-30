"""Canonical raw-byte argument frames for cognitive EDIT actions.

The control token itself remains the separate internal ID 262. It is followed
by at least three literal bytes: ``position_low, opcode_flags, value``. For
positions 0..255 and a single-byte insert/replace/delete, opcodes are exactly
0/1/2. MOVE (opcode 3) additionally encodes its destination, since three bytes
cannot describe both an arbitrary span and a destination. Extended position,
span length and multi-byte payloads use canonical unsigned LEB128 extensions.
Coordinates are relative to the currently loaded workspace, as in apply_edit.
"""
from __future__ import annotations

from .cognitive import CognitiveAction
from .procedural import raw_bytes

EDIT_TOKEN = 262
KINDS = ("insert", "delete", "replace", "move")
_POSITION_EXT = 0x80
_RANGE_EXT = 0x40
_PAYLOAD_EXT = 0x20
_ALLOWED_FLAGS = _POSITION_EXT | _RANGE_EXT | _PAYLOAD_EXT | 0x03
_MAX_INTEGER = (1 << 63) - 1


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= _MAX_INTEGER:
        raise ValueError(f"{name} must be a non-negative integer within 63 bits")
    return value


def _encode_uvarint(value: int) -> bytes:
    value = _integer(value, "varint")
    out = bytearray()
    while value >= 128:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _decode_uvarint(data: bytes, offset: int) -> tuple[int, int]:
    start, value, shift = offset, 0, 0
    for _ in range(10):
        if offset >= len(data):
            raise ValueError("truncated edit argument extension")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if value > _MAX_INTEGER or data[start:offset] != _encode_uvarint(value):
                raise ValueError("non-canonical or oversized edit argument extension")
            return value, offset
        shift += 7
    raise ValueError("oversized edit argument extension")


def encode_edit_arguments(action: CognitiveAction) -> bytes:
    """Encode a typed EDIT action without ever serializing a control as data.

    Unused sampled fields (e.g. a destination on REPLACE) are intentionally
    not encoded. A payload of exactly one byte and span of one byte have no
    extensions, yielding ``bytes([pos, op, value])`` for positions <= 255.
    """
    if not isinstance(action, CognitiveAction) or action.token != EDIT_TOKEN:
        raise ValueError("expected a cognitive EDIT_OP action")
    if action.kind not in KINDS:
        raise ValueError("unknown edit kind")
    if not isinstance(action.data, bytes):
        raise TypeError("edit payload must be raw bytes")
    start = _integer(action.start, "start")
    kind = action.kind
    flags = KINDS.index(kind)
    extensions = bytearray()
    if start > 255:
        flags |= _POSITION_EXT
        extensions.extend(_encode_uvarint(start >> 8))
    if kind != "insert":
        end = _integer(action.end, "end")
        if end < start:
            raise ValueError("end must not precede start")
        length = end - start
        if length != 1:
            flags |= _RANGE_EXT
            extensions.extend(_encode_uvarint(length))
    if kind in ("insert", "replace"):
        payload = action.data
        first = payload[0] if payload else 0
        if len(payload) != 1:
            flags |= _PAYLOAD_EXT
            extensions.extend(_encode_uvarint(len(payload)))
            extensions.extend(payload[1:])
    else:
        if action.data:
            raise ValueError("delete and move cannot carry payload bytes")
        first = 0
    if kind == "move":
        destination = _integer(action.destination, "destination")
        if start < destination < end:
            raise ValueError("move destination cannot lie inside its source span")
        extensions.extend(_encode_uvarint(destination))
    return bytes((start & 0xFF, flags, first)) + bytes(extensions)


def decode_edit_arguments(data: bytes | bytearray, *, max_position: int = 4096,
                          max_payload: int = 4096) -> CognitiveAction:
    """Parse and validate one complete canonical EDIT argument frame."""
    frame = raw_bytes(data, "edit argument bytes")
    if len(frame) < 3:
        raise ValueError("EDIT_OP requires position, opcode and byte")
    max_position = _integer(max_position, "max_position")
    max_payload = _integer(max_payload, "max_payload")
    start, flags, first = frame[:3]
    if flags & ~_ALLOWED_FLAGS:
        raise ValueError("unknown edit argument flags")
    kind = KINDS[flags & 0x03]
    offset = 3
    if flags & _POSITION_EXT:
        high, offset = _decode_uvarint(frame, offset)
        if high == 0:
            raise ValueError("position extension must be necessary")
        start |= high << 8
    if start > max_position:
        raise ValueError("edit position exceeds the configured canvas limit")
    end: int | None = None
    if kind == "insert":
        if flags & _RANGE_EXT:
            raise ValueError("insert cannot encode a source range")
    else:
        length = 1
        if flags & _RANGE_EXT:
            length, offset = _decode_uvarint(frame, offset)
            if length == 1:
                raise ValueError("range extension must be necessary")
        end = start + length
        if end > max_position:
            raise ValueError("edit range exceeds the configured canvas limit")
    payload = b""
    if kind in ("insert", "replace"):
        length = 1
        if flags & _PAYLOAD_EXT:
            length, offset = _decode_uvarint(frame, offset)
            if length == 1 or length > max_payload:
                raise ValueError("invalid or unnecessary payload extension")
            if length == 0 and first != 0:
                raise ValueError("empty payload must have a zero placeholder")
        if length > max_payload:
            raise ValueError("edit payload exceeds the configured canvas limit")
        if length:
            tail_length = length - 1
            if offset + tail_length > len(frame):
                raise ValueError("truncated edit payload")
            payload = bytes([first]) + frame[offset:offset + tail_length]
            offset += tail_length
    elif first != 0 or flags & _PAYLOAD_EXT:
        raise ValueError("delete/move require a zero byte placeholder")
    destination: int | None = None
    if kind == "move":
        destination, offset = _decode_uvarint(frame, offset)
        if destination > max_position or (end is not None and start < destination < end):
            raise ValueError("invalid move destination")
    if offset != len(frame):
        raise ValueError("unexpected bytes after edit argument frame")
    return CognitiveAction(EDIT_TOKEN, start=start, end=end, kind=kind, data=payload, destination=destination)
