"""Bounded conversational sessions over raw bytes and explicit persistent memory.

The learned classifier runs before action selection. It is not an authority:
side effects require narrowly parsed, affirmative commands and edit results are
verified before being returned. User IDs are namespaces, NOT authentication;
the host must authenticate users and protect the unencrypted memory file.
"""
from __future__ import annotations

import copy
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from .cognitive import CognitiveAction
from .cognitive_arguments import encode_edit_arguments
from .conversation_intents import ConversationIntent, INTENT_NAMES
from .dialogue_response import ByteDialogueDecoder
from .generator_conditioned import GeneratorConditioned
from .holdout import HoldoutViolation
from .memory_model import MemoryModel
from .procedural import raw_bytes


@dataclass(frozen=True)
class ConversationTurn:
    user: bytes
    assistant: bytes


def pack_conversation(message: bytes, history: tuple[ConversationTurn, ...] | list[ConversationTurn],
                      max_len: int) -> bytes:
    """Keep the complete current user turn and as many whole recent turns as fit."""
    msg = raw_bytes(message, "message")
    if not isinstance(max_len, int) or isinstance(max_len, bool) or max_len < 3:
        raise ValueError("max_len must be at least three")
    current = b"U:" + msg
    if not msg or len(current) > max_len:
        raise ValueError("the current message is empty or exceeds the byte context limit")
    prefix = b""
    for turn in reversed(history):
        if not isinstance(turn, ConversationTurn):
            raise TypeError("history must contain ConversationTurn objects")
        prior = b"U:" + raw_bytes(turn.user, "history user") + b"\nA:" + raw_bytes(turn.assistant, "history assistant") + b"\n"
        if len(prior) + len(prefix) + len(current) > max_len:
            break
        prefix = prior + prefix
    return prefix + current


class ConversationMemory:
    """Exact, user-scoped fact slots in the existing differentiable memory bank.

    Persistence is opt-in via ``path``. If the file exists it is audited and
    loaded at construction. Saves use atomic replacement; a failed save rolls
    back the modified procedural bank in this process. The lock coordinates
    sessions sharing this object, not other processes or other wrappers.
    """

    def __init__(self, memory: MemoryModel, path: str | Path | None = None) -> None:
        if not isinstance(memory, MemoryModel):
            raise TypeError("memory must be a MemoryModel")
        self.memory = memory
        self.path = None if path is None else Path(path)
        self._lock = threading.RLock()
        if self.path is not None and self.path.exists():
            self.memory.load(str(self.path), map_location=self.memory.procedural.keys.device)

    @staticmethod
    def _frame(user_id: bytes, key: bytes) -> bytes:
        user_id = raw_bytes(user_id, "user_id")
        key = raw_bytes(key, "fact key")
        if not 1 <= len(user_id) <= 64 or not 1 <= len(key) <= 64:
            raise ValueError("user IDs and fact keys must each contain 1 to 64 bytes")
        return b"conversation.fact.v1\0" + len(user_id).to_bytes(2, "big") + user_id + len(key).to_bytes(2, "big") + key

    def _atomic_save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".conversation-", suffix=".pt", dir=str(path.parent))
        os.close(fd)
        try:
            self.memory.save(tmp)
            # mkstemp uses private permissions; the saved file is unencrypted.
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def save(self, path: str | Path | None = None) -> None:
        with self._lock:
            target = self.path if path is None else Path(path)
            if target is None:
                raise ValueError("a path is required to persist this memory")
            self._atomic_save(target)

    def _mutate(self, operation):
        """Rollback in-process procedural state if an opted-in save fails."""
        with self._lock:
            bank = self.memory.procedural
            snapshot = {name: value.detach().clone() for name, value in bank.named_buffers()}
            metadata = copy.deepcopy(bank.metadata)
            tick = self.memory.session_tick
            try:
                result = operation()
                if self.path is not None:
                    self._atomic_save(self.path)
                return result
            except Exception:
                with torch.no_grad():
                    for name, value in bank.named_buffers():
                        value.copy_(snapshot[name])
                bank.metadata = metadata
                self.memory.session_tick = tick
                raise

    def remember(self, user_id: bytes, key: bytes, value: bytes, *, source: bytes) -> None:
        frame = self._frame(user_id, key)
        value = raw_bytes(value, "fact value")
        source = raw_bytes(source, "explicit source")
        if not value or len(value) > min(256, self.memory.max_pattern_bytes):
            raise ValueError("fact value is empty or exceeds the memory limit")
        if len(frame) > self.memory.max_pattern_bytes:
            raise ValueError("framed fact key exceeds the memory limit")
        metadata = {"type": "conversation_fact", "version": 1}
        # Audit the full request as well as the actual stored key and value.
        self.memory.validate_holdout_write(frame, value, metadata=metadata, extra=(source,))
        trace = torch.empty((0, self.memory.d_model), device=self.memory.procedural.keys.device)
        def write() -> None:
            bank = self.memory.procedural
            matches = any(
                int(bank.input_lens[index].item()) == len(frame)
                and bytes(bank.input_bytes[index, :len(frame)].cpu().tolist()) == frame
                for index in (bank.used & bank.structured).nonzero(as_tuple=False).flatten().tolist()
            )
            # Do not let one user's new fact evict another user's (or a skill)
            # through this API. Existing facts can be replaced in their own slot.
            if bool(bank.used.all()) and not matches:
                raise ValueError("conversation memory is full; forget an existing fact first")
            if matches:
                self.memory.delete_procedures(frame)
            self.memory.write_procedure(frame, value, trace, metadata=metadata)
        self._mutate(write)

    def recall(self, user_id: bytes, key: bytes) -> bytes | None:
        frame = self._frame(user_id, key)
        with self._lock:
            try:
                return self.memory.execute_procedure(frame).output
            except LookupError:
                return None

    def forget(self, user_id: bytes, key: bytes) -> int:
        frame = self._frame(user_id, key)
        return int(self._mutate(lambda: self.memory.delete_procedures(frame)))


@dataclass(frozen=True)
class ConversationResult:
    response: bytes
    intent: str | None
    confidence: float | None
    action: Literal["answer", "ask", "edit", "remember", "recall", "forget"]
    edited_document: bytes | None = None
    response_source: str = "policy"
    memory_persisted: bool = False
    edit_events: tuple[tuple[int, ...], ...] = ()
    # Phase 11 diagnostics; Phase 10 callers retain identical defaults.
    retrieved_episode_count: int = 0
    episode_recorded: bool = False
    episode_persisted: bool = False
    routing_reason: str = "model"
    # Phase 12 opt-in diagnostics; a vector-only answer never consumed episodes.
    retrieved_vector_count: int = 0
    vector_only: bool = False


# Side-effect grammar is intentionally much narrower than classifier vocabulary.
_KEY = r"([a-z][a-z0-9 -]{0,47}?)"
_FLAGS = re.IGNORECASE | re.ASCII  # Avoid Unicode case-fold matches that are not ASCII keys.
_REMEMBER = re.compile(r"^(?:(?:please|could you)\s+)?remember\s+(?:that\s+)?my\s+" + _KEY + r"\s+(?:is|as|=)\s+(.+?)[.!?]?$", _FLAGS)
_RECALL = re.compile(r"^(?:what\s+(?:is|was)|do\s+you\s+(?:remember|know)|(?:can\s+you|please)\s+(?:tell\s+me|recall)|tell\s+me)\s+my\s+" + _KEY + r"[.!?]?$", _FLAGS)
_FORGET = re.compile(r"^(?:(?:please|could you|can you)\s+)?(?:forget\s+my|(?:delete|remove|erase)\s+my\s+saved)\s+" + _KEY + r"[.!?]?$", _FLAGS)
_EDIT = re.compile(r"^(?:(?:please|could you|can you)\s+)?(?:replace|change)\s+(?:the\s+)?byte\s+(?:(?:at\s+(?:zero-based\s+)?position|number)\s+)?(\d{1,5})\s+(?:with|to)\s+(?:byte\s+)?(\d{1,3})[.!?]?$", _FLAGS)
_CONFIRM = re.compile(r"^(?:please\s+)?(?:apply|do)\s+that\s+edit[.!?]?$", _FLAGS)

# Deliberately tiny, explicit reference answers. These are host-supplied
# knowledge, not claims that the neural decoder learned open-domain facts.
_KNOWN_ANSWERS: dict[str, bytes] = {
    "what is a byte": b"A byte is a unit of data that usually contains eight bits.",
    "how many bits are in a byte": b"A byte usually contains eight bits.",
    "what is a bit": b"A bit is a binary digit: either zero or one.",
    "what is python": b"Python is a general-purpose programming language.",
}


def _key(text: str) -> bytes:
    return " ".join(text.lower().split()).encode("ascii")


def _visible_text(data: bytes) -> bool:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return bool(text.strip()) and all(character.isprintable() or character in "\n\t" for character in text)


class ConversationSession:
    """Single user's bounded dialogue; no background logging or implicit writes."""

    def __init__(self, model: GeneratorConditioned, decoder: ByteDialogueDecoder,
                 memory: ConversationMemory, user_id: bytes, *, min_confidence: float = 0.55,
                 max_history_turns: int = 8) -> None:
        ConversationMemory._frame(user_id, b"check")
        if not bool(model.intent_trained):
            raise RuntimeError("the intent classifier is untrained; train or load a Phase 10 checkpoint")
        if not bool(decoder.response_trained):
            raise RuntimeError("the response decoder is untrained")
        if decoder.context_dim != model.d_model:
            raise ValueError("decoder and generator context dimensions differ")
        if decoder.expected_generator_heads is not None and decoder.expected_generator_heads != model.n_heads:
            raise ValueError("decoder checkpoint expects a different generator attention-head count")
        if decoder.byte_head.weight.device != model.byte_emb.weight.device:
            raise ValueError("decoder and generator must share a device")
        if not 0 <= min_confidence <= 1 or not isinstance(max_history_turns, int) or not 0 <= max_history_turns <= 100:
            raise ValueError("invalid confidence or history limit")
        self.model, self.decoder, self.memory = model.eval(), decoder.eval(), memory
        self.user_id = bytes(user_id)
        self.min_confidence = float(min_confidence)
        self.max_history_turns = max_history_turns
        self.history: list[ConversationTurn] = []
        self._pending_edit: tuple[int, int, bytes] | None = None
        self._pending_document: bytes | None = None
        self._lock = threading.RLock()

    def _finish(self, message: bytes, result: ConversationResult) -> ConversationResult:
        # Do not retain rejected, malformed or oversized input in the session.
        if (self.max_history_turns and len(message) + 2 <= self.model.max_len
                and _visible_text(message) and len(result.response) <= self.model.max_len):
            self.history.append(ConversationTurn(message, result.response))
            self.history = self.history[-self.max_history_turns:]
        return result

    @torch.no_grad()
    def respond(self, message: bytes | bytearray, *, document: bytes | bytearray | None = None) -> ConversationResult:
        msg = raw_bytes(message, "message")
        doc = None if document is None else raw_bytes(document, "document")
        if doc is not None and len(doc) > self.model.max_canvas_bytes:
            raise ValueError("document exceeds the configured canvas limit")
        with self._lock:
            if len(msg) + 2 > self.model.max_len:
                return self._finish(msg, ConversationResult(b"Please make your message shorter so I can process it.", None, None, "ask"))
            if not _visible_text(msg):
                return self._finish(msg, ConversationResult(b"Please send a nonempty UTF-8 text message.", None, None, "ask"))
            try:
                packed = pack_conversation(msg, self.history, self.model.max_len)
            except ValueError:
                return self._finish(msg, ConversationResult(b"Please make your message shorter so I can process it.", None, None, "ask"))
            device = self.model.byte_emb.weight.device
            ids = torch.tensor([list(packed)], dtype=torch.long, device=device)
            lengths = torch.tensor([len(packed)], dtype=torch.long, device=device)
            focus = torch.tensor([len(packed) - len(msg)], dtype=torch.long, device=device)
            features = self.model.conversation_features(ids, lengths, focus_start=focus)
            logits = self.model.intent_classifier(features)
            if not bool(torch.isfinite(logits).all()):
                return self._finish(msg, ConversationResult(b"I couldn't classify that safely. Could you rephrase?", None, None, "ask"))
            probabilities = torch.softmax(logits, dim=-1)[0]
            confidence, index = probabilities.max(dim=0)
            intent = ConversationIntent(int(index.item()))
            name, score = INTENT_NAMES[int(intent)], float(confidence.item())
            text = msg.decode("utf-8").strip()

            # A pending edit is executed only upon an explicit, exact confirmation
            # and a supplied document. Classification still precedes this decision.
            pending = self._pending_edit
            pending_doc = self._pending_document
            self._pending_edit = None
            self._pending_document = None
            if pending is not None and _CONFIRM.fullmatch(text) and doc is not None:
                return self._finish(msg, self._do_edit(doc, *pending, name=name, score=score))
            if score < self.min_confidence:
                return self._finish(msg, ConversationResult(b"I'm not sure what you want me to do. Could you clarify?", name, score, "ask"))
            if intent == ConversationIntent.CHAT:
                response, complete = self.decoder.generate(features)
                if not complete or not _visible_text(response):
                    return self._finish(msg, ConversationResult(b"I'm here. What would you like to talk about?", name, score, "answer", response_source="fallback"))
                return self._finish(msg, ConversationResult(response, name, score, "answer", response_source="byte_decoder"))
            if intent == ConversationIntent.QUESTION:
                normalized = " ".join(text.casefold().rstrip(".!? ").split())
                answer = _KNOWN_ANSWERS.get(normalized)
                if answer is not None:
                    return self._finish(msg, ConversationResult(answer, name, score, "answer", response_source="verified_reference"))
                return self._finish(msg, ConversationResult(b"I don't have a reliable answer to that yet. Could you share more context?", name, score, "ask"))
            if intent == ConversationIntent.CLARIFY:
                return self._finish(msg, ConversationResult(b"Could you tell me what you need and give me a little more detail?", name, score, "ask"))
            if intent == ConversationIntent.REMEMBER:
                match = _REMEMBER.fullmatch(text)
                if match is None or not _visible_text(match.group(2).encode("utf-8")):
                    return self._finish(msg, ConversationResult(b"What exactly would you like me to remember? Say, for example, 'remember my name is Ada'.", name, score, "ask"))
                key, value = _key(match.group(1)), match.group(2).strip().encode("utf-8")
                try:
                    self.memory.remember(self.user_id, key, value, source=msg)
                except (HoldoutViolation, ValueError, OSError, RuntimeError):
                    return self._finish(msg, ConversationResult(b"I could not safely save that memory.", name, score, "ask"))
                return self._finish(msg, ConversationResult(b"I'll remember your " + key + b".", name, score, "remember", memory_persisted=self.memory.path is not None))
            if intent == ConversationIntent.RECALL:
                match = _RECALL.fullmatch(text)
                if match is None:
                    return self._finish(msg, ConversationResult(b"Which saved detail would you like me to recall?", name, score, "ask"))
                key = _key(match.group(1))
                value = self.memory.recall(self.user_id, key)
                if value is None:
                    return self._finish(msg, ConversationResult(b"I don't have your " + key + b" saved. What would you like me to remember?", name, score, "ask"))
                if not _visible_text(value):
                    return self._finish(msg, ConversationResult(b"That saved value isn't displayable text; please clarify what you need.", name, score, "ask"))
                return self._finish(msg, ConversationResult(b"Your " + key + b" is " + value + b".", name, score, "recall"))
            if intent == ConversationIntent.FORGET:
                match = _FORGET.fullmatch(text)
                if match is None:
                    return self._finish(msg, ConversationResult(b"Which saved detail should I forget?", name, score, "ask"))
                key = _key(match.group(1))
                try:
                    removed = self.memory.forget(self.user_id, key)
                except (HoldoutViolation, OSError, RuntimeError):
                    return self._finish(msg, ConversationResult(b"I could not safely remove that memory.", name, score, "ask"))
                response = b"I removed your saved " + key + b"." if removed else b"I don't have your " + key + b" saved."
                return self._finish(msg, ConversationResult(response, name, score, "forget", memory_persisted=self.memory.path is not None))
            if intent == ConversationIntent.EDIT:
                match = _EDIT.fullmatch(text)
                if match is None:
                    if doc is not None:
                        self._pending_document = doc
                    return self._finish(msg, ConversationResult(b"What should I change? Please specify a zero-based byte position and replacement byte.", name, score, "ask"))
                position, value = int(match.group(1)), int(match.group(2))
                if value > 255:
                    return self._finish(msg, ConversationResult(b"The replacement byte must be between 0 and 255.", name, score, "ask"))
                doc = doc if doc is not None else pending_doc
                if doc is None:
                    self._pending_edit = (position, value, msg)
                    return self._finish(msg, ConversationResult(b"Please provide the document, then say 'apply that edit'.", name, score, "ask"))
                return self._finish(msg, self._do_edit(doc, position, value, msg, name=name, score=score))
            raise AssertionError("unhandled intent")

    def _do_edit(self, document: bytes, position: int, value: int, instruction: bytes, *, name: str, score: float) -> ConversationResult:
        if not 0 <= position < len(document) or not 0 <= value <= 255:
            return ConversationResult(b"That byte position or value is outside the document's valid range.", name, score, "ask")
        if document[position] == value:
            return ConversationResult(b"That byte already has the requested value; nothing changed.", name, score, "answer")
        if not bool(self.model.structured_edit_trained_kinds[2]):
            return ConversationResult(b"The one-byte edit model is not trained; the document was not changed.", name, score, "ask")
        action = CognitiveAction(self.model.EDIT_OP, start=position, end=position + 1, kind="replace", data=bytes([value]))
        request = encode_edit_arguments(action)
        result = self.model.cognitive_generate(document, instruction_bytes=instruction, structured_request=request,
                                               max_steps=12, temperature=0.0)
        expected = document[:position] + bytes([value]) + document[position + 1:]
        events = tuple(event.control_and_arguments for event in result.events)
        if not result.done or result.canvas != expected:
            return ConversationResult(b"I couldn't safely complete that edit; the document is unchanged.", name, score, "ask", edit_events=events)
        return ConversationResult(b"I made the requested one-byte edit.", name, score, "edit", edited_document=result.canvas, edit_events=events)
