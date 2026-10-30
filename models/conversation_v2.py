"""Phase 11 conversational session with scoped episodic retrieval.

The host must authenticate user IDs and protect any unencrypted memory file.
Automatic episode persistence is opt-in via a memory path; disable episodes for
no implicit storage. Exact read-only fact grounding and fail-closed action
checks are safety boundaries, not claims of general language understanding.
"""
from __future__ import annotations

import math
import re
from dataclasses import replace

import torch

from .contextual_response import ContextualByteDecoder, pack_response_context
from .conversation import (ConversationMemory, ConversationResult, ConversationSession, ConversationTurn,
                           _CONFIRM, _EDIT, _FLAGS, _FORGET, _KEY, _KNOWN_ANSWERS, _RECALL, _REMEMBER,
                           _key, _visible_text, pack_conversation)
from .conversation_intents import ConversationIntent, INTENT_NAMES
from .episodic_conversation import EpisodicConversationMemory, RetrievedEpisodes
from .holdout import HoldoutViolation
from .procedural import raw_bytes


_STATEMENT = re.compile(r"^my\s+" + _KEY + r"\s+is\s+(.+?)[.!?]?$", _FLAGS)
_RECALL_EXTRA = re.compile(r"^(?:what's|what\s+are|what\s+were)\s+my\s+" + _KEY + r"[.!?]?$", _FLAGS)
# Conservative: quoted occurrences may also disable this turn's storage.
_NO_STORAGE = re.compile(
    r"\b(?:(?:do\s+not|don't|dont|never)\s+(?:ever\s+)?(?:save|store|remember|keep|retain|record|log)"
    r"|(?:do\s+not|don't|dont)\s+want\s+(?:you\s+to\s+)?(?:save|store|remember|keep|retain|record|log)"
    r"|(?:do\s+not|don't|dont)\s+want\s+(?:this|that|it|my\s+message|our\s+conversation|this\s+conversation)\s+"
    r"(?:to\s+be\s+)?(?:saved|stored|remembered|kept|retained|recorded|logged)"
    r"|(?:stop|avoid)\s+(?:saving|storing|remembering|keeping|retaining|recording|logging)"
    r"|off\s+the\s+record|no\s+(?:logging|storage|recording)"
    r"|without\s+(?:saving|storing|remembering|retaining|recording|logging))\b", _FLAGS)
_NO_STORAGE_SESSION = re.compile(
    r"\b(?:(?:do\s+not|don't|dont|never)\s+(?:ever\s+)?(?:want\s+you\s+to\s+)?"
    r"(?:save|store|remember|keep|retain|record|log)\s+"
    r"(?:anything|any\s+(?:messages|conversation|turns|information)|our\s+conversation|this\s+conversation|my\s+messages|future\s+turns)"
    r"|(?:do\s+not|don't|dont)\s+want\s+(?:our|this)\s+conversation\s+(?:to\s+be\s+)?"
    r"(?:saved|stored|remembered|kept|retained|recorded|logged)"
    r"|stop\s+(?:saving|storing|remembering|keeping|retaining|recording|logging)"
    r"|no\s+(?:logging|storage|recording))\b", _FLAGS)


class ContextualConversationSession(ConversationSession):
    """Predict intent, retrieve scoped episodes, generate and guard actions."""

    def __init__(self, model, decoder: ContextualByteDecoder, memory: EpisodicConversationMemory,
                 user_id: bytes, *, min_confidence: float = 0.55, max_history_turns: int = 8,
                 episode_limit: int = 5, record_episodes: bool = True, temperature: float = 0.0) -> None:
        if not isinstance(memory, EpisodicConversationMemory):
            raise TypeError("Phase 11 requires EpisodicConversationMemory")
        if not isinstance(decoder, ContextualByteDecoder):
            raise TypeError("Phase 11 requires a ContextualByteDecoder")
        if not isinstance(episode_limit, int) or isinstance(episode_limit, bool) or not 1 <= episode_limit <= 5:
            raise ValueError("episode_limit must be in 1..5")
        if not isinstance(record_episodes, bool) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("invalid episode-storage or temperature setting")
        super().__init__(model, decoder, memory, user_id, min_confidence=min_confidence,
                         max_history_turns=max_history_turns)
        self.episode_limit, self.record_episodes, self.temperature = episode_limit, record_episodes, float(temperature)
        self._privacy_disabled = False

    def _disable_session_storage(self) -> None:
        self.record_episodes = False
        self._privacy_disabled = True
        self.history.clear()
        self.max_history_turns = 0
        self._pending_edit = self._pending_document = None

    def _finish_v2(self, message: bytes, result: ConversationResult, *, recent_count: int,
                   record: bool = True) -> ConversationResult:
        valid = (len(message) + 2 <= self.model.max_len and _visible_text(message)
                 and len(result.response) <= 512 and _visible_text(result.response))
        if not valid or not record:
            return replace(result, retrieved_episode_count=recent_count)
        recorded = False
        if self.record_episodes:
            try:
                self.memory.record_episode(self.user_id, ConversationTurn(message, result.response))
                recorded = True
            except (HoldoutViolation, ValueError, OSError, RuntimeError):
                # Never claim retention when the bounded bank or atomic save fails.
                result = replace(result, response=result.response + b" I could not save this conversation turn.")
        if self.max_history_turns:
            self.history.append(ConversationTurn(message, result.response))
            self.history = self.history[-self.max_history_turns:]
        return replace(result, retrieved_episode_count=recent_count, episode_recorded=recorded,
                       episode_persisted=recorded and self.memory.path is not None)

    def _generate(self, context: torch.Tensor, intent: ConversationIntent, message: bytes,
                  history: tuple[ConversationTurn, ...]) -> tuple[bytes, bool]:
        packed, segments = pack_response_context(message, history, max_bytes=self.decoder.max_input_bytes)
        device = context.device
        source = torch.tensor([list(packed)], dtype=torch.long, device=device)
        seg = torch.tensor([list(segments)], dtype=torch.long, device=device)
        length = torch.tensor([len(packed)], dtype=torch.long, device=device)
        label = torch.tensor([int(intent)], dtype=torch.long, device=device)
        return self.decoder.generate(context, label, source, seg, length, temperature=self.temperature)

    @staticmethod
    def _episode_fact(key: bytes, turns: tuple[ConversationTurn, ...]) -> bytes | None:
        for turn in reversed(turns):
            try:
                text = turn.user.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            # Only affirmative, narrowly anchored text is treated as evidence.
            match = _STATEMENT.fullmatch(text) or _REMEMBER.fullmatch(text)
            if match is not None and _key(match.group(1)) == key:
                value = match.group(2).strip().encode("utf-8")
                if _visible_text(value):
                    return value
        return None

    @staticmethod
    def _contains_value(response: bytes, value: bytes) -> bool:
        try:
            return bool(re.search(r"(?<!\w)" + re.escape(value.decode("utf-8")) + r"(?!\w)",
                                  response.decode("utf-8"), re.IGNORECASE))
        except UnicodeDecodeError:
            return False

    @torch.no_grad()
    def respond(self, message: bytes | bytearray, *, document: bytes | bytearray | None = None) -> ConversationResult:
        msg = raw_bytes(message, "message")
        doc = None if document is None else raw_bytes(document, "document")
        if doc is not None and len(doc) > self.model.max_canvas_bytes:
            raise ValueError("document exceeds the configured canvas limit")
        with self._lock:
            if len(msg) + 2 > self.model.max_len:
                return ConversationResult(b"Please make your message shorter so I can process it.", None, None, "ask")
            if not _visible_text(msg):
                return ConversationResult(b"Please send a nonempty UTF-8 text message.", None, None, "ask")
            text = msg.decode("utf-8").strip()
            if self.record_episodes:
                recent = self.memory.recent(self.user_id, msg, limit=self.episode_limit)
                history = recent.turns
            else:
                recent = RetrievedEpisodes((), torch.empty((0, self.model.d_model)), ())
                history = tuple(self.history[-self.episode_limit:])
            count = len(recent.turns)
            # Intent classification uses the current turn alone, matching its
            # protected held-out evaluation. The response feature pass injects
            # the actual retrieved values as memory_context when available.
            packed = pack_conversation(msg, (), self.model.max_len)
            device = self.model.byte_emb.weight.device
            ids = torch.tensor([list(packed)], dtype=torch.long, device=device)
            lengths = torch.tensor([len(packed)], dtype=torch.long, device=device)
            focus = torch.tensor([2], dtype=torch.long, device=device)
            base_features = self.model.conversation_features(ids, lengths, focus_start=focus)
            logits = self.model.intent_classifier(base_features)
            if not bool(torch.isfinite(logits).all()):
                if _NO_STORAGE.search(text):
                    if _NO_STORAGE_SESSION.search(text):
                        self._disable_session_storage()
                    result = ConversationResult(b"I won't save this turn. Existing saved records remain.", None, None,
                                                "answer", response_source="privacy_guard")
                    return self._finish_v2(msg, result, recent_count=count, record=False)
                result = ConversationResult(b"I couldn't classify that safely. Could you rephrase?", None, None, "ask")
                return self._finish_v2(msg, result, recent_count=count)
            probabilities = torch.softmax(logits, dim=-1)[0]
            confidence, index = probabilities.max(dim=0)
            intent = ConversationIntent(int(index.item()))
            name, score = INTENT_NAMES[int(intent)], float(confidence.item())
            features = base_features
            if count:
                vectors = recent.vectors.unsqueeze(0).to(device=device, dtype=base_features.dtype)
                features = self.model.conversation_features(ids, lengths, focus_start=focus, memory_context=vectors)
            finish = lambda result, record=True: self._finish_v2(msg, result, recent_count=count, record=record)

            # An explicit per-turn privacy request wins over all other actions;
            # it is not placed in episodic or local transcript history.
            if _NO_STORAGE.search(text):
                self._pending_edit = self._pending_document = None
                if _NO_STORAGE_SESSION.search(text):
                    self._disable_session_storage()
                    response = (b"I won't store future turns in this session. Existing saved records remain; "
                                b"ask me to forget a detail if you want the stored conversation cleared.")
                else:
                    response = b"I won't save or remember this turn."
                return finish(ConversationResult(response, name, score, "answer",
                                                 response_source="privacy_guard"), record=False)

            pending, pending_doc = self._pending_edit, self._pending_document
            self._pending_edit = self._pending_document = None
            if pending is not None and _CONFIRM.fullmatch(text) and doc is not None:
                return finish(self._do_edit(doc, *pending, name=name, score=score))

            # Exact read-only grounding is a safe policy override even if the
            # classifier mislabels the request; the predicted intent remains in
            # the result and the override is explicitly reported.
            recall_match = _RECALL.fullmatch(text) or _RECALL_EXTRA.fullmatch(text)
            if recall_match is not None:
                key = _key(recall_match.group(1))
                value = self._episode_fact(key, history)
                if value is None:
                    value = self.memory.recall(self.user_id, key)
                routing = "model" if intent == ConversationIntent.RECALL else "exact_read_only_recall"
                if value is None:
                    return finish(ConversationResult(b"I don't have that detail in your recent conversation or saved facts. What is it?",
                                                     name, score, "ask", routing_reason=routing))
                if not _visible_text(value):
                    return finish(ConversationResult(b"That saved value isn't displayable text; please clarify what you need.",
                                                     name, score, "ask", routing_reason=routing))
                response, complete = self._generate(features, ConversationIntent.RECALL, msg, history)
                if complete and _visible_text(response) and self._contains_value(response, value):
                    return finish(ConversationResult(response, name, score, "recall", response_source="contextual_byte_decoder",
                                                     routing_reason=routing))
                # This explicit, evidence-checked fallback is counted separately
                # from decoder success and cannot fabricate the target value.
                return finish(ConversationResult(b"You said " + value + b".", name, score, "recall",
                                                 response_source="grounded_fallback", routing_reason=routing))
            # Unambiguous, anchored affirmative memory commands are stronger
            # evidence than an uncalibrated classifier. In particular, never
            # silently prevent a user from deleting their own stored history
            # because the model guessed CHAT. Negations and quoted text do not
            # match these exact grammars, and the privacy guard ran first.
            remember_match = _REMEMBER.fullmatch(text)
            if remember_match is not None and _visible_text(remember_match.group(2).encode("utf-8")):
                routing = "model" if intent == ConversationIntent.REMEMBER and score >= self.min_confidence else "exact_remember_command"
                if self._privacy_disabled:
                    return finish(ConversationResult(b"This session is in no-storage mode, so I won't save a detail.",
                                                     name, score, "ask", response_source="privacy_guard", routing_reason=routing), record=False)
                key, value = _key(remember_match.group(1)), remember_match.group(2).strip().encode("utf-8")
                try:
                    self.memory.remember(self.user_id, key, value, source=msg)
                except (HoldoutViolation, ValueError, OSError, RuntimeError):
                    return finish(ConversationResult(b"I could not safely save that memory.", name, score, "ask", routing_reason=routing), record=False)
                response, complete = self._generate(features, ConversationIntent.REMEMBER, msg, history)
                if not complete or not _visible_text(response) or b"?" in response:
                    response, source = b"I'll remember your " + key + b".", "fallback"
                else:
                    source = "contextual_byte_decoder"
                return finish(ConversationResult(response, name, score, "remember", response_source=source,
                                                 memory_persisted=self.memory.path is not None, routing_reason=routing))
            forget_match = _FORGET.fullmatch(text)
            if forget_match is not None:
                routing = "model" if intent == ConversationIntent.FORGET and score >= self.min_confidence else "exact_forget_command"
                key = _key(forget_match.group(1))
                try:
                    facts, episodes = self.memory.forget_and_purge_history(self.user_id, key)
                except (HoldoutViolation, OSError, RuntimeError):
                    return finish(ConversationResult(b"I could not safely remove that memory.", name, score, "ask", routing_reason=routing), record=False)
                self.history.clear()
                if facts:
                    response = b"I removed the saved detail and cleared your stored conversation history."
                elif episodes:
                    response = b"I cleared your stored conversation history; no separate saved fact matched that key."
                else:
                    response = b"I found no saved detail or conversation history to remove."
                # Do not immediately re-create an episode containing the forget request.
                return finish(ConversationResult(response, name, score, "forget", memory_persisted=self.memory.path is not None,
                                                 routing_reason=routing), record=False)
            if score < self.min_confidence:
                return finish(ConversationResult(b"I'm not sure what you want me to do. Could you clarify?", name, score, "ask"))
            if intent == ConversationIntent.CHAT:
                response, complete = self._generate(features, intent, msg, history)
                if complete and _visible_text(response):
                    return finish(ConversationResult(response, name, score, "answer", response_source="contextual_byte_decoder"))
                return finish(ConversationResult(b"I'm here. What would you like to talk about?", name, score, "answer"))
            if intent == ConversationIntent.QUESTION:
                normalized = " ".join(text.casefold().rstrip(".!? ").split())
                answer = _KNOWN_ANSWERS.get(normalized)
                if answer is not None:
                    return finish(ConversationResult(answer, name, score, "answer", response_source="verified_reference"))
                response, complete = self._generate(features, intent, msg, history)
                lower = response.lower()
                if complete and _visible_text(response) and any(s in lower for s in (b"not sure", b"don't know", b"cannot", b"unsure")):
                    return finish(ConversationResult(response, name, score, "ask", response_source="contextual_byte_decoder"))
                return finish(ConversationResult(b"I don't have a reliable answer to that yet. Could you share more context?", name, score, "ask"))
            if intent == ConversationIntent.CLARIFY:
                response, complete = self._generate(features, intent, msg, history)
                if complete and _visible_text(response) and b"?" in response:
                    return finish(ConversationResult(response, name, score, "ask", response_source="contextual_byte_decoder"))
                return finish(ConversationResult(b"Could you tell me what you need and give me a little more detail?", name, score, "ask"))
            if intent == ConversationIntent.REMEMBER:
                return finish(ConversationResult(b"What exactly would you like me to remember? For example, 'remember my name is Ada'.", name, score, "ask"))
            if intent == ConversationIntent.RECALL:
                return finish(ConversationResult(b"Which saved detail would you like me to recall?", name, score, "ask"))
            if intent == ConversationIntent.FORGET:
                return finish(ConversationResult(b"Which saved detail should I forget?", name, score, "ask"))
            if intent == ConversationIntent.EDIT:
                match = _EDIT.fullmatch(text)
                if match is None:
                    if doc is not None:
                        self._pending_document = doc
                    return finish(ConversationResult(b"What should I change? Please specify a zero-based byte position and replacement byte.", name, score, "ask"))
                position, value = int(match.group(1)), int(match.group(2))
                if value > 255:
                    return finish(ConversationResult(b"The replacement byte must be between 0 and 255.", name, score, "ask"))
                doc = doc if doc is not None else pending_doc
                if doc is None:
                    self._pending_edit = (position, value, msg)
                    return finish(ConversationResult(b"Please provide the document, then say 'apply that edit'.", name, score, "ask"))
                result = self._do_edit(doc, position, value, msg, name=name, score=score)
                if result.action == "edit":
                    response, complete = self._generate(features, intent, msg, history)
                    if complete and _visible_text(response) and b"?" not in response and any(
                        word in response.lower() for word in (b"done", b"made", b"updated", b"changed", b"all set")
                    ):
                        result = replace(result, response=response, response_source="contextual_byte_decoder")
                return finish(result)
            raise AssertionError("unhandled intent")
