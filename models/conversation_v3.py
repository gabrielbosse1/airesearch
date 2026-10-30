"""Phase 12 conversational session with an auditable vector-only recall route.

Narrow English grammars choose a user-scoped structured fact slot. For supported
recall questions, the model receives only the current question and retrieved
vectors: no transcript, expected answer, procedural lookup or grounded fallback
is supplied. Other Phase 11 actions and safety checks remain in place.
"""
from __future__ import annotations

import math
from dataclasses import replace

import torch

from .conversation import ConversationResult, _visible_text, pack_conversation
from .conversation_intents import ConversationIntent, INTENT_NAMES
from .conversation_v2 import ContextualConversationSession
from .diverse_response import DiverseByteDecoder
from .procedural import raw_bytes
from .vector_memory import (VectorConversationMemory, VectorFactCodec, VectorMemoryReader,
                            question_topic, statement_fact, workplace_forget_topic)


class VectorConversationSession(ContextualConversationSession):
    def __init__(self, model, decoder, memory: VectorConversationMemory, reader: VectorMemoryReader,
                 diverse_decoder: DiverseByteDecoder, user_id: bytes, *, min_confidence: float = 0.55,
                 max_history_turns: int = 8, episode_limit: int = 5, record_episodes: bool = True,
                 temperature: float = 0.15, sample_responses: bool = True, seed: int | None = None) -> None:
        if not isinstance(memory, VectorConversationMemory) or not isinstance(reader, VectorMemoryReader):
            raise TypeError("Phase 12 requires vector conversation memory and reader")
        if not isinstance(diverse_decoder, DiverseByteDecoder):
            raise TypeError("Phase 12 requires a diverse byte decoder")
        if not bool(reader.reader_trained) or not bool(diverse_decoder.response_trained):
            raise ValueError("Phase 12 response readers must be trained")
        if reader.context_dim != model.d_model or diverse_decoder.context_dim != model.d_model:
            raise ValueError("reader/generator width mismatch")
        if not math.isfinite(temperature) or not 0 < temperature <= 2:
            raise ValueError("Phase 12 temperature must be in (0, 2]")
        super().__init__(model, decoder, memory, user_id, min_confidence=min_confidence,
                         max_history_turns=max_history_turns, episode_limit=episode_limit,
                         record_episodes=record_episodes, temperature=0.0)
        self.reader = reader.eval()
        self.diverse_decoder = diverse_decoder.eval()
        self.response_temperature = float(temperature)
        self.sample_responses = bool(sample_responses)
        self.rng = torch.Generator(device=model.byte_emb.weight.device)
        if seed is None:
            self.rng.seed()
        else:
            self.rng.manual_seed(seed)

    def _finish_v2(self, message, result, *, recent_count, record=True):
        if result.intent == "chat" and result.action == "answer" and result.response_source == "contextual_byte_decoder":
            result = replace(result, response_source="diverse_byte_decoder")
        fact = statement_fact(message) if self.record_episodes and record else None
        if fact is not None and len(fact[1]) > VectorFactCodec.MAX_BYTES:
            result = replace(result, response=result.response + b" That value is too long for vector recall.")
        return super()._finish_v2(message, result, recent_count=recent_count, record=record)

    def _generate(self, context, intent, message, history):
        if intent != ConversationIntent.CHAT:
            return super()._generate(context, intent, message, history)
        # Match response training: the only source bytes are the current turn.
        device = self.model.byte_emb.weight.device
        packed = pack_conversation(message, (), self.model.max_len)
        ids = torch.tensor([list(packed)], dtype=torch.long, device=device)
        length = torch.tensor([len(packed)], dtype=torch.long, device=device)
        features = self.model.conversation_features(ids, length, focus_start=torch.tensor([2], device=device))
        response, complete, _style = self.diverse_decoder.generate(features, temperature=self.response_temperature,
                                                                    sample=self.sample_responses, rng=self.rng)
        return response, complete

    @torch.no_grad()
    def respond(self, message: bytes | bytearray, *, document: bytes | bytearray | None = None) -> ConversationResult:
        msg = raw_bytes(message, "message")
        doc = None if document is None else raw_bytes(document, "document")
        if doc is not None and len(doc) > self.model.max_canvas_bytes:
            raise ValueError("document exceeds the configured canvas limit")
        forget_topic = workplace_forget_topic(msg) if _visible_text(msg) and len(msg) + 2 <= self.model.max_len else None
        if forget_topic is not None:
            with self._lock:
                device = self.model.byte_emb.weight.device
                payload = pack_conversation(msg, (), self.model.max_len)
                ids = torch.tensor([list(payload)], dtype=torch.long, device=device)
                lengths = torch.tensor([len(payload)], dtype=torch.long, device=device)
                features = self.model.conversation_features(ids, lengths, focus_start=torch.tensor([2], device=device))
                logits = self.model.intent_classifier(features)
                if not bool(torch.isfinite(logits).all()):
                    return self._finish_v2(msg, ConversationResult(b"I couldn't classify that safely. Could you rephrase?",
                                                                   None, None, "ask"), recent_count=0, record=False)
                score, prediction = torch.softmax(logits, dim=-1)[0].max(dim=0)
                intent = ConversationIntent(int(prediction.item()))
                name, confidence = INTENT_NAMES[int(intent)], float(score.item())
                routing = "model" if intent == ConversationIntent.FORGET and confidence >= self.min_confidence else "exact_forget_workplace_command"
                try:
                    vectors, episodes = self.memory.forget_topic_and_purge_history(self.user_id, forget_topic)
                except (ValueError, OSError, RuntimeError):
                    return self._finish_v2(msg, ConversationResult(b"I could not safely remove that workplace memory.",
                                                                   name, confidence, "ask", routing_reason=routing),
                                           recent_count=0, record=False)
                self.history.clear()
                self._pending_edit = self._pending_document = None
                if vectors:
                    response = b"I removed that workplace vector and cleared your stored conversation history."
                elif episodes:
                    response = b"I cleared your stored conversation history; no workplace vector matched."
                else:
                    response = b"I found no workplace vector or conversation history to remove."
                return self._finish_v2(msg, ConversationResult(response, name, confidence, "forget", routing_reason=routing,
                                                               memory_persisted=self.memory.path is not None),
                                       recent_count=0, record=False)
        # A disabled implicit-memory mode retains Phase 11's explicit-only
        # semantics and is not counted as a vector-only recall evaluation.
        if (not self.record_episodes or not _visible_text(msg) or len(msg) + 2 > self.model.max_len
                or question_topic(msg) is None):
            return super().respond(msg, document=doc)
        with self._lock:
            device = self.model.byte_emb.weight.device
            payload = pack_conversation(msg, (), self.model.max_len)
            ids = torch.tensor([list(payload)], dtype=torch.long, device=device)
            lengths = torch.tensor([len(payload)], dtype=torch.long, device=device)
            focus = torch.tensor([2], dtype=torch.long, device=device)
            base = self.model.conversation_features(ids, lengths, focus_start=focus)
            logits = self.model.intent_classifier(base)
            if not bool(torch.isfinite(logits).all()):
                return self._finish_v2(msg, ConversationResult(b"I couldn't classify that safely. Could you rephrase?",
                                                               None, None, "ask"), recent_count=0)
            score, prediction = torch.softmax(logits, dim=-1)[0].max(dim=0)
            intent = ConversationIntent(int(prediction.item()))
            name, confidence = INTENT_NAMES[int(intent)], float(score.item())
            routing = "vector_recall" if intent == ConversationIntent.RECALL else "exact_vector_read_only_recall"
            self._pending_edit = self._pending_document = None
            try:
                retrieved = self.memory.retrieve_fact(self.user_id, msg)
            except (ValueError, RuntimeError):
                retrieved = None
            if retrieved is None:
                result = ConversationResult(b"I don't have a matching vector fact available. What is it?", name, confidence,
                                            "ask", response_source="vector_missing", routing_reason=routing,
                                            vector_only=True)
                return self._finish_v2(msg, result, recent_count=0)
            vectors = retrieved.vectors.unsqueeze(0).to(device=device, dtype=base.dtype)
            context = self.model.conversation_features(ids, lengths, focus_start=focus, memory_context=vectors)
            response, complete = self.reader.generate(context, vectors)
            if not complete or not _visible_text(response):
                result = ConversationResult(b"I couldn't read that vector fact reliably. Could you clarify?", name, confidence,
                                            "ask", response_source="vector_reader_failed", routing_reason=routing,
                                            retrieved_vector_count=vectors.shape[1], vector_only=True)
            else:
                result = ConversationResult(response, name, confidence, "recall", response_source="vector_reader",
                                            routing_reason=routing, retrieved_vector_count=vectors.shape[1], vector_only=True)
            # The output may be recorded only after generation. It was not an
            # input to this query; no transcript or procedural value was read.
            return self._finish_v2(msg, result, recent_count=0)
