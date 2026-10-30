"""Scoped recent conversational episodes using the existing memory bank.

Episodes are user-provided data, not trusted instructions. They are never
promoted to semantic memory; a forget operation can therefore purge the live
user-scoped transcript. Persistent storage is unencrypted and requires host
access control. One wrapper lock does not synchronize independent processes.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import torch

from .conversation import ConversationMemory, ConversationTurn
from .memory_model import TIER_EPISODIC
from .procedural import byte_pattern_embedding, raw_bytes


_EPISODE_TYPE = "conversation_episode_v2"


@dataclass(frozen=True)
class RetrievedEpisodes:
    turns: tuple[ConversationTurn, ...]
    vectors: torch.Tensor  # (K, d_model), oldest to newest
    slots: tuple[int, ...]


class EpisodicConversationMemory(ConversationMemory):
    """Store and retrieve the latest scoped turns via ``MemoryModel.retrieve``."""

    @staticmethod
    def scope(user_id: bytes) -> str:
        user_id = raw_bytes(user_id, "user_id")
        ConversationMemory._frame(user_id, b"check")
        frame = b"conversation.episode.person.v1\0" + len(user_id).to_bytes(2, "big") + user_id
        return "conversation:" + hashlib.sha256(frame).hexdigest()

    def _mutate_banks(self, operation):
        """Atomically roll back affected live banks on ordinary exceptions."""
        with self._lock:
            banks = (self.memory.episodic, self.memory.procedural)
            snapshots = [{name: value.detach().clone() for name, value in bank.named_buffers()} for bank in banks]
            metadata = [copy.deepcopy(bank.metadata) for bank in banks]
            tick = self.memory.session_tick
            try:
                result = operation()
                if self.path is not None:
                    self._atomic_save(self.path)
                return result
            except Exception:
                with torch.no_grad():
                    for bank, snapshot, saved_metadata in zip(banks, snapshots, metadata):
                        for name, value in bank.named_buffers():
                            value.copy_(snapshot[name])
                        bank.metadata = saved_metadata
                    self.memory.session_tick = tick
                raise

    def _write_episode(self, user_id: bytes, turn: ConversationTurn) -> None:
        """Write one episode inside the caller's bank transaction."""
        scope = self.scope(user_id)
        if not isinstance(turn, ConversationTurn):
            raise TypeError("turn must be a ConversationTurn")
        user = raw_bytes(turn.user, "episode user")
        assistant = raw_bytes(turn.assistant, "episode assistant")
        if not user or len(user) > 512 or len(assistant) > 512:
            raise ValueError("episode fields must be nonempty/bounded raw bytes")
        metadata = {"type": _EPISODE_TYPE, "person": scope, "modality": "text",
                    "user_bytes": user, "assistant_bytes": assistant, "no_promote": True}
        self.memory.validate_holdout_write(user, assistant, metadata=metadata)
        device = self.memory.episodic.keys.device
        content = byte_pattern_embedding(user + b"\n" + assistant, self.memory.d_model, device=device)
        # Never use the global eviction policy to remove another user's entry.
        if bool(self.memory.episodic.used.all()):
            removed = self.memory.delete_episodic_entries(person=scope, event_type=_EPISODE_TYPE, limit=1)
            if not removed:
                raise ValueError("episodic memory is full; no owned episode can be recycled")
        self.memory.write(content, metadata, tier=TIER_EPISODIC, raw_input=user, raw_output=assistant)

    def record_episode(self, user_id: bytes, turn: ConversationTurn) -> None:
        self._mutate_banks(lambda: self._write_episode(user_id, turn))

    @torch.no_grad()
    def recent(self, user_id: bytes, query: bytes, *, limit: int = 5) -> RetrievedEpisodes:
        scope = self.scope(user_id)
        query = raw_bytes(query, "episode query")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 5:
            raise ValueError("episode limit must be between 1 and 5")
        device = self.memory.episodic.keys.device
        query_emb = byte_pattern_embedding(query, self.memory.d_model, device=device)
        with self._lock:
            # Retrieve all matching owned episodic candidates, then pick the most
            # recent by write tick. Similarity top-K alone is not chronological.
            with self.memory.read_only():
                entries, _vectors = self.memory.retrieve(query_emb, top_k=self.memory.epi_cap,
                                                          person_filter=scope, tier_filter=TIER_EPISODIC)
            selected = [entry for entry in entries if entry["metadata"].get("type") == _EPISODE_TYPE]
            selected.sort(key=lambda entry: (int(entry["metadata"].get("written_at", -1)), int(entry["slot"])))
            selected = selected[-limit:]
            turns: list[ConversationTurn] = []
            values: list[torch.Tensor] = []
            slots: list[int] = []
            for entry in selected:
                metadata = entry["metadata"]
                user, assistant = metadata.get("user_bytes"), metadata.get("assistant_bytes")
                # Reject malformed/legacy data, rather than treating arbitrary
                # metadata as a conversation or executable instruction.
                if not isinstance(user, bytes) or not isinstance(assistant, bytes):
                    continue
                turns.append(ConversationTurn(user, assistant))
                values.append(entry["value"].detach().clone())
                slots.append(int(entry["slot"]))
            vectors = torch.stack(values) if values else torch.empty((0, self.memory.d_model), device=device)
            return RetrievedEpisodes(tuple(turns), vectors, tuple(slots))

    def forget_and_purge_history(self, user_id: bytes, key: bytes) -> tuple[int, int]:
        """Forget the exact fact and all this user's stored conversation turns.

        Clearing the full transcript avoids retaining the forgotten detail in a
        previous message or assistant reply. It cannot erase external backups.
        """
        frame, scope = self._frame(user_id, key), self.scope(user_id)
        def remove() -> tuple[int, int]:
            facts = self.memory.delete_procedures(frame)
            episodes = self.memory.delete_episodic_entries(person=scope, event_type=_EPISODE_TYPE)
            return facts, episodes
        return self._mutate_banks(remove)
