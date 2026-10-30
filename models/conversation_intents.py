"""Stable labels for the byte-level conversational intent head.

These IDs are internal class labels, never serialized as raw user bytes.
Changing their order would make existing Phase 10 checkpoints incompatible.
"""
from enum import IntEnum


class ConversationIntent(IntEnum):
    CHAT = 0
    QUESTION = 1
    EDIT = 2
    REMEMBER = 3
    RECALL = 4
    FORGET = 5
    CLARIFY = 6


INTENT_NAMES: tuple[str, ...] = tuple(member.name.lower() for member in ConversationIntent)
NUM_INTENTS = len(INTENT_NAMES)
