"""Specific/missing-argument raw-byte intent pairs for conversational refinement.

The label EDIT means an edit is requested, not that it is safe to execute;
actual edits still need explicit, valid arguments. CLARIFY is used for an
underspecified request when no resolving context is provided. Splits use
separate wording and entities; they are small synthetic English examples.
"""
from __future__ import annotations

from dataclasses import dataclass

from models.conversation_intents import ConversationIntent


@dataclass(frozen=True)
class IntentPair:
    vague: bytes
    specific: bytes
    specific_intent: ConversationIntent


# User-provided examples deliberately belong to training, never held-out tests.
_TRAIN_EDIT = [
    ("Fix this", "Fix the typo at byte 9"),
    ("Fix it", "Fix the typo at byte 14"),
    ("Please fix that", "Please fix the typo at position 7"),
    ("Could you correct it", "Correct the misspelling on line 12"),
    ("Edit this", "Edit line 3 of the document"),
    ("Change it", "Change byte 4 to 65"),
    ("Can you change this", "Change the first byte to 90"),
    ("Remove it", "Remove the first word from the note"),
    ("Delete that", "Delete the third sentence from the draft"),
    ("Make it better", "Replace the word 'teh' with 'the'"),
    ("Revise this", "Revise the title of the article"),
    ("Please update that", "Update the year to 2025 in the letter"),
    ("Replace it", "Replace byte 6 with 100"),
    ("Can you improve this", "Correct the typo on the second line"),
    ("Move it", "Move the first paragraph below the second"),
    ("Shorten this", "Shorten the opening paragraph of the essay"),
    ("Take care of that", "Remove the final character from the document"),
    ("Work on this", "Change the sentence ending to a period"),
    ("Fix the issue", "Repair the typo at offset 3"),
    ("Do the edit", "Insert the word hello after the greeting"),
]
_TRAIN_QUESTION = [
    ("Tell me about it", "What did Alice say?"),
    ("What did they say", "What did Bob say about the meeting"),
    ("Explain that", "Explain how a byte stores data"),
    ("What is that", "What is a computer network"),
    ("Who was it", "Who invented the telephone"),
    ("What happened there", "What happened during the moon landing"),
    ("Why did that happen", "Why does rain fall from clouds"),
    ("How does it work", "How does a bicycle move"),
    ("Tell me more", "Tell me more about coral reefs"),
    ("Can you explain it", "Can you explain how plants grow"),
    ("When did that happen", "When did the first moon landing happen"),
    ("Where was it", "Where is the Eiffel Tower"),
    ("What about them", "What is known about honey bees"),
    ("What does that mean", "What does the word gravity mean"),
    ("Who said that", "Who said the quote about imagination"),
    ("Tell me what happened", "Tell me what happened in the Apollo mission"),
    ("Which one is it", "Which planet is closest to the Sun"),
    ("How do I do that", "How do I boil water safely"),
    ("Can you answer this", "Can you explain photosynthesis"),
    ("Why is it like that", "Why is the ocean salty"),
]
_TRAIN_OTHER = [
    ("Do it", "What is my favorite color", ConversationIntent.RECALL),
    ("Please handle this", "Remember my name is Avery", ConversationIntent.REMEMBER),
    ("Sort this out", "Forget my favorite food", ConversationIntent.FORGET),
    ("I need help", "Hello, how are you", ConversationIntent.CHAT),
    ("I don't know what to do", "What is a binary digit", ConversationIntent.QUESTION),
    ("I am confused", "Change byte 2 to 33", ConversationIntent.EDIT),
    ("Try again", "What was my pet", ConversationIntent.RECALL),
    ("Please do that", "Remember my city is Prague", ConversationIntent.REMEMBER),
    ("Can you help with this", "Delete my saved hobby", ConversationIntent.FORGET),
    ("I need something", "Thanks for your time", ConversationIntent.CHAT),
    ("I don't understand", "Why do leaves change color", ConversationIntent.QUESTION),
    ("This needs work", "Replace byte 5 with 98", ConversationIntent.EDIT),
    ("What should we do", "What is my project", ConversationIntent.RECALL),
    ("Something is wrong", "Remember my hobby is baking", ConversationIntent.REMEMBER),
    ("That's not right", "Forget my favorite color", ConversationIntent.FORGET),
    ("Can you do that", "Good morning friend", ConversationIntent.CHAT),
]
_VALIDATION = [
    ("Could you fix that for me?", "Correct the typo at byte 23.", ConversationIntent.EDIT),
    ("Change this please.", "Change the fourth character to A.", ConversationIntent.EDIT),
    ("Make this nicer.", "Replace the misspelled word in line 11.", ConversationIntent.EDIT),
    ("Could you remove it?", "Remove the first line of the draft.", ConversationIntent.EDIT),
    ("What was said?", "What did Clara say about dinner?", ConversationIntent.QUESTION),
    ("Tell me about that.", "Tell me about volcanic islands.", ConversationIntent.QUESTION),
    ("Explain this to me.", "Explain how a steam engine works.", ConversationIntent.QUESTION),
    ("What is it about?", "What is a cloud made of?", ConversationIntent.QUESTION),
    ("Can you take care of that?", "What is my saved name?", ConversationIntent.RECALL),
    ("I need assistance with it.", "Please forget my saved project.", ConversationIntent.FORGET),
]
_TEST = [
    ("Please repair this.", "Fix the typo in the ninth byte.", ConversationIntent.EDIT),
    ("Can you modify that?", "Change the byte at offset 13 to 74.", ConversationIntent.EDIT),
    ("Make that clearer.", "Correct the spelling in line 6.", ConversationIntent.EDIT),
    ("Could you delete this?", "Remove the second word from the note.", ConversationIntent.EDIT),
    ("What did someone say?", "What did Elena say about lunch?", ConversationIntent.QUESTION),
    ("I'd like to hear about it.", "Tell me about the planet Neptune.", ConversationIntent.QUESTION),
    ("Explain the thing.", "Explain how a bicycle brake works.", ConversationIntent.QUESTION),
    ("Why did it happen?", "Why do birds migrate south?", ConversationIntent.QUESTION),
    ("Could you deal with it?", "Can you recall my saved nickname?", ConversationIntent.RECALL),
    ("I need help with that.", "Please erase my saved hometown.", ConversationIntent.FORGET),
]


def _variants(text: str) -> tuple[bytes, ...]:
    # Augmentation is limited to train: held-out wording remains separate.
    value = text.strip().rstrip(".!?")
    return tuple(dict.fromkeys((value.encode(), (value + ".").encode(),
                                (value[:1].lower() + value[1:]).encode(),
                                (value[:1].upper() + value[1:] + "?").encode())))


def vague_intent_pairs(split: str) -> list[IntentPair]:
    if split == "train":
        raw = [(a, b, ConversationIntent.EDIT) for a, b in _TRAIN_EDIT]
        raw += [(a, b, ConversationIntent.QUESTION) for a, b in _TRAIN_QUESTION]
        raw += _TRAIN_OTHER
        return [IntentPair(a, b, intent) for vague, specific, intent in raw
                for a in _variants(vague) for b in _variants(specific)]
    if split not in ("validation", "test"):
        raise ValueError("split must be train, validation or test")
    rows = _VALIDATION if split == "validation" else _TEST
    return [IntentPair(a.encode("utf-8"), b.encode("utf-8"), intent) for a, b, intent in rows]


def hard_negative_examples(split: str) -> list[tuple[bytes, ConversationIntent]]:
    # Deduplicate cross-product augmentation by actual (bytes, intent).
    result: dict[bytes, ConversationIntent] = {}
    for pair in vague_intent_pairs(split):
        for payload, label in ((pair.vague, ConversationIntent.CLARIFY), (pair.specific, pair.specific_intent)):
            previous = result.get(payload)
            if previous is not None and previous != label:
                raise ValueError(f"contradictory hard-negative label: {payload!r}")
            result[payload] = label
    return list(result.items())
