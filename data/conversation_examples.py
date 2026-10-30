"""Deterministic synthetic raw-byte dialogue-intent examples.

Splits use disjoint wording templates, not just a different random seed. The
corpus is deliberately small, English-only and synthetic; it is not evidence
of broad dialogue understanding. Social response targets are recurring canned
training strings, so response evaluation must not claim unseen-target language
generation. All natural-language fields are UTF-8 bytes, without a tokenizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from models.conversation_intents import ConversationIntent


@dataclass(frozen=True)
class ConversationExample:
    message: bytes
    intent: ConversationIntent
    response: bytes | None = None
    response_category: str | None = None


_SOCIAL = {
    "greeting": b"Hello! What would you like to talk about?",
    "thanks": b"You are welcome! What else can I help with?",
    "wellbeing": b"I am here and ready to talk. How are you?",
}

_CHAT = {
    "train": [
        ("hello", "greeting"), ("hi", "greeting"), ("hey", "greeting"),
        ("good morning", "greeting"), ("good afternoon", "greeting"),
        ("good evening", "greeting"), ("nice to meet you", "greeting"),
        ("hello there", "greeting"), ("hey friend", "greeting"),
        ("thanks", "thanks"), ("thank you", "thanks"), ("thanks for helping", "thanks"),
        ("thank you so much", "thanks"), ("many thanks", "thanks"),
        ("how are you", "wellbeing"), ("how are you doing", "wellbeing"),
        ("are you doing well", "wellbeing"),
    ],
    "validation": [
        ("hi there", "greeting"), ("greetings", "greeting"),
        ("thanks a lot", "thanks"), ("appreciate your help", "thanks"),
        ("how is it going", "wellbeing"),
    ],
    "test": [
        ("hello friend", "greeting"), ("good to meet you", "greeting"),
        ("thank you for your help", "thanks"), ("I appreciate it", "thanks"),
        ("how is your day", "wellbeing"),
    ],
}

_TEMPLATES: dict[str, dict[ConversationIntent, list[str]]] = {
    "train": {
        ConversationIntent.QUESTION: ["what is {topic}", "explain {topic}", "how does {topic} work",
            "tell me about {topic}", "why is {topic} important", "what does {topic} mean",
            "could you explain {topic}", "describe {topic}"],
        ConversationIntent.EDIT: ["replace byte at position {pos} with {byte}",
            "replace byte {pos} with {byte}", "change byte {pos} to {byte}",
            "please edit the document", "fix this typo", "modify this text",
            "delete the first byte", "insert a character into this text", "change this sentence"],
        ConversationIntent.REMEMBER: ["remember my {key} is {value}", "remember that my {key} is {value}",
            "please remember my {key} is {value}", "remember my {key} = {value}"],
        ConversationIntent.RECALL: ["what is my {key}", "do you remember my {key}",
            "tell me my {key}", "what was my {key}"],
        ConversationIntent.FORGET: ["forget my {key}", "please forget my {key}",
            "delete my saved {key}", "remove my saved {key}"],
        ConversationIntent.CLARIFY: ["do it", "make it better", "help me", "I need some help",
            "that is not right", "what next", "could you help", "I am not sure", "maybe",
            "I want something", "can you do that", "make a change", "this needs work",
            "can we try again", "I have a problem", "not like that"],
    },
    "validation": {
        ConversationIntent.QUESTION: ["what can you tell me about {topic}", "how would you describe {topic}",
            "why does {topic} matter"],
        ConversationIntent.EDIT: ["please replace byte number {pos} with {byte}",
            "can you change this document", "correct the misspelling here", "remove the last character"],
        ConversationIntent.REMEMBER: ["could you remember that my {key} is {value}",
            "remember my {key} as {value}"],
        ConversationIntent.RECALL: ["can you tell me my {key}", "do you know my {key}"],
        ConversationIntent.FORGET: ["could you forget my {key}", "erase my saved {key}"],
        ConversationIntent.CLARIFY: ["please handle this", "something is wrong", "I need advice",
            "what do you think", "do the thing", "I am confused"],
    },
    "test": {
        ConversationIntent.QUESTION: ["could you explain {topic} to me", "what do you know about {topic}",
            "give me an explanation of {topic}"],
        ConversationIntent.EDIT: ["please replace byte at position {pos} with {byte}",
            "could you fix this text", "revise this document", "remove the second byte"],
        ConversationIntent.REMEMBER: ["please remember that my {key} is {value}",
            "could you remember my {key} is {value}"],
        ConversationIntent.RECALL: ["please tell me my {key}", "can you recall my {key}"],
        ConversationIntent.FORGET: ["please remove my saved {key}", "can you forget my {key}"],
        ConversationIntent.CLARIFY: ["sort it out", "I want you to help", "something needs to change",
            "what can we do", "make this work", "can you take care of it"],
    },
}

_TOPICS = {
    "train": ["a byte", "gravity", "a computer", "Python", "rain", "memory", "a network", "music", "an atom", "a tree"],
    "validation": ["the ocean", "photosynthesis", "a database", "electricity", "language"],
    "test": ["a planet", "entropy", "a bicycle", "the internet", "a river"],
}
_KEYS = ["name", "favorite color", "favorite food", "project", "preferred language", "city", "pet", "hobby"]
_VALUES = {
    "train": ["Ada", "blue", "pasta", "Orion", "Python", "Dallas", "cat", "painting"],
    "validation": ["Mira", "green", "rice", "Atlas", "Rust", "Paris", "dog", "music"],
    "test": ["Lina", "violet", "tacos", "Nova", "Go", "Oslo", "bird", "cycling"],
}
_NUMBERS = {
    "train": [(0, 65), (1, 90), (2, 33), (3, 32), (4, 97), (5, 98), (9, 100), (12, 101), (16, 255), (64, 0)],
    "validation": [(7, 88), (10, 33), (30, 200)],
    "test": [(6, 44), (11, 71), (15, 128)],
}


def conversation_examples(split: str) -> list[ConversationExample]:
    if split not in _TEMPLATES:
        raise ValueError("split must be train, validation or test")
    rows: dict[bytes, ConversationExample] = {}
    # Case and punctuation vary, but templates themselves are disjoint.
    def variants(text: str) -> list[str]:
        if split == "train":
            return [text, text + ".", text + "!", text[:1].upper() + text[1:] + "?"]
        return [text, text[:1].upper() + text[1:] + "?"]

    def add(text: str, intent: ConversationIntent, response: bytes | None = None,
            category: str | None = None) -> None:
        for version in variants(text):
            message = version.encode("utf-8")
            prior = rows.get(message)
            item = ConversationExample(message, intent, response, category)
            if prior is not None and prior != item:
                raise ValueError(f"conflicting synthetic labels for {message!r}")
            rows[message] = item

    for template, category in _CHAT[split]:
        add(template, ConversationIntent.CHAT, _SOCIAL[category], category)
    for intent, templates in _TEMPLATES[split].items():
        for template in templates:
            if "{topic}" in template:
                replacements = ({"topic": topic} for topic in _TOPICS[split])
            elif "{pos}" in template:
                replacements = ({"pos": pos, "byte": byte} for pos, byte in _NUMBERS[split])
            elif "{value}" in template:
                replacements = ({"key": key, "value": value} for key, value in product(_KEYS, _VALUES[split]))
            elif "{key}" in template:
                replacements = ({"key": key} for key in _KEYS)
            else:
                replacements = ({},)
            for values in replacements:
                add(template.format(**values), intent)
    return list(rows.values())


def response_templates() -> dict[str, bytes]:
    return dict(_SOCIAL)
