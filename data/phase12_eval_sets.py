"""Predeclared Phase 12 intent, vector-memory and response fixtures.

These small synthetic English fixtures are protected before Phase 12 training.
The public blue and Alice examples are demonstrations, not hidden tests. Exact
phrases and values are disjoint from the new training split; common grammar and
question wording can recur, so these are not open-domain evaluations.
"""
from __future__ import annotations

from dataclasses import dataclass

from models.conversation_intents import ConversationIntent as I


_FINAL_INTENTS = {
    I.CHAT: ["Hello, I am glad to see you.", "Hey, thanks for joining me.", "Good afternoon, pal.",
             "Thanks for staying with me.", "It is lovely to chat today.", "Hello, hope you are well.",
             "My favorite fruit is plum.", "My hobby is sailing.", "My hometown is Bilbao.", "My dog is named Sunny."],
    I.QUESTION: ["What is a comet?", "How does a solar panel work?", "Explain how magnets attract.",
                 "Why do leaves fall in autumn?", "Who invented the printing press?", "What does a database do?",
                 "Tell me about alpine flowers.", "Can you describe a volcano?", "How is cheese made?", "What is an ecosystem?"],
    I.EDIT: ["Replace byte 22 with 84.", "Change byte 17 to 112.", "Delete the final word from the note.",
             "Fix the typo on line 31.", "Remove the comma from the title.", "Insert a period at the end of the sentence.",
             "Move the third paragraph above the second.", "Replace the word 'small' with 'tiny' in the letter.",
             "Correct the spelling on line 19.", "Edit line 21 of the draft."],
    I.REMEMBER: ["Remember my nickname is Finch.", "Please remember my hometown is Bilbao.",
                 "Remember that my favorite fruit is plum.", "Remember my favorite drink is chai.",
                 "Remember my team is West.", "Remember my instrument is accordion.",
                 "Remember that my pet is Sunny.", "Please remember my editor is Emacs.",
                 "Remember my school is Ashwood.", "Remember my favorite season is spring."],
    I.RECALL: ["Please tell me my nickname.", "Do you know my hometown?", "What is my favorite fruit?",
               "Can you recall my favorite drink?", "What was my team?", "What was my instrument?",
               "Do you remember my editor?", "What is my school?", "Can you tell me my favorite season?", "What is my animal?"],
    I.FORGET: ["Please forget my alias.", "Can you forget my hometown?", "Forget my favorite fruit.",
               "Please remove my saved preferred drink.", "Could you forget my team?", "Delete my saved instrument.",
               "Can you forget my editor?", "Forget my school.", "Please remove my saved favorite season.", "Erase my saved animal."],
    I.CLARIFY: ["Could you sort this problem?", "I would like you to do it.", "Please tell me about that thing.",
                "What did the person say?", "Can you improve it for me?", "Go ahead and change that.",
                "Would you handle this?", "What do we do about it?", "I need help with something.", "Please explain the matter."],
}


@dataclass(frozen=True)
class MemoryCase:
    key: bytes
    value: bytes
    statement: bytes
    question: bytes


# Phase 11's inspected retention cases are not reused as fresh final cases.
_MEMORY = (
    ("favorite color", "indigo", "What is my favorite color?"),
    ("name", "Soraya", "What is my name?"),
    ("favorite food", "mango", "What is my favorite food?"),
    ("pet", "a ferret", "Do you remember my pet?"),
    ("hometown", "Valencia", "Can you recall my hometown?"),
    ("favorite drink", "rooibos", "What is my favorite drink?"),
    ("project", "Helios", "What was my project?"),
    ("hobby", "origami", "Do you remember my hobby?"),
    ("preferred language", "Swahili", "What is my preferred language?"),
    ("instrument", "mandolin", "Tell me my instrument."),
)


def intent_test_examples() -> list[tuple[bytes, I]]:
    return [(message.encode("utf-8"), intent) for intent, messages in _FINAL_INTENTS.items() for message in messages]


def memory_test_cases() -> list[MemoryCase]:
    return [MemoryCase(key.encode(), value.encode(), f"My {key} is {value}.".encode(), question.encode())
            for key, value, question in _MEMORY]


# Test prompts are never written into the response-model training dataset.
DIVERSITY_TEST_PROMPT = b"Good to see you again."
DIVERSITY_VALIDATION_PROMPTS = (b"Hello and welcome.", b"Good afternoon, friend.")
