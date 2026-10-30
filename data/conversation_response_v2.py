"""Synthetic diverse responses and held-out two-turn fact conversations.

Training responses have varied human-written styles and the neural decoder is
conditioned on intent, current UTF-8 byte input and retrieved episode bytes.
Values in the ten retention fixtures are disjoint from the corresponding Phase
11 training values; this is still a small synthetic, English-only evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random

from models.conversation import ConversationTurn
from models.conversation_intents import ConversationIntent as I


@dataclass(frozen=True)
class ResponseExample:
    message: bytes
    history: tuple[ConversationTurn, ...]
    intent: I
    response: bytes


@dataclass(frozen=True)
class RetentionCase:
    key: bytes
    value: bytes
    statement: bytes
    question: bytes


_KEYS = ("favorite color", "name", "favorite food", "pet", "hometown", "favorite drink", "project", "hobby", "preferred language", "instrument")
_TRAIN_VALUES = {
    "favorite color": ("red", "green", "gold", "orange", "purple", "white", "black", "silver"),
    "name": ("Ada", "Mia", "Nora", "Theo", "Ezra", "Leah", "Arlo", "Nina"),
    "favorite food": ("pasta", "bread", "soup", "pizza", "salad", "rice", "beans", "apples"),
    "pet": ("a cat", "a dog", "a horse", "a turtle", "a bird", "a fish", "a hamster", "a frog"),
    "hometown": ("Paris", "Lyon", "Rome", "Oslo", "Berlin", "Lima", "Cairo", "Dublin"),
    "favorite drink": ("water", "coffee", "juice", "milk", "lemonade", "soda", "smoothies", "cider"),
    "project": ("Orion", "Lumen", "Delta", "Cedar", "Iris", "Cobalt", "Echo", "Willow"),
    "hobby": ("drawing", "music", "painting", "reading", "gardening", "running", "cycling", "baking"),
    "preferred language": ("Python", "Rust", "French", "Italian", "German", "Arabic", "English", "Korean"),
    "instrument": ("piano", "guitar", "flute", "drums", "harp", "cello", "clarinet", "trumpet"),
}
_VALIDATION_VALUES = {
    "favorite color": ("pink", "bronze"), "name": ("Zara", "Luca"), "favorite food": ("tacos", "corn"),
    "pet": ("a parrot", "a gecko"), "hometown": ("Madrid", "Perth"),
    "favorite drink": ("tea", "tonic"), "project": ("Nova", "Pine"), "hobby": ("sewing", "dancing"),
    "preferred language": ("Japanese", "Turkish"), "instrument": ("banjo", "oboe"),
}
# These first cases were examined after the initial decoder and are therefore
# development examples, not part of the final retention score.
_DEVELOPMENT = (
    ("favorite color", "blue", "My favorite color is blue.", "What is my favorite color?"),
    ("name", "Amara", "My name is Amara.", "What is my name?"),
    ("favorite food", "sushi", "My favorite food is sushi.", "What is my favorite food?"),
    ("pet", "a rabbit", "My pet is a rabbit.", "Do you remember my pet?"),
    ("hometown", "Recife", "My hometown is Recife.", "Can you recall my hometown?"),
    ("favorite drink", "cocoa", "My favorite drink is cocoa.", "What is my favorite drink?"),
    ("project", "Atlas", "My project is Atlas.", "What was my project?"),
    ("hobby", "pottery", "My hobby is pottery.", "Do you remember my hobby?"),
    ("preferred language", "Spanish", "My preferred language is Spanish.", "What is my preferred language?"),
    ("instrument", "violin", "My instrument is violin.", "Tell me my instrument."),
)

# Fresh, protected before copy-alignment training/model selection. These values
# differ from the training and response-validation values for the same key.
_TEST = (
    ("favorite color", "teal", "My favorite color is teal.", "What is my favorite color?"),
    ("name", "Imani", "My name is Imani.", "What is my name?"),
    ("favorite food", "ramen", "My favorite food is ramen.", "What is my favorite food?"),
    ("pet", "a lizard", "My pet is a lizard.", "Do you remember my pet?"),
    ("hometown", "Quito", "My hometown is Quito.", "Can you recall my hometown?"),
    ("favorite drink", "matcha", "My favorite drink is matcha.", "What is my favorite drink?"),
    ("project", "Zephyr", "My project is Zephyr.", "What was my project?"),
    ("hobby", "sculpting", "My hobby is sculpting.", "Do you remember my hobby?"),
    ("preferred language", "Portuguese", "My preferred language is Portuguese.", "What is my preferred language?"),
    ("instrument", "saxophone", "My instrument is saxophone.", "Tell me my instrument."),
)

_SOCIAL_PROMPTS = ["Hi", "Hello there", "Hey friend", "Good morning", "Nice to see you",
                   "Thanks", "Thank you so much", "How are you", "Good evening", "Hi, how's it going"]
_SOCIAL_RESPONSES = ["Hello! What would you like to talk about?", "Hi there, what is on your mind?",
                     "Hey! How can I help you today?", "Good to see you. What shall we discuss?",
                     "I am here. What would you like to explore?", "Hello! I am listening."]
_ACKNOWLEDGEMENTS = ["Thanks for sharing that.", "Good to know, thanks for telling me.",
                     "I appreciate you sharing that.", "Got it, thanks for letting me know."]
_CLARIFY_PROMPTS = ["Fix this", "Tell me about it", "Do it", "Make it better", "Explain that",
                    "What happened there", "Help with this", "Could you handle it", "Change that", "What does that mean"]
_CLARIFY_RESPONSES = ["Could you give me a little more detail?", "What would you like me to do exactly?",
                      "Can you tell me which part you mean?", "Could you clarify what you need?",
                      "I am not sure what you mean. Can you explain?", "What should I focus on?"]
_UNCERTAIN_RESPONSES = ["I am not sure about that. Could you share more context?", "I don't know enough to answer that yet.",
                        "I cannot give a reliable answer to that right now.", "I am unsure. What more can you tell me?"]
_ACTION_RESPONSES = {
    I.EDIT: ["All set, I made that change.", "Done, the requested byte is updated.", "Got it, I made the edit.", "The byte has been changed."] ,
    I.REMEMBER: ["I will keep that in mind.", "Got it, I will remember that.", "Thanks, I have saved that detail.", "Okay, I will remember it."] ,
    I.FORGET: ["I removed the saved detail.", "Done, I have forgotten that.", "Okay, I cleared that memory.", "That saved detail has been removed."] ,
}


def retention_cases() -> list[RetentionCase]:
    return [RetentionCase(key.encode(), value.encode(), statement.encode(), question.encode())
            for key, value, statement, question in _TEST]


def developmental_retention_cases() -> list[RetentionCase]:
    """Inspected initial fixtures, including the user-requested blue example."""
    return [RetentionCase(key.encode(), value.encode(), statement.encode(), question.encode())
            for key, value, statement, question in _DEVELOPMENT]


def copy_generalization_examples(*, count: int = 400, seed: int = 119011) -> list[ResponseExample]:
    """Synthetic byte strings with supervised source positions, train split only.

    The strings are deliberately unlike the protected development/validation/
    test values. Alignment targets are derived from these training examples,
    never from a held-out answer at inference.
    """
    if not isinstance(count, int) or not 0 <= count <= 10000:
        raise ValueError("invalid synthetic example count")
    rng = random.Random(seed)
    syllables = ("ba", "ke", "ri", "mo", "su", "ta", "ve", "li", "do", "na", "fi", "po", "za", "gu", "we", "ya")
    protected = {case.value.lower() for case in retention_cases() + developmental_retention_cases()}
    protected |= {value.encode().lower() for values in _VALIDATION_VALUES.values() for value in values}
    examples: list[ResponseExample] = []
    for index in range(count):
        key = _KEYS[index % len(_KEYS)]
        while True:
            value = "".join(rng.choice(syllables) for _ in range(rng.randint(2, 5)))
            if index % 5 == 0:
                value = "a " + value
            if value.encode().lower() not in protected:
                break
        statement = f"My {key} is {value}.".encode()
        history = (ConversationTurn(statement, b"Thanks for sharing."),)
        question = rng.choice((f"What is my {key}?", f"Do you remember my {key}?", f"Tell me my {key}."))
        response = rng.choice((f"You told me {value}.", f"It is {value}.", f"You said {value}.", f"Your {key} is {value}."))
        examples.append(ResponseExample(question.encode(), history, I.RECALL, response.encode()))
    return examples


def response_examples(split: str) -> list[ResponseExample]:
    if split not in ("train", "validation"):
        raise ValueError("response training has train/validation splits; retention_cases are test")
    rows: list[ResponseExample] = []
    values_by_key = _TRAIN_VALUES if split == "train" else _VALIDATION_VALUES
    questions = ("What is my {key}?", "Do you remember my {key}?", "Tell me my {key}.") if split == "train" else (
        "Can you recall my {key}?", "What was my {key}?")
    answer_forms = ("Your {key} is {value}.", "It is {value}.", "You said {value}.",
                    "You told me {value}.", "I remember {value}.")
    for key in _KEYS:
        for value, question, answer in product(values_by_key[key], questions, answer_forms):
            statement = f"My {key} is {value}.".encode("utf-8")
            # The response to the preceding user statement may vary in reality.
            history = (ConversationTurn(statement, b"Thanks for sharing."),)
            rows.append(ResponseExample(question.format(key=key).encode(), history, I.RECALL,
                                        answer.format(key=key, value=value).encode()))
        if split == "train":
            for value, answer in product(values_by_key[key], _ACKNOWLEDGEMENTS):
                rows.append(ResponseExample(f"My {key} is {value}.".encode(), (), I.CHAT, answer.encode()))
    if split == "train":
        for prompt, response in product(_SOCIAL_PROMPTS, _SOCIAL_RESPONSES):
            rows.append(ResponseExample(prompt.encode(), (), I.CHAT, response.encode()))
        for prompt, response in product(_CLARIFY_PROMPTS, _CLARIFY_RESPONSES):
            rows.append(ResponseExample(prompt.encode(), (), I.CLARIFY, response.encode()))
        for prompt, response in product(("What is gravity?", "Who wrote Hamlet?", "Explain entropy.", "What is music?"), _UNCERTAIN_RESPONSES):
            rows.append(ResponseExample(prompt.encode(), (), I.QUESTION, response.encode()))
        action_prompts = {I.EDIT: ("Replace byte 1 with 100.", "Change byte 4 to 90."),
                          I.REMEMBER: ("Remember my name is Ada.", "Remember my hobby is music."),
                          I.FORGET: ("Forget my name.", "Forget my hobby.")}
        for intent, prompts in action_prompts.items():
            for prompt, response in product(prompts, _ACTION_RESPONSES[intent]):
                rows.append(ResponseExample(prompt.encode(), (), intent, response.encode()))
    else:
        for prompt, response in product(("Greetings", "Hello friend", "Good afternoon"), _SOCIAL_RESPONSES):
            rows.append(ResponseExample(prompt.encode(), (), I.CHAT, response.encode()))
        for prompt, response in product(("Could you help with this?", "Tell me what it was."), _CLARIFY_RESPONSES):
            rows.append(ResponseExample(prompt.encode(), (), I.CLARIFY, response.encode()))
        for prompt, response in product(("What is astronomy?", "How does a river flow?"), _UNCERTAIN_RESPONSES):
            rows.append(ResponseExample(prompt.encode(), (), I.QUESTION, response.encode()))
    return rows
