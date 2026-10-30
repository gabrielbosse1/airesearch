"""Small predeclared validation/test sets for Phase 11 intent evaluation.

The test is held back from epoch selection and from all Phase 11 training.
These manually authored English examples are not a representative population.
"""
from __future__ import annotations

from models.conversation_intents import ConversationIntent as I


_VALIDATION = {
    I.CHAT: ["Greetings, nice to meet you.", "Thanks for helping me today.", "Hello, good to see you.", "My favorite sport is tennis."],
    I.QUESTION: ["How does a wind turbine work?", "Explain what a galaxy is.", "What is an algorithm?", "Why do clouds form?"],
    I.EDIT: ["Fix the error on line 18.", "Replace byte 10 with 80.", "Delete the third word from the article.", "Change the first letter to M."],
    I.REMEMBER: ["Remember my nickname is Mica.", "Please remember my hometown is Lyon.", "Remember my team is North.", "Remember that my favorite drink is juice."],
    I.RECALL: ["What is my nickname?", "Tell me my hometown.", "Can you recall my team?", "Do you remember my favorite drink?"],
    I.FORGET: ["Forget my nickname.", "Delete my saved hometown.", "Please forget my team.", "Remove my saved favorite drink."],
    I.CLARIFY: ["Can you sort this out?", "I'd like you to fix something.", "Tell me what it was.", "Please handle the issue."],
}

_TEST = {
    I.CHAT: ["Hello, it is good to see you.", "Hey, nice to chat with you.", "Good morning, friend!", "Thank you for listening.",
             "Thanks for your patience.", "Hello there, how are things?", "My favorite sport is rowing.", "My pet is called Dingo.",
             "My hometown is Bergen.", "My favorite color is amber."],
    I.QUESTION: ["What is a satellite?", "How does a submarine work?", "Explain quantum energy.", "Why do oceans have tides?",
                 "Who discovered penicillin?", "What does a compiler do?", "Tell me about migratory birds.",
                 "Can you explain the water cycle?", "What causes an earthquake?", "Where do penguins live?"],
    I.EDIT: ["Replace byte 8 with 72.", "Change byte 11 to 45.", "Fix the typo at position 21.",
             "Delete the first paragraph from the report.", "Remove the word 'really' from the sentence.",
             "Insert a comma after the first word.", "Correct the spelling on line 15.",
             "Move the second line below the third.", "Replace byte 7 with 120.", "Edit line 8 of the draft."],
    I.REMEMBER: ["Remember my nickname is River.", "Remember my hometown is Bergen.", "Remember my favorite sport is rowing.",
                 "Remember my favorite drink is cocoa.", "Remember my team is East.", "Remember my instrument is violin.",
                 "Remember that my pet is Dingo.", "Please remember my editor is Vim.",
                 "Remember my school is Cedar.", "Remember my favorite season is winter."],
    I.RECALL: ["What is my alias?", "What is my hometown?", "What is my favorite sport?", "What is my favorite drink?",
               "Do you remember my team?", "Tell me my instrument.", "Can you recall my editor?",
               "What was my school?", "Do you know my favorite season?", "What is my favorite animal?"],
    I.FORGET: ["Forget my alias.", "Forget my hometown.", "Forget my favorite sport.", "Forget my favorite drink.",
               "Delete my saved club.", "Remove my saved instrument.", "Please forget my animal.",
               "Could you forget my editor?", "Erase my saved college.", "Can you forget my favorite season?"],
    I.CLARIFY: ["Could you fix this part?", "I need you to do something.", "Tell me about the thing.",
                "What did they mention?", "Please improve that.", "Change it for me.", "Can you deal with this?",
                "What should we try next?", "I'd like help with something.", "Could you explain it a bit?"],
}


def overall_intent_examples(split: str) -> list[tuple[bytes, I]]:
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    data = _VALIDATION if split == "validation" else _TEST
    return [(message.encode("utf-8"), intent) for intent, messages in data.items() for message in messages]
