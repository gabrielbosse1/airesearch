"""Synthetic raw-byte CHAT supervision, with one learned latent style per reply.

The response texts recur across prompts by design. This small data set teaches
surface variation, not open-domain semantic competence or safety.
"""
from __future__ import annotations

RESPONSES = (
    "Thanks for sharing. Want to say more?",
    "I hear you. How are you feeling about it?",
    "That makes sense. What would help now?",
    "Feel free to tell me more.",
    "What's been on your mind?",
    "How has the day been for you?",
    "I'm here. Would you like to talk more?",
    "Sure, let's chat. What's on your mind?",
    "Tell me a little more if you'd like.",
    "It's good to hear from you. What's up?",
    "Would you like to share more about that?",
    "Go on, I'm listening.",
    "Thanks for telling me. How can I help?",
    "Let's take it one step at a time.",
    "I'm listening. What would be useful?",
    "You can share as much as you like.",
    "What would you like to talk about?",
    "Happy to listen. What comes next?",
    "Take your time. I'm here to listen.",
    "How can I support you right now?",
    "I appreciate you sharing that.",
    "Would it help to talk this through?",
    "All right. What are you thinking about?",
    "That sounds important to you.",
    "Can you tell me a little more?",
    "I'm glad you told me. What's next?",
    "I see. Want to expand on that?",
    "Thanks for opening up. What's on your mind?",
    "No rush. What would you like to share?",
    "Let's talk. Where would you like to start?",
    "Of course. Tell me what's on your mind.",
    "I'm here with you. What feels important?",
)

TRAIN_PROMPTS = (
    "Hello, I'd like a chat.", "Hey there, can we talk?", "I'd like to say hello.",
    "Hi, I'm around for a while.", "Hello, how are things?", "Hey, I'm here again.",
    "I wanted to check in.", "I'm just stopping by to chat.", "Can we chat for a minute?",
    "I'd like to talk about my day.", "Hi, I have a few thoughts.", "Hello again, how's it going?",
    "I'm here and ready to chat.", "I wanted to talk with you.", "I've got a moment to talk.",
    "Hi there, what are you up to?", "Let's have a little conversation.", "Hey, want to talk?",
    "Today has been interesting.", "I have something on my mind.", "This day has been a lot.",
    "I'd like someone to listen.", "It's been a long day.", "Hi, I could use a chat.",
    "Hello, I'm feeling thoughtful.", "I wanted to share something.", "May I talk with you?",
    "Hey, are you around?", "I have a few minutes to chat.", "Hello, let's talk for a bit.",
    "I have some time to talk.", "I was thinking about my day.",
)


def response_bytes() -> tuple[bytes, ...]:
    return tuple(response.encode("utf-8") for response in RESPONSES)


def training_prompts() -> tuple[bytes, ...]:
    return tuple(prompt.encode("utf-8") for prompt in TRAIN_PROMPTS)
