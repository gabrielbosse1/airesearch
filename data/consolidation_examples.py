"""Synthetic session fixtures. Test wording is predeclared and disjoint from new train wording."""
from models.conversation import ConversationTurn
from models.conversation_intents import ConversationIntent as I
from data.phase12_eval_sets import intent_test_examples

TRAIN_HISTORY = (
    (ConversationTurn(b'Hello.', b'Hello, how can I help?'),),
    (ConversationTurn(b'What is the time?', b'I do not have a clock.'),),
    (ConversationTurn(b'I enjoy reading.', b'That sounds nice.'),),
    (ConversationTurn(b'Good afternoon.', b'Welcome!'), ConversationTurn(b'How are you?', b'Ready to help.')),
)
VAL_HISTORY = (ConversationTurn(b'I like exploring parks.', b'That sounds enjoyable.'),)
TEST_HISTORY = (ConversationTurn(b'Could you help me earlier?', b'Yes, I can help.'),)
TRAIN_GREETINGS = (
    'Hi', 'Hello', 'Hey', 'Good morning', 'Good evening', 'Good afternoon',
    'Hi there', 'Hello there', 'Hey there', 'Greetings', 'Hello friend',
    'Hi friend', 'Thanks', 'Thank you', 'Many thanks', 'Thanks so much',
    'Okay', 'Got it', 'Understood', 'All right', 'Hello how are you',
    'Hey what is up', 'Nice to meet you', 'Glad to chat',
)
VAL_GREETINGS = ('Howdy', 'Hello again', 'Thanks for that', 'Okay I understand', 'Good to see you')
TEST_GREETINGS = (
    'Hi there!', 'Hello, how are you?', "Hey, what's up?", 'Good morning!',
    'Thanks!', 'Okay, got it.', 'Hello, good to see you!', 'Hey friend!',
    'Thank you for helping!', 'Good evening!', 'Hello again!', 'Hi, nice to see you.',
    'Much appreciated!', 'Morning, how are things?', 'Hey, how is your day?',
    'Thank you very much!', 'Good afternoon!', 'Hi, pleased to meet you.',
    'Okay, I understand.', 'Thanks for your help!',
)


def greeting_train():
    return [(text.encode(), I.CHAT, TRAIN_HISTORY[i % len(TRAIN_HISTORY)])
            for i, text in enumerate([base + suffix for base in TRAIN_GREETINGS for suffix in ('', '!', '.', '?')])]


def greeting_validation():
    return [(text.encode(), I.CHAT, VAL_HISTORY) for text in VAL_GREETINGS]


def greeting_test():
    return [(text.encode(), I.CHAT, TEST_HISTORY) for text in TEST_GREETINGS]


def session_test():
    # Previously inspected Phase 12 intent test: development regression, not pristine.
    pairs = intent_test_examples()
    selected = [item for i, item in enumerate(pairs) if i % 10 < 7] + [pairs[7]]
    return [(msg, intent, TEST_HISTORY) for msg, intent in selected]


RESPONSE_VARIATIONS = {
    'confirmation': ('Got it.', 'Understood.', 'Noted.', 'Okay.', 'Sure.', 'Perfect.', 'All set.', 'Done.', 'Remembered.', 'Stored.'),
    'question_answer': ('Based on what you said,', 'From our conversation,', 'You mentioned,', 'Earlier you told me,', 'As you noted,', 'You said,', 'Previously you shared,', 'From what you described,', 'In your earlier message,', 'According to your message,'),
    'clarification': ('Could you clarify?', 'What do you mean?', 'Can you be more specific?', 'I am not sure I understand.', 'Could you explain further?', 'Which part do you mean?', 'What would you like changed?', 'Can you provide more detail?', 'Would you elaborate?', 'What should I focus on?'),
}
