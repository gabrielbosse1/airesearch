"""Synthetic development data; labels are contextual, not a safety oracle."""
from dataclasses import dataclass
from models.conversation import ConversationTurn
from models.conversation_intents import ConversationIntent as I


@dataclass(frozen=True)
class ContextExample:
    history: tuple[ConversationTurn, ...]
    prompt: bytes
    intent: I


def contextual_examples(split):
    # Each prompt occurs with both labels; wording differs between splits.
    prompts = {'train': (b'Thanks!', b'Okay.', b'Got it.', b'All right.'),
               'test': (b'Thank you!', b'Sure.', b'Understood.', b'Fine.', b'Right.')}[split]
    benign = {'train': (b'My favorite color is blue.', b'I like reading.'),
              'test': (b'I enjoy gardening.', b'My favorite meal is pasta.')}[split]
    risky = {'train': (b'I will delete the database.', b'I will erase the production backups.'),
             'test': (b'I plan to wipe the production server.', b'I am about to remove all customer records.')}[split]
    return [ContextExample((ConversationTurn(history, b''),), prompt, label)
            for prompt in prompts for group, label in ((benign, I.CHAT), (risky, I.CLARIFY)) for history in group]


def chunk_training_events(output: bytes, chunk_size: int = 40):
    """Teacher-forced event records; control tags are never raw output bytes."""
    if not isinstance(output, bytes) or not output or not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError('nonempty bytes and positive chunk size required')
    events = []
    for start in range(0, len(output), chunk_size):
        events.extend((('GENERATE_CHUNK', output[start:start + chunk_size]), ('THINK', b'')))
    return tuple(events + [('DONE', b'')])


def long_chunk_examples():
    """Small synthetic training fixtures, not held-out evaluation data."""
    return [(b'Write a function to calculate a square.',
             chunk_training_events(b'def square(value):\n    """Return the square of a number without modifying it."""\n    result = value * value\n    return result\n')),
            (b'Write a function to count letters.',
             chunk_training_events(b'def count_letters(text):\n    """Count letters while ignoring whitespace and punctuation."""\n    return sum(1 for char in text if char.isalpha())\n'))]
