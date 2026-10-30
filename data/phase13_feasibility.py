"""Predeclared synthetic feasibility fixtures; not a benchmark of safety or coding."""
from models.conversation import ConversationTurn
from models.conversation_intents import ConversationIntent as I
from data.phase13_examples import chunk_training_events

TRAIN_PROMPTS = ('Thanks', 'Okay', 'Understood clearly', 'Right', 'Fine', 'Noted', 'All right', 'I see clearly', 'Got it', 'Sure',
                 'Very well', 'I understand', 'That sounds acceptable', 'Okay then', 'Good', 'Agreed', 'I follow', 'Yes indeed',
                 'Appreciated indeed', 'Thank you', 'That helps', 'I hear you', 'Alright then noted', 'Fair enough', 'Okay thanks')
VALID_PROMPTS = ('Much appreciated', 'Thanks kindly', 'I understand now', 'That is clear', 'Certainly',
                 'Sounds fine', 'I got that', 'I see now', 'Alright then', 'Thanks again')
TRAIN_BENIGN = ('I planted flowers.', 'I enjoy hiking.', 'I enjoy woodworking.', 'I made a cake.', 'I like jazz.')
TRAIN_RISK = ('I will delete the database.', 'I plan to erase the backups.', 'I will wipe the server.',
              'I plan to remove customer records.', 'I will destroy the only archive.')
VALID_BENIGN = ('I enjoy cooking.', 'I collected seashells.', 'My hobby is cycling.')
VALID_RISK = ('I will remove production logs.', 'I am about to delete all account records.', 'I plan to erase the storage volume.')


def routing_examples(split):
    prompts, benign, risky = ((TRAIN_PROMPTS, TRAIN_BENIGN, TRAIN_RISK) if split == 'train' else
                              (VALID_PROMPTS, VALID_BENIGN, VALID_RISK) if split == 'validation' else (None, None, None))
    if prompts is None: raise ValueError('unknown split')
    return [(prompt.encode(), label, (ConversationTurn(history.encode(), b'I hear you.'),))
            for i, prompt in enumerate(prompts)
            for history, label in ((benign[i % len(benign)], I.CHAT), (risky[i % len(risky)], I.CLARIFY))]


def chunk_examples(split):
    numbers = range(20) if split == 'train' else range(20, 25) if split == 'validation' else None
    if numbers is None: raise ValueError('unknown split')
    rows=[]
    for i in numbers:
        prompt = f'Write Python function scale_{i:02d} that multiplies its argument by {i + 2}.'.encode()
        body = f'def scale_{i:02d}(value):\n    """Scale the input number by {i + 2}."""\n    result = value * {i + 2}\n    return result\n'.encode()
        # Valid Python comment padding keeps every target exactly 120 bytes.
        target = body + b'# ' + b'x' * (120-len(body)-3) + b'\n'
        assert len(target) == 120
        rows.append((prompt, target, chunk_training_events(target, 30)))
    return rows
