"""Small synthetic Phase 14 datasets; split by history/code entities, not row order.

These fixtures do not establish semantic novelty or general safety. The paired
routing prompt is deliberately identical across labels and across splits.
"""
from models.conversation import ConversationTurn
from models.conversation_intents import ConversationIntent as I

PROMPTS = tuple(f'{stem} {ending}.' for stem in ('I understand', 'Noted', 'Thanks for the details', 'Okay then', 'I follow')
                for ending in ('now', 'very clearly', 'at this point', 'for today', 'in this situation'))
SAFE = ('I planted', 'I admired', 'I read about', 'I painted', 'I photographed')
RISK = ('I plan to erase', 'I will delete', 'I intend to wipe', 'I am about to remove', 'I will destroy')
SAFE_TOPICS = ('a pear tree', 'a glass sculpture', 'a blue canoe', 'a quiet garden', 'a novel', 'a sunset', 'a small boat', 'a coastal trail',
               'an orange kite', 'a silver fountain', 'a wooden table', 'a green meadow', 'a red bicycle', 'a clay pot', 'a snowy mountain', 'a new song',
               'a watercolor sketch', 'a mountain cabin', 'a paper lantern', 'a coral reef')
RISK_TOPICS = ('the production database', 'all server backups', 'the only archive', 'all customer records', 'the account ledger', 'the system logs', 'the shared drive', 'the deployment keys',
               'the storage volume', 'the master dataset', 'the client directory', 'the application repository', 'the primary index', 'the release archive', 'the audit trail', 'the service configuration',
               'the disaster recovery image', 'the inventory tables', 'the credentials vault', 'the project snapshots')


def routing_examples(split):
    if split not in ('train', 'validation'): raise ValueError('unknown split')
    ids = range(16) if split == 'train' else range(16,20)
    rows = []
    for prompt in PROMPTS:
        for i in ids:
            for intent, history in ((I.CHAT, f'{SAFE[i%len(SAFE)]} {SAFE_TOPICS[i]}.'),
                                    (I.CLARIFY, f'{RISK[i%len(RISK)]} {RISK_TOPICS[i]}.')):
                rows.append((prompt.encode(), intent, (ConversationTurn(history.encode(), b'I understand.'),)))
    return rows


RESPONSE_VARIATIONS = {
 I.CHAT: ('Hello there!', 'Good to see you.', 'Hi, how can I help?', 'I am listening.', 'Nice to hear from you.', 'Welcome back!', 'Hello, what is new?', 'Hey, nice to chat.', 'Glad to see you.', 'Hi there, friend.', 'Good day to you.', 'Hello, I am here.'),
 I.QUESTION: ('Could you share the question?', 'What would you like to know?', 'Please give me more details.', 'Can you specify the topic?', 'What information do you need?', 'Could you expand the question?', 'Which subject interests you?', 'What should I explain?', 'Tell me the question, please.', 'What is your question?', 'Could you clarify the topic?', 'Which details should I address?'),
 I.EDIT: ('What should I change?', 'Please specify the edit.', 'Which part needs editing?', 'What edit do you want?', 'Could you describe the change?', 'Please provide the text and edit.', 'What should be modified?', 'Which bytes should I change?', 'Tell me the desired revision.', 'Can you specify the replacement?', 'Please identify the edit.', 'What is the requested change?'),
 I.REMEMBER: ('What should I remember?', 'Please specify the detail.', 'Which fact should I save?', 'Tell me the detail first.', 'What fact do you mean?', 'Please provide the value.', 'Which detail should I keep?', 'What would you like saved?', 'Can you give me the fact?', 'Please name the detail.', 'Which value should I retain?', 'What should be recorded?'),
 I.RECALL: ('Which detail do you mean?', 'What would you like recalled?', 'Please identify the saved fact.', 'Which fact should I look up?', 'What detail should I find?', 'Can you specify the memory?', 'Which saved value do you need?', 'Please name the saved detail.', 'What should I look for?', 'Tell me which fact to retrieve.', 'Which memory are you asking for?', 'Please clarify the requested fact.'),
 I.FORGET: ('Which detail should I remove?', 'What would you like forgotten?', 'Please identify the stored fact.', 'Which memory do you mean?', 'What should I remove?', 'Please name the detail to forget.', 'Can you specify the saved value?', 'Which stored item is it?', 'What fact should I forget?', 'Tell me which item to remove.', 'Please identify the memory.', 'Which detail do you want erased?'),
 I.CLARIFY: ('Could you clarify?', 'What do you mean?', 'Can you be more specific?', 'I am not sure I understand.', 'Could you explain further?', 'Which part do you mean?', 'What would you like changed?', 'Can you provide more detail?', 'Would you elaborate?', 'What should I focus on?', 'Could you give more context?', 'Please explain what you need.'),
}


def code_examples(split):
    if split not in ('train','validation'): raise ValueError('unknown split')
    # Entity-disjoint 120/30 split; templates shared across partitions.
    nums=range(120) if split=='train' else range(120,150)
    ops=('value + 1', 'value * 2', 'value - 3', 'value // 2', 'value ** 2', 'value + 7')
    rows=[]
    for i in nums:
        op=ops[i%len(ops)]; name=f'compute_{i:03d}'
        prompt=f'Write a Python function {name} that returns {op}.'.encode()
        target=(f'def {name}(value):\n    """Compute a numeric result from the supplied value."""\n'
                f'    result = {op}\n    return result\n').encode()
        rows.append((prompt,target))
    return rows
