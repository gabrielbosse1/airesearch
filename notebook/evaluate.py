"""Free-running smoke probes. These are diagnostics, not proof of capability.

Mineflayer probes ask the model for player actions, extract the ``Action:``
block and validate it with the static checker, ``node --check`` and the mock
bot. A pass means "the mock accepted this call and it matched the request",
not "the model can play Minecraft".
"""
import re
import torch

from notebook.phone_model import generate
from notebook.tools import reward
from notebook.test_mineflayer import TEST_CASES, extract_action, validate_code

CASES = [
 ('pt-br', 'Request: Olá, como você está?\nAnswer: '),
 ('code', 'Request: Write a Python function to sort a list\nCode: '),
 ('search', 'Request: What is the weather in São Paulo?\nAction: '),
 ('edit', 'Request: Corrija: voce\nDraft: voce\n'),
 ('memory', 'Memory:\nMeu animal favorito: capivara\nRequest: Qual é meu animal favorito?\nAnswer: '),
]
MINEFLAYER_CASES = 4


def _mineflayer_rows(model, max_new):
    rows = []
    for case in TEST_CASES[:MINEFLAYER_CASES]:
        prompt = f"Request: {case['request']}\nThought: "
        output = generate(model, prompt, max_new=max_new)
        action = extract_action(output)
        if action:
            report = validate_code(action, case['mock'], case['expect'])
        else:
            report = {'ok': False, 'issues': ['no Action block'], 'syntax': None,
                      'expectation': None}
        rows.append({'case': 'mineflayer-' + case['name'], 'pass': bool(report['ok']),
                     'output': output, 'action': action, 'issues': report['issues'],
                     'syntax': report['syntax'], 'expectation': report['expectation']})
    return rows


def evaluate(model, max_new=128):
    rows = []
    for kind, prompt in CASES[:2]:
        output = generate(model, prompt, max_new=max_new)
        if kind == 'pt-br':
            ok = bool(re.search(r'\b(estou|bem|olá|obrigad[oa]|você)\b', output.lower()))
        else:
            ok = bool(reward('code', output))
        rows.append({'case': kind, 'pass': bool(ok), 'output': output})
    with torch.no_grad():
        rows.extend(_mineflayer_rows(model, max_new))
    for kind, prompt in CASES[2:]:
        output = generate(model, prompt, max_new=max_new)
        if kind == 'search':
            ok = 'search_ddg(' in output
        elif kind == 'edit':
            ok = bool(reward('edit', output, 'voce', 'você'))
        else:
            ok = 'capivara' in output.lower()
        rows.append({'case': kind, 'pass': bool(ok), 'output': output})
    return rows
