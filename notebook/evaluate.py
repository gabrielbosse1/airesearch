"""Free-running smoke probes. These are diagnostics, not proof of capability."""
import ast
import re
from notebook.phone_model import generate
from notebook.tools import Memory, minecraft_valid, reward

CASES = [
 ('pt-br', 'Request: Olá, como você está?\nAnswer: '),
 ('code', 'Request: Write a Python function to sort a list\nCode: '),
 ('minecraft', 'Request: Build a house at 0 64 0\nAction: '),
 ('search', 'Request: What is the weather in São Paulo?\nAction: '),
 ('edit', 'Request: Corrija: voce\nDraft: voce\n'),
 ('memory', 'Memory:\nMeu animal favorito: capivara\nRequest: Qual é meu animal favorito?\nAnswer: '),
]

def evaluate(model, max_new=128):
    rows = []
    for kind, prompt in CASES:
        output = generate(model, prompt, max_new=max_new)
        if kind == 'pt-br': ok = bool(re.search(r'\b(estou|bem|olá|obrigad[oa]|você)\b', output.lower()))
        elif kind == 'code': ok = bool(reward('code', output))
        elif kind == 'minecraft': ok = minecraft_valid(output)
        elif kind == 'search': ok = 'search_ddg(' in output
        elif kind == 'edit': ok = bool(reward('edit', output, 'voce', 'você'))
        else: ok = 'capivara' in output.lower()
        rows.append({'case': kind, 'pass': bool(ok), 'output': output})
    return rows
