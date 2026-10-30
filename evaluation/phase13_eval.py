"""Phase 13 development benchmark: session routing and long byte generation.

Targets are only used for scoring. This is not a held-out safety evaluation.
Run memory_eval separately to include its fresh-process and 500-turn checks.
"""
import argparse
import ast
import json
from pathlib import Path
import torch
from data.phase13_examples import contextual_examples
from evaluation.memory_eval import _load, _session
from models.chunked_decoder import generate_diverse_in_chunks
from models.conversation_intents import INTENT_NAMES

TASKS = [
 ('reverse a string', 'def reverse_string(text):\n    """Return the input string in reverse order."""\n    reversed_text = text[::-1]\n    return reversed_text\n'),
 ('sum a list', 'def sum_list(values):\n    """Return the sum of a sequence of numbers."""\n    total = 0\n    for value in values:\n        total += value\n    return total\n'),
 ('find the maximum', 'def find_maximum(values):\n    """Return the largest value in a nonempty sequence."""\n    if not values:\n        raise ValueError("empty sequence")\n    return max(values)\n'),
 ('count words', 'def count_words(text):\n    """Count whitespace-separated words in the input text."""\n    words = text.split()\n    return len(words)\n'),
 ('check a palindrome', 'def is_palindrome(text):\n    """Check whether text reads the same in both directions."""\n    return text == text[::-1]\n'),
 ('filter even numbers', 'def even_numbers(values):\n    """Return only even numbers from the given iterable."""\n    return [value for value in values if value % 2 == 0]\n'),
 ('compute factorial', 'def factorial(n):\n    """Compute the factorial of a nonnegative integer."""\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n'),
 ('remove duplicates', 'def unique_values(values):\n    """Remove duplicate hashable items while preserving order."""\n    return list(dict.fromkeys(values))\n'),
 ('calculate mean', 'def mean(values):\n    """Return the arithmetic mean of a nonempty sequence."""\n    if not values:\n        raise ValueError("empty sequence")\n    return sum(values) / len(values)\n'),
 ('count vowels', 'def count_vowels(text):\n    """Count English vowels in text without regard to case."""\n    return sum(1 for char in text.lower() if char in "aeiou")\n'),
]


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j-1] + (x != y)))
        row = nxt
    return row[-1]


def evaluate(args):
    torch.set_num_threads(1)
    components = _load(args)
    model, _, _, _, decoder = components
    rows = []
    for case in contextual_examples('test'):
        session, _ = _session(components, b'phase13-evaluation')
        session.max_history_turns = 2
        session.history = list(case.history)
        result = session.respond(case.prompt)
        expected = INTENT_NAMES[int(case.intent)]
        rows.append({'history': case.history[0].user.decode(), 'prompt': case.prompt.decode(),
                     'expected': expected, 'predicted': result.intent, 'action': result.action,
                     'correct': result.intent == expected})
    gen = []
    for task, reference in TASKS:
        prompt = ('Write a Python function to ' + task + '.').encode()
        ids = torch.tensor([list(b'U:' + prompt)], dtype=torch.long)
        lengths = torch.tensor([ids.shape[1]])
        with torch.no_grad():
            context = model.conversation_features(ids, lengths, focus_start=torch.tensor([2]))
        # Accept-all is intentionally not a coherence verifier; report syntax separately.
        result = generate_diverse_in_chunks(decoder, context, lambda _prefix, _chunk: True,
                                           target_length=200, chunk_size=40, max_chunks=10)
        try:
            tree = ast.parse(result.output.decode('utf-8'))
            structural = len(result.output) >= 100 and any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in tree.body)
        except (UnicodeDecodeError, SyntaxError, ValueError):
            structural = False
        ref = reference.encode()
        edit = distance(result.output, ref)
        gen.append({'task': task, 'output_hex': result.output.hex(), 'output': result.output.decode('utf-8', 'replace'),
                    'length': len(result.output), 'halt_reason': result.halt_reason, 'edit_distance': edit,
                    'byte_similarity': 1 - edit / max(len(ref), len(result.output)), 'structural': structural})
    return {'routing': {'count': len(rows), 'correct': sum(x['correct'] for x in rows), 'cases': rows},
            'generation': {'count': len(gen), 'structurally_coherent': sum(x['structural'] for x in gen),
                           'mean_byte_similarity': sum(x['byte_similarity'] for x in gen) / len(gen), 'cases': gen,
                           'caveat': 'Syntax and length proxy only; no semantic or functional verification. Accept-all verifier.'},
            'caveat': 'Synthetic development benchmark; targets and examples have not been independently validated.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generator', default='runs/phase12/generator.pt')
    p.add_argument('--response', default='runs/phase12/response.pt')
    p.add_argument('--vector-reader', default='runs/phase12/vector_reader.pt')
    p.add_argument('--diverse-response', default='runs/phase12/diverse_response.pt')
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--output', default='runs/phase13/benchmark.json')
    args = p.parse_args()
    result = evaluate(args)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'routing': f"{result['routing']['correct']}/{result['routing']['count']}",
                      'generation': f"{result['generation']['structurally_coherent']}/{result['generation']['count']}",
                      'mean_byte_similarity': result['generation']['mean_byte_similarity'], 'output': str(path)}, indent=2))

if __name__ == '__main__':
    main()
