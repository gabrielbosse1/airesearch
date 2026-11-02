"""Host-side tools. No generated Python or Minecraft calls are executed here."""
import ast
import re
from notebook.preprocess import apply_edits

def search_ddg(query):
    """Return at most three live result snippets as UTF-8 bytes; network required."""
    from ddgs import DDGS
    results = list(DDGS().text(query, max_results=3))
    return '\n'.join(f"{r.get('title', '')} | {r.get('href', '')} | {r.get('body', '')}" for r in results).encode('utf8')

def minecraft_valid(text):
    """Legacy Phase 19 command probe, kept only for the old reward test."""
    return bool(re.search(r'minecraft\.place_block\("[a-z_]+",\s*-?\d+,\s*-?\d+,\s*-?\d+\)', text))

def mineflayer_valid(text):
    """Static-check Mineflayer JavaScript. No parsing proof, no execution."""
    from notebook.test_mineflayer import static_issues
    return not static_issues(text)

def grammar_score(text):
    # Toy heuristic, not a Portuguese grammar checker.
    return bool(re.search(r'\b(você|olá|obrigado|estou)\b', text.lower())) and not bool(re.search(r'\b(voce|ola|casaa|diaa)\b', text.lower()))

def reward(kind, text, draft=None, target=None):
    if kind == 'code':
        source = text.strip().removeprefix('```python').removesuffix('```').strip()
        try:
            tree = ast.parse(source)
            return float(any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)))
        except (SyntaxError, ValueError): return 0.0
    if kind == 'minecraft': return float(minecraft_valid(text))
    if kind == 'mineflayer': return float(mineflayer_valid(text))
    if kind == 'pt': return .5 * grammar_score(text)
    if kind == 'edit' and draft is not None:
        match = re.search(r'\[EDIT pos=\d+ delete=\d+ bytes=[0-9a-f]*\]', text)
        if not match: return 0.0
        try:
            edited = apply_edits(draft, match.group())
            # Exact target is a proxy for reducing errors, not an independent evaluator.
            return .5 * (edited == target and edited != draft)
        except (ValueError, UnicodeError): return 0.0
    return 0.0

class Memory:
    """Explicit host-managed session memory, not learned or persistent."""
    def __init__(self): self.items = {}
    def remember(self, key, value): self.items[key] = value
    def recall(self, key): return self.items.get(key)
    def prefix(self): return '\n'.join(f'{k}: {v}' for k,v in self.items.items())

def run_search_action(generated, search=search_ddg):
    """Run only an explicit, single literal search_ddg call, never eval arbitrary code."""
    line = next((x[len('Action: '):].strip() for x in generated.splitlines() if x.startswith('Action: search_ddg(')), None)
    if line is None: return None
    try:
        node = ast.parse(line, mode='eval').body
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'search_ddg' and len(node.args) == 1 and not node.keywords
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                and 0 < len(node.args[0].value) <= 200):
            return None
        return search(node.args[0].value)
    except (SyntaxError, ValueError): return None

def respond(model, request, memory=None, use_search=False, max_new=96):
    from notebook.phone_model import generate
    context = ('Memory:\n' + memory.prefix() + '\n') if memory and memory.items else ''
    prompt = context + 'Request: ' + request + '\n'
    output = generate(model, prompt, max_new=max_new)
    observation = run_search_action(output) if use_search else None
    if observation is not None:
        output += '\nObservation: ' + observation.decode('utf8', 'replace') + '\nAnswer: '
        output += generate(model, prompt + output, max_new=max_new)
    # The caller chooses whether to apply edits; no implicit manipulation of output.
    return output
