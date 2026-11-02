"""Phase 19.4 — validate generated Mineflayer code without executing Minecraft.

Three independent checks, each weaker than the next:

1. ``static_issues``: a small JavaScript tokenizer/guard. Rejects syntax debris,
   Python-isms and host capability words; cannot prove correctness.
2. ``syntax_check``: ``node --check`` on the code wrapped in an async function.
   Parses only; no user code runs.
3. ``run_mock_batch``: executes the code in ``notebook/mineflayer_mock.js``, a
   Node mock of the Mineflayer API inside ``node:vm`` with no ``require``,
   ``process`` or network, a wall-clock race, and no real Minecraft server.
   It records API calls and mutates a tiny mock world so expectations such as
   "dug diamond_ore at (50, 30, 75)" can be checked.

**Never ``exec()`` model output.** Isolation here is for non-adversarial model
text only; it is not a security boundary for untrusted third-party code.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from notebook.mineflayer_training_data import (ITEM_IDS, Trajectory,
                                               mineflayer_trajectories)

HARNESS = Path(__file__).with_name('mineflayer_mock.js')

BLOCKED = {
    'require': r'\brequire\s*\(',
    'process': r'\bprocess\b',
    'eval': r'\beval\s*\(',
    'Function constructor': r'\bnew\s+Function\b|\bFunction\s*\(',
    'constructor escape': r'\bconstructor\b',
    'python dunder escape': r'__import__|__builtins__|__class__|__subclasses__|__globals__',
    'dynamic import': r'\bimport\s*\(',
    'globalThis': r'\bglobalThis\b',
    'prototype access': r'__proto__|\.prototype\b',
    'node fs/path/os': r'\b(require\(|child_process|\bfs\.|\bos\.)',
    'network': r'\bfetch\s*\(|XMLHttpRequest|WebSocket',
}
PYTHON_ISMS = {
    'python print': r'\bprint\s*\(',
    'python def': r'^\s*def\s+\w+\s*\(',
    'python import': r'^\s*(import|from)\s+\w+',
    'python literals': r'\b(True|False|None)\b',
    'python control flow': r'^\s*(elif|else:)\b',
    'python self': r'\bself\.',
    'wrong SDK': r'mineflayer\.Item|mineflayer\.createBot\s*\(',
}
AMBIGUOUS = {
    'unbounded loop': r'\bwhile\s*\(|\bfor\s*\(\s*;\s*;',
    'nearestEntity needs a filter function': r"bot\.nearestEntity\s*\(\s*['\"]",
    'blockAt expects a Vec3': r'bot\.blockAt\s*\([^)]*,',
    'pathfinder.goto expects a Goal': r'bot\.pathfinder\.goto\s*\((?!\s*new\s+\w*Goal)',
    'missing trailing semicolon': r'[^;{}\s]\s*$',
}


def _strip_strings_and_comments(code: str) -> tuple[str, bool]:
    """Remove string/comment contents so keywords inside chat text are ignored."""
    out, i, n, quote = [], 0, len(code), None
    terminated = True
    while i < n:
        char = code[i]
        if quote is None:
            if char in "'\"`":
                quote = char
                out.append(' ')
            elif code.startswith('//', i):
                end = code.find('\n', i)
                i = n if end < 0 else end
                continue
            elif code.startswith('/*', i):
                end = code.find('*/', i + 2)
                i = n if end < 0 else end + 2
                continue
            else:
                out.append(char)
            i += 1
        else:
            if char == '\\':
                i += 2
                continue
            if char == quote:
                quote = None
            i += 1
    if quote is not None:
        terminated = False
    return ''.join(out), terminated


def _balanced(code: str) -> list[str]:
    pairs = {')': '(', ']': '[', '}': '{'}
    stack: list[str] = []
    for char in code:
        if char in '([{':
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return ['unbalanced brackets']
    return ['unbalanced brackets'] if stack else []


def static_issues(code: str) -> list[str]:
    """Blocking problems: wrong language, host capabilities, broken structure."""
    text = (code or '').strip()
    if not text:
        return ['empty action']
    stripped, terminated = _strip_strings_and_comments(text)
    issues = []
    if not terminated:
        issues.append('unterminated string')
    issues.extend(_balanced(stripped))
    if 'bot.' not in stripped:
        issues.append('no bot API call')
    for label, pattern in BLOCKED.items():
        if re.search(pattern, stripped, re.MULTILINE):
            issues.append(label)
    for label, pattern in PYTHON_ISMS.items():
        if re.search(pattern, stripped, re.MULTILINE):
            issues.append(label)
    return issues


def static_warnings(code: str) -> list[str]:
    stripped, _ = _strip_strings_and_comments((code or '').strip())
    return [label for label, pattern in AMBIGUOUS.items()
            if re.search(pattern, stripped, re.MULTILINE)]


def syntax_check(code: str, timeout: float = 10.0) -> bool | None:
    """``node --check`` wrapped in an async function. None when node is absent."""
    node = shutil.which('node')
    if node is None:
        return None
    wrapped = 'async function __mineflayer_main() {\n' + code + '\n}\n'
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf8') as handle:
        handle.write(wrapped)
        path = handle.name
    try:
        result = subprocess.run([node, '--check', path], capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        Path(path).unlink(missing_ok=True)


def run_mock_batch(cases: list[dict], timeout: float = 180.0) -> list[dict] | None:
    """Execute cases in the Node mock. None when node or the harness is absent."""
    node = shutil.which('node')
    if node is None or not HARNESS.exists():
        return None
    prepared = []
    for case in cases:
        mock = dict(case.get('mock') or {})
        mock.setdefault('registry', ITEM_IDS)
        prepared.append({'id': case.get('id'), 'code': case.get('code', ''), 'mock': mock})
    payload = json.dumps({'cases': prepared})
    try:
        result = subprocess.run([node, str(HARNESS)], input=payload, capture_output=True,
                                text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return [{'ok': False, 'error': f'harness failed: {type(exc).__name__}: {exc}', 'actions': []}] * len(cases)
    if not result.stdout.strip():
        return [{'ok': False, 'error': f'harness produced no output: {result.stderr[-300:]}', 'actions': []}] * len(cases)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return [{'ok': False, 'error': 'harness output was not JSON', 'actions': []}] * len(cases)
    if data.get('fatal'):
        return [{'ok': False, 'error': str(data['fatal'])[:300], 'actions': []}] * len(cases)
    return data.get('results', [])


def run_mock(code: str, mock: dict | None = None) -> dict | None:
    results = run_mock_batch([{'code': code, 'mock': mock or {}}])
    return None if results is None else results[0]


def _positions_match(actual, expected, xz_only: bool = False) -> bool:
    if actual is None or expected is None:
        return False
    for index, (a, e) in enumerate(zip(actual, expected)):
        if e is None or (xz_only and index == 1):
            continue
        if abs(float(a) - float(e)) > 1e-6:
            return False
    return True


def check_expect(expect: dict, result: dict) -> tuple[bool, str]:
    """Compare recorded mock actions with the trajectory's declared intent."""
    actions = result.get('actions') or []
    kind = expect.get('type')

    def held(name, **extra):
        for action in actions:
            if action.get('type') != name:
                continue
            for key, value in extra.items():
                if action.get(key) != value:
                    break
            else:
                return action
        return None

    if kind == 'goto':
        if expect.get('follow'):
            match = [a for a in actions if a.get('type') == 'goto' and a.get('follow') == expect['follow']]
        else:
            match = [a for a in actions if a.get('type') == 'goto'
                     and _positions_match(a.get('target'), expect.get('target'), expect.get('xz', False))]
        if not match:
            return False, 'no matching pathfinder.goto action'
        if expect.get('within') and match[0].get('within') not in (expect['within'], None):
            return False, f"goal radius {match[0].get('within')} != {expect['within']}"
        return True, 'goto matched'
    if kind == 'setGoal':
        if expect.get('follow'):
            match = [a for a in actions if a.get('type') == 'setGoal' and a.get('follow') == expect['follow']]
        else:
            match = [a for a in actions if a.get('type') == 'setGoal'
                     and _positions_match(a.get('target'), expect.get('target'))]
        return (True, 'setGoal matched') if match else (False, 'no matching pathfinder.setGoal')
    if kind in ('stop', 'consume', 'activateItem'):
        return (True, f'{kind} recorded') if held(kind) else (False, f'no {kind} action')
    if kind == 'dig':
        return (True, 'dig matched') if held('dig', block=expect['block']) else (False, 'dig did not target the block')
    if kind == 'collect':
        return (True, 'collect matched') if held('dig', block=expect['block']) else (False, 'collect did not target the block')
    if kind == 'craft':
        match = [a for a in actions if a.get('type') == 'craft' and a.get('item') == expect['item']]
        if not match:
            return False, 'craft did not produce the item'
        if 'count' in expect and match[0].get('count') != expect['count']:
            return False, f"craft count {match[0].get('count')} != {expect['count']}"
        return True, 'craft matched'
    if kind in ('attack', 'pvp_attack'):
        return (True, f'{kind} matched') if held(kind, entity=expect['entity']) else (False, 'attack did not target the entity')
    if kind == 'place':
        match = [a for a in actions if a.get('type') == 'placeBlock' and a.get('block') == expect['block']]
        if len(match) < expect.get('min_count', 1):
            return False, f"placed {len(match)} {expect['block']}, expected {expect.get('min_count', 1)}"
        if 'position' in expect and not _positions_match(match[0].get('position'), expect['position']):
            return False, 'placement was at the wrong position'
        return True, 'placement matched'
    if kind == 'equip':
        action = held('equip', item=expect['item'])
        if not action:
            return False, 'item was not equipped'
        if expect.get('destination') and action.get('destination') != expect['destination']:
            return False, 'equipped to the wrong slot'
        return True, 'equip matched'
    if kind in ('toss', 'deposit', 'withdraw'):
        action = held(kind, item=expect['item'])
        if not action:
            return False, f'{kind} did not use the item'
        if expect.get('count') is not None and action.get('count') != expect['count']:
            return False, f"{kind} count {action.get('count')} != {expect['count']}"
        return True, f'{kind} matched'
    if kind == 'chat':
        messages = ' '.join(a.get('message', '') for a in actions if a.get('type') == 'chat')
        return (True, 'chat matched') if expect['contains'] in messages else (False, 'chat text missing')
    if kind == 'whisper':
        match = [a for a in actions if a.get('type') == 'whisper' and a.get('player') == expect['player']
                 and expect['contains'] in a.get('message', '')]
        return (True, 'whisper matched') if match else (False, 'whisper missing')
    if kind == 'listen':
        return (True, 'listener registered') if held('listen', event=expect['event']) else (False, 'no listener')
    if kind == 'activateBlock':
        return (True, 'activateBlock matched') if held('activateBlock', block=expect['block']) else (False, 'wrong block activated')
    return False, f'unknown expectation {kind}'


TEST_CASES = [
    {'name': 'navigate', 'skill': 'navigation',
     'request': 'Walk to (100, 64, 200) and stop there.',
     'reference': 'await bot.pathfinder.goto(new GoalNear(100, 64, 200, 1));',
     'mock': {'position': [0, 64, 0]},
     'expect': {'type': 'goto', 'target': [100, 64, 200], 'within': 1}},
    {'name': 'mine_diamond', 'skill': 'mining',
     'request': 'Mine the diamond ore at (50, 30, 75).',
     'reference': ("const block = bot.blockAt(new Vec3(50, 30, 75));\n"
                   "if (!block || block.name !== 'diamond_ore') throw new Error('expected diamond_ore');\n"
                   "await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['diamond_pickaxe'].id, null), 'hand');\n"
                   'await bot.dig(block);'),
     'mock': {'position': [0, 64, 0], 'world': {'50,30,75': 'diamond_ore'},
              'inventory': ['diamond_pickaxe']},
     'expect': {'type': 'dig', 'block': 'diamond_ore'}},
    {'name': 'craft_pickaxe', 'skill': 'crafting',
     'request': 'Craft a wooden pickaxe.',
     'reference': ("const craftingTable = bot.blockAt(new Vec3(1, 64, 1));\n"
                   "const recipe = bot.recipesFor(bot.registry.itemsByName['wooden_pickaxe'].id, null, 1, craftingTable)[0];\n"
                   "if (!recipe) throw new Error('missing materials');\n"
                   'await bot.craft(recipe, 1, craftingTable);'),
     'mock': {'position': [0, 64, 0], 'world': {'1,64,1': 'crafting_table'},
              'inventory': ['oak_planks', 'stick']},
     'expect': {'type': 'craft', 'item': 'wooden_pickaxe', 'count': 1}},
    {'name': 'attack_zombie', 'skill': 'combat',
     'request': 'There is a zombie nearby. Attack it.',
     'reference': ("const target = bot.nearestEntity(e => e.type === 'mob' && e.name === 'zombie');\n"
                   "if (!target) throw new Error('no zombie in range');\n"
                   "await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['iron_sword'].id, null), 'hand');\n"
                   'await bot.attack(target);'),
     'mock': {'position': [0, 64, 0], 'inventory': ['iron_sword'],
              'entities': [{'id': 7, 'name': 'zombie', 'type': 'mob', 'position': [3, 64, 2]}]},
     'expect': {'type': 'attack', 'entity': 'zombie'}},
    {'name': 'place_torch', 'skill': 'building',
     'request': 'Place a torch at (10, 65, 20).',
     'reference': ("await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['torch'].id, null), 'hand');\n"
                   'await bot.placeBlock(bot.blockAt(new Vec3(10, 64, 20)), new Vec3(0, 1, 0));'),
     'mock': {'position': [0, 64, 0], 'world': {'10,64,20': 'stone'}, 'inventory': ['torch']},
     'expect': {'type': 'place', 'block': 'torch', 'position': [10, 65, 20]}},
    {'name': 'equip_sword', 'skill': 'inventory',
     'request': 'Equip the diamond sword in my hand.',
     'reference': ("await bot.equip(bot.inventory.findInventoryItem(bot.registry.itemsByName['diamond_sword'].id, null), 'hand');"),
     'mock': {'position': [0, 64, 0], 'inventory': ['diamond_sword']},
     'expect': {'type': 'equip', 'item': 'diamond_sword', 'destination': 'hand'}},
    {'name': 'chat', 'skill': 'chat',
     'request': 'Say hello to the other players.',
     'reference': "bot.chat('Hello everyone!');",
     'mock': {'position': [0, 64, 0]},
     'expect': {'type': 'chat', 'contains': 'Hello'}},
    {'name': 'harvest_replant', 'skill': 'farming',
     'request': 'Harvest the mature wheat at (4, 64, 0) and replant it.',
     'reference': ("const crop = bot.findBlock({ matching: b => b.name === 'wheat' && b.getProperties().age === 7, maxDistance: 16 });\n"
                   "if (!crop) throw new Error('no mature wheat');\n"
                   'await bot.pathfinder.goto(new GoalBlock(crop.position.x, crop.position.y, crop.position.z));\n'
                   'await bot.dig(crop);\n'
                   "const seeds = bot.inventory.findInventoryItem(bot.registry.itemsByName['wheat_seeds'].id, null);\n"
                   "if (!seeds) throw new Error('no seeds');\n"
                   "await bot.equip(seeds, 'hand');\n"
                   'await bot.placeBlock(bot.blockAt(crop.position.offset(0, -1, 0)), new Vec3(0, 1, 0));'),
     'mock': {'position': [4, 64, 0],
              'world': {'4,64,0': 'wheat', '4,63,0': 'farmland'},
              'inventory': ['wheat_seeds']},
     'expect': {'type': 'place', 'block': 'wheat_seeds'}},
]


def validate_code(code: str, mock: dict | None = None, expect: dict | None = None,
                  use_sandbox: bool = True) -> dict:
    """Full validation of one snippet; every layer is reported separately."""
    issues = static_issues(code)
    warnings = static_warnings(code)
    syntax = syntax_check(code)
    result = run_mock(code, mock) if use_sandbox and not issues else None
    expectation = None
    if result is not None and result.get('ok') and expect:
        expectation = check_expect(expect, result)
    expectation_ok = expectation is None or expectation[0]
    if use_sandbox:
        ok = (not issues and syntax is not False and result is not None
              and result.get('ok') and expectation_ok)
    else:
        ok = not issues and syntax is not False
    return {'ok': ok, 'issues': issues, 'warnings': warnings, 'syntax': syntax,
            'result': result, 'expectation': expectation}


def self_test(cases: list[dict] | None = None) -> list[dict]:
    """Run every reference implementation through all three layers."""
    rows = []
    for case in (cases or TEST_CASES):
        report = validate_code(case['reference'], case['mock'], case['expect'])
        rows.append({'case': case['name'], 'pass': bool(report['ok']), 'issues': report['issues'],
                     'warnings': report['warnings'], 'syntax': report['syntax'],
                     'expectation': report['expectation'],
                     'error': None if report['result'] is None else report['result'].get('error')})
    return rows


def validate_trajectories(rows: list[Trajectory] | None = None, use_sandbox: bool = True) -> dict:
    """Execute every generated trajectory in the mock and check its expectation."""
    rows = rows or mineflayer_trajectories()
    static_failures = []
    for row in rows:
        issues = static_issues(row.action)
        if issues:
            static_failures.append({'skill': row.skill, 'scenario': row.scenario, 'issues': issues})
    results = None
    if use_sandbox:
        results = run_mock_batch([{'code': row.action, 'mock': row.mock} for row in rows])
    runtime_failures = []
    if results is not None:
        for row, result in zip(rows, results):
            if not result.get('ok'):
                runtime_failures.append({'skill': row.skill, 'scenario': row.scenario,
                                         'error': result.get('error')})
                continue
            ok, reason = check_expect(row.expect, result)
            if not ok:
                runtime_failures.append({'skill': row.skill, 'scenario': row.scenario, 'error': reason})
    passed = len(rows) - len(static_failures) - len(runtime_failures)
    return {'total': len(rows), 'passed': passed, 'static_failures': static_failures[:20],
            'runtime_failures': runtime_failures[:20],
            'static_failure_count': len(static_failures),
            'runtime_failure_count': len(runtime_failures),
            'sandbox': results is not None}


def extract_action(text: str) -> str:
    """Pull the ``Action:`` block out of a free-running model response."""
    lines = (text or '').splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith('Action:'):
            body = [line.split('Action:', 1)[1].strip()]
            for follow in lines[index + 1:]:
                if re.match(r'^(Observation|Answer|Thought|Request):', follow.strip()):
                    break
                body.append(follow)
            action = '\n'.join(body).strip()
            action = re.sub(r'^```(?:javascript|js)?\s*|\s*```$', '', action.strip())
            return action.strip()
    return ''


def evaluate_mineflayer(model, max_new: int = 192, use_sandbox: bool = True) -> list[dict]:
    """Free-running probes: prompt, generate, extract, validate. Not a benchmark."""
    from notebook.phone_model import generate

    rows = []
    for case in TEST_CASES:
        prompt = f"Request: {case['request']}\nThought: "
        output = generate(model, prompt, max_new=max_new)
        action = extract_action(output)
        report = validate_code(action, case['mock'], case['expect'], use_sandbox=use_sandbox) \
            if action else {'ok': False, 'issues': ['no Action block'], 'warnings': [],
                            'syntax': None, 'result': None, 'expectation': None}
        rows.append({'case': case['name'], 'skill': case['skill'], 'pass': bool(report['ok']),
                     'output': output, 'action': action, 'issues': report['issues'],
                     'warnings': report['warnings'], 'syntax': report['syntax'],
                     'expectation': report['expectation'],
                     'error': None if report['result'] is None else report['result'].get('error')})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='check the 8 reference snippets')
    parser.add_argument('--validate-data', action='store_true', help='run all generated trajectories')
    parser.add_argument('--show', type=int, default=3)
    args = parser.parse_args(argv)
    code = 0
    if args.validate_data:
        summary = validate_trajectories()
        print(json.dumps({k: v for k, v in summary.items() if not k.endswith('failures')}, indent=2))
        for row in (summary['static_failures'] + summary['runtime_failures'])[:args.show]:
            print('FAIL', row)
        code |= int(summary['passed'] != summary['total'])
    if args.self_test or not args.validate_data:
        rows = self_test()
        for row in rows:
            print(f"{row['case']:<16} {'PASS' if row['pass'] else 'FAIL'} {row['issues'] or ''} "
                  f"{row['expectation'] or ''} {row['error'] or ''}")
        print('passed', sum(r['pass'] for r in rows), '/', len(rows))
        code |= int(any(not r['pass'] for r in rows))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
