"""Phase 15 synthetic tool-use tasks (template-bound; not general code ability).

Every row is a supervised mapping

    (request bytes, optional prior tool-result bytes) -> route + argument bytes

Two row kinds exist:

* **call rows** (``result is None``) ask for the first tool call for a request.
* **follow-up rows** carry the bytes the sandbox actually returned. A successful
  result is labelled ``none`` (stop); a failing result is labelled with the
  corrected call. Follow-up results are produced by really executing the code in
  :class:`models.sandbox.ByteSandbox`, never by hand-writing plausible text.

Entity pools (number pairs, pseudo-words, ranges, triples, filenames) are large
and split by *instance*, so a held-out prompt contains digits and letter
sequences the head has never seen: the only way to lower training loss is to
copy the operand out of the request, not to memorise a table of answers. Prompt
templates, code templates and repair edits are still shared across splits, so
this measures tool routing plus template-bound copying -- **not** code
understanding and **not** the ability to debug arbitrary programs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from models.sandbox import ByteSandbox, looks_failed
from models.tool_use import MAX_REQUEST_BYTES, ROUTE_NAMES

EXEC_PYTHON = b"exec_python"
READ_FILE = b"read_file"
WRITE_FILE = b"write_file"

_CONSONANTS = "bcdfghjklmnpqrstvwxz"
_VOWELS = "aeiou"
_EXTENSIONS = ("txt", "md", "csv", "log", "json")


def _pseudo_words(count: int, seed: int) -> tuple[str, ...]:
    """Deterministic nonsense words: unfamiliar letter sequences force copying."""
    rng = random.Random(seed)
    words: set[str] = set()
    while len(words) < count:
        syllables = rng.randint(2, 3)
        candidate = "".join(rng.choice(_CONSONANTS) + rng.choice(_VOWELS) for _ in range(syllables))
        words.add(candidate)
    return tuple(sorted(words))


def _triples(count: int, seed: int) -> tuple[tuple[int, int, int], ...]:
    rng = random.Random(seed)
    out: set[tuple[int, int, int]] = set()
    while len(out) < count:
        out.add(tuple(rng.sample(range(1, 60), 3)))  # type: ignore[arg-type]
    return tuple(sorted(out))


def _filenames(count: int, seed: int) -> tuple[str, ...]:
    words = _pseudo_words(count, seed)
    return tuple(f"{word}.{_EXTENSIONS[index % len(_EXTENSIONS)]}" for index, word in enumerate(words))


def _sentences(count: int, seed: int) -> tuple[str, ...]:
    words = _pseudo_words(3 * count, seed + 7)
    return tuple(" ".join(words[index * 3:index * 3 + 3]) for index in range(count))


POOL_SIZES = {"train": 320, "validation": 40, "test": 40}
PAIRS = tuple((a, b) for a in range(2, 24) for b in range(2, 24))
WORDS = _pseudo_words(400, 20251030)
RANGES = tuple(range(2, 402))
TRIPLES = _triples(400, 515)
FILENAMES = _filenames(400, 9091)
SENTENCES = _sentences(200, 4242)


def _slice(pool, split: str):
    size = POOL_SIZES[split]
    if split == "train":
        return pool[:size]
    start = POOL_SIZES["train"] if split == "validation" else POOL_SIZES["train"] + POOL_SIZES["validation"]
    return pool[start:start + size]


@dataclass(frozen=True)
class ToolExample:
    request: bytes
    tool: bytes            # b"" means "no tool call"
    argument: bytes
    result: bytes | None   # prior tool result in the context; None for a first turn

    @property
    def route(self) -> int:
        return 0 if not self.tool else ROUTE_NAMES.index(self.tool.decode("ascii"))


ARITHMETIC_PROMPTS = ("Calculate {a} {op} {b}.", "What is {a} {op} {b}?", "Compute {a} {op} {b}.",
                      "Work out {a} {op} {b} please.", "Give me {a} {op} {b}.", "Evaluate {a} {op} {b}.")
OPS = ("*", "+", "-")
REVERSE_PROMPTS = ("Reverse the string '{word}'.", "Flip the text '{word}' backwards.",
                   "Show '{word}' reversed.", "Print '{word}' in reverse.")
UPPER_PROMPTS = ("Uppercase the text '{word}'.", "Convert '{word}' to upper case.",
                 "Shout the word '{word}'.", "Capitalise '{word}'.")
LENGTH_PROMPTS = ("How many characters are in '{word}'?", "Count the characters of '{word}'.",
                  "What is the length of '{word}'?", "Measure the length of '{word}'.")
RANGE_PROMPTS = ("Sum the numbers from 1 to {value}.", "Add up every number from 1 to {value}.",
                 "What is the total of 1 to {value}?", "Total the integers 1 through {value}.")
TRIPLE_PROMPTS = ("Sort the numbers {a}, {b} and {c}.", "Put {a}, {b}, {c} in ascending order.",
                  "Order these numbers: {a}, {b}, {c}.", "Sort this list: {a}, {b}, {c}.")
SENTENCE_PROMPTS = ("Count the words in '{sentence}'.", "How many words are in '{sentence}'?",
                    "Word count for '{sentence}' please.", "Tell me how many words '{sentence}' has.")
READ_PROMPTS = ("Read the file {name}.", "Show the contents of {name}.", "Open and print {name}.",
                "What is inside {name}?")
WRITE_PROMPTS = ("Write 'hello sandbox' to {name}.", "Save the text hello sandbox in {name}.",
                 "Create {name} containing hello sandbox.", "Put hello sandbox into {name}.")
LISTDIR_PROMPTS = ("List the files in the current directory.", "Show me the files here.",
                   "What files are in this folder?", "List the current directory contents.",
                   "Print the filenames in this directory.", "Show the directory listing.",
                   "Which files exist here?", "List everything in this folder.",
                   "Give me a listing of this directory.", "What is in the working directory?",
                   "List files here.", "Show what files this folder holds.",
                   "Print the directory contents.", "Enumerate the files in this directory.",
                   "What does this folder contain?", "List out the files in this sandbox.",
                   "Display the current folder listing.", "Read out the filenames here.",
                   "Tell me the files in this directory.", "List the sandbox directory.",
                   "Which entries are in this folder?", "Show the sandbox file list.",
                   "Report the files in the working folder.", "List the directory entries here.")
LISTDIR_CODE = b"import os\nprint(sorted(os.listdir()))"
LISTDIR_SPLITS = {"train": range(0, 12), "validation": range(12, 18), "test": range(18, 24)}

REPAIR_PROMPTS = ("Run this Python and fix any error: {code}",
                  "Execute this and repair it if it fails: {code}",
                  "Please run: {code}")

#: Repair supervision is **generated**, not hand-listed, so every training
#: snippet contains different content: the head can only succeed by copying the
#: pasted code out of the request and applying a category-level edit, never by
#: recalling a memorised snippet. Categories are A misspelled builtin,
#: B missing colon, C missing closing paren, D type coercion,
#: E misspelled variable, F invalid literal. They recur across splits with
#: different content; the held-out tasks are unseen instances, not unseen skills.
REPAIR_CATEGORIES = ("A", "B", "C", "D", "E", "F")
REPAIR_PER_SPLIT = {"train": 16, "validation": 2, "test": 1}
#: Share of successful-result ("stop") rows, sampled deterministically.
STOP_FRACTION = 0.25
_BUILTIN_TYPOS = ("pritn", "prnt", "prin", "pirnt")
_EXPRESSIONS = ("1 + 1", "2 * 3", "40 - 8", "7 // 2", "'sandbox'", "'byte core'", "len('abc')",
                "sum([1, 2, 3])", "sorted([3, 1, 2])", "max(4, 9)", "min(6, 2)", "abs(-5)",
                "round(3.7)", "str(12)", "int('7')", "list('abc')", "tuple([1, 2])", "bool(0)")
_CONDITIONS = ("2 > 1", "5 == 5", "len('abc') > 2", "10 < 3", "7 % 2 == 1")
_INNERS = ("len('{word}')", "'{word}'.upper()", "sum([{a}, {b}])", "int({n})", "sorted([{a}, {b}])",
           "max({a}, {n})", "'{word}'.strip()")
_NAMES = ("total", "value", "result", "count", "score", "amount", "index", "length", "offset", "width",
          "height", "radius", "weight", "margin", "budget", "sample", "window", "cursor", "buffer", "signal",
          "packet", "metric", "vector", "matrix", "kernel", "channel", "region", "segment", "cluster", "module",
          "record", "bundle", "stream", "tensor", "anchor", "branch", "cycle", "domain", "entry", "factor",
          "gateway", "handler", "input", "jitter", "label", "marker", "node", "option", "pixel", "query",
          "range", "slot", "token", "unit", "vertex", "widget", "yield", "zone", "alpha", "beta")


def _scramble(name: str, rng: random.Random) -> str:
    """Swap two adjacent letters so the typo is a realistic misspelling."""
    position = rng.randrange(max(len(name) - 1, 1))
    letters = list(name)
    letters[position], letters[position + 1] = letters[position + 1], letters[position]
    return "".join(letters)


def _category_pairs(category: str, count: int, seed: int) -> list[tuple[bytes, bytes]]:
    rng = random.Random(seed + 17 * REPAIR_CATEGORIES.index(category))
    pairs: list[tuple[bytes, bytes]] = []
    for index in range(count):
        if category == "A":
            expression = _EXPRESSIONS[index % len(_EXPRESSIONS)]
            typo = _BUILTIN_TYPOS[index % len(_BUILTIN_TYPOS)]
            pairs.append((f"{typo}({expression})".encode(), f"print({expression})".encode()))
        elif category == "B":
            # Single-line compound statements: the interactive loop reads one
            # line at a time, so multi-line pastes are out of scope.
            condition = _CONDITIONS[index % len(_CONDITIONS)]
            head = (f"if {condition}", f"for i in range({3 + index % 5})",
                    f"while {condition}")[index % 3]
            body = f"print({_EXPRESSIONS[index % len(_EXPRESSIONS)]})"
            pairs.append((f"{head} {body}".encode(), f"{head}: {body}".encode()))
        elif category == "C":
            template = _INNERS[index % len(_INNERS)]
            inner = template.format(word=_NAMES[index % len(_NAMES)], a=2 + index % 9,
                                    b=5 + index % 7, n=3 + index % 6)
            pairs.append((f"print({inner}".encode(), f"print({inner})".encode()))
        elif category == "D":
            if index % 2:
                pairs.append((f"print('{_NAMES[index % len(_NAMES)]}' + {3 + index % 6})".encode(),
                              f"print('{_NAMES[index % len(_NAMES)]}' + str({3 + index % 6}))".encode()))
            else:
                pairs.append((f"print({3 + index % 6} + '{2 + index % 8}')".encode(),
                              f"print({3 + index % 6} + int('{2 + index % 8}'))".encode()))
        elif category == "E":
            name = _NAMES[(index * 3) % len(_NAMES)]
            wrong = _scramble(name, rng)
            if wrong == name:
                wrong = name[::-1]
            value = 10 + index % 40
            pairs.append((f"{name} = {value}; print({wrong})".encode(),
                          f"{name} = {value}; print({name})".encode()))
        else:
            digits = f"{100 + index % 800}"
            pairs.append((f"print(int('{digits}a'))".encode(), f"print(int('{digits}'))".encode()))
    return pairs


def _split_pairs(pairs, split: str):
    train = REPAIR_PER_SPLIT["train"]
    validation = REPAIR_PER_SPLIT["validation"]
    if split == "train":
        return pairs[:train]
    if split == "validation":
        return pairs[train:train + validation]
    start = train + validation
    return pairs[start:start + REPAIR_PER_SPLIT["test"]]


def _arithmetic_rows(split: str) -> list[tuple[bytes, bytes]]:
    pairs = _slice(PAIRS, split)
    rows = []
    for position, (a, b) in enumerate(pairs):
        op = OPS[position % len(OPS)]
        prompt = ARITHMETIC_PROMPTS[position % len(ARITHMETIC_PROMPTS)].format(a=a, op=op, b=b)
        rows.append((prompt.encode("utf-8"), f"print({a}{op}{b})".encode("utf-8")))
    return rows


def _word_rows(split: str, prompts: tuple[str, ...], code_template: str) -> list[tuple[bytes, bytes]]:
    words = _slice(WORDS, split)
    return [(prompts[position % len(prompts)].format(word=word).encode("utf-8"),
             code_template.format(word=word).encode("utf-8")) for position, word in enumerate(words)]


def _range_rows(split: str) -> list[tuple[bytes, bytes]]:
    values = _slice(RANGES, split)
    return [(RANGE_PROMPTS[position % len(RANGE_PROMPTS)].format(value=value).encode("utf-8"),
             f"print(sum(range(1, {value}+1)))".encode("utf-8")) for position, value in enumerate(values)]


def _triple_rows(split: str) -> list[tuple[bytes, bytes]]:
    triples = _slice(TRIPLES, split)
    return [(TRIPLE_PROMPTS[position % len(TRIPLE_PROMPTS)].format(a=a, b=b, c=c).encode("utf-8"),
             f"print(sorted([{a}, {b}, {c}]))".encode("utf-8"))
            for position, (a, b, c) in enumerate(triples)]


def _sentence_rows(split: str) -> list[tuple[bytes, bytes]]:
    sentences = _slice(SENTENCES, split)
    return [(SENTENCE_PROMPTS[position % len(SENTENCE_PROMPTS)].format(sentence=sentence).encode("utf-8"),
             f"print(len('{sentence}'.split()))".encode("utf-8"))
            for position, sentence in enumerate(sentences)]


def _listdir_rows(split: str) -> list[tuple[bytes, bytes]]:
    return [(LISTDIR_PROMPTS[index % len(LISTDIR_PROMPTS)].encode("utf-8"), LISTDIR_CODE)
            for index in LISTDIR_SPLITS[split]]


def _file_rows(split: str, prompts: tuple[str, ...], tool: bytes,
               argument_template: str) -> list[tuple[bytes, bytes, bytes]]:
    names = _slice(FILENAMES, split)
    return [(prompts[position % len(prompts)].format(name=name).encode("utf-8"), tool,
             argument_template.format(name=name).encode("utf-8")) for position, name in enumerate(names)]


def code_rows(split: str) -> list[tuple[bytes, bytes]]:
    """All templated ``exec_python`` rows for a split: (request, target code)."""
    if split not in POOL_SIZES:
        raise ValueError("unknown split")
    rows = _arithmetic_rows(split)
    rows += _word_rows(split, REVERSE_PROMPTS, "print('{word}'[::-1])")
    rows += _word_rows(split, UPPER_PROMPTS, "print('{word}'.upper())")
    rows += _word_rows(split, LENGTH_PROMPTS, "print(len('{word}'))")
    rows += _range_rows(split)
    rows += _triple_rows(split)
    rows += _sentence_rows(split)
    rows += _listdir_rows(split)
    return rows


def file_rows(split: str) -> list[tuple[bytes, bytes, bytes]]:
    """``read_file`` / ``write_file`` rows: (request, tool, argument)."""
    if split not in POOL_SIZES:
        raise ValueError("unknown split")
    rows = _file_rows(split, READ_PROMPTS, READ_FILE, "{name}")
    rows += _file_rows(split, WRITE_PROMPTS, WRITE_FILE, "{name}\nhello sandbox")
    return rows


#: Valid snippets that the head must copy verbatim on request ("run this").
RUN_TEMPLATES = ("print({expr})", "x = {a}; print(x * {b})", "print(sorted([{a}, {b}, {c}]))",
                 "print(len('{word}'))", "print(list(range({n})))",
                 "value = {a}; print(value + {b})", "print('{word}'.upper())",
                 "print({a} ** 2 + {b})", "text = '{word}'; print(text * 2)")
RUN_PROMPTS = ("Run this Python: {code}", "Execute: {code}", "Please run: {code}")


def _run_pool(total: int = 400) -> list[tuple[bytes, bytes]]:
    """Unique verbatim-copy items; duplicates are dropped, order is preserved."""
    rng = random.Random(77000)
    rows: list[tuple[bytes, bytes]] = []
    seen: set[bytes] = set()
    while len(rows) < total:
        template = rng.choice(RUN_TEMPLATES)
        code = template.format(expr=rng.choice(_EXPRESSIONS), a=rng.randint(2, 13),
                               b=rng.randint(3, 11), c=rng.randint(7, 11), n=rng.randint(2, 7),
                               word=rng.choice(_NAMES))
        prompt = rng.choice(RUN_PROMPTS).format(code=code).encode("utf-8")
        if prompt in seen:
            continue
        seen.add(prompt)
        rows.append((prompt, code.encode("utf-8")))
    return rows


_RUN_OFFSETS = {"train": 0, "validation": 140, "test": 160}


def run_rows(split: str, count: int | None = None) -> list[tuple[bytes, bytes]]:
    """(request, code) pairs where the correct call copies the code verbatim.

    Items come from one deduplicated pool sliced per split, so a held-out
    request never repeats a training request.
    """
    if split not in _RUN_OFFSETS:
        raise ValueError("unknown split")
    size = count if count is not None else {"train": 140, "validation": 20, "test": 20}[split]
    start = _RUN_OFFSETS[split]
    return _run_pool()[start:start + size]


_SYLLABLES = ("ba", "ce", "di", "fo", "gu", "ja", "ke", "li", "mo", "nu", "po",
              "ra", "se", "ti", "vo", "xu", "ze", "tra", "ble", "son")
_WILD_PER_SPLIT = {"train": 500, "validation": 60}
_WILD_OFFSET = {"train": 0, "validation": 500}


def _wild_word(rng: random.Random) -> str:
    """Random pronounceable word: the space is far too large to memorise."""
    return "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 3)))


def _wild_pair(category: str, rng: random.Random) -> tuple[bytes, bytes]:
    """One-line (broken, corrected) Python from a wide randomized space.

    The space is deliberately too large to memorise, so the only way to score
    these rows is to copy the pasted code out of the request and apply the
    category-level edit.
    """
    word = _wild_word(rng)
    number, other = rng.randint(2, 900), rng.randint(2, 90)
    if category == "A":
        typo = rng.choice(_BUILTIN_TYPOS + ("pint", "prnit", "prtin"))
        body = rng.choice((f"{number} {rng.choice(OPS)} {other}", f"'{word}'",
                           f"len('{word}')", f"sum([{number}, {other}])",
                           f"sorted([{other}, {number}])", f"max({number}, {other})",
                           f"{number} ** 2", f"str({number})", f"list('{word[:4]}')"))
        return f"{typo}({body})".encode(), f"print({body})".encode()
    if category == "B":
        head = rng.choice((f"if {number} > {other}", f"if {number} == {number}",
                           f"for i in range({other % 9 + 2})", f"while {other} < {number}"))
        tail = f"print({number} {rng.choice(OPS)} {other})"
        return f"{head} {tail}".encode(), f"{head}: {tail}".encode()
    if category == "C":
        inner = rng.choice((f"len('{word}')", f"'{word}'.upper()",
                            f"sum([{number}, {other}])", f"sorted([{other}, {number}])",
                            f"max({number}, {other})", f"int({other})"))
        return f"print({inner}".encode(), f"print({inner})".encode()
    if category == "D":
        if rng.random() < 0.5:
            return f"print({number} + '{other}')".encode(), f"print({number} + int('{other}'))".encode()
        return f"print('{word}' + {number})".encode(), f"print('{word}' + str({number}))".encode()
    if category == "E":
        name = rng.choice(_NAMES)
        wrong = _scramble(name, rng)
        if wrong == name:
            wrong = name[::-1]
        return (f"{name} = {number}; print({wrong})".encode(),
                f"{name} = {number}; print({name})".encode())
    return (f"print(int('{number}{rng.choice('xyz')}'))".encode(),
            f"print(int('{number}'))".encode())


def _wild_pool(total: int = 560) -> list[tuple[bytes, bytes]]:
    """Unique (broken, corrected) pairs never used as repair targets."""
    rng = random.Random(31337)
    taken = {broken for split in REPAIR_PER_SPLIT for broken, _ in repair_snippets(split)}
    taken |= {code for _, code in _run_pool()}
    rows: list[tuple[bytes, bytes]] = []
    seen: set[bytes] = set()
    while len(rows) < total:
        broken, fixed = _wild_pair(rng.choice(REPAIR_CATEGORIES), rng)
        if broken in seen or broken in taken or b"\n" in broken or b"\n" in fixed:
            continue
        seen.add(broken)
        rows.append((broken, fixed))
    return rows


def wild_pairs(split: str) -> list[tuple[bytes, bytes]]:
    """Wide, non-memorisable (broken, corrected) pairs for a split."""
    if split not in _WILD_PER_SPLIT:
        return []
    start = _WILD_OFFSET[split]
    return _wild_pool()[start:start + _WILD_PER_SPLIT[split]]


def wild_requests(split: str) -> list[tuple[bytes, bytes, bytes]]:
    """``(request, broken, corrected)`` over a space too wide to memorise."""
    rows: list[tuple[bytes, bytes, bytes]] = []
    for index, (broken, fixed) in enumerate(wild_pairs(split)):
        prompt = REPAIR_PROMPTS[index % len(REPAIR_PROMPTS)].format(code=broken.decode("utf-8"))
        rows.append((prompt.encode("utf-8"), broken, fixed))
    return rows


def wild_rows(split: str, sandbox: ByteSandbox) -> list[ToolExample]:
    """First-attempt and correction rows over the wide space.

    Labels are validated in the sandbox exactly like the repair family: a pair
    whose broken code does not fail, or whose correction does not run, is
    dropped rather than taught.
    """
    rows: list[ToolExample] = []
    for request, broken, fixed in wild_requests(split):
        failure = sandbox.call(EXEC_PYTHON, broken)
        success = sandbox.call(EXEC_PYTHON, fixed)
        if not looks_failed(failure) or looks_failed(success):
            continue
        rows.append(ToolExample(request, EXEC_PYTHON, broken, None))
        rows.append(ToolExample(request, EXEC_PYTHON, fixed, failure))
        rows.append(ToolExample(request, b"", b"", success))
    return rows


def repair_snippets(split: str) -> list[tuple[bytes, bytes]]:
    """Generated (broken, corrected) pairs for a split, across all categories."""
    if split not in REPAIR_PER_SPLIT:
        raise ValueError("unknown split")
    total = sum(REPAIR_PER_SPLIT.values())
    pairs: list[tuple[bytes, bytes]] = []
    for offset, category in enumerate(REPAIR_CATEGORIES):
        pairs.extend(_split_pairs(_category_pairs(category, total, 5150 + offset), split))
    return pairs


def repair_rows(split: str, sandbox: ByteSandbox) -> list[ToolExample]:
    """First-attempt, correction and stop rows built from real sandbox output.

    Every snippet is paired with every prompt template, and the failure text is
    the sandbox's own output for the broken code -- never a hand-written guess.
    """
    rows: list[ToolExample] = []
    for position, (broken, fixed) in enumerate(repair_snippets(split)):
        failure = sandbox.call(EXEC_PYTHON, broken)
        success = sandbox.call(EXEC_PYTHON, fixed)
        # Labels must be real: skip any generated pair where the broken code
        # does not actually fail or the correction does not actually run.
        if not looks_failed(failure) or looks_failed(success):
            continue
        prompt = REPAIR_PROMPTS[position % len(REPAIR_PROMPTS)].format(code=broken.decode("utf-8"))
        request = prompt.encode("utf-8")
        rows.append(ToolExample(request, EXEC_PYTHON, broken, None))
        rows.append(ToolExample(request, EXEC_PYTHON, fixed, failure))
        rows.append(ToolExample(request, b"", b"", success))
    return rows


def stop_rows(split: str, sandbox: ByteSandbox, fraction: float = STOP_FRACTION) -> list[ToolExample]:
    """Successful-result rows: the head should stop calling tools.

    Only a deterministic sample is executed, so building a split stays fast.
    """
    rows: list[ToolExample] = []
    candidates = code_rows(split)
    step = max(int(1 / max(fraction, 1e-6)), 1)
    for index, (request, code) in enumerate(candidates):
        if index % step:
            continue
        rows.append(ToolExample(request, b"", b"", sandbox.call(EXEC_PYTHON, code)))
    return rows


def decline_prompts(split: str, *, conversational_only: bool = False, limit: int = 400) -> list[bytes]:
    """Phase 10 dialogue prompts that carry no tool request.

    ``conversational_only`` keeps greetings/thanks/wellbeing wording; the full
    set also contains Phase 10 EDIT/RECALL wording that *looks* tool-shaped but
    is not servable by these three tools, which makes it a harder measurement.
    """
    from data.conversation_examples import conversation_examples
    from models.conversation_intents import ConversationIntent

    examples = conversation_examples(split)
    if conversational_only:
        examples = [row for row in examples if row.intent in (ConversationIntent.CHAT,
                                                              ConversationIntent.QUESTION,
                                                              ConversationIntent.CLARIFY)]
    step = max(len(examples) // limit, 1)
    prompts = []
    for example in examples[::step][:limit]:
        message = example.message.strip()
        if message and len(message) <= MAX_REQUEST_BYTES:
            prompts.append(message)
    return prompts


def decline_rows(split: str, limit: int = 400) -> list[ToolExample]:
    """Conversational prompts labelled "no tool" so the router can decline.

    These are the existing Phase 10 synthetic dialogue prompts (chat, question,
    remember/recall/forget and clarification wordings). Without them the router
    has only ever seen tool-shaped requests and would happily route "hello" to
    a tool. They are reminders, not new language: the messages are the Phase 10
    corpus, unmodified.
    """
    return [ToolExample(message, b"", b"", None) for message in decline_prompts(split, limit=limit)]


def tool_examples(split: str, sandbox: ByteSandbox | None = None) -> list[ToolExample]:
    """Complete rows for a split. ``sandbox`` is required for real results."""
    if split not in POOL_SIZES:
        raise ValueError("unknown split")
    rows = [ToolExample(request, EXEC_PYTHON, code, None) for request, code in code_rows(split)]
    rows += [ToolExample(request, EXEC_PYTHON, code, None) for request, code in run_rows(split)]
    rows += [ToolExample(request, tool, argument, None) for request, tool, argument in file_rows(split)]
    rows += decline_rows(split)
    if sandbox is None:
        return rows
    rows += stop_rows(split, sandbox)
    rows += repair_rows(split, sandbox)
    rows += wild_rows(split, sandbox)
    return rows


def routing_prompts(split: str = "test", count: int = 20) -> list[tuple[bytes, bytes, bytes]]:
    """Held-out routing prompts: (request, tool, argument), no prior result."""
    candidates: list[tuple[bytes, bytes, bytes]] = []
    for request, code in code_rows(split):
        candidates.append((request, EXEC_PYTHON, code))
    for request, tool, argument in file_rows(split):
        candidates.append((request, tool, argument))
    if len(candidates) < count:
        raise ValueError("not enough held-out routing prompts")
    step = max(len(candidates) // count, 1)
    return candidates[::step][:count]


def debugging_tasks(split: str = "test", count: int = 5) -> list[tuple[bytes, bytes, bytes]]:
    """(request, broken code, corrected code) triples from the held-out split.

    The Phase 15 benchmark fixes the task count at five; the generated pool may
    hold one more category than the benchmark scores.
    """
    tasks = []
    for position, (broken, fixed) in enumerate(repair_snippets(split)[:count]):
        prompt = REPAIR_PROMPTS[position % len(REPAIR_PROMPTS)].format(code=broken.decode("utf-8"))
        tasks.append((prompt.encode("utf-8"), broken, fixed))
    return tasks
