"""Regression tests for the retained byte-level runtime (Level 4 of the audit).

The runtime is three pieces of host code, no learned parameters:

* :mod:`models.react` -- one regex intercepts ``Action: tool("arg")`` in the byte
  stream; everything after the action on that turn is discarded, the tool runs,
  and a real ``Observation:`` is appended.
* :mod:`models.sandbox` -- the tool executor: directory-confined, time- and
  size-bounded.
* :mod:`models.foundation_memory` -- memory enters the stream as
  ``[MEM] topic = value`` prefix bytes; nothing else crosses the boundary.

A scripted stand-in model is used throughout: these tests are about the
protocol, not about generation quality.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.foundation import decode_ids
from models.foundation_memory import (MEMORY_MARK, MemoryEntry, PrefixMemory,
                                      ScriptedRetriever, topic_of_question)
from models.react import (
    OBSERVATION, REF_CLOSE, REF_OPEN, ActionCall, ReactLoop,
    format_action, format_observation, last_action, observation_line,
    parse_action, reference_spans, reference_tools, render_prompt,
    resolve_arguments, resolve_reference, sandbox_observation,
    strip_references, truncate_at_action,
)
from models.sandbox import ByteSandbox, looks_failed


class ScriptedModel:
    """Minimal ``generate`` contract: return the next scripted byte string."""

    def __init__(self, outputs: list[bytes], max_seq_len: int = 512) -> None:
        self.outputs = list(outputs)
        self.config = SimpleNamespace(max_seq_len=max_seq_len)
        self.prompts: list[bytes] = []

    def generate(self, prefix: torch.Tensor, *, max_new_bytes: int = 192,
                 temperature: float = 0.0) -> torch.Tensor:
        self.prompts.append(decode_ids(prefix))
        text = self.outputs.pop(0) if self.outputs else b""
        return torch.tensor(list(text), dtype=torch.long)


# ---------------------------------------------------------------------------
# Action grammar
# ---------------------------------------------------------------------------
def test_parse_action_accepts_only_complete_terminated_calls():
    call = parse_action(b' Thought: need it\nAction: add("2", "3")')
    assert isinstance(call, ActionCall)
    assert call.tool == b"add" and call.args == (b"2", b"3")
    assert call.name == "add"
    # A half-written call never reaches a tool.
    assert parse_action(b'Action: add("2", "3"') is None
    assert parse_action(b'Action: add(2, 3)') is None       # unquoted arguments
    assert parse_action(b'Action: add("2") trailing') is None
    assert parse_action(b"Thought: no action here") is None
    assert parse_action(b"") is None
    with pytest.raises(TypeError):
        parse_action("not bytes")


def test_last_action_ignores_text_after_the_call():
    text = b' Thought: ok\nAction: double("21")\nObservation: I made this up'
    call = last_action(text)
    assert call is not None and call.tool == b"double" and call.args == (b"21",)
    assert last_action(b"Thought: nothing") is None


def test_format_action_escapes_quotes_and_backslashes():
    rendered = format_action(b"echo", (b'a"b', b"c\\d"))
    assert rendered == b'Action: echo("a\\"b", "c\\\\d")'
    # Plain arguments round-trip through the parser unchanged.
    plain = format_action(b"echo", (b"one", b"two words"))
    again = parse_action(plain)
    assert again is not None and again.args == (b"one", b"two words")
    # A call whose arguments contain an escaped quote fails closed rather than
    # reaching a tool with mangled bytes (the parser rejects odd quote counts).
    assert parse_action(rendered) is None


def test_observations_are_collapsed_to_one_bounded_line():
    assert format_observation(b"line one\nAction: fake\nline two") == \
        b"line one Action: fake line two"
    long = b"x" * 200
    assert format_observation(long, limit=10) == b"x" * 10 + b"..."
    assert format_observation(b"   ") == b"(empty)"
    assert observation_line(b"42") == OBSERVATION + b" 42"


def test_sandbox_observation_reduces_the_envelope_to_one_line():
    envelope = b"4\n---STDERR---\n\n---EXIT---\n0"
    assert sandbox_observation(envelope) == b"4"
    failing = b"\n---STDERR---\nTraceback (most recent call last):\nValueError: bad\n---EXIT---\n1"
    assert sandbox_observation(failing) == b"ValueError: bad"


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def test_reactloop_executes_the_action_and_discards_fabricated_observations():
    model = ScriptedModel([
        b' I will compute it.\nAction: double("21")\nObservation: 999 (invented)',
        b"\nAnswer: 42",
    ])
    loop = ReactLoop(model, {b"double": lambda x: str(int(x) * 2).encode()},
                     max_rounds=3)
    run = loop.run(b"What is 21 doubled?")
    assert run.answer == b"42"
    assert run.tool_calls == 1 and run.actions == ["double"]
    assert run.steps[0].result == b"42" and run.steps[0].observation == b"42"
    assert b"invented" not in run.transcript, "the host must not keep model-written observations"
    assert run.transcript.count(OBSERVATION) == 1
    assert not run.declined
    # The prompt the model saw is the request plus a Thought cue.
    assert model.prompts[0].startswith(b"Request: What is 21 doubled?\nThought:")


def test_reactloop_fails_closed_on_unknown_tools_and_bad_arities():
    model = ScriptedModel([b'Action: rm_rf("/")\n', b'Action: double()\n'])
    loop = ReactLoop(model, {b"double": lambda x: x}, max_rounds=1)
    run = loop.run(b"do something")
    assert run.steps[0].result == b"ERROR: unknown tool: rm_rf"
    assert b"ERROR: unknown tool" in run.transcript
    assert run.answer == b""

    model = ScriptedModel([b'Action: double()\n'])
    loop = ReactLoop(model, {b"double": lambda x: x}, max_rounds=1)
    run = loop.run(b"do something")
    assert run.steps[0].result == b"ERROR: wrong arguments for tool: double"


def test_reactloop_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        ReactLoop(ScriptedModel([]), {}, max_rounds=0)
    with pytest.raises(ValueError):
        ReactLoop(ScriptedModel([]), {}, max_new_bytes=0)


def test_truncate_at_action_drops_the_continuation():
    text = b'Thought: x\nAction: f("1")\nObservation: invented'
    call = last_action(text)
    assert call is not None
    assert truncate_at_action(text, call) == b'Thought: x\nAction: f("1")'


def test_render_prompt_puts_the_memory_prefix_before_the_request():
    prompt = render_prompt(b"What is my name?", b"[MEM] my:name = Ada")
    assert prompt == b"[MEM] my:name = Ada\nRequest: What is my name?\nThought:"


# ---------------------------------------------------------------------------
# References: the model points, the host resolves
# ---------------------------------------------------------------------------
def test_reference_spans_and_stripping():
    marked = b"compare " + REF_OPEN + b"main.py" + REF_CLOSE + b" and " \
        + REF_OPEN + b"utils.py" + REF_CLOSE
    assert reference_spans(marked) == [b"main.py", b"utils.py"]
    assert strip_references(marked) == b"compare main.py and utils.py"


def test_reference_resolution_accepts_an_index_or_literal_bytes_only():
    request = b"open " + REF_OPEN + b"main.py" + REF_CLOSE
    assert resolve_reference(b"1", request) == b"main.py"
    assert resolve_reference(b"main.py", request) == b"main.py"
    with pytest.raises(Exception):
        resolve_reference(b"2", request)      # out of range
    with pytest.raises(Exception):
        resolve_reference(b"other.py", request)  # never shown to the model


def test_reference_tools_turn_failures_into_error_observations():
    request = b"open " + REF_OPEN + b"main.py" + REF_CLOSE
    tools = reference_tools({b"cat": lambda path: b"contents of " + path}, request)
    assert tools[b"cat"](b"[REF]1[/REF]") == b"contents of main.py"
    failure = tools[b"cat"](b"[REF]7[/REF]")
    assert failure.startswith(b"ERROR: unresolved reference:")
    assert resolve_arguments((b"[REF]1[/REF]",), request) == (b"main.py",)


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------
def test_sandbox_executes_python_and_reports_failure(tmp_path):
    with ByteSandbox(tmp_path / "box") as sandbox:
        assert b"4" in sandbox.execute_python(b"print(2 + 2)")
        failed = sandbox.execute_python(b"raise ValueError('boom')")
        assert looks_failed(failed) and b"ValueError: boom" in failed
        assert looks_failed(b"ERROR: nope") and looks_failed(b"")
        assert not looks_failed(b"4\n---STDERR---\n\n---EXIT---\n0")


def test_sandbox_confines_paths_and_bounds_sizes(tmp_path):
    with ByteSandbox(tmp_path / "box") as sandbox:
        assert sandbox.write_file(b"note.txt", b"hello").startswith(b"OK")
        assert sandbox.read_file(b"note.txt") == b"hello"
        assert sandbox.read_file(b"../escape").startswith(b"ERROR")
        assert sandbox.write_file(b"/etc/passwd", b"x").startswith(b"ERROR")
        assert sandbox.execute_python(b"print('x')" * 10_000) .startswith(b"ERROR")


def test_disabled_sandbox_refuses_to_execute(tmp_path):
    with ByteSandbox(tmp_path / "box", enabled=False) as sandbox:
        assert sandbox.execute_python(b"print(1)") == b"ERROR: sandbox execution is disabled."


# ---------------------------------------------------------------------------
# Memory as prefix bytes
# ---------------------------------------------------------------------------
def test_prefix_memory_renders_retrieved_entries_as_bytes():
    store = ScriptedRetriever({b"my:favorite_color": b"blue",
                               b"person:alice:workplace": b"Acme Labs"})
    memory = PrefixMemory(store, top_k=3)
    prefix = memory.prefix(b"What is my favorite color?")
    assert prefix == MEMORY_MARK + b" my:favorite_color = blue"
    assert topic_of_question(b"What is my favorite color?") == b"my:favorite_color"
    assert topic_of_question(b"unrelated") == b""


def test_prefix_memory_is_bounded_and_fails_safe():
    def broken(_query: bytes):
        raise RuntimeError("store offline")

    assert PrefixMemory(broken).prefix(b"anything") == b""
    entries = [MemoryEntry(topic=b"t", value=b"v" * 200)]
    memory = PrefixMemory(ScriptedRetriever({}), top_k=1, max_entry_bytes=8)
    assert memory.render(entries) == MEMORY_MARK + b" t = " + b"v" * 8
    many = [MemoryEntry(topic=f"t{index}".encode(), value=b"v") for index in range(5)]
    capped = PrefixMemory(lambda _query: many, top_k=2)
    assert len(capped.retrieve(b"q")) == 2
    assert capped.prefix(b"q").count(b"\n") == 1
    with pytest.raises(ValueError):
        PrefixMemory(broken, top_k=0)
    with pytest.raises(TypeError):
        PrefixMemory(ScriptedRetriever({})).retrieve("text")
