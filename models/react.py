"""ReAct tool use as plain bytes: ``Thought -> Action -> Observation -> Answer``.

The prototype needed the model to emit exact control-byte frames; one wrong
byte cost the entire turn, and a separate channel had to be plumbed through
every module. Here tool use is ordinary text in the byte stream:

    Thought: I need the product.
    Action: exec_python("print(12 * 12)")
    Observation: 144
    Answer: 144

The host intercepts an ``Action:`` line with one regex, runs the tool, and
appends a real ``Observation:``. There is no control token, no frame parser and
no separate result channel -- the observation is just more bytes for the same
next-byte objective.

Two rules keep the loop honest:

1. **The model never fabricates an observation.** Whatever the model emits
   after an ``Action:`` line is truncated before the host appends the real
   result, so every ``Observation:`` in a transcript came from a tool.
2. **Unknown tools and unparsable calls fail closed.** No guess is executed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Mapping

import torch

if TYPE_CHECKING:  # pragma: no cover
    pass

THOUGHT = b"Thought:"
ACTION = b"Action:"
OBSERVATION = b"Observation:"
ANSWER = b"Answer:"
REQUEST = b"Request:"
MEMORY_PREFIX = b"[MEM]"

#: ``Action: tool("arg")`` or ``Action: tool("a", "b")``, at the end of a line.
ACTION_RE = re.compile(rb"Action:[ \t]*([a-z_][a-z0-9_]*)\(([^\n]*)\)[ \t]*$")
QUOTED_ARG_RE = re.compile(rb'"((?:[^"\\]|\\.)*)"')
MAX_OBSERVATION_BYTES = 160


@dataclass(frozen=True)
class ActionCall:
    """A parsed ``Action:`` line."""

    tool: bytes
    args: tuple[bytes, ...]
    raw: bytes

    @property
    def name(self) -> str:
        return self.tool.decode("utf-8", "replace")


def _match_action(line: bytes) -> ActionCall | None:
    """Parse one line as an action, or return ``None``.

    Accepts only a *complete* argument list: quoted strings separated by
    commas. An unterminated quote or a stray token is rejected, so a
    half-written call never reaches a tool.
    """
    match = ACTION_RE.search(line)
    if match is None:
        return None
    inner = match.group(2)
    if inner.count(b'"') % 2:
        return None  # an unterminated string
    remainder = QUOTED_ARG_RE.sub(b"", inner).replace(b",", b"").strip()
    if remainder:
        return None  # something other than quoted, comma-separated arguments
    args = tuple(re.sub(rb'\\(.)', rb"\1", arg) for arg in QUOTED_ARG_RE.findall(inner))
    return ActionCall(tool=match.group(1), args=args, raw=line)


@dataclass
class ReactStep:
    """One executed round: the action and the bytes the tool returned."""

    action: ActionCall
    result: bytes
    observation: bytes


@dataclass
class ReactRun:
    """A complete trajectory: prompt, transcript, executed steps, answer."""

    request: bytes
    transcript: bytes
    answer: bytes
    steps: list[ReactStep] = field(default_factory=list)
    rounds: int = 0
    declined: bool = False

    @property
    def tool_calls(self) -> int:
        return len(self.steps)

    @property
    def actions(self) -> list[str]:
        return [step.action.name for step in self.steps]


def parse_action(text: bytes) -> ActionCall | None:
    """Parse the final complete ``Action:`` line, or return ``None``.

    Only a *terminated* call is accepted: the closing paren must be the last
    non-space byte of its line. A half-written call never reaches a tool.
    """
    if not isinstance(text, (bytes, bytearray)):
        raise TypeError("text must be bytes")
    for line in reversed(bytes(text).split(b"\n")):
        stripped = line.strip()
        if not stripped:
            continue
        return _match_action(stripped)  # the last non-empty line decides
    return None


def last_action(text: bytes) -> ActionCall | None:
    """Last parsable ``Action:`` line anywhere in ``text``.

    The loop uses this rather than :func:`parse_action` because a model that
    emits the action *and* a continuation in one chunk still deserves to have
    its action executed -- and the continuation discarded, so a fabricated
    ``Observation:`` can never survive.
    """
    for line in reversed(bytes(text).split(b"\n")):
        stripped = line.strip()
        if not stripped:
            continue
        action = _match_action(stripped)
        if action is not None:
            return action
    return None


def format_action(tool: bytes, args: tuple[bytes, ...]) -> bytes:
    """Render an action line the way the dataset and the model both see it."""
    rendered = b", ".join(b'"' + arg.replace(b"\\", b"\\\\").replace(b'"', b'\\"') + b'"' for arg in args)
    return ACTION + b" " + tool + b"(" + rendered + b")"


def sandbox_observation(result: bytes) -> bytes:
    """Reduce the sandbox envelope to what the stream should carry.

    Success: the program's stdout. Failure: the last traceback line, so the
    model sees ``NameError: name 'pritn' is not defined`` instead of a
    multi-line traceback with a temporary path in it. One function is used by
    the dataset and by the runtime, so training observations and live
    observations are byte-identical.
    """
    if not isinstance(result, (bytes, bytearray)):
        raise TypeError("result must be bytes")
    stdout, _, rest = bytes(result).partition(b"\n---STDERR---\n")
    stderr = rest.split(b"\n---EXIT---\n")[0]
    if stderr.strip():
        lines = [line.strip() for line in stderr.strip().split(b"\n") if line.strip()]
        return lines[-1] if lines else b"ERROR: execution failed"
    return stdout.strip() if stdout.strip() else b"(no output)"


def format_observation(result: bytes, limit: int = MAX_OBSERVATION_BYTES) -> bytes:
    """Collapse tool output to one bounded line.

    Newlines become spaces so an observation can never start a fake
    ``Action:`` or ``Answer:`` line, and truncation is marked explicitly.
    """
    if not isinstance(result, (bytes, bytearray)):
        raise TypeError("result must be bytes")
    if limit < 1:
        raise ValueError("limit must be positive")
    collapsed = b" ".join(bytes(result).split())
    if len(collapsed) > limit:
        collapsed = collapsed[:limit] + b"..."
    return collapsed if collapsed else b"(empty)"


def observation_line(result: bytes, limit: int = MAX_OBSERVATION_BYTES) -> bytes:
    return OBSERVATION + b" " + format_observation(result, limit)


def answer_line(answer: bytes) -> bytes:
    return ANSWER + b" " + b" ".join(answer.split())


def render_prompt(request: bytes, memory: bytes = b"") -> bytes:
    """Bytes the model conditions on: optional memory prefix plus the request."""
    parts: list[bytes] = []
    if memory:
        parts.append(memory.rstrip(b"\n"))
    parts.append(REQUEST + b" " + b" ".join(request.split()))
    parts.append(THOUGHT)
    return b"\n".join(parts)


def render_trajectory(request: bytes, body: bytes, memory: bytes = b"") -> bytes:
    """Full training text: the prompt (ending in ``Thought:``) plus the body."""
    prompt = render_prompt(request, memory)
    if not body:
        return prompt
    body = bytes(body).lstrip(b"\n")
    if not body.startswith(b" "):
        body = b" " + body
    return prompt + body


def extract_answer(transcript: bytes) -> bytes:
    """Return the last ``Answer:`` line, or empty bytes if there is none."""
    found = b""
    for line in transcript.split(b"\n"):
        if line.startswith(ANSWER):
            found = line[len(ANSWER):].strip()
    return found


def truncate_at_action(transcript: bytes, action: ActionCall) -> bytes:
    """Drop anything the model emitted after its action line.

    Guarantees the next ``Observation:`` is the host's, not the model's.
    """
    marker = action.raw
    index = transcript.rfind(marker)
    if index < 0:
        return transcript
    return transcript[: index + len(marker)]


class ReactLoop:
    """Generate -> intercept one action -> execute -> append the observation.

    The loop is deliberately dumb: it owns no policy about *what* to run. It
    parses one regex, refuses what it cannot parse, executes through an
    injected tool mapping, and feeds real bytes back.
    """

    def __init__(self, model, tools: Mapping[bytes, Callable[..., bytes]], *,
                 max_rounds: int = 3, max_new_bytes: int = 160,
                 temperature: float = 0.0, memory: Callable[[bytes], bytes] | None = None) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if max_new_bytes < 1:
            raise ValueError("max_new_bytes must be positive")
        self.model = model
        # Tool names are bytes in the stream; accept str keys for convenience.
        self.tools = {name.encode("utf-8") if isinstance(name, str) else name: handler
                      for name, handler in tools.items()}
        self.max_rounds = max_rounds
        self.max_new_bytes = max_new_bytes
        self.temperature = temperature
        self.memory = memory

    def execute(self, action: ActionCall) -> bytes:
        """Run a parsed action. Unknown tools and bad arities fail closed."""
        handler = self.tools.get(action.tool)
        if handler is None:
            return b"ERROR: unknown tool: " + action.tool
        try:
            result = handler(*action.args)
        except TypeError:
            return b"ERROR: wrong arguments for tool: " + action.tool
        except Exception as error:  # a tool may raise; the loop must not
            return b"ERROR: " + type(error).__name__.encode("utf-8", "replace")
        if not isinstance(result, (bytes, bytearray)):
            raise TypeError("tools must return bytes")
        return bytes(result)

    def run(self, request: bytes) -> ReactRun:
        """Run one request to an answer or to the round budget."""
        if not isinstance(request, (bytes, bytearray)):
            raise TypeError("request must be bytes")
        request = b" ".join(bytes(request).split())
        memory = self.memory(request) if self.memory is not None else b""
        transcript = render_prompt(request, memory)
        steps: list[ReactStep] = []
        rounds = 0
        for _ in range(self.max_rounds):
            ids = _encode(transcript, self.model)
            generated = self.model.generate(ids, max_new_bytes=self.max_new_bytes,
                                            temperature=self.temperature)
            text = _decode(generated)
            rounds += 1
            if not text.strip():
                break
            transcript = transcript + text
            action = last_action(text)
            if action is None:
                break
            # Rule 1: the model's continuation after the action is discarded.
            transcript = truncate_at_action(transcript, action)
            result = self.execute(action)
            observation = format_observation(result)
            transcript = transcript + b"\n" + OBSERVATION + b" " + observation
            steps.append(ReactStep(action=action, result=result, observation=observation))
            if any(line.startswith(ANSWER) for line in transcript.split(b"\n")):
                break
            transcript = transcript + b"\n" + THOUGHT
        answer = extract_answer(transcript)
        return ReactRun(request=request, transcript=transcript, answer=answer,
                        steps=steps, rounds=rounds, declined=not steps and not answer)


def _encode(text: bytes, model) -> torch.Tensor:
    from .foundation import encode_bytes

    limit = int(model.config.max_seq_len)
    ids = encode_bytes(text, bos=True)
    return ids[-limit:]


def _decode(ids) -> bytes:
    from .foundation import decode_ids

    return decode_ids(ids)
