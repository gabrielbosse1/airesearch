"""Interactive ByteCore tool agent: two generations behind one sandbox.

**Default (Phase 16, ByteCore v2).** The foundation model is a standard causal
byte LM and tool use is ReAct text. Each turn the model continues a plain byte
stream and the host intercepts one ``Action:`` line with a regex::

    Request: Calculate 6 * 7.
    Thought: ...
    Action: exec_python("print(6 * 7)")
    Observation: 42
    Answer: 42

The sandbox runs the call, the host appends the real ``Observation:`` (anything
the model emitted after its action is discarded, so observations are never
fabricated), and the model finishes with ``Answer:``. No control frames, no copy
head, no separate result channel.

**``--legacy-frames`` (Phase 15, superseded).** The frozen pointer-generator
policy that emits ``[TOOL_CALL] name [TOOL_ARG_START] argument [TOOL_ARG_END]``
frames. It is kept runnable for comparison and for reproducing published
measurements; no Phase 16 module imports it.

Run it with::

    python -m demo.tool_agent --once "Calculate 6 * 7."
    python -m demo.tool_agent --with-memory --user alice
    python -m demo.tool_agent --legacy-frames --once "Calculate 6 * 7."
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch

from models.react import ReactLoop, sandbox_observation
from models.sandbox import STDERR_MARK, ByteSandbox, looks_failed
from models.tool_use import (MAX_ARGUMENT_BYTES, ROUTE_NAMES, ToolCall, ToolCallPolicy,
                             find_tool_call, load_tool_policy, sanitize_result)
from training.train_confidence import load_generator


@dataclass
class AgentTurn:
    """One user turn: what was executed, what the sandbox returned, what to say."""

    response: bytes
    route: str
    source: str
    rounds: int
    calls: list[bytes] = field(default_factory=list)
    arguments: list[bytes] = field(default_factory=list)
    results: list[bytes] = field(default_factory=list)
    frames: list[tuple[int, ...]] = field(default_factory=list)
    repaired: bool = False

    @property
    def tool_calls(self) -> int:
        return len(self.calls)


class ToolAgent:
    """Route a request to the sandbox, then read the bytes the sandbox returned."""

    def __init__(self, policy: ToolCallPolicy, sandbox: ByteSandbox, *, session=None,
                 max_rounds: int = 3, threshold: float = 0.6, max_bytes: int = MAX_ARGUMENT_BYTES) -> None:
        # Duck-typed on generate() so the loop can be exercised with scripted
        # heads in tests; a wrong object still fails immediately.
        if not callable(getattr(policy, "generate", None)):
            raise TypeError("policy must expose a generate(request, result) method")
        if not isinstance(sandbox, ByteSandbox):
            raise TypeError("sandbox must be a ByteSandbox")
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least one")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be within [0, 1]")
        if not 1 <= max_bytes <= MAX_ARGUMENT_BYTES:
            raise ValueError("max_bytes must be within the argument limit")
        self.policy = policy
        self.sandbox = sandbox
        self.session = session
        self.max_rounds = int(max_rounds)
        self.threshold = float(threshold)
        self.max_bytes = int(max_bytes)
        self.transcript: list[tuple[bytes, bytes, bytes]] = []

    # ------------------------------------------------------------- internals
    def _emit(self, request: bytes, result: bytes | None):
        """Ask the head for one frame; returns (generation, call or None)."""
        generation = self.policy.generate(request, result, threshold=self.threshold, max_bytes=self.max_bytes)
        if generation.call is None:
            return generation, None
        if not generation.complete:
            # Fail closed: an unterminated argument is never dispatched.
            return generation, None
        try:
            call, _ = find_tool_call(generation.frame)
        except ValueError:
            return generation, None
        if call is None or call != generation.call:
            # The parser and the emitted frame must agree before execution.
            return generation, None
        return generation, call

    def _render(self, call: ToolCall, result: bytes) -> bytes:
        if looks_failed(result):
            stderr = result.split(STDERR_MARK)[-1].split(b"\n---EXIT---\n")[0]
            tail = [line for line in stderr.strip().splitlines() if line.strip()]
            reason = tail[-1] if tail else result.strip()
            return b"The sandbox reported an error: " + reason
        # Only the sandbox's stdout is reported; the stderr envelope stays in
        # the transcript that the head reads on the next turn.
        body = result.split(STDERR_MARK)[0].strip()
        if not body:
            body = result.strip()
        if call.tool == b"read_file":
            return b"File contents: " + body
        return body

    # ------------------------------------------------------------------ main
    def respond(self, message: bytes) -> AgentTurn:
        if isinstance(message, str):
            message = message.encode("utf-8")
        if not isinstance(message, (bytes, bytearray)) or not message.strip():
            raise ValueError("message must be non-empty bytes")
        request = bytes(message)
        generation, call = self._emit(request, None)
        if call is None:
            if self.session is not None:
                outcome = self.session.respond(request)
                return AgentTurn(response=outcome.response, route="none",
                                 source="phase12_conversation_session", rounds=0)
            return AgentTurn(response=b"I have no tool for that request.", route=ROUTE_NAMES[generation.route],
                             source="tool_router_declined", rounds=0)
        result: bytes | None = None
        turn = AgentTurn(response=b"", route=generation.route_name, source="sandbox", rounds=0)
        for _ in range(self.max_rounds):
            assert call is not None
            turn.calls.append(call.tool)
            turn.arguments.append(call.argument)
            turn.frames.append(generation.frame)
            result = self.sandbox.call(call.tool, call.argument)
            turn.results.append(result)
            turn.rounds += 1
            self.transcript.append((call.tool, call.argument, result))
            if turn.rounds > 1:
                turn.repaired = True
            if not looks_failed(result):
                break
            # Inject the sandbox bytes for the next routing decision.
            next_generation, next_call = self._emit(request, result)
            generation = next_generation
            if next_call is None:
                break
            turn.route = next_generation.route_name
            call = next_call
        turn.response = self._render(call, result if result is not None else b"")
        return turn

    def injected_context(self) -> bytes:
        """The exact bytes the head would see as ``[TOOL_RESULT]`` next turn."""
        if not self.transcript:
            return b""
        return sanitize_result(self.transcript[-1][2])


def build_session(args, generator):
    """Optional Phase 12 conversational fallback (unchanged routing inside)."""
    from models.contextual_response import load_contextual_decoder
    from models.conversation_v3 import VectorConversationSession
    from models.diverse_response import load_diverse_decoder
    from models.memory_model import MemoryModel
    from models.vector_memory import VectorConversationMemory, load_vector_reader

    decoder = load_contextual_decoder(args.response, generator_path=args.generator)
    codec, reader = load_vector_reader(args.vector_reader, generator_path=args.generator)
    diverse = load_diverse_decoder(args.diverse_response, generator_path=args.generator)
    bank = MemoryModel(d_model=generator.d_model, capacity=192)
    memory = VectorConversationMemory(bank, codec, args.memory)
    return VectorConversationSession(generator, decoder, memory, reader, diverse, args.user.encode("utf-8"),
                                     min_confidence=args.threshold, episode_limit=args.episode_limit,
                                     record_episodes=not args.no_episode_storage)


def _legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", default="runs/phase12/generator.pt")
    parser.add_argument("--policy", default="runs/phase15/tool_policy.pt")
    parser.add_argument("--sandbox-dir", default="runs/phase15/sandbox")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--with-memory", action="store_true", help="fall back to the Phase 12 session for non-tool turns")
    parser.add_argument("--memory", help="opt-in UNENCRYPTED Phase 12 memory checkpoint")
    parser.add_argument("--response", default="runs/phase12/response.pt")
    parser.add_argument("--vector-reader", default="runs/phase12/vector_reader.pt")
    parser.add_argument("--diverse-response", default="runs/phase12/diverse_response.pt")
    parser.add_argument("--user", default="local")
    parser.add_argument("--episode-limit", type=int, default=5)
    parser.add_argument("--no-episode-storage", action="store_true")
    return parser


def main_legacy(args) -> None:
    torch.set_num_threads(1)

    generator = load_generator(args.generator, "cpu", args.heads)
    policy = load_tool_policy(args.policy, generator=generator, generator_path=args.generator)
    session = build_session(args, generator) if args.with_memory else None
    sandbox = ByteSandbox(args.sandbox_dir, timeout=args.timeout)
    agent = ToolAgent(policy, sandbox, session=session, max_rounds=args.max_rounds, threshold=args.threshold)

    print("ByteCore tool agent. The head routes; the sandbox verifies. /quit exits; /files lists the sandbox.")
    print(f"Sandbox root: {sandbox.root}")
    while True:
        try:
            line = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line == "/quit":
            break
        if line == "/files":
            print("Sandbox files:", sorted(path.name for path in sandbox.root.iterdir()))
            continue
        turn = agent.respond(line.encode("utf-8"))
        print("Assistant:", turn.response.decode("utf-8", errors="replace"))
        detail = [f"route={turn.route}", f"rounds={turn.rounds}", f"source={turn.source}"]
        if turn.tool_calls:
            detail.append("calls=" + ",".join(call.decode() for call in turn.calls))
        if turn.repaired:
            detail.append("self-corrected=yes")
        if agent.injected_context():
            detail.append("injected=" + repr(agent.injected_context()[:48]))
        print("  [" + ", ".join(detail) + "]")




# ==========================================================================
# Phase 16: ReAct agent over the foundation model.
# ==========================================================================
class ReactAgent:
    """Foundation model + sandbox + optional memory prefix."""

    def __init__(self, model, sandbox: ByteSandbox, *, memory=None, max_rounds: int = 3,
                 max_new_bytes: int = 200, temperature: float = 0.0) -> None:
        self.model = model
        self.sandbox = sandbox
        self.memory = memory
        self.tools = {
            b"exec_python": self._exec_python,
            b"read_file": self.sandbox.read_file,
            b"write_file": self.sandbox.write_file,
            b"recall": self._recall,
        }
        self.loop = ReactLoop(model, self.tools, max_rounds=max_rounds,
                              max_new_bytes=max_new_bytes, temperature=temperature,
                              memory=memory.prefix if memory is not None else None)

    def _exec_python(self, code: bytes) -> bytes:
        return sandbox_observation(self.sandbox.execute_python(code))

    def _recall(self, topic: bytes) -> bytes:
        """Answer a recall action from the prefix store, never from weights."""
        if self.memory is None:
            return b"ERROR: memory is disabled"
        key = topic[3:] if topic.startswith(b"my:") else topic
        queries = (b"What is my " + key.replace(b"_", b" ") + b"?",
                   key.replace(b":", b" ").replace(b"_", b" ") + b"?")
        for query in queries:
            for entry in self.memory.retrieve(query):
                return entry.value
        return b"ERROR: no stored fact for " + topic

    def respond(self, request: bytes):
        from models.foundation_memory import MemoryPrefixReport

        run = self.loop.run(request)
        report = MemoryPrefixReport(query=request, entries=list(self.memory.retrieve(request))
                                    if self.memory is not None else [], prefix=b"")
        return run, report


def _build_prefix_memory(args) -> "object | None":
    """Wire Phase 12 vector facts into the prefix interface (optional)."""
    try:
        from models.foundation_memory import PrefixMemory, VectorFactRetriever
        from models.memory_model import MemoryModel
        from models.vector_memory import VectorConversationMemory, load_vector_reader
        from training.train_confidence import load_generator

        generator = load_generator(args.generator, "cpu", args.heads).eval()
        codec, reader = load_vector_reader(args.vector_reader, generator_path=args.generator)
        bank = MemoryModel(d_model=generator.d_model, capacity=192, n_heads=generator.n_heads).eval()
        store = VectorConversationMemory(bank, codec, args.memory)
        retriever = VectorFactRetriever(generator, store, reader, user_id=args.user.encode(),
                                        top_k=args.top_k)
        return PrefixMemory(retriever, top_k=args.top_k)
    except Exception as error:  # memory is optional; never break the agent
        print(f"[memory unavailable: {type(error).__name__}: {error}]")
        return None


def main() -> None:
    parser = _legacy_parser()
    parser.add_argument("--model", default="runs/phase16/foundation.pt")
    parser.add_argument("--max-new-bytes", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--once", default=None, help="run a single request and exit")
    parser.add_argument("--legacy-frames", action="store_true",
                        help="use the superseded Phase 15 control-frame loop")
    args = parser.parse_args()
    if args.legacy_frames:
        main_legacy(args)
        return

    from models.foundation import load_foundation

    torch.set_num_threads(1)
    model = load_foundation(args.model, device="cpu")
    sandbox = ByteSandbox(args.sandbox_dir or "runs/phase16/sandbox", timeout=args.timeout)
    memory = _build_prefix_memory(args) if args.with_memory else None
    agent = ReactAgent(model, sandbox, memory=memory, max_rounds=args.max_rounds,
                       max_new_bytes=args.max_new_bytes, temperature=args.temperature)
    print("ByteCore v2 tool agent (ReAct). The model proposes; the sandbox verifies. /quit exits.")

    def once(text: str) -> None:
        run, report = agent.respond(text.encode("utf-8"))
        print(run.transcript.decode("utf-8", "replace"))
        detail = [f"rounds={run.rounds}"]
        if run.actions:
            detail.append("calls=" + ",".join(run.actions))
        if report.entries:
            detail.append("memory=" + ",".join(entry.topic.decode() for entry in report.entries))
        print("  [" + ", ".join(detail) + "]")

    if args.once:
        once(args.once)
        sandbox.close()
        return

    while True:
        try:
            line = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line == "/quit":
            break
        if line == "/files":
            print("Sandbox files:", sorted(path.name for path in sandbox.root.iterdir()))
            continue
        once(line)
    sandbox.close()


if __name__ == "__main__":
    main()
