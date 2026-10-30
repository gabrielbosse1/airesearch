"""Interactive Phase 15 tool-use agent: the model routes, the sandbox verifies.

Loop per user turn (see AGENTS.md, Phase 15):

1. The user request is packed and the frozen backbone produces features.
2. The tool head routes (no-tool / exec_python / read_file / write_file) and,
   when a tool is chosen, decodes the argument bytes into a frame
   ``[TOOL_CALL] name [TOOL_ARG_START] argument [TOOL_ARG_END]``.
3. The host parses that frame (incomplete frames are never executed), runs it
   in :class:`models.sandbox.ByteSandbox`, and injects the real result bytes
   back as ``[TOOL_RESULT] ...`` on the next turn.
4. The loop stops when the sandbox result is error free, when the head declines
   to call another tool, or when ``--max-rounds`` is reached.

Nothing here evaluates whether the code is *right*: the sandbox decides that and
returns bytes. The final reply is rendered by the host from those bytes -- the
head emits tool calls, not prose.

Run it with:
    python -m demo.tool_agent --policy runs/phase15/tool_policy.pt
    python -m demo.tool_agent --policy runs/phase15/tool_policy.pt --with-memory
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch

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


def main() -> None:
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
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
