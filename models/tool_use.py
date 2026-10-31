"""Phase 15: external tool-use frames, routing head and argument decoder.

The model does **not** verify anything. It maps a request (and, on later turns,
the bytes an external sandbox returned) onto an explicit tool-call frame:

    [TOOL_CALL] <tool name bytes> [TOOL_ARG_START] <argument bytes> [TOOL_ARG_END]

The host intercepts that frame, executes the tool in :mod:`models.sandbox` and
injects ``[TOOL_RESULT] <result bytes>`` back as a *separate* input channel.

Safety properties of this module:

* A frame is executed only when it is complete. Truncated or unterminated
  frames raise :class:`ToolFrameTruncated` and are never dispatched.
* Argument bytes must be raw data (0-255). A control ID inside the argument
  region is a hard error, so a tool result or user string can never forge a
  nested ``[TOOL_CALL]``.
* Tool names are allow-listed; arguments and results are length capped.
* Tool results are fed to this head only. The Phase 10/12 seven-way intent
  classifier keeps receiving exactly the user message bytes it was trained on,
  so its calibration and weights are unaffected by tool traffic.

This is a routing/decoding head over frozen generator features, not a language
model: it is trained on template-bound synthetic tasks and does not understand
the code it emits.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .generator_conditioned import GeneratorConditioned
from .procedural import raw_bytes

TOOL_CALL: int = GeneratorConditioned.TOOL_CALL
TOOL_ARG_START: int = GeneratorConditioned.TOOL_ARG_START
TOOL_ARG_END: int = GeneratorConditioned.TOOL_ARG_END
TOOL_RESULT: int = GeneratorConditioned.TOOL_RESULT

#: Index 0 means "no tool call"; the rest match :data:`TOOL_NAMES`.
ROUTE_NAMES: tuple[str, ...] = ("none", "exec_python", "read_file", "write_file")
TOOL_NAMES: tuple[bytes, ...] = (b"exec_python", b"read_file", b"write_file")
NO_TOOL: int = 0

MAX_ARGUMENT_BYTES: int = 512
MAX_RESULT_BYTES: int = 4096
#: Bounds for the packed request/result context fed to the frozen backbone.
MAX_REQUEST_BYTES: int = 96
MAX_CONTEXT_RESULT_BYTES: int = 48
REQUEST_PREFIX: bytes = b"T:"
RESULT_PREFIX: bytes = b"\nR:"
TRUNCATION_MARK: bytes = b"..."


class ToolFrameError(ValueError):
    """A malformed, unsupported or unsafe tool frame. Never execute it."""


class ToolFrameTruncated(ToolFrameError):
    """The frame ended before ``TOOL_ARG_END``. Ask the model to continue."""


class ToolFileError(RuntimeError):
    """The tool head checkpoint is unusable or was not trained."""


@dataclass(frozen=True)
class ToolCall:
    """One complete, allow-listed tool invocation."""

    tool: bytes
    argument: bytes

    def __post_init__(self) -> None:
        if self.tool not in TOOL_NAMES:
            raise ToolFrameError(f"unsupported tool: {self.tool!r}")
        if not isinstance(self.argument, (bytes, bytearray)):
            raise ToolFrameError("argument must be bytes")
        if len(self.argument) > MAX_ARGUMENT_BYTES:
            raise ToolFrameError(f"argument exceeds {MAX_ARGUMENT_BYTES} bytes")

    @property
    def name(self) -> str:
        return self.tool.decode("ascii")


@dataclass(frozen=True)
class ToolGeneration:
    """A decoded model turn: routing decision plus the emitted frame."""

    route: int
    route_confidence: float
    call: ToolCall | None
    frame: tuple[int, ...]
    complete: bool

    @property
    def route_name(self) -> str:
        return ROUTE_NAMES[self.route]


def encode_tool_call(call: ToolCall) -> list[int]:
    """Serialize a call as control/byte IDs (control IDs are never data)."""
    if not isinstance(call, ToolCall):
        raise TypeError("call must be a ToolCall")
    return ([TOOL_CALL] + list(call.tool) + [TOOL_ARG_START]
            + list(call.argument) + [TOOL_ARG_END])


def encode_tool_result(result: bytes) -> list[int]:
    """Serialize a sandbox result for injection into the model's context."""
    payload = raw_bytes(result, "result")
    return [TOOL_RESULT] + list(payload[:MAX_RESULT_BYTES])


def sanitize_result(result: bytes, *, limit: int = MAX_CONTEXT_RESULT_BYTES) -> bytes:
    """Cap a result for the context window, keeping the informative tail.

    Tracebacks put the actual error on the last line, so the tail is kept and
    the head is marked as truncated rather than silently dropped.
    """
    payload = raw_bytes(result, "result")
    if len(payload) <= limit:
        return payload
    keep = max(limit - len(TRUNCATION_MARK), 0)
    return TRUNCATION_MARK + payload[len(payload) - keep:]


def _check_ids(ids) -> list[int]:
    if not isinstance(ids, (list, tuple)) and not (isinstance(ids, torch.Tensor) and ids.ndim == 1):
        raise TypeError("ids must be a one-dimensional sequence of token IDs")
    values = [int(x) for x in ids]
    if any(value < 0 or value >= GeneratorConditioned.VOCAB_SIZE for value in values):
        raise ToolFrameError("tool frame contains out-of-range token IDs")
    return values


def parse_tool_call(ids) -> ToolCall:
    """Parse one complete frame. Raises on anything incomplete or malformed.

    Only the first ``TOOL_CALL`` is honoured; the scan never re-enters itself,
    and control IDs inside the name/argument regions are rejected outright.
    """
    values = _check_ids(ids)
    try:
        start = values.index(TOOL_CALL)
    except ValueError as exc:
        raise ToolFrameError("no TOOL_CALL token in the generated sequence") from exc
    name_bytes = bytearray()
    index = start + 1
    while index < len(values):
        token = values[index]
        if token == TOOL_ARG_START:
            break
        if token >= GeneratorConditioned.VOCAB_BYTES:
            raise ToolFrameError("control token inside the tool name region")
        name_bytes.append(token)
        index += 1
    else:
        raise ToolFrameTruncated("frame ended before TOOL_ARG_START")
    index += 1
    argument = bytearray()
    while index < len(values):
        token = values[index]
        if token == TOOL_ARG_END:
            tool = bytes(name_bytes)
            if tool not in TOOL_NAMES:
                raise ToolFrameError(f"unsupported tool: {tool!r}")
            if len(argument) > MAX_ARGUMENT_BYTES:
                raise ToolFrameError(f"argument exceeds {MAX_ARGUMENT_BYTES} bytes")
            return ToolCall(tool, bytes(argument))
        if token >= GeneratorConditioned.VOCAB_BYTES:
            raise ToolFrameError("control token inside the tool argument region")
        argument.append(token)
        index += 1
    raise ToolFrameTruncated("frame ended before TOOL_ARG_END")


def find_tool_call(ids) -> tuple[ToolCall, int] | None:
    """Return the first complete frame and its end index, or ``None``.

    ``None`` means "nothing actionable yet"; a malformed frame still raises so
    the host can surface the failure instead of silently ignoring it.
    """
    values = _check_ids(ids)
    try:
        start = values.index(TOOL_CALL)
    except ValueError:
        return None
    try:
        call = parse_tool_call(values[start:])
    except ToolFrameTruncated:
        return None
    end = start + len(encode_tool_call(call))
    return call, end


def pack_tool_request(request: bytes, result: bytes | None, *, max_len: int) -> tuple[bytes, int]:
    """Pack a request and optional prior tool result into one bounded frame.

    Returns the packed bytes and the index where the request starts, so callers
    control the focus window explicitly instead of trusting delimiters.
    """
    message = raw_bytes(request, "request")
    if not message:
        raise ValueError("request must not be empty")
    if not isinstance(max_len, int) or isinstance(max_len, bool) or max_len < 4:
        raise ValueError("max_len must be at least four")
    if len(message) > MAX_REQUEST_BYTES:
        raise ValueError(f"request exceeds {MAX_REQUEST_BYTES} bytes")
    frame = bytearray(REQUEST_PREFIX + message)
    if result:
        tail = sanitize_result(result)
        budget = max_len - len(frame) - len(RESULT_PREFIX)
        if budget > 0:
            # Keep the tail of the tail: the informative line is the last one.
            frame += RESULT_PREFIX + tail[max(len(tail) - budget, 0):]
    if len(frame) > max_len:
        raise ValueError("packed tool request exceeds the generator context limit")
    return bytes(frame), len(REQUEST_PREFIX)


class ToolRouteHead(nn.Module):
    """Classify frozen request features into no-tool / tool-name routes."""

    def __init__(self, context_dim: int, hidden_dim: int = 96, num_routes: int = len(ROUTE_NAMES)) -> None:
        super().__init__()
        if context_dim < 1 or hidden_dim < 1 or num_routes < 2:
            raise ValueError("route head dimensions must be positive")
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_routes)
        self.register_buffer("tool_trained", torch.zeros((), dtype=torch.bool))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("context must have shape (batch, context_dim)")
        return self.classifier(self.net(context))


class ToolArgumentDecoder(nn.Module):
    """Autoregressive raw-byte argument decoder with attention over the request.

    EOS is an internal stop class (256) and is never serialized into a frame.
    Attention lets the decoder copy literals (numbers, filenames) out of the
    request instead of reconstructing them from one pooled vector.
    """

    EOS: int = 256
    START: int = 257
    IGNORE: int = -100

    def __init__(self, context_dim: int, hidden_dim: int = 96, byte_dim: int = 48) -> None:
        super().__init__()
        if min(context_dim, hidden_dim, byte_dim) < 1:
            raise ValueError("decoder dimensions must be positive")
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.byte_embedding = nn.Embedding(258, byte_dim)
        self.route_embedding = nn.Embedding(len(ROUTE_NAMES), byte_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.attention_query = nn.Linear(hidden_dim, hidden_dim)
        self.attention_key = nn.Linear(context_dim, hidden_dim)
        self.attention_score = nn.Linear(hidden_dim, 1)
        self.attention_value = nn.Linear(context_dim, hidden_dim)
        self.initial = nn.Linear(hidden_dim + byte_dim, hidden_dim)
        # Two batched passes: the first reads the bytes emitted so far, the
        # second also consumes the request bytes attended from the first pass.
        # Both training and generation run the same two passes, so teacher
        # forcing is the only difference between them.
        self.read_pass = nn.GRU(byte_dim * 2 + hidden_dim, hidden_dim, batch_first=True)
        self.recurrent = nn.GRU(byte_dim * 2 + hidden_dim * 2, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 257)
        # Pointer-generator gate: mixes the byte vocabulary with a copy
        # distribution over request positions, so copying a literal is one
        # decision instead of a chain of per-byte predictions.
        self.copy_gate = nn.Linear(hidden_dim * 2, 1)
        self.register_buffer("tool_trained", torch.zeros((), dtype=torch.bool))

    def _initial_state(self, projected: torch.Tensor, route_vector: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.initial(torch.cat((projected, route_vector), dim=-1))).unsqueeze(0)

    def _attention(self, states: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor):
        """Additive attention: ``states`` (B, S, H) over ``memory`` (B, L, C).

        Returns the attended vectors and the attention distribution, which the
        pointer-generator output reuses as a copy distribution over request
        positions.
        """
        if states.ndim != 3 or memory.ndim != 3:
            raise ValueError("attention expects (batch, length, width) tensors")
        # The query length (decoder steps) and the memory length (packed
        # request) are independent axes, so each is broadcast on the other.
        scores = self.attention_score(
            torch.tanh(self.attention_query(states).unsqueeze(2) + self.attention_key(memory).unsqueeze(1))
        ).squeeze(-1)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights, self.attention_value(memory)), weights

    def _inputs(self, previous: torch.Tensor, route_vector: torch.Tensor,
                projected: torch.Tensor) -> torch.Tensor:
        steps = previous.shape[1]
        return torch.cat((self.byte_embedding(previous),
                          route_vector.unsqueeze(1).expand(-1, steps, -1),
                          projected.unsqueeze(1).expand(-1, steps, -1)), dim=-1)

    def _validate(self, context, route, previous, memory, memory_mask) -> int:
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("context must have shape (batch, context_dim)")
        batch = context.shape[0]
        if memory.ndim != 3 or memory.shape[0] != batch or memory.shape[2] != self.context_dim:
            raise ValueError("memory must have shape (batch, length, context_dim)")
        if memory_mask.shape != memory.shape[:2] or memory_mask.dtype != torch.bool:
            raise ValueError("memory_mask must be a boolean (batch, length) tensor")
        if route.dtype != torch.long or route.shape != (batch,):
            raise ValueError("route must be one long ID per batch element")
        if bool(((route < 0) | (route >= len(ROUTE_NAMES))).any()):
            raise ValueError("route IDs must match the supported routes")
        if (previous.dtype != torch.long or previous.ndim != 2 or previous.shape[0] != batch
                or bool(((previous < 0) | (previous > self.START)).any())):
            raise ValueError("previous must contain byte/START IDs")
        return batch

    def _shifted_read_states(self, read_states: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
        """States that attention may use at each step: h_{t-1}, never h_t."""
        return torch.cat((initial.transpose(0, 1), read_states[:, :-1]), dim=1)

    def _mix(self, states: torch.Tensor, attended: torch.Tensor, copy_weights: torch.Tensor,
             memory_bytes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Log probability of the next byte: generated, copied, or EOS."""
        if memory_bytes.shape != mask.shape:
            raise ValueError("memory_bytes must match the memory mask")
        if bool(((memory_bytes < 0) | (memory_bytes > 255)).any()):
            raise ValueError("memory_bytes must contain raw request bytes")
        vocabulary = torch.softmax(self.output(states), dim=-1)
        gate = torch.sigmoid(self.copy_gate(torch.cat((states, attended), dim=-1)))
        mixed = vocabulary * gate
        weights = (1.0 - gate) * copy_weights * mask.unsqueeze(1).to(copy_weights.dtype)
        index = memory_bytes.unsqueeze(1).expand(-1, states.shape[1], -1)
        mixed = torch.scatter_add(mixed, 2, index, weights)
        return torch.log(mixed.clamp_min(1e-8))

    def forward(self, context: torch.Tensor, route: torch.Tensor, previous: torch.Tensor,
                memory: torch.Tensor, memory_mask: torch.Tensor, memory_bytes: torch.Tensor) -> torch.Tensor:
        """Return log probabilities over 256 byte classes plus EOS."""
        self._validate(context, route, previous, memory, memory_mask)
        projected = torch.tanh(self.context_projection(context))
        route_vector = self.route_embedding(route)
        initial = self._initial_state(projected, route_vector)
        embedded = self._inputs(previous, route_vector, projected)
        read_states, _ = self.read_pass(embedded, initial)
        attended, copy_weights = self._attention(self._shifted_read_states(read_states, initial),
                                                 memory, memory_mask)
        states, _ = self.recurrent(torch.cat((embedded, attended), dim=-1), initial)
        return self._mix(states, attended, copy_weights, memory_bytes, memory_mask)

    @torch.no_grad()
    def generate(self, context: torch.Tensor, route: torch.Tensor, memory: torch.Tensor,
                 memory_mask: torch.Tensor, memory_bytes: torch.Tensor,
                 *, max_bytes: int = MAX_ARGUMENT_BYTES) -> tuple[bytes, bool]:
        """Greedy argument bytes and whether EOS was reached in budget."""
        if context.shape != (1, self.context_dim) or route.shape != (1,):
            raise ValueError("generation requires one context and one route")
        if not 1 <= max_bytes <= MAX_ARGUMENT_BYTES:
            raise ValueError("max_bytes must be within the argument limit")
        projected = torch.tanh(self.context_projection(context))
        route_vector = self.route_embedding(route)
        initial = self._initial_state(projected, route_vector)
        read_state = initial
        state = initial
        previous = torch.tensor([[self.START]], dtype=torch.long, device=context.device)
        result = bytearray()
        for _ in range(max_bytes + 1):
            embedded = torch.cat((self.byte_embedding(previous), route_vector.unsqueeze(1),
                                  projected.unsqueeze(1)), dim=-1)
            # Same two passes as training: attention reads the previous step's
            # read-pass state, then the byte is emitted from the combined pass.
            attended, copy_weights = self._attention(read_state.transpose(0, 1), memory, memory_mask)
            hidden, state = self.recurrent(torch.cat((embedded, attended), dim=-1), state)
            _, read_state = self.read_pass(embedded, read_state)
            log_probabilities = self._mix(hidden, attended, copy_weights, memory_bytes, memory_mask)
            token = int(log_probabilities[0, -1].argmax(dim=-1).item())
            if token == self.EOS:
                return bytes(result), True
            if len(result) == max_bytes:
                return bytes(result), False
            result.append(token)
            previous = torch.tensor([[token]], dtype=torch.long, device=context.device)
        raise AssertionError("unreachable")


class ToolCallPolicy:
    """Frozen generator features plus the two trainable Phase 15 heads.

    The generator is treated as a read-only feature extractor: nothing in this
    class trains or stores backbone weights, so Phase 10/12 decoders that are
    bound to the generator checkpoint hash keep working unchanged.
    """

    def __init__(self, generator: GeneratorConditioned, route_head: ToolRouteHead | None = None,
                 argument_decoder: ToolArgumentDecoder | None = None, *, route_hidden_dim: int = 96,
                 argument_hidden_dim: int = 128, byte_dim: int = 64) -> None:
        if not isinstance(generator, GeneratorConditioned):
            raise TypeError("generator must be a GeneratorConditioned")
        self.generator = generator.eval()
        context_dim = generator.d_model
        self.route_head = route_head if route_head is not None else ToolRouteHead(context_dim, route_hidden_dim)
        self.argument_decoder = argument_decoder if argument_decoder is not None else ToolArgumentDecoder(
            context_dim, argument_hidden_dim, byte_dim)
        self.route_head.eval()
        self.argument_decoder.eval()

    @property
    def trained(self) -> bool:
        return bool(self.route_head.tool_trained) and bool(self.argument_decoder.tool_trained)

    def parameters(self):
        return list(self.route_head.parameters()) + list(self.argument_decoder.parameters())

    def encode(self, requests, results=None, *, width: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (ids, lengths, hidden, mask) for a batch of packed requests.

        ``width`` lets callers pad every chunk to one shared length so cached
        tensors can be concatenated across batches.
        """
        results = [None] * len(requests) if results is None else list(results)
        if len(results) != len(requests):
            raise ValueError("results must be None or match the number of requests")
        packed = [pack_tool_request(request, result, max_len=self.generator.max_len)
                  for request, result in zip(requests, results)]
        natural = max(len(frame) for frame, _ in packed)
        if width is None:
            width = natural
        elif width < natural:
            raise ValueError("width must cover the longest packed request")
        device = self.generator.byte_emb.weight.device
        ids = torch.zeros((len(packed), width), dtype=torch.long, device=device)
        lengths = torch.tensor([len(frame) for frame, _ in packed], dtype=torch.long, device=device)
        for row, (frame, _) in enumerate(packed):
            ids[row, :len(frame)] = torch.tensor(list(frame), dtype=torch.long, device=device)
        state = self.generator(ids, return_state=True)
        positions = torch.arange(width, device=device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        return ids, lengths, state.hidden, mask

    def pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths
        last = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths.squeeze(1) - 1]
        return (mean + last) * 0.5

    def hidden_states(self, requests, results=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (context, memory, mask, packed request ids)."""
        ids, _, hidden, mask = self.encode(requests, results)
        return self.pool(hidden, mask), hidden, mask, ids

    @torch.no_grad()
    def generate(self, request: bytes, result: bytes | None = None, *,
                 threshold: float = 0.5, max_bytes: int = MAX_ARGUMENT_BYTES) -> ToolGeneration:
        """Route one request and, when a tool is selected, emit its frame."""
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be within [0, 1]")
        context, hidden, mask, ids = self.hidden_states([request], [result])
        logits = self.route_head(context)
        probabilities = torch.softmax(logits, dim=-1)[0]
        confidence, route = float(probabilities.max().item()), int(probabilities.argmax().item())
        if route == NO_TOOL or confidence < threshold:
            return ToolGeneration(route=NO_TOOL, route_confidence=confidence, call=None,
                                  frame=(), complete=True)
        route_tensor = torch.tensor([route], dtype=torch.long, device=context.device)
        argument, complete = self.argument_decoder.generate(context, route_tensor, hidden, mask, ids,
                                                            max_bytes=max_bytes)
        call = ToolCall(TOOL_NAMES[route - 1], argument)
        return ToolGeneration(route=route, route_confidence=confidence, call=call,
                              frame=tuple(encode_tool_call(call)), complete=complete)

    def save(self, path: str | Path, *, generator_path: str | Path) -> None:
        payload = {
            "schema_version": 1,
            "generator_sha256": hashlib.sha256(Path(generator_path).read_bytes()).hexdigest(),
            "context_dim": self.route_head.context_dim,
            "route_hidden_dim": self.route_head.hidden_dim,
            "route_num_routes": int(self.route_head.classifier.out_features),
            "argument_hidden_dim": self.argument_decoder.hidden_dim,
            "argument_byte_dim": self.argument_decoder.byte_embedding.embedding_dim,
            "tool_names": [name.decode("ascii") for name in TOOL_NAMES],
            "state_dict": {**{f"route_head.{key}": value for key, value in self.route_head.state_dict().items()},
                           **{f"argument_decoder.{key}": value for key, value in self.argument_decoder.state_dict().items()}},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load(self, state_dict) -> None:
        """Restore both heads from a prefixed checkpoint mapping."""
        if not isinstance(state_dict, dict):
            raise TypeError("state_dict must be a mapping")
        route = {key[len("route_head."):]: value for key, value in state_dict.items()
                 if key.startswith("route_head.")}
        argument = {key[len("argument_decoder."):]: value for key, value in state_dict.items()
                    if key.startswith("argument_decoder.")}
        if not route or not argument:
            raise ToolFileError("tool head checkpoint is missing one of the two heads")
        self.route_head.load_state_dict(route, strict=True)
        self.argument_decoder.load_state_dict(argument, strict=True)


def load_tool_policy(path: str | Path, *, generator: GeneratorConditioned,
                     generator_path: str | Path) -> ToolCallPolicy:
    """Load a trained tool head, bound to the exact generator it was paired with.

    The generator weights themselves are never rewritten by Phase 15, so the
    checkpoint hash still identifies the decoders Phase 10/12 depend on.
    """
    if not isinstance(generator, GeneratorConditioned):
        raise TypeError("generator must be a GeneratorConditioned")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ToolFileError("unsupported tool head checkpoint schema")
    if [name.encode("ascii") for name in state.get("tool_names", [])] != list(TOOL_NAMES):
        raise ToolFileError("tool head checkpoint targets a different tool set")
    expected = state.get("generator_sha256")
    actual = hashlib.sha256(Path(generator_path).read_bytes()).hexdigest()
    if not isinstance(expected, str) or expected != actual:
        raise ToolFileError("tool head checkpoint does not match this generator")
    context_dim = state.get("context_dim")
    if not isinstance(context_dim, int) or isinstance(context_dim, bool) or context_dim != generator.d_model:
        raise ToolFileError("tool head context width does not match this generator")
    policy = ToolCallPolicy(
        generator,
        ToolRouteHead(context_dim, int(state["route_hidden_dim"])),
        ToolArgumentDecoder(context_dim, int(state["argument_hidden_dim"]), int(state["argument_byte_dim"])),
    )
    policy.load(state["state_dict"])
    if not policy.trained:
        raise ToolFileError("tool head checkpoint is not marked trained")
    return policy
