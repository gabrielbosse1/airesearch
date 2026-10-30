"""Supervised training of the raw-byte cognitive control and editing heads.

The inference loop is non-differentiable at the action boundary. This helper
teacher-forces a labelled action through the differentiable policy instead; it
does not require an external tokenizer or execute any stored code.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F

from models.cognitive import CognitiveAction, apply_edit
from models.generator_conditioned import GeneratorConditioned
from models.procedural import byte_pattern_embedding, raw_bytes


def edit_intent_target(generator: GeneratorConditioned, current: bytes, target: bytes) -> torch.Tensor:
    """Fixed, order-sensitive byte-sketch difference, used as a label only.

    The deterministic sketch has no trainable parameters and is not a perfect
    encoding of edits; in particular, collisions are possible. Detaching the
    label prevents the model from reducing MSE by moving its own embeddings.
    """
    current = raw_bytes(current, "current")
    target = raw_bytes(target, "target")
    weight = generator.byte_emb.weight
    with torch.no_grad():
        current_embedding = byte_pattern_embedding(current, generator.d_model, device=weight.device)
        target_embedding = byte_pattern_embedding(target, generator.d_model, device=weight.device)
        return (target_embedding - current_embedding).to(dtype=weight.dtype).detach()


def reasoning_loss_from_thoughts(
    generator: GeneratorConditioned,
    thoughts: list[tuple[torch.Tensor, bytes]],
    target: bytes,
) -> torch.Tensor:
    """Mean MSE at post-THINK states; the desired bytes never enter the policy."""
    raw_bytes(target, "target")
    if not thoughts:
        return generator.byte_emb.weight.new_zeros(())
    losses = []
    for scratchpad, current in thoughts:
        prediction = generator.predict_edit_intent(scratchpad)
        if prediction.shape[0] != 1:
            raise ValueError("thought supervision currently requires one canvas")
        label = edit_intent_target(generator, current, target)
        losses.append(F.mse_loss(prediction[0], label))
    return torch.stack(losses).mean()


def cognitive_supervision_loss(
    generator: GeneratorConditioned,
    canvas_view: bytes,
    scratchpad: torch.Tensor,
    action: CognitiveAction,
    *,
    workspace: bytes | None = None,
    memory_context: torch.Tensor | list[dict[str, Any]] | tuple[Any, ...] | None = None,
    memory_model: Any | None = None,
    dirty: bool = False,
    last_token: int | None = None,
    structured_request: bytes | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Teacher-force one labelled control and its applicable edit arguments.

    For model-generated insert/replace, this policy predicts one payload byte
    per EDIT_OP. Host controllers may submit arbitrary-length byte payloads;
    split such training targets into single-byte actions when needed.
    """
    if not generator.LOAD_SPAN <= action.token <= generator.DONE:
        raise ValueError("action is not a cognitive control token")
    if action.token in (generator.SAVE_SPAN, generator.EDIT_OP) and workspace is None:
        raise ValueError("the action requires a loaded workspace")
    pointer_bytes = workspace if action.token == generator.EDIT_OP else canvas_view
    policy = generator.cognitive_policy(
        canvas_view, scratchpad, pointer_bytes=pointer_bytes,
        memory_context=memory_context, memory_model=memory_model,
        loaded=workspace is not None, dirty=dirty, last_token=last_token,
        **({"structured_request": structured_request} if structured_request is not None else {}),
    )

    def cross_entropy(logits: torch.Tensor, target: int, name: str) -> torch.Tensor:
        if not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < logits.numel():
            raise ValueError(f"invalid {name} target")
        return F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=logits.device))

    terms = {"action": cross_entropy(policy.action_logits, action.token - generator.LOAD_SPAN, "action")}
    if action.token in (generator.LOAD_SPAN, generator.EDIT_OP):
        if action.start is None:
            raise ValueError("start is required")
        terms["start"] = cross_entropy(policy.start_logits, action.start, "start")
        if action.token == generator.LOAD_SPAN or action.kind in ("delete", "replace", "move"):
            if action.end is None or action.end < action.start:
                raise ValueError("end must not precede start")
            terms["end"] = cross_entropy(policy.end_logits[action.start:], action.end - action.start, "end")
    if action.token == generator.EDIT_OP:
        kinds = ("insert", "delete", "replace", "move")
        if action.kind not in kinds:
            raise ValueError("unknown edit kind")
        terms["kind"] = cross_entropy(policy.kind_logits, kinds.index(action.kind), "kind")
        if action.kind in ("insert", "replace"):
            if not isinstance(action.data, bytes) or len(action.data) != 1:
                raise ValueError("model-generated edit targets must contain exactly one byte")
            terms["byte"] = cross_entropy(policy.byte_logits, action.data[0], "byte")
        if action.kind == "move":
            if action.destination is None:
                raise ValueError("move destination is required")
            if action.end is not None and action.start is not None and action.start < action.destination < action.end:
                raise ValueError("move destination cannot be inside the source span")
            terms["destination"] = cross_entropy(policy.destination_logits, action.destination, "destination")
    return sum(terms.values()), terms


def validate_action_trace(generator: GeneratorConditioned, source: bytes, actions: list[CognitiveAction], *,
                          raw_observer: Callable[[bytes], None] | None = None) -> bytes:
    """Validate/replay controls symbolically, optionally auditing intermediate bytes."""
    if not isinstance(source, bytes) or not actions or len(actions) > 4096:
        raise ValueError("trace requires byte input and 1..4096 actions")
    data, workspace, span, dirty = source, None, None, False
    payload_run = bytearray()
    if raw_observer is not None:
        raw_observer(source)
    if len(source) > generator.max_canvas_bytes:
        raise ValueError("source exceeds generator canvas limit")
    for index, action in enumerate(actions):
        if not isinstance(action, CognitiveAction):
            raise TypeError("trace actions must be CognitiveAction objects")
        if action.token == generator.EDIT_OP and action.kind in ("insert", "replace"):
            payload_run.extend(action.data)
        elif payload_run:
            if raw_observer is not None:
                raw_observer(bytes(payload_run))
            payload_run.clear()
        if action.token == generator.LOAD_SPAN:
            if dirty or not isinstance(action.start, int) or isinstance(action.start, bool) or not isinstance(action.end, int) or isinstance(action.end, bool) or not 0 <= action.start <= action.end <= len(data):
                raise ValueError("invalid LOAD_SPAN in trace")
            span = (action.start, action.end); workspace = data[span[0]:span[1]]
        elif action.token == generator.THINK:
            if not isinstance(action.think_steps, int) or isinstance(action.think_steps, bool) or not 1 <= action.think_steps <= generator.max_think_steps:
                raise ValueError("invalid THINK in trace")
        elif action.token == generator.EDIT_OP:
            if workspace is None or span is None:
                raise ValueError("EDIT_OP requires a loaded workspace")
            if action.kind in ("insert", "replace") and len(action.data) != 1:
                raise ValueError("model-supervised insert/replace must use a single byte")
            updated = apply_edit(workspace, action, max_length=generator.max_canvas_bytes - (len(data) - (span[1] - span[0])))
            dirty = dirty or updated != workspace; workspace = updated
        elif action.token == generator.SAVE_SPAN:
            if workspace is None or span is None:
                raise ValueError("SAVE_SPAN requires a loaded workspace")
            data = data[:span[0]] + workspace + data[span[1]:]
            span = (span[0], span[0] + len(workspace)); dirty = False
        elif action.token == generator.DONE:
            if dirty or index != len(actions) - 1:
                raise ValueError("DONE must be last, with no pending edits")
            return data
        else:
            raise ValueError("unsupported cognitive control")
        if raw_observer is not None:
            if workspace is not None and span is not None:
                raw_observer(workspace)
                raw_observer(data[:span[0]] + workspace + data[span[1]:])
            else:
                raw_observer(data)
    raise ValueError("trace must end with DONE")


def cognitive_trace_supervision_loss(
    generator: GeneratorConditioned,
    source: bytes,
    actions: list[CognitiveAction],
    *,
    memory_context: torch.Tensor | None = None,
    instruction_bytes: bytes | None = None,
    target_bytes: bytes | None = None,
    return_reasoning: bool = False,
    structured_request: bytes | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, int]:
    """Teacher-force a variable-length trace with differentiable THINK states.

    Optional reasoning targets are used solely for an auxiliary MSE after each
    THINK, never as policy input. The default scalar return is backward-compatible.
    """
    actual_target = validate_action_trace(generator, source, actions)
    if return_reasoning:
        if target_bytes is None or raw_bytes(target_bytes, "target_bytes") != actual_target:
            raise ValueError("reasoning target must match the validated trace output")
    device = generator.byte_emb.weight.device
    data, workspace, span, dirty = source, None, None, False
    scratch = generator.scratchpad.clone()
    if instruction_bytes is not None:
        instruction = raw_bytes(instruction_bytes, "instruction_bytes")
        if len(instruction) > generator.max_canvas_bytes:
            raise ValueError("instruction exceeds max_canvas_bytes")
        if instruction:
            scratch = generator.read_scratchpad(
                scratch, torch.tensor([list(instruction)], dtype=torch.long, device=device)
            )
    previous = None
    losses = []
    thoughts: list[tuple[torch.Tensor, bytes]] = []
    for action in actions:
        view = data if workspace is None or span is None else data[:span[0]] + workspace + data[span[1]:]
        loss, _ = cognitive_supervision_loss(generator, view, scratch, action, workspace=workspace,
                                             memory_context=memory_context, dirty=dirty, last_token=previous,
                                             structured_request=structured_request)
        losses.append(loss)
        if action.token == generator.LOAD_SPAN:
            assert action.start is not None and action.end is not None
            span = (action.start, action.end); workspace = data[span[0]:span[1]]
            scratch = generator.read_scratchpad(scratch, torch.tensor([list(workspace)], dtype=torch.long, device=device))
        elif action.token == generator.THINK:
            policy = generator.cognitive_policy(view, scratch, pointer_bytes=data if workspace is None else workspace,
                                                memory_context=memory_context, loaded=workspace is not None,
                                                dirty=dirty, last_token=previous,
                                                **({"structured_request": structured_request} if structured_request is not None else {}))
            scratch = generator.think_scratchpad(scratch, policy.context.unsqueeze(0), steps=action.think_steps).scratchpad
            if return_reasoning:
                thoughts.append((scratch, view))
        elif action.token == generator.EDIT_OP:
            assert workspace is not None and span is not None
            updated = apply_edit(workspace, action, max_length=generator.max_canvas_bytes - (len(data) - (span[1] - span[0])))
            dirty = dirty or updated != workspace; workspace = updated
        elif action.token == generator.SAVE_SPAN:
            assert workspace is not None and span is not None
            data = data[:span[0]] + workspace + data[span[1]:]
            span = (span[0], span[0] + len(workspace)); dirty = False
        previous = action.token
    trace_loss = torch.stack(losses).mean()
    if return_reasoning:
        assert target_bytes is not None
        return trace_loss, reasoning_loss_from_thoughts(generator, thoughts, target_bytes), len(thoughts)
    return trace_loss
