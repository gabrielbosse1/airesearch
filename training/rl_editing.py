"""Memory-free REINFORCE for autonomous raw-byte cognitive editing.

The deployed cognitive loop samples discrete actions under no_grad. We replay
those exact actions against the same model in evaluation mode to recover a
*gradient-bearing* log probability, including all sampled arguments and the
many-to-one destination remapping. Targets are used for training reward and
auxiliary supervision, never as rollout inputs. No policy memory, stored
traces, controller, tokenizer, or target-derived rollout budget is used.

Run from the repository root, for example:
    python -m training.rl_editing --epochs 20 --examples-per-stage 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data.edit_curriculum import curriculum_examples
from data.holdout_registry import synthetic_holdouts
from data.ingest_real_traces import load_pairs, make_control_trace
from models.cognitive import CognitiveResult, action_from_record, action_to_record, apply_edit
from models.cognitive_arguments import decode_edit_arguments, encode_edit_arguments
from models.generator_conditioned import GeneratorConditioned
from models.holdout import HoldoutGuard
from models.procedural import raw_bytes
from training.co_train import CURRICULUM_STAGES, curriculum_indices, curriculum_stage
from training.cognitive_train import (cognitive_trace_supervision_loss, reasoning_loss_from_thoughts,
                                      validate_action_trace)
from training.train_cognitive import freeze_backbone
from training.train_confidence import load_generator


@dataclass(frozen=True)
class PolicyRollout:
    result: CognitiveResult
    log_probability: torch.Tensor
    thoughts: list[tuple[torch.Tensor, bytes]]


_ATTENTION_LOCK = threading.RLock()


@contextmanager
def consistent_attention_math():
    """Use the same MHA path for no-grad sampling and differentiable replay.

    Newer PyTorch versions use a numerically different MHA inference fast path
    when gradients are disabled. The switch is process-global, restored even
    on exceptions; concurrent calls may temporarily lose that optimization.
    """
    with _ATTENTION_LOCK:
        backend = getattr(torch.backends, "mha", None)
        if backend is None or not hasattr(backend, "get_fastpath_enabled"):
            yield
            return
        previous = backend.get_fastpath_enabled()
        backend.set_fastpath_enabled(False)
        try:
            yield
        finally:
            backend.set_fastpath_enabled(previous)


def byte_levenshtein(source: bytes, target: bytes, *, max_cells: int = 1_000_000) -> int:
    """Exact unit-cost byte Levenshtein using linear memory and bounded work."""
    source = raw_bytes(source, "source")
    target = raw_bytes(target, "target")
    if max_cells < 1 or (len(source) + 1) * (len(target) + 1) > max_cells:
        raise ValueError("edit-distance computation exceeds max_cells")
    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for i, a in enumerate(source, 1):
        current = [i]
        for j, b in enumerate(target, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def editing_reward(result: CognitiveResult, target: bytes, generator: GeneratorConditioned) -> dict[str, float | int]:
    """The requested task reward; only committed canvas bytes are scored."""
    target = raw_bytes(target, "target")
    distance = byte_levenshtein(result.canvas, target)
    length = max(len(result.canvas), len(target), 1)
    thoughts = sum(event.thought_steps for event in result.events)
    edits = sum(event.action.token == generator.EDIT_OP for event in result.events)
    similarity = 1.0 - distance / length
    return {"reward": similarity - 0.01 * (thoughts + edits), "similarity": similarity,
            "distance": distance, "think_steps": thoughts, "edit_steps": edits}


def hybrid_weights(epoch: int, total_epochs: int) -> tuple[float, float]:
    """Linearly anneal supervised/RL weights from 0.8/0.2 to 0.2/0.8."""
    if (not isinstance(epoch, int) or isinstance(epoch, bool) or not isinstance(total_epochs, int)
            or isinstance(total_epochs, bool) or total_epochs < 2 or not 0 <= epoch < total_epochs):
        raise ValueError("epoch and total_epochs must describe at least two epochs")
    supervised = 0.8 - 0.6 * epoch / (total_epochs - 1)
    return supervised, 1.0 - supervised


def destination_log_probability(logits: torch.Tensor, start: int, end: int, destination: int,
                                temperature: float) -> torch.Tensor:
    """Probability of the *recorded* destination after the loop's remapping.

    The inference loop maps all indices strictly inside [start, end] to end,
    even for edit kinds whose destination is ignored by apply_edit. Their
    probabilities must be summed rather than treating the remapped index as the
    only possible sampled value.
    """
    if (logits.ndim != 1 or not math.isfinite(temperature) or temperature <= 0
            or not 0 <= start <= end < logits.numel() or not 0 <= destination < logits.numel()
            or start < destination < end):
        raise ValueError("invalid sampled destination or temperature")
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    if destination == end and end > start:
        return torch.logsumexp(log_probs[start + 1:end + 1], dim=0)
    return log_probs[destination]


def replay_action_log_probability(
    generator: GeneratorConditioned,
    source: bytes,
    result: CognitiveResult,
    *,
    temperature: float,
    instruction_bytes: bytes | None = None,
    structured_request: bytes | None = None,
    guard: HoldoutGuard | None = None,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, bytes]]]:
    """Differentiably rescore one action sequence from cognitive_generate.

    Call before any parameter change, with the same instruction and positive
    sampling temperature. Sampling and replay both use eval mode so dropout
    cannot change the distribution. When calling this directly, enclose both
    the original sampling and replay in ``consistent_attention_math()``.
    The byte environment is rebuilt from sampled actions; it must have been
    sampled without a host controller, memory or adaptive halting.
    """
    source = raw_bytes(source, "source")
    instruction = None if instruction_bytes is None else raw_bytes(instruction_bytes, "instruction_bytes")
    request = None if structured_request is None else raw_bytes(structured_request, "structured_request")
    if request is not None:
        decode_edit_arguments(request, max_position=generator.max_canvas_bytes, max_payload=generator.max_canvas_bytes)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("sampling temperature must be finite and positive")
    if not result.events:
        raise ValueError("a sampled rollout must contain an action")
    if not torch.is_grad_enabled():
        raise RuntimeError("differentiable replay requires gradients enabled")
    generator.eval()
    if guard is not None:
        guard.validate(source, metadata={"trace_actions": [action_to_record(e.action) for e in result.events],
                                         "instruction_bytes": instruction, "structured_request": request})
    device = generator.byte_emb.weight.device
    scratch = generator.scratchpad.clone().to(device=device, dtype=generator.byte_emb.weight.dtype)
    if instruction:
        scratch = generator.read_scratchpad(scratch, torch.tensor([list(instruction)], device=device, dtype=torch.long))
    data, workspace, span, dirty = source, None, None, False
    previous: int | None = None
    log_terms: list[torch.Tensor] = []
    thoughts: list[tuple[torch.Tensor, bytes]] = []
    done = False
    kinds = ("insert", "delete", "replace", "move")

    def log_prob(logits: torch.Tensor, index: int) -> torch.Tensor:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < logits.numel():
            raise ValueError("sampled index is outside its policy distribution")
        return F.log_softmax(logits / temperature, dim=-1)[index]

    for event_index, event in enumerate(result.events):
        action = event.action
        view = data if workspace is None or span is None else data[:span[0]] + workspace + data[span[1]:]
        pointer = data if workspace is None else workspace
        request_kwargs = {"structured_request": request} if request is not None else {}
        policy = generator.cognitive_policy(view, scratch, pointer_bytes=pointer,
                                            loaded=workspace is not None, dirty=dirty, last_token=previous,
                                            **request_kwargs)
        masked = policy.action_logits.clone()
        if workspace is None:
            masked[generator.SAVE_SPAN - generator.LOAD_SPAN] = -torch.inf
            masked[generator.EDIT_OP - generator.LOAD_SPAN] = -torch.inf
        if dirty:
            masked[generator.LOAD_SPAN - generator.LOAD_SPAN] = -torch.inf
            masked[generator.DONE - generator.LOAD_SPAN] = -torch.inf
        log_terms.append(log_prob(masked, action.token - generator.LOAD_SPAN))
        if action.token == generator.LOAD_SPAN:
            if workspace is not None:
                policy = generator.cognitive_policy(view, scratch, pointer_bytes=data, loaded=True,
                                                    dirty=False, last_token=previous, **request_kwargs)
            if action.start is None or action.end is None:
                raise ValueError("sampled LOAD lacks pointers")
            log_terms.append(log_prob(policy.start_logits, action.start))
            log_terms.append(log_prob(policy.end_logits[action.start:], action.end - action.start))
            if not 0 <= action.start <= action.end <= len(data):
                raise ValueError("sampled LOAD pointers are invalid")
            span = (action.start, action.end)
            workspace = data[span[0]:span[1]]
            scratch = generator.read_scratchpad(scratch, torch.tensor([list(workspace)], device=device, dtype=torch.long))
        elif action.token == generator.THINK:
            if event.thought_steps != action.think_steps:
                raise ValueError("adaptive or host-controlled thoughts are not supported by this rollout")
            scratch = generator.think_scratchpad(scratch, policy.context.unsqueeze(0),
                                                  steps=action.think_steps).scratchpad
            thoughts.append((scratch, view))
        elif action.token == generator.EDIT_OP:
            if workspace is None or span is None or action.kind not in kinds:
                raise ValueError("sampled EDIT requires a loaded span and valid kind")
            if action.start is None or action.end is None or action.destination is None:
                raise ValueError("sampled EDIT lacks pointers")
            log_terms.append(log_prob(policy.kind_logits, kinds.index(action.kind)))
            log_terms.append(log_prob(policy.start_logits, action.start))
            log_terms.append(log_prob(policy.end_logits[action.start:], action.end - action.start))
            log_terms.append(destination_log_probability(policy.destination_logits, action.start, action.end,
                                                          action.destination, temperature))
            if action.kind in ("insert", "replace"):
                if len(action.data) != 1:
                    raise ValueError("sampled insert/replace must contain one raw byte")
                log_terms.append(log_prob(policy.byte_logits, action.data[0]))
            elif action.data:
                raise ValueError("delete/move cannot carry sampled payload")
            max_workspace = generator.max_canvas_bytes - (len(data) - (span[1] - span[0]))
            applied_action = action
            if event.argument_bytes is not None:
                if event.argument_bytes != encode_edit_arguments(action):
                    raise ValueError("sampled EDIT argument bytes do not match its policy action")
                applied_action = decode_edit_arguments(event.argument_bytes, max_position=generator.max_canvas_bytes,
                                                       max_payload=generator.max_canvas_bytes)
            updated = apply_edit(workspace, applied_action, max_length=max_workspace)
            dirty = dirty or updated != workspace
            workspace = updated
        elif action.token == generator.SAVE_SPAN:
            if workspace is None or span is None:
                raise ValueError("sampled SAVE requires a loaded span")
            data = data[:span[0]] + workspace + data[span[1]:]
            span = (span[0], span[0] + len(workspace))
            dirty = False
        elif action.token == generator.DONE:
            if dirty or event_index != len(result.events) - 1:
                raise ValueError("DONE must be last and cannot discard pending edits")
            done = True
        else:
            raise ValueError("unknown sampled cognitive control")
        if guard is not None:
            actual_view = data if workspace is None or span is None else data[:span[0]] + workspace + data[span[1]:]
            guard.validate(actual_view)
            if workspace is not None:
                guard.validate(workspace)
        previous = action.token

    canvas_mismatch = data != result.canvas
    done_mismatch = done != result.done
    workspace_mismatch = (workspace if dirty else None) != result.pending_workspace
    span_mismatch = (span if dirty else None) != result.pending_span
    scratch_mismatch = not torch.allclose(scratch.detach(), result.scratchpad, rtol=1e-5, atol=1e-6)
    if canvas_mismatch or done_mismatch or workspace_mismatch or span_mismatch or scratch_mismatch:
        max_delta = float((scratch.detach() - result.scratchpad).abs().max())
        raise RuntimeError("rollout/replay divergence: "
                           f"canvas={canvas_mismatch}, done={done_mismatch}, workspace={workspace_mismatch}, "
                           f"span={span_mismatch}, scratch={scratch_mismatch} (max delta={max_delta:.8g}); "
                           "do not use these gradients")
    total_log_prob = torch.stack(log_terms).sum()
    if not bool(torch.isfinite(total_log_prob)):
        raise RuntimeError("sampled sequence has non-finite probability")
    if not total_log_prob.requires_grad:
        raise RuntimeError("policy parameters must require gradients for REINFORCE")
    return total_log_prob, thoughts


def autonomous_rollout(
    generator: GeneratorConditioned,
    source: bytes,
    *,
    max_steps: int,
    temperature: float,
    instruction_bytes: bytes | None = None,
    structured_request: bytes | None = None,
    guard: HoldoutGuard | None = None,
) -> PolicyRollout:
    """Sample the deployed, memory-free loop and replay it with gradients.

    As with ``cognitive_generate``, the generator is left in evaluation mode;
    gradients still flow through replayed policy computations.
    """
    source = raw_bytes(source, "source")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= 4096:
        raise ValueError("max_steps must be in [1, 4096]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("training rollout temperature must be finite and positive")
    # Every model-generated edit can increase length by at most one byte.
    # This conservative precondition makes all sampled edits safe without
    # silently resampling or changing their probability distribution.
    if len(source) + max_steps > generator.max_canvas_bytes:
        raise ValueError("source plus rollout budget can exceed the canvas capacity")
    with consistent_attention_math():
        result = generator.cognitive_generate(source, max_steps=max_steps, temperature=temperature,
                                               instruction_bytes=instruction_bytes, structured_request=structured_request)
        log_probability, thoughts = replay_action_log_probability(generator, source, result, temperature=temperature,
                                                                   instruction_bytes=instruction_bytes,
                                                                   structured_request=structured_request, guard=guard)
    return PolicyRollout(result, log_probability, thoughts)


def reinforce_loss(log_probability: torch.Tensor, reward: float, *, baseline: float = 0.0) -> torch.Tensor:
    """Score-function loss; a sample-independent baseline reduces variance."""
    if log_probability.ndim != 0 or not math.isfinite(reward) or not math.isfinite(baseline):
        raise ValueError("log probability, reward and baseline must be finite scalars")
    return -(float(reward) - float(baseline)) * log_probability


def _real_training_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for record in records:
        if record.get("split") != "train":
            raise ValueError("only train records can provide supervised traces or RL rewards")
        _, actions = make_control_trace(record["input"], record["output"])
        prepared.append({**record, "instruction_bytes": None,
                         "trace_actions": [action_to_record(action) for action in actions]})
    return prepared


def preflight_training_examples(generator: GeneratorConditioned, examples: list[dict[str, Any]],
                                guard: HoldoutGuard) -> None:
    """Check all train bytes, trace payloads and intermediate canvases first."""
    for example in examples:
        if example.get("split") != "train":
            raise ValueError("RL and supervised labels must come from training data")
        source, target = example["input"], example["output"]
        records = example["trace_actions"]
        instruction = example.get("instruction_bytes")
        request = example.get("structured_request")
        if request is not None:
            decode_edit_arguments(request, max_position=generator.max_canvas_bytes, max_payload=generator.max_canvas_bytes)
        guard.validate(source, target, metadata={"split": "train", "trace_actions": records,
                                                 "instruction_bytes": instruction, "structured_request": request})
        actions = [action_from_record(record) for record in records]
        output = validate_action_trace(generator, source, actions, raw_observer=lambda raw: guard.validate(raw))
        if output != target:
            raise ValueError("training trace does not reproduce its target")


@torch.no_grad()
def evaluate_autonomous(generator: GeneratorConditioned, records: list[dict[str, Any]], *, budget: int) -> dict[str, Any]:
    """Greedy memory-free evaluation; target bytes are accessed only after inference."""
    if not 1 <= budget <= 4096 or not records:
        raise ValueError("evaluation requires a nonempty split and fixed valid budget")
    generator.eval()
    details = []
    for record in records:
        result = generator.cognitive_generate(record["input"], max_steps=budget,
                                               instruction_bytes=record.get("instruction_bytes"))
        target = record["output"]  # scoring only, after generation
        reward = editing_reward(result, target, generator)
        details.append({"id": record["id"], "kind": record.get("kind", "unknown"),
                        "exact": bool(result.done and result.canvas == target), "done": result.done,
                        "halt_reason": result.halt_reason, "steps": len(result.events),
                        "budget": budget, **reward,
                        "output_hex": result.canvas.hex(), "target_hex": target.hex(),
                        "actions": [action_to_record(event.action) for event in result.events]})
    count = len(details)
    return {"count": count, "exact": sum(int(item["exact"]) for item in details),
            "exact_fraction": sum(int(item["exact"]) for item in details) / count,
            "done_fraction": sum(int(item["done"]) for item in details) / count,
            "mean_similarity": sum(float(item["similarity"]) for item in details) / count,
            "mean_reward": sum(float(item["reward"]) for item in details) / count,
            "examples": details}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Train a curriculum with validation-only selection and a final blind test."""
    if (args.epochs < 5 or args.examples_per_stage < 1 or args.real_anchors < 0
            or not 1 <= args.rollout_budget <= 4096 or not 1 <= args.eval_budget <= 4096
            or not math.isfinite(args.temperature) or args.temperature <= 0
            or not math.isfinite(args.lr) or args.lr <= 0 or args.threads < 1):
        raise ValueError("invalid training options; use at least five epochs and positive budgets/rate/temperature")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    source_path = Path(args.generator)
    generator = load_generator(str(source_path), args.device, args.generator_heads)
    baseline = load_generator(str(source_path), args.device, args.generator_heads)
    real_train = load_pairs(args.pairs, split="train")
    real_validation = load_pairs(args.pairs, split="validation")
    real_test = load_pairs(args.pairs, split="test")
    if not real_train or not real_validation or len(real_test) != 8:
        raise ValueError("expected nonempty train/validation and exactly eight real held-out test records")
    synthetic_train = curriculum_examples(split="train", count_per_stage=args.examples_per_stage, seed=args.seed)
    synthetic_validation = curriculum_examples(split="validation", count_per_stage=2, seed=args.seed)
    # Fail closed when the known processed holdout registry is missing, unless
    # the caller explicitly accepts and records an incomplete registry.
    allow_missing = bool(getattr(args, "allow_missing_processed_holdouts", False))
    registered_synthetic = synthetic_holdouts(args.processed_dir, require=not allow_missing)
    guard = HoldoutGuard.from_records(real_validation + real_test + synthetic_validation + registered_synthetic)
    anchors = _real_training_examples(real_train)
    preflight_training_examples(generator, synthetic_train + anchors, guard)
    if any(len(example["input"]) + args.rollout_budget > generator.max_canvas_bytes
           for example in synthetic_train + anchors):
        raise ValueError("a training rollout could exceed the canvas capacity")
    # Fail before the first gradient if a stage is missing.
    for epoch in range(args.epochs):
        curriculum_indices(synthetic_train, epoch, args.epochs)

    generator.train()  # invalidate any prior confidence calibration before optimization
    parameters = freeze_backbone(generator)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    logs: list[dict[str, Any]] = []
    best_rank: tuple[float, float] | None = None
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(args.epochs):
        stage = curriculum_stage(epoch, args.epochs)
        indices = curriculum_indices(synthetic_train, epoch, args.epochs)
        examples = [synthetic_train[i] for i in indices]
        # Fixed train-only anchors maintain exposure to the real fixture without
        # placing its validation/test examples or their traces in the optimizer.
        if args.real_anchors:
            examples += [anchors[(epoch * args.real_anchors + i) % len(anchors)] for i in range(args.real_anchors)]
        random.Random(args.seed + epoch).shuffle(examples)
        supervised_weight, rl_weight = hybrid_weights(epoch, args.epochs)
        results = []
        for example in examples:
            source, target = example["input"], example["output"]
            instruction = example.get("instruction_bytes")
            # Self-critical greedy reward is a sample-independent baseline,
            # conditioned on the same train example and current policy.
            greedy = generator.cognitive_generate(source, max_steps=args.rollout_budget,
                                                  instruction_bytes=instruction)
            baseline_reward = float(editing_reward(greedy, target, generator)["reward"])
            sampled = autonomous_rollout(generator, source, max_steps=args.rollout_budget,
                                         temperature=args.temperature, instruction_bytes=instruction, guard=guard)
            reward = editing_reward(sampled.result, target, generator)
            actions = [action_from_record(record) for record in example["trace_actions"]]
            supervised, teacher_reasoning, teacher_thoughts = cognitive_trace_supervision_loss(
                generator, source, actions, instruction_bytes=instruction,
                target_bytes=target, return_reasoning=True,
            )
            sampled_reasoning = reasoning_loss_from_thoughts(generator, sampled.thoughts, target)
            reasoning = ((teacher_reasoning + sampled_reasoning) * 0.5
                         if sampled.thoughts and teacher_thoughts else teacher_reasoning)
            rl_loss = reinforce_loss(sampled.log_probability, float(reward["reward"]), baseline=baseline_reward)
            total = supervised_weight * supervised + rl_weight * rl_loss + 0.5 * reasoning
            if not bool(torch.isfinite(total)):
                raise RuntimeError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            results.append({"reward": float(reward["reward"]), "baseline_reward": baseline_reward,
                            "similarity": float(reward["similarity"]), "think_steps": int(reward["think_steps"]),
                            "edit_steps": int(reward["edit_steps"]),
                            "exact": float(sampled.result.done and sampled.result.canvas == target),
                            "supervised_loss": float(supervised.detach()), "rl_loss": float(rl_loss.detach()),
                            "reasoning_loss": float(reasoning.detach()), "total_loss": float(total.detach()),
                            "log_probability": float(sampled.log_probability.detach()), "gradient_norm": float(grad_norm)})
        mean = lambda key: sum(float(result[key]) for result in results) / len(results)
        val = evaluate_autonomous(generator, real_validation, budget=args.eval_budget)
        # Predeclared validation selection is restricted to the final stage so
        # the selected checkpoint has seen all five curriculum difficulties.
        # Test results are not computed or consulted until training is complete.
        rank = (float(val["exact_fraction"]), float(val["mean_similarity"]))
        if stage == CURRICULUM_STAGES[-1] and (best_rank is None or rank > best_rank):
            best_rank = rank
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in generator.state_dict().items()}
        log = {"epoch": epoch + 1, "curriculum_stage": stage, "examples": len(examples),
               "supervised_weight": supervised_weight, "rl_weight": rl_weight,
               **{key: mean(key) for key in ("reward", "baseline_reward", "similarity", "think_steps", "edit_steps",
                                              "exact", "supervised_loss", "rl_loss", "reasoning_loss", "total_loss",
                                              "log_probability", "gradient_norm")},
               "validation_exact": val["exact"], "validation_count": val["count"],
               "validation_mean_similarity": val["mean_similarity"]}
        logs.append(log)
        print(f"epoch {epoch + 1}/{args.epochs} {stage}: reward={log['reward']:.4f} "
              f"baseline={log['baseline_reward']:.4f} val={val['exact']}/{val['count']} "
              f"rl_weight={rl_weight:.3f}", flush=True)
    assert best_state is not None and best_rank is not None
    last_state = {key: value.detach().cpu().clone() for key, value in generator.state_dict().items()}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(last_state, output / "last_epoch.pt")
    generator.load_state_dict(best_state, strict=True)
    generator.eval()
    torch.save(generator.state_dict(), output / "generator.pt")
    # Only after all training/validation decisions are fixed do these three
    # predeclared policies see test inputs. No test score selects a checkpoint.
    baseline_test = evaluate_autonomous(baseline, real_test, budget=args.eval_budget)
    selected_test = evaluate_autonomous(generator, real_test, budget=args.eval_budget)
    validation_final = evaluate_autonomous(generator, real_validation, budget=args.eval_budget)
    synthetic_validation_final = evaluate_autonomous(generator, synthetic_validation, budget=args.eval_budget)
    if best_epoch == args.epochs:
        final_test = selected_test
    else:
        generator.load_state_dict(last_state, strict=True)
        final_test = evaluate_autonomous(generator, real_test, budget=args.eval_budget)
    metrics: dict[str, Any] = {
        "protocol": {
            "seed": args.seed, "device": args.device, "generator_checkpoint": str(source_path),
            "generator_checkpoint_sha256": _sha256(source_path), "pairs": args.pairs,
            "epochs": args.epochs, "examples_per_stage": args.examples_per_stage, "real_anchors_per_epoch": args.real_anchors,
            "rollout_budget": args.rollout_budget, "evaluation_budget": args.eval_budget,
            "temperature": args.temperature, "learning_rate": args.lr,
            "training": "train-only synthetic curriculum plus real train anchors; no memory or stored traces in policy rollout",
            "supervision": "teacher-forced training-only traces; fixed target-minus-current byte sketch after THINK",
            "loss": "supervised_weight*trace_CE + rl_weight*[-(reward-greedy_baseline)*sampled_log_probability] + 0.5*reasoning_MSE",
            "reward": "1 - byte_levenshtein(committed_canvas,target)/max(len(committed_canvas),len(target),1) - 0.01*(THINK steps + EDIT actions)",
            "validation_selection": "only final-stage epochs; maximize real validation exact fraction, break ties by normalized byte similarity; earliest on ties",
            "test_inference": "greedy autonomous cognitive_generate; no memory, controller, stored trace, instruction, target or target-derived budget; baseline, validation-selected and last-epoch policies evaluated after selection",
            "real_fixture_limitation": "the real held-out pairs do not include actionable edit requests; source alone underdetermines the desired target",
            "guard_limit": "hash guard checks registered raw bytes and trace states, not model-weight history or semantic duplicates",
            "registered_processed_synthetic_holdouts": len(registered_synthetic),
            "allow_missing_processed_holdouts": allow_missing,
            "synthetic_curriculum_validation_examples": len(synthetic_validation),
            "best_epoch": best_epoch, "best_validation_rank": list(best_rank),
            "generator_parameters": sum(parameter.numel() for parameter in generator.parameters()),
            "reasoning_predictor_parameters": sum(parameter.numel() for parameter in generator.reasoning_predictor.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        },
        "counts": {"real_train": len(real_train), "real_validation": len(real_validation), "real_test": len(real_test),
                   "synthetic_train": len(synthetic_train)},
        "reward_curve": logs, "baseline_test": baseline_test, "selected_test": selected_test, "final_test": final_test,
        "selected_validation": validation_final, "synthetic_validation": synthetic_validation_final,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "baseline_test": f"{baseline_test['exact']}/{baseline_test['count']}",
                      "selected_test": f"{selected_test['exact']}/{selected_test['count']}",
                      "final_epoch_test": f"{final_test['exact']}/{final_test['count']}",
                      "synthetic_validation": f"{synthetic_validation_final['exact']}/{synthetic_validation_final['count']}"},
                     indent=2), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="runs/real_traces/generator.pt")
    parser.add_argument("--generator-heads", type=int, default=4)
    parser.add_argument("--pairs", default="data/real_edit_pairs.jsonl")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--allow-missing-processed-holdouts", action="store_true",
                        help="explicitly accept an incomplete processed holdout registry")
    parser.add_argument("--output-dir", default="runs/phase8")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--examples-per-stage", type=int, default=8)
    parser.add_argument("--real-anchors", type=int, default=2)
    parser.add_argument("--rollout-budget", type=int, default=16)
    parser.add_argument("--eval-budget", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=881)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
