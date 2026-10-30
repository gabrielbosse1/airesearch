"""Co-training loop: MemoryModel <-> GeneratorConditioned.

The two models improve *each other* rather than being trained in isolation:

  1. Generator -> Memory pressure: the generator must produce output that is
     consistent with what memory knows about the input. The consistency loss
     pushes the generator's expected output embedding toward the memory
     context retrieved for that input — so the generator learns to be
     "memorable" and consistent with long-term memory.

  2. Memory -> Generator pressure: the memory model's retrieved context must
     actually help the generator predict the right next bytes. The memory
     encoders receive gradients through the generator's loss so memory learns
     to be a better context provider.

This mirrors the user's request: "the model that generates later is trained so
it understands better what the model that remember does, and the opposite."

External examples adjust memory *and* knowledge: new (input, output, context)
triples are written to memory (fast write) and then both models co-train on
them, so a skill learned from several examples becomes stable knowledge
without full retraining.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.memory_model import MemoryModel
from models.generator_conditioned import GeneratorConditioned, GeneratorState
from models.cognitive import action_from_record
from training.cognitive_train import cognitive_trace_supervision_loss, validate_action_trace
from training.replay import ReplayBuffer, ReplayExample, mixed_examples


CURRICULUM_STAGES = (
    "single_byte_substitution",
    "single_word_replacement",
    "single_line_edit",
    "multiline_insert_delete",
    "code_block_move",
)


def curriculum_stage(epoch: int, total_epochs: int) -> str:
    """Return the stage for a zero-based epoch, in five equal progress bands."""
    if (not isinstance(epoch, int) or isinstance(epoch, bool)
            or not isinstance(total_epochs, int) or isinstance(total_epochs, bool)
            or total_epochs < 1 or not 0 <= epoch < total_epochs):
        raise ValueError("epoch must be in [0, total_epochs), with total_epochs positive")
    return CURRICULUM_STAGES[min(4, 5 * epoch // total_epochs)]


def curriculum_indices(examples: list[dict[str, Any]], epoch: int, total_epochs: int) -> list[int]:
    """Select labelled training examples for this epoch; missing stages fail closed."""
    stage = curriculum_stage(epoch, total_epochs)
    selected = [i for i, example in enumerate(examples) if example.get("curriculum_stage") == stage]
    if not selected:
        raise ValueError(f"no training examples for curriculum stage {stage!r}")
    return selected


# --------------------------------------------------------------------------- #
# Small helpers (defined first so they can be used below)
# --------------------------------------------------------------------------- #
def math_sqrt(x: float) -> float:
    return float(x) ** 0.5


def _byte_mean_embedding(ids, byte_emb_weight, valid_mask=None):
    """Mean byte embedding; byte 0 is valid and masks indicate padding."""
    mask = torch.ones_like(ids, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    weights = mask.to(byte_emb_weight.dtype).unsqueeze(-1)
    emb = byte_emb_weight[ids.clamp(min=0, max=255)] * weights
    return emb.sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def _expected_embedding(logits, byte_emb_weight, valid_mask=None):
    """Differentiable expected byte embedding from generator logits.

    logits: (B, L, V). Returns (B, d) mean expected byte embedding.
    """
    probs = F.softmax(logits, dim=-1)  # (B, L, V)
    byte_probs = probs[..., : GeneratorConditioned.VOCAB_BYTES]  # (B, L, 256)
    byte_probs = byte_probs / byte_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    emb = byte_probs @ byte_emb_weight  # (B, L, d)
    if valid_mask is None:
        return emb.mean(dim=1)
    weights = valid_mask.to(emb.dtype).unsqueeze(-1)
    return (emb * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def prepare_forced_memory_examples(memory, examples: list[dict[str, bytes]]) -> list[dict[str, Any]]:
    """Write observed episodes and cache actual scoped ``MemoryModel.retrieve`` values.

    The target is checked for provenance and later used only as a training
    label. The cached model input contains the question and retrieved vectors,
    never the episode bytes or target text. Call on an isolated training bank.
    """
    from models.conversation import ConversationTurn
    from models.vector_memory import VectorConversationMemory, question_topic, statement_fact
    if not isinstance(memory, VectorConversationMemory) or not examples:
        raise ValueError("a vector memory and nonempty examples are required")
    # Validate the whole batch before its first persistent write.
    for example in examples:
        statement, question, target = example["statement"], example["question"], example["target"]
        if not all(isinstance(x, bytes) and x for x in (statement, question, target)):
            raise TypeError("statements, questions and targets must be nonempty raw bytes")
        fact = statement_fact(statement)
        if fact is None or fact[0] != question_topic(question) or fact[1] != target:
            raise ValueError("training label must match the affirmative stored statement and question")
        memory.memory.validate_holdout_write(statement, target, metadata={"split": "train"}, extra=(question,))
    prepared = []
    for example in examples:
        user = b"phase12-training"
        memory.record_episode(user, ConversationTurn(example["statement"], b""))
        retrieved = memory.retrieve_fact(user, example["question"])
        if retrieved is None:
            raise RuntimeError("training fact was not retrieved")
        prepared.append({"question": example["question"], "vectors": retrieved.vectors.detach().clone(),
                         "target": example["target"]})
    return prepared


def forced_memory_logits(generator: GeneratorConditioned, memory, reader,
                         batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Return response distributions on questions and retrieved vectors only.

    Use ``prepare_forced_memory_examples`` to write/retrieve first. Generator
    input consists exclusively of ``U:`` plus the current question; actual
    vectors are also passed into its ``memory_context``. Labels are consumed
    only by the loss. Direct memory writes remain gradient-free.
    """
    from models.vector_memory import VectorConversationMemory, VectorFactCodec, VectorMemoryReader
    if not isinstance(memory, VectorConversationMemory) or not isinstance(reader, VectorMemoryReader) or not batch:
        raise ValueError("a vector memory, reader and nonempty prepared batch are required")
    device = generator.byte_emb.weight.device
    if reader.projection.weight.device != device:
        raise ValueError("generator and reader must share a device")
    prompts = [b"U:" + row["question"] for row in batch]
    if any(len(prompt) > generator.max_len for prompt in prompts):
        raise ValueError("question exceeds generator context limit")
    max_prompt = max(map(len, prompts))
    ids = torch.zeros((len(batch), max_prompt), dtype=torch.long, device=device)
    lengths = torch.tensor([len(prompt) for prompt in prompts], dtype=torch.long, device=device)
    focus = torch.full((len(batch),), 2, dtype=torch.long, device=device)
    max_slots = max(row["vectors"].shape[0] for row in batch)
    if not 1 <= max_slots <= VectorFactCodec.MAX_SLOTS:
        raise ValueError("invalid number of vector slots")
    vectors = torch.zeros((len(batch), max_slots, VectorFactCodec.DIM), device=device)
    mask = torch.zeros((len(batch), max_slots), dtype=torch.bool, device=device)
    targets = torch.full((len(batch), max_slots * VectorFactCodec.BYTES_PER_SLOT), VectorMemoryReader.IGNORE,
                         dtype=torch.long, device=device)
    for i, (row, prompt) in enumerate(zip(batch, prompts)):
        ids[i, :len(prompt)] = torch.tensor(list(prompt), dtype=torch.long, device=device)
        value, retrieved = row["target"], row["vectors"]
        if not isinstance(value, bytes) or not 1 <= len(value) <= VectorFactCodec.MAX_BYTES:
            raise ValueError("training target must be 1..40 bytes")
        if retrieved.ndim != 2 or retrieved.shape[1] != VectorFactCodec.DIM or len(value) > retrieved.shape[0] * 8:
            raise ValueError("retrieved vector/target shape mismatch")
        vectors[i, :len(retrieved)] = retrieved.to(device=device)
        mask[i, :len(retrieved)] = True
        targets[i, :len(value)] = torch.tensor(list(value), dtype=torch.long, device=device)
        if len(value) < len(retrieved) * VectorFactCodec.BYTES_PER_SLOT:
            targets[i, len(value)] = VectorMemoryReader.EOS
    with torch.no_grad():
        context = generator.conversation_features(ids, lengths, focus_start=focus, memory_context=vectors, memory_mask=mask)
    output = reader(context.detach(), vectors, mask)
    return output, targets, {"batch_size": len(batch), "retrieved_slots": int(mask.sum()),
                             "episode_bytes_in_generator_input": 0}


def forced_memory_training_step(generator: GeneratorConditioned, memory, reader, batch: list[dict[str, Any]],
                                optimizer: torch.optim.Optimizer | None = None) -> tuple[torch.Tensor, dict[str, int]]:
    """Train a neural byte reader from actual cached retrievals without episode bytes."""
    from models.vector_memory import VectorMemoryReader
    output, targets, stats = forced_memory_logits(generator, memory, reader, batch)
    loss = F.nll_loss(output.reshape(-1, 257), targets.reshape(-1), ignore_index=VectorMemoryReader.IGNORE)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("non-finite forced-memory loss")
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), 2.0, error_if_nonfinite=True)
        optimizer.step()
    return loss, stats


def mixed_training_step(new_examples: list[ReplayExample], replay_buffer: ReplayBuffer,
                        loss_fn: Callable[[list[ReplayExample]], torch.Tensor],
                        optimizer: torch.optim.Optimizer, *, batch_size: int,
                        rng: random.Random, old_fraction: float = 0.30) -> dict[str, float | int]:
    """Perform one genuinely 70/30 new/replay update with caller-supplied loss.

    ``loss_fn`` must return one differentiable scalar averaged over the mixed
    examples. Replay is checked at insertion/loading, not silently populated
    from user transcripts. A replay buffer cannot guarantee no forgetting.
    """
    batch = mixed_examples(new_examples, replay_buffer, batch_size=batch_size, rng=rng, old_fraction=old_fraction)
    loss = loss_fn(batch)
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise ValueError("mixed loss must be one finite scalar")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    old_count = round(batch_size * old_fraction)
    return {"loss": float(loss.detach()), "new_count": batch_size - old_count, "replay_count": old_count}


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ExampleDataset(Dataset):
    """External examples for skill/memory acquisition.

    Each item: {"input": [byte ids], "output": [byte ids], "context": dict}
    where context holds {person, modality, session, type, ...}.
    """

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch or any(len(e["output"]) == 0 for e in batch):
        raise ValueError("co-training requires non-empty batches and outputs")
    max_in = max(1, max(len(e["input"]) for e in batch))
    max_out = max(len(e["output"]) for e in batch)
    input_ids = torch.zeros((len(batch), max_in), dtype=torch.long)
    output_ids = torch.zeros((len(batch), max_out), dtype=torch.long)
    input_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    output_mask = torch.zeros_like(output_ids, dtype=torch.bool)
    contexts = []
    for i, e in enumerate(batch):
        inp = torch.tensor(list(e["input"]), dtype=torch.long)
        out = torch.tensor(list(e["output"]), dtype=torch.long)
        if bool(((inp < 0) | (inp > 255)).any() or ((out < 0) | (out > 255)).any()):
            raise ValueError("co-training examples must contain raw bytes only")
        input_ids[i, :len(inp)] = inp
        output_ids[i, :len(out)] = out
        input_mask[i, :len(inp)] = True
        output_mask[i, :len(out)] = True
        contexts.append(e.get("context", {}))
    return {"input_ids": input_ids, "output_ids": output_ids, "input_mask": input_mask,
            "output_mask": output_mask, "contexts": contexts,
            "instruction_bytes": [e.get("instruction_bytes") for e in batch],
            "trace_actions": [e.get("trace_actions") for e in batch]}


def encode_text(text: str) -> list[int]:
    return list(text.encode("utf-8"))


# --------------------------------------------------------------------------- #
# Core co-training step
# --------------------------------------------------------------------------- #
def co_train_step(
    memory: MemoryModel,
    generator: GeneratorConditioned,
    batch: dict[str, Any],
    device: str | torch.device,
    opt_mem: torch.optim.Optimizer,
    opt_gen: torch.optim.Optimizer,
    lambda_consistency: float = 1.0,
    lambda_retrieval: float = 1.0,
    lambda_confidence: float = 0.1,
    cognitive_trace_weight: float = 0.0,
    reasoning_weight: float = 0.5,
) -> dict[str, float]:
    """Co-train on actual top-K persistent reads, live encodings and labels.

    Byte zero is never a padding sentinel: explicit masks distinguish valid
    bytes from batch padding. A single backward pass updates both models, so
    neither optimizer mutates parameters needed by a second backward pass.
    """
    input_ids = batch["input_ids"].to(device)
    output_ids = batch["output_ids"].to(device)
    input_mask = batch.get("input_mask", torch.ones_like(input_ids, dtype=torch.bool)).to(device).bool()
    output_mask = batch.get("output_mask", torch.ones_like(output_ids, dtype=torch.bool)).to(device).bool()
    contexts = batch["contexts"]
    b = input_ids.shape[0]
    if input_mask.shape != input_ids.shape or output_mask.shape != output_ids.shape or not bool(output_mask.any()):
        raise ValueError("co-training masks must match the byte tensors and include an output")
    if len(contexts) != b or output_ids.shape[0] != b:
        raise ValueError("batch dimensions and contexts must match")
    if bool(((input_ids < 0) | (input_ids > 255)).any() or ((output_ids < 0) | (output_ids > 255)).any()):
        raise ValueError("co-training inputs and targets must be raw bytes")
    if not 0 <= cognitive_trace_weight < float("inf") or not 0 <= reasoning_weight < float("inf"):
        raise ValueError("trace and reasoning weights must be finite and non-negative")
    # Check the entire batch before a memory write, optimizer step or mode change.
    batch_traces = batch.get("trace_actions")
    instructions = batch.get("instruction_bytes", [None] * b)
    if len(instructions) != b or (batch_traces is not None and len(batch_traces) != b):
        raise ValueError("batch trace and instruction counts must match batch size")
    for instruction in instructions:
        if instruction is not None and not isinstance(instruction, bytes):
            raise TypeError("instruction_bytes must be raw bytes")
        if instruction is not None and len(instruction) > generator.max_canvas_bytes:
            raise ValueError("instruction exceeds max_canvas_bytes")
    for i in range(b):
        source = bytes(input_ids[i][input_mask[i]].detach().cpu().tolist())
        target = bytes(output_ids[i][output_mask[i]].detach().cpu().tolist())
        # Trace payloads can contain more bytes than the final target, including
        # temporary insert/delete edits. Inspect them before training or writes.
        metadata = {"context": contexts[i], "trace_actions": None if batch_traces is None else batch_traces[i],
                    "instruction_bytes": instructions[i]}
        memory.validate_holdout_write(source, target, metadata=metadata)
    decoded_actions = []
    any_traces = batch_traces is not None and any(trace is not None for trace in batch_traces)
    use_traces = bool(cognitive_trace_weight or (reasoning_weight and any_traces))
    if use_traces:
        traces = batch_traces
        if traces is None or (cognitive_trace_weight and any(trace is None for trace in traces)):
            raise ValueError("trace-supervised cognitive co-training requires a trace for every example")
        for i, records in enumerate(traces):
            if records is None:
                continue  # Auxiliary reasoning can use a partially labelled batch.
            source = bytes(input_ids[i][input_mask[i]].detach().cpu().tolist())
            target = bytes(output_ids[i][output_mask[i]].detach().cpu().tolist())
            actions = [action_from_record(record) for record in records]
            observer = (lambda raw: memory.validate_holdout_write(raw)) if memory.heldout_guard is not None else None
            if validate_action_trace(generator, source, actions, raw_observer=observer) != target:
                raise ValueError("cognitive trace does not reproduce its target")
            decoded_actions.append((i, actions))

    generator.train()
    memory.train()
    byte_emb_weight = generator.byte_emb.weight
    input_emb = _byte_mean_embedding(input_ids, byte_emb_weight, input_mask)
    with torch.no_grad():
        for i in range(b):
            pattern = bytes(input_ids[i][input_mask[i]].detach().cpu().tolist())
            try:
                # Keep an existing structured skill active instead of flooding
                # the procedural bank with duplicate unstructured batch writes.
                memory.execute_procedure(pattern)
            except LookupError:
                target = bytes(output_ids[i][output_mask[i]].detach().cpu().tolist())
                memory.write(input_emb[i], {**contexts[i], "type": "example_input"},
                             raw_input=pattern, raw_output=target)

    # Persistent multi-vector retrieval carries a straight-through gradient to
    # the memory router/query/importance scorer. A live in-batch memory token
    # additionally trains key/value encoders without mutating stored buffers.
    mem_tokens, top_weights, mem_mask = memory.retrieve_differentiable(input_emb, top_k=generator.memory_top_k)
    keys, values = memory.encode(input_emb)
    queries = memory.query_encoder(input_emb)
    live_attn = F.softmax((queries @ keys.T) / math_sqrt(memory.d_model), dim=-1)
    live_read = live_attn @ values
    mem_tokens = torch.cat([mem_tokens, live_read.unsqueeze(1)], dim=1)
    mem_mask = torch.cat([mem_mask, torch.ones(b, 1, dtype=torch.bool, device=device)], dim=1)
    context = 0.75 * (mem_tokens[:, :-1] * top_weights.unsqueeze(-1)).sum(dim=1) + 0.25 * live_read

    start = torch.full((b, 1), GeneratorConditioned.FUNC_CALL, dtype=torch.long, device=device)
    gen_input = torch.cat([start, output_ids[:, :-1]], dim=1)
    state = generator(gen_input, mem_tokens, memory_mask=mem_mask, return_state=True)
    assert isinstance(state, GeneratorState)
    gen_logits = state.output
    losses = F.cross_entropy(gen_logits.reshape(-1, generator.VOCAB_SIZE), output_ids.reshape(-1), reduction="none")
    gen_loss = losses.reshape_as(output_ids)[output_mask].mean()
    generation_logits = gen_logits.detach().clone()
    generation_logits[..., generator.LOAD_SPAN:] = -torch.inf
    correct = (generation_logits.argmax(dim=-1) == output_ids).to(gen_logits.dtype)
    confidence_loss = F.binary_cross_entropy_with_logits(state.confidence_logits[output_mask], correct[output_mask])

    real_out_emb = _byte_mean_embedding(output_ids, byte_emb_weight, output_mask)
    context_norm = F.normalize(context, dim=-1)
    real_norm = F.normalize(real_out_emb, dim=-1)
    retrieval_sim = (context_norm * real_norm).sum(dim=-1).mean()
    retrieval_loss = -retrieval_sim
    gen_out_emb = _expected_embedding(gen_logits, byte_emb_weight, output_mask)
    consistency_sim = (F.normalize(gen_out_emb, dim=-1) * F.normalize(context.detach(), dim=-1)).sum(dim=-1).mean()
    consistency_loss = -consistency_sim

    trace_loss = gen_loss.new_zeros(())
    reasoning_loss = gen_loss.new_zeros(())
    thought_count = 0
    if use_traces:
        trace_losses, reasoning_losses = [], []
        for i, actions in decoded_actions:
            source = bytes(input_ids[i][input_mask[i]].detach().cpu().tolist())
            target = bytes(output_ids[i][output_mask[i]].detach().cpu().tolist())
            # Filter invalid retrieved slots while retaining gradients through valid reads.
            context_i = mem_tokens[i:i + 1, mem_mask[i]]
            # The live training read is not available at deployment. Train both
            # the conditioned and unconditioned policy rather than relying on it.
            conditioned = cognitive_trace_supervision_loss(
                generator, source, actions, memory_context=context_i,
                instruction_bytes=instructions[i], target_bytes=target, return_reasoning=True,
            )
            unconditioned = cognitive_trace_supervision_loss(
                generator, source, actions, instruction_bytes=instructions[i],
                target_bytes=target, return_reasoning=True,
            )
            trace_losses.append((conditioned[0] + unconditioned[0]) * 0.5)
            reasoning_losses.append((conditioned[1] + unconditioned[1]) * 0.5)
            thought_count += conditioned[2]
        trace_loss = torch.stack(trace_losses).mean()
        reasoning_loss = torch.stack(reasoning_losses).mean()
    total = (gen_loss + lambda_consistency * consistency_loss + lambda_retrieval * retrieval_loss
             + lambda_confidence * confidence_loss + cognitive_trace_weight * trace_loss
             + reasoning_weight * reasoning_loss)
    opt_gen.zero_grad()
    opt_mem.zero_grad()
    total.backward()
    opt_gen.step()
    opt_mem.step()
    memory.consolidate(lr=0.005)
    return {
        "gen_loss": float(gen_loss.item()),
        "retrieval_sim": float(retrieval_sim.item()),
        "consistency_sim": float(consistency_sim.item()),
        "consistency_loss": float(consistency_loss.item()),
        "confidence_loss": float(confidence_loss.item()),
        "cognitive_trace_loss": float(trace_loss.detach().item()),
        "reasoning_loss": float(reasoning_loss.detach().item()),
        "reasoning_think_events": float(thought_count),
        "generator_total_loss": float(total.detach().item()),
        "confidence_accuracy": float(((state.confidence >= 0.5) == correct.bool())[output_mask].float().mean().item()),
    }



# --------------------------------------------------------------------------- #
# External example ingestion
# --------------------------------------------------------------------------- #
def ingest_examples(
    memory: MemoryModel,
    generator: GeneratorConditioned,
    examples: list[dict[str, Any]],
    device: str | torch.device,
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 1e-3,
    cognitive_trace_weight: float = 0.0,
    reasoning_weight: float = 0.5,
    curriculum: bool = False,
) -> list[dict[str, Any]]:
    """Ingest examples and co-train, optionally following the five-stage curriculum.

    With curriculum enabled, examples must be labelled with ``curriculum_stage``;
    only the current stage is written or trained in that epoch. All stages and
    holdout-sensitive bytes are preflighted before the first persistent write.
    The legacy all-examples ingestion behavior remains the default.
    """
    if (not examples or epochs < 0 or batch_size < 1 or lr <= 0
            or not 0 <= cognitive_trace_weight < float("inf")
            or not 0 <= reasoning_weight < float("inf")):
        raise ValueError("examples, batch_size, learning rate, epochs and loss weights must be valid")
    if curriculum and 0 < epochs < len(CURRICULUM_STAGES):
        raise ValueError("a complete curriculum needs at least five epochs")
    epoch_indices = ([curriculum_indices(examples, epoch, epochs) for epoch in range(epochs)]
                     if curriculum and epochs else [])
    if memory.d_model != generator.d_model:
        raise ValueError("memory and generator must share d_model")
    # Validate the entire batch before the first persistent write.
    prepared: list[tuple[bytes, bytes, dict[str, Any], bytes, torch.Tensor | None]] = []
    for example in examples:
        inp, out = bytes(example["input"]), bytes(example["output"])
        context = dict(example.get("context", {}))
        explanation = context.get("explanation_bytes", b"")
        instruction = example.get("instruction_bytes")
        trace = example.get("latent_trace")
        records = example.get("trace_actions")
        if not isinstance(explanation, bytes):
            raise TypeError("explanation_bytes must be bytes")
        if instruction is not None:
            if not isinstance(instruction, bytes):
                raise TypeError("instruction_bytes must be bytes")
            if len(instruction) > generator.max_canvas_bytes:
                raise ValueError("instruction exceeds max_canvas_bytes")
            # Guard and persist the exact conditioning bytes alongside the trace.
            context["instruction_bytes"] = instruction
        if (epochs and not out) or len(inp) > memory.max_pattern_bytes or len(out) > memory.max_pattern_bytes:
            raise ValueError("training needs non-empty outputs; patterns must fit memory limits")
        if len(explanation) > memory.max_explanation_bytes:
            raise ValueError("explanation exceeds memory limits")
        if trace is not None and (not torch.is_tensor(trace) or trace.ndim != 2 or trace.shape[1] != memory.d_model
                                  or trace.shape[0] > memory.max_trace_steps or not bool(torch.isfinite(trace).all())):
            raise ValueError("latent_trace must be finite with shape (steps, d_model) within memory limits")
        if records is not None:
            actions = [action_from_record(record) for record in records]
            observer = (lambda raw: memory.validate_holdout_write(raw)) if memory.heldout_guard is not None else None
            if validate_action_trace(generator, inp, actions, raw_observer=observer) != out:
                raise ValueError("trace actions do not reproduce the supplied output")
            context["trace_actions"] = records
        elif cognitive_trace_weight and epochs:
            raise ValueError("trace-supervised co-training requires actions for every example")
        # Include trace payloads and the default type exactly as write_procedure
        # will store them, so no later example can fail after earlier writes.
        guarded_context = dict(context)
        guarded_context.setdefault("type", "procedure")
        memory.validate_holdout_write(inp, out, metadata=guarded_context, extra=(explanation,))
        prepared.append((inp, out, context, explanation, trace))

    # Direct writes enable deterministic replay even if generation is imperfect.
    def summarize(data: bytes) -> torch.Tensor:
        if not data:
            return torch.zeros(generator.d_model, device=device)
        ids = torch.tensor(list(data), dtype=torch.long, device=device)
        return generator.byte_emb(ids).mean(dim=0)

    written: set[int] = set()

    def write_prepared(indices: list[int]) -> None:
        with torch.no_grad():
            for index in indices:
                if index in written:
                    continue
                inp, out, context, explanation, supplied_trace = prepared[index]
                trace = (torch.stack([summarize(inp), summarize(out)])[:memory.max_trace_steps]
                         if supplied_trace is None else supplied_trace)
                memory.write_procedure(inp, out, trace, explanation, metadata=context)
                written.add(index)

    # Ordinary ingestion preserves its historical write-all behavior. With a
    # curriculum, future-stage examples must not enter persistent memory early.
    if not curriculum or epochs == 0:
        write_prepared(list(range(len(examples))))
    if epochs == 0:
        return []

    opt_mem = torch.optim.Adam(memory.parameters(), lr=lr)
    opt_gen = torch.optim.Adam(generator.parameters(), lr=lr)

    logs = []
    for epoch in range(epochs):
        indices = epoch_indices[epoch] if curriculum else list(range(len(examples)))
        if curriculum:
            write_prepared(indices)
        ds = ExampleDataset([examples[i] for i in indices])
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
        epoch_logs = []
        for batch in dl:
            step = co_train_step(
                memory, generator, batch, device, opt_mem, opt_gen,
                cognitive_trace_weight=cognitive_trace_weight, reasoning_weight=reasoning_weight,
            )
            epoch_logs.append(step)
        avg: dict[str, Any] = {
            k: sum(d[k] for d in epoch_logs) / len(epoch_logs) for k in epoch_logs[0]
        } if epoch_logs else {}
        if curriculum:
            avg["curriculum_stage"] = curriculum_stage(epoch, epochs)
            avg["curriculum_examples"] = len(indices)
        logs.append(avg)
        print(
            f"[ingest epoch {epoch+1}/{epochs}] "
            f"gen_loss {avg.get('gen_loss',0):.4f} "
            f"retrieval_sim {avg.get('retrieval_sim',0):.4f} "
            f"consistency_sim {avg.get('consistency_sim',0):.4f}",
            flush=True,
        )
    return logs


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_system(memory: MemoryModel, generator: GeneratorConditioned, dirpath: str) -> None:
    import os
    os.makedirs(dirpath, exist_ok=True)
    memory.save(os.path.join(dirpath, "memory.pt"))
    torch.save(generator.state_dict(), os.path.join(dirpath, "generator.pt"))


def load_system(
    memory: MemoryModel, generator: GeneratorConditioned, dirpath: str,
    map_location: str | torch.device = "cpu",
) -> tuple[MemoryModel, GeneratorConditioned]:
    memory.load(os.path.join(dirpath, "memory.pt"), map_location=map_location)
    generator.load_state_dict(
        torch.load(os.path.join(dirpath, "generator.pt"), map_location=map_location)
    )
    return memory, generator
