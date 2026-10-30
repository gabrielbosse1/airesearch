"""Ingest real before/after byte pairs as optimal edit traces and procedures.

Levenshtein distance uses unit-cost insertion, deletion, replacement and free
matches. It is optimal for that cost model; a MOVE is *not* a unit-cost
Levenshtein operation. Inputs and outputs stay raw bytes and are never run as
code. The JSONL input accepts hex-encoded bytes, so NUL and arbitrary binary
content survive transport without a tokenizer.
"""
from __future__ import annotations

import argparse
import json
import sys
from array import array
from pathlib import Path
from typing import Any, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.holdout_registry import synthetic_holdouts
from models.cognitive import CognitiveAction, CognitiveObservation, apply_edit, action_to_record
from models.generator_conditioned import GeneratorConditioned
from models.memory_model import MemoryModel
from models.holdout import HoldoutGuard
from training.co_train import ingest_examples
from training.cognitive_train import validate_action_trace
from training.train_confidence import load_generator


def optimal_edit_actions(source: bytes, target: bytes, *, max_cells: int = 1_000_000) -> tuple[int, list[CognitiveAction]]:
    """Exact minimum unit-cost edit distance and left-to-right workspace actions.

    Coordinates address the *current* workspace. Tie-breaking is deterministic
    (replace, delete, insert). The quadratic bound is explicit and prevents
    accidentally allocating an unbounded matrix for large documents.
    """
    if not isinstance(source, bytes) or not isinstance(target, bytes):
        raise TypeError("source and target must be bytes")
    n, m = len(source), len(target)
    if max_cells < 1 or (n + 1) * (m + 1) > max_cells:
        raise ValueError("edit-distance matrix exceeds max_cells; split the document first")
    rows = [array("I", [0]) * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        rows[i][m] = n - i
    for j in range(m + 1):
        rows[n][j] = m - j
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if source[i] == target[j]:
                rows[i][j] = rows[i + 1][j + 1]
            else:
                rows[i][j] = 1 + min(rows[i + 1][j + 1], rows[i + 1][j], rows[i][j + 1])
    i = j = pos = 0
    actions: list[CognitiveAction] = []
    while i < n or j < m:
        if i < n and j < m and source[i] == target[j] and rows[i][j] == rows[i + 1][j + 1]:
            i += 1; j += 1; pos += 1
        elif i < n and j < m and rows[i][j] == 1 + rows[i + 1][j + 1]:
            actions.append(CognitiveAction(GeneratorConditioned.EDIT_OP, start=pos, end=pos + 1,
                                           kind="replace", data=target[j:j + 1]))
            i += 1; j += 1; pos += 1
        elif i < n and rows[i][j] == 1 + rows[i + 1][j]:
            actions.append(CognitiveAction(GeneratorConditioned.EDIT_OP, start=pos, end=pos + 1, kind="delete"))
            i += 1
        elif j < m and rows[i][j] == 1 + rows[i][j + 1]:
            actions.append(CognitiveAction(GeneratorConditioned.EDIT_OP, start=pos, kind="insert", data=target[j:j + 1]))
            j += 1; pos += 1
        else:
            raise RuntimeError("invalid edit-distance traceback")
    check = source
    for action in actions:
        check = apply_edit(check, action, max_length=max(n, m) + len(actions))
    if check != target or len(actions) != rows[0][0]:
        raise AssertionError("edit-distance traceback did not reproduce its target")
    return int(rows[0][0]), actions


def make_control_trace(source: bytes, target: bytes) -> tuple[int, list[CognitiveAction]]:
    distance, edits = optimal_edit_actions(source, target)
    return distance, [CognitiveAction(GeneratorConditioned.LOAD_SPAN, start=0, end=len(source)),
                      CognitiveAction(GeneratorConditioned.THINK), *edits,
                      CognitiveAction(GeneratorConditioned.SAVE_SPAN), CognitiveAction(GeneratorConditioned.DONE)]


def load_pairs(path: str | Path, *, split: str | None = None) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if split is not None and item.get("split") != split:
                continue
            try:
                source, target = bytes.fromhex(item["input_hex"]), bytes.fromhex(item["output_hex"])
                explanation = item.get("explanation", "").encode("utf-8")
            except (KeyError, ValueError, AttributeError) as exc:
                raise ValueError(f"invalid byte-pair record at {path}:{line_no}") from exc
            records.append({"id": str(item.get("id", line_no)), "input": source, "output": target,
                            "explanation": explanation, "split": item.get("split", "unspecified"),
                            "kind": item.get("kind", "unknown"), "source": item.get("source", "unspecified")})
    return records


def prepare_trace(record: dict[str, Any], generator: GeneratorConditioned, memory: MemoryModel) -> dict[str, Any]:
    """Execute the trace with a host controller and capture real scratchpad states.

    States are compressed by uniformly selecting at most max_trace_steps.
    Host-directed correctness here is not model-generated edit accuracy.
    """
    inp, out = record["input"], record["output"]
    if len(inp) > memory.max_pattern_bytes or len(out) > memory.max_pattern_bytes:
        raise ValueError("pair exceeds procedural memory pattern limits")
    distance, actions = make_control_trace(inp, out)
    if memory.heldout_guard is not None:
        validate_action_trace(generator, inp, actions, raw_observer=lambda raw: memory.validate_holdout_write(raw))
    if len(actions) > 4096:
        raise ValueError("trace exceeds the cognitive loop's action limit")
    states: list[torch.Tensor] = []
    def controller(obs: CognitiveObservation) -> CognitiveAction:
        states.append(obs.scratchpad.mean(dim=1).squeeze(0).cpu())
        return actions[obs.step]
    result = generator.cognitive_generate(inp, controller=controller, max_steps=len(actions))
    if not result.done or result.canvas != out:
        raise AssertionError("host-controlled trace failed to reproduce target")
    states.append(result.scratchpad.mean(dim=1).squeeze(0).cpu())
    count = min(memory.max_trace_steps, len(states))
    indices = [0] if count == 1 else [round(i * (len(states) - 1) / (count - 1)) for i in range(count)]
    trace = torch.stack([states[i] for i in indices]) if count else torch.empty(0, memory.d_model)
    metadata = {"type": "real_edit", "id": record["id"], "source": record["source"], "kind": record["kind"],
                "split": record["split"], "distance": distance, "distance_cost_model": "unit Levenshtein; moves not primitive",
                "trace_actions": [action_to_record(a) for a in actions], "explanation_bytes": record["explanation"]}
    return {"input": inp, "output": out, "context": metadata, "latent_trace": trace,
            "trace_actions": metadata["trace_actions"]}


def ingest_real_traces(memory: MemoryModel, generator: GeneratorConditioned, records: Iterable[dict[str, Any]], *,
                       epochs: int = 0, batch_size: int = 4, lr: float = 1e-3,
                       cognitive_trace_weight: float = 1.0) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    records = list(records)
    # Host trace execution itself may consult the generator; reject the entire
    # batch before it, training, or any persistent write.
    for record in records:
        metadata = {name: record.get(name, "unspecified") for name in ("id", "split", "kind", "source")}
        memory.validate_holdout_write(record["input"], record["output"],
                                     metadata=metadata, extra=(record.get("explanation", b""),))
    prepared = [prepare_trace(record, generator, memory) for record in records]
    if not prepared:
        raise ValueError("no byte pairs to ingest")
    logs = ingest_examples(memory, generator, prepared, generator.byte_emb.weight.device,
                           epochs=epochs, batch_size=batch_size, lr=lr,
                           cognitive_trace_weight=cognitive_trace_weight if epochs else 0.0)
    return prepared, logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/real_edit_pairs.jsonl")
    parser.add_argument("--split", choices=("train",), default="train",
                        help="only the training split may be persisted")
    parser.add_argument("--generator", default="runs/dual_memory/generator.pt")
    parser.add_argument("--processed-dir", default="data/processed", help="optional synthetic validation/test registry")
    parser.add_argument("--output-dir", default="runs/real_trace_ingestion")
    parser.add_argument("--epochs", type=int, default=0, help="optional joint co-training epochs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    generator = load_generator(args.generator, args.device, 4)
    records = load_pairs(args.pairs, split=args.split)
    heldout = load_pairs(args.pairs, split="validation") + load_pairs(args.pairs, split="test")
    synthetic = synthetic_holdouts(args.processed_dir, require=False)
    memory = MemoryModel(d_model=generator.d_model, capacity=max(128, len(records) * 5)).to(args.device)
    memory.set_holdout_guard(HoldoutGuard.from_records(heldout + synthetic))
    prepared, logs = ingest_real_traces(memory, generator, records, epochs=args.epochs,
                                         batch_size=args.batch_size, lr=args.lr)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    memory.save(str(out / "memory.pt")); torch.save(generator.state_dict(), out / "generator.pt")
    summary = {"count": len(prepared), "total_optimal_byte_edits": sum(e["context"]["distance"] for e in prepared),
               "co_training": logs, "known_real_heldout_records": len(heldout),
               "known_synthetic_heldout_records": len(synthetic), "holdout_audit": memory.audit_holdouts(),
               "warning": "Host-directed trace replay is not learned edit generation; exact hashes do not detect near-duplicates."}
    (out / "ingestion.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
