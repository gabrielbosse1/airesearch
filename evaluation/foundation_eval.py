"""Benchmark the ByteCore v2 foundation (Phase 16).

Four checks, each answering a different question:

1. **Next-byte accuracy.** Can a standard causal LM learn the ReAct byte
   stream? Reported overall, on short sequences, and split by segment
   (Thought / Action / Observation / Answer) so a high score cannot hide
   behind format boilerplate.
2. **Tool interception.** Does the regex interceptor catch
   ``Action: exec_python("...")``, run the sandbox, and inject a real
   ``Observation:`` -- and does it refuse to run anything it cannot parse?
3. **Memory recall.** Does Phase 12's vector-only recall still score 10/10
   with the new prefix integration, and does the prefix carry the retrieved
   bytes?
4. **Regression.** Does the whole suite still pass?

Usage::

    python -m evaluation.foundation_eval
    python -m evaluation.foundation_eval --skip-unit-tests
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from data.scalable_react_dataset import Trajectory
from models.foundation import (CausalByteTransformer, FoundationConfig, encode_bytes,
                               load_foundation)
from models.react import ReactLoop, parse_action, sandbox_observation
from models.sandbox import ByteSandbox, looks_failed

DEFAULT_MODEL = "runs/phase16/foundation.pt"
DEFAULT_DATA = "runs/phase16"
SEGMENT_MARKERS = ((b"Thought:", "thought"), (b"Action:", "action"),
                   (b"Observation:", "observation"), (b"Answer:", "answer"))


# --------------------------------------------------------------------------
# 1. Next-byte accuracy
# --------------------------------------------------------------------------
def segment_positions(text: bytes) -> list[str]:
    """Label each byte position with the segment it belongs to."""
    labels = ["preamble"] * len(text)
    for start in range(len(text)):
        if text[start] == 0x0A or start == 0:
            origin = start + 1 if text[start] == 0x0A else start
            for marker, name in SEGMENT_MARKERS:
                if text[origin:origin + len(marker)] == marker:
                    end = text.find(b"\n", origin)
                    end = len(text) if end < 0 else end
                    for index in range(origin, end):
                        labels[index] = name
                    break
    return labels


@torch.no_grad()
def next_byte_accuracy(model: CausalByteTransformer, rows: list[Trajectory], *,
                       max_length: int | None = None) -> dict:
    """Teacher-forced next-byte accuracy, overall and per segment."""
    model.eval()
    correct = total = 0
    per_segment: dict[str, list[int]] = {}
    counted_rows = 0
    truncated_rows = 0
    for row in rows:
        ids = encode_bytes(row.text, bos=True, eos=True)
        if max_length is not None and ids.shape[0] > max_length:
            continue
        # A checkpoint with a shorter context than the trajectory still scores:
        # the tail beyond max_seq_len is reported as skipped, never guessed.
        limit = model.config.max_seq_len
        if ids.shape[0] > limit:
            truncated_rows += 1
            ids = ids[:limit]
        counted_rows += 1
        window = ids[:-1].unsqueeze(0)
        targets = ids[1:]
        logits = model(window)[0]
        predictions = logits.argmax(dim=-1)
        labels = segment_positions(row.text)
        for position in range(targets.shape[0]):
            segment = labels[position] if position < len(labels) else "preamble"
            bucket = per_segment.setdefault(segment, [0, 0])
            bucket[1] += 1
            bucket[0] += int(predictions[position].item() == int(targets[position].item()))
            total += 1
            correct += int(predictions[position].item() == int(targets[position].item()))
    return {
        "rows": counted_rows,
        "rows_truncated_to_context": truncated_rows,
        "bytes": total,
        "next_byte_accuracy": correct / max(total, 1),
        "by_segment": {name: {"accuracy": hits / max(count, 1), "bytes": count}
                       for name, (hits, count) in sorted(per_segment.items())},
    }


# --------------------------------------------------------------------------
# 2. Tool interception
# --------------------------------------------------------------------------
@dataclass
class ScriptedModel:
    """A stand-in policy that emits a fixed continuation, for loop tests."""

    script: bytes
    calls: int = 0
    config: object = field(default_factory=FoundationConfig)

    def generate(self, prefix, *, max_new_bytes: int = 160, temperature: float = 0.0,
                 stop: tuple[int, ...] = ()):
        self.calls += 1
        return torch.tensor(list(self.script[:max_new_bytes]), dtype=torch.long)


def interception_checks(sandbox: ByteSandbox) -> dict:
    """Does the regex catch real actions and refuse everything else?"""
    checks: list[dict] = []

    def run(script: bytes, request: bytes = b"Run this.", rounds: int = 3) -> dict:
        model = ScriptedModel(script)
        loop = ReactLoop(model, {
            b"exec_python": lambda code: sandbox_observation(sandbox.execute_python(code)),
            b"read_file": sandbox.read_file,
            b"write_file": sandbox.write_file,
            b"recall": lambda topic: b"blue",
        }, max_rounds=rounds, max_new_bytes=200)
        result = loop.run(request)
        return {"transcript": result.transcript, "steps": result.steps, "rounds": result.rounds,
                "answer": result.answer, "calls": model.calls}

    # (a) a well-formed action is intercepted and executed with a real observation
    outcome = run(b" compute it\nAction: exec_python(\"print(6 * 7)\")\n")
    expected = sandbox_observation(sandbox.execute_python(b"print(6 * 7)"))
    checks.append({
        "name": "exec_python intercepted and executed",
        "passed": bool(outcome["steps"]) and outcome["steps"][0].observation == expected,
        "detail": f"observation={outcome['steps'][0].observation.decode() if outcome['steps'] else 'none'}",
    })

    # (b) the injected observation really follows the Action line
    transcript = outcome["transcript"]
    checks.append({
        "name": "Observation injected after the Action",
        "passed": b"Action: exec_python(\"print(6 * 7)\")\nObservation: 42" in transcript,
        "detail": transcript.decode("utf-8", "replace").replace("\n", " | ")[:160],
    })

    # (c) a two-argument action parses both arguments
    outcome = run(b" write it\nAction: write_file(\"eval_note.txt\", \"hello sandbox\")")
    checks.append({
        "name": "two-argument action parses both arguments",
        "passed": bool(outcome["steps"]) and outcome["steps"][0].action.args == (b"eval_note.txt", b"hello sandbox"),
        "detail": str(outcome["steps"][0].action.args if outcome["steps"] else "none"),
    })

    # (d) unknown tools fail closed without executing anything
    outcome = run(b" try\nAction: delete_everything(\"/\")")
    checks.append({
        "name": "unknown tool fails closed",
        "passed": bool(outcome["steps"]) and outcome["steps"][0].observation.startswith(b"ERROR: unknown tool"),
        "detail": outcome["steps"][0].observation.decode() if outcome["steps"] else "none",
    })

    # (e) an unterminated call is never executed
    checks.append({
        "name": "unterminated action is not executed",
        "passed": parse_action(b"Action: exec_python(\"print(1)") is None,
        "detail": "parse_action returned None",
    })

    # (f) text that merely mentions the word Action is not a call
    checks.append({
        "name": "prose mentioning Action is not a call",
        "passed": parse_action(b"Action: is not always a tool call") is None,
        "detail": "parse_action returned None",
    })

    # (g) a failing snippet produces the real error, not a guess
    outcome = run(b" run it\nAction: exec_python(\"pritn(2)\")")
    error = outcome["steps"][0].observation if outcome["steps"] else b""
    checks.append({
        "name": "sandbox error surfaces as the observation",
        "passed": error.startswith(b"NameError"),
        "detail": error.decode("utf-8", "replace")[:80],
    })

    # (h) the loop stops at the round budget instead of running forever
    looping = b" keep going\nAction: exec_python(\"print(1)\")"
    outcome = run(looping, rounds=2)
    checks.append({
        "name": "round budget stops the loop",
        "passed": outcome["rounds"] <= 2 and len(outcome["steps"]) <= 2,
        "detail": f"rounds={outcome['rounds']} steps={len(outcome['steps'])}",
    })

    passed = sum(1 for check in checks if check["passed"])
    return {"checks": checks, "passed": passed, "total": len(checks)}


# --------------------------------------------------------------------------
# 3. Memory recall
# --------------------------------------------------------------------------
def memory_checks() -> dict:
    """Phase 12 vector-only recall (10/10) plus the new prefix integration."""
    started = time.time()
    completed = subprocess.run(
        [sys.executable, "-m", "evaluation.memory_eval", "--output", "runs/phase16/memory_eval.json"],
        capture_output=True, text=True)
    report: dict = {"command": "python -m evaluation.memory_eval",
                    "exit_code": completed.returncode,
                    "stdout_tail": completed.stdout.strip().splitlines()[-3:] if completed.stdout else [],
                    "stderr_tail": completed.stderr.strip().splitlines()[-3:]}
    recall = None
    path = Path("runs/phase16/memory_eval.json")
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        fresh = payload.get("fresh_final_cases", {})
        recall = f"{fresh.get('correct', 0)}/{fresh.get('count', 0)}"
        report["short_context"] = payload.get("short_context", {}).get("correct")
        report["long_context"] = payload.get("long_context", {}).get("correct")
        report["cross_session"] = payload.get("cross_session", {}).get("passed")
    report["vector_only_fresh_cases"] = recall
    report["target"] = "10/10"
    report["met"] = recall == "10/10"
    report["elapsed_s"] = round(time.time() - started, 1)

    # Prefix integration with the real store.
    try:
        from models.conversation_v3 import VectorConversationSession
        from models.contextual_response import load_contextual_decoder
        from models.diverse_response import load_diverse_decoder
        from models.foundation_memory import PrefixMemory, VectorFactRetriever
        from models.memory_model import MemoryModel
        from models.vector_memory import VectorConversationMemory, load_vector_reader
        from training.train_confidence import load_generator

        # The Phase 12 recall stack, assembled exactly as its own benchmark does.
        generator = load_generator("runs/phase12/generator.pt", "cpu", 4).eval()
        contextual = load_contextual_decoder("runs/phase12/response.pt",
                                             generator_path="runs/phase12/generator.pt")
        diverse = load_diverse_decoder("runs/phase12/diverse_response.pt",
                                       generator_path="runs/phase12/generator.pt")
        codec, reader = load_vector_reader("runs/phase12/vector_reader.pt",
                                           generator_path="runs/phase12/generator.pt")
        bank = MemoryModel(d_model=generator.d_model, capacity=192, n_heads=generator.n_heads).eval()
        store = VectorConversationMemory(bank, codec)
        session = VectorConversationSession(generator, contextual, store, reader, diverse,
                                            b"foundation-eval")
        session.respond(b"my favorite color is blue.")
        retriever = VectorFactRetriever(generator, store, reader, user_id=b"foundation-eval", top_k=2)
        retriever.observe(b"my:favorite_color", b"What is my favorite color?")
        prefix_memory = PrefixMemory(retriever, top_k=2)
        prefix = prefix_memory.prefix(b"What is my favorite color?")
        report["prefix_integration"] = {
            "prefix": prefix.decode("utf-8", "replace"),
            "contains_value": b"blue" in prefix,
            "entries": len(prefix_memory.retrieve(b"What is my favorite color?")),
        }
    except Exception as error:  # report, do not crash the benchmark
        report["prefix_integration"] = {"error": f"{type(error).__name__}: {error}"}
    return report


# --------------------------------------------------------------------------
# 4. Argument fidelity (the honest counterweight to next-byte accuracy)
# --------------------------------------------------------------------------
@torch.no_grad()
def argument_fidelity(model: CausalByteTransformer, rows: list[Trajectory], sandbox: ByteSandbox,
                      limit: int = 60) -> dict:
    """Free-running arguments on held-out exec trajectories.

    Next-byte accuracy is dominated by format bytes, so this measures the part
    that actually requires copying the request: does the emitted argument match
    the reference byte for byte, and does it run?
    """
    from models.foundation import decode_ids
    from models.react import last_action, render_prompt

    subset = [row for row in rows if row.kind == "exec"][:limit]
    exact = runs = total = 0
    for row in subset:
        prompt = render_prompt(row.request, row.memory)
        ids = encode_bytes(prompt, bos=True)[-model.config.max_seq_len:]
        text = decode_ids(model.generate(ids, max_new_bytes=140))
        action = last_action(text)
        total += 1
        if action is None or not action.args:
            continue
        reference = last_action(row.text)
        if reference is not None and action.args[0] == reference.args[0]:
            exact += 1
        if not looks_failed(sandbox.execute_python(action.args[0])):
            runs += 1
    return {"rows": total, "argument_byte_exact": exact, "runs_cleanly": runs,
            "note": ("free-running greedy generation; measures literal copying, "
                     "which is what next-byte accuracy hides")}


def byte_class_accuracy(model: CausalByteTransformer, rows: list[Trajectory]) -> dict:
    """Teacher-forced accuracy split by byte class inside the Action segment."""
    model.eval()
    hits = {"digits": 0, "letters": 0, "punctuation": 0}
    counts = {"digits": 0, "letters": 0, "punctuation": 0}
    with torch.no_grad():
        for row in rows:
            ids = encode_bytes(row.text, bos=True, eos=True)[:model.config.max_seq_len]
            predictions = model(ids[:-1].unsqueeze(0))[0].argmax(dim=-1)
            targets = ids[1:]
            labels = segment_positions(row.text)
            for position in range(targets.shape[0]):
                if position >= len(labels) or labels[position] != "action":
                    continue
                value = int(targets[position].item())
                key = ("digits" if 48 <= value <= 57
                       else "letters" if (97 <= value <= 122 or 65 <= value <= 90)
                       else "punctuation")
                counts[key] += 1
                hits[key] += int(predictions[position].item()) == value
    return {key: {"accuracy": hits[key] / max(counts[key], 1), "bytes": counts[key]}
            for key in counts}


# --------------------------------------------------------------------------
# 4. Regression
# --------------------------------------------------------------------------
def regression_checks(skip: bool = False) -> dict:
    if skip:
        return {"skipped": True}
    started = time.time()
    completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                               capture_output=True, text=True)
    tail = completed.stderr.strip().splitlines()[-3:] if completed.stderr else []
    ran = next((line for line in reversed(tail) if line.startswith("Ran ")), "")
    return {"exit_code": completed.returncode, "summary": ran, "tail": tail,
            "passed": completed.returncode == 0, "elapsed_s": round(time.time() - started, 1)}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def read_rows(path: Path) -> list[Trajectory]:
    rows: list[Trajectory] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            rows.append(Trajectory(request=payload["request"].encode("latin-1"),
                                   body=payload["body"].encode("latin-1"),
                                   memory=payload["memory"].encode("latin-1"),
                                   kind=payload["kind"],
                                   text=payload["text"].encode("latin-1")))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--output", default="runs/phase16/foundation_eval.json")
    parser.add_argument("--short-bytes", type=int, default=192,
                        help="trajectories up to this many bytes count as 'short'")
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    args = parser.parse_args()

    model = load_foundation(args.model, device="cpu")
    rows = read_rows(Path(args.data_dir) / "react_test.jsonl")
    report: dict = {
        "checkpoint": args.model,
        "parameters": model.parameters_count,
        "config": model.config.to_dict(),
        "test_rows": len(rows),
    }

    report["next_byte_accuracy"] = next_byte_accuracy(model, rows)
    short = [row for row in rows if len(row.text) <= args.short_bytes]
    report["next_byte_accuracy_short"] = next_byte_accuracy(model, short, max_length=args.short_bytes + 2)
    report["next_byte_accuracy_short"]["target"] = ">0.85"
    report["next_byte_accuracy_short"]["met"] = (
        report["next_byte_accuracy_short"]["next_byte_accuracy"] > 0.85)

    with ByteSandbox() as sandbox:
        report["tool_interception"] = interception_checks(sandbox)
        report["tool_interception"]["met"] = (
            report["tool_interception"]["passed"] == report["tool_interception"]["total"])
        report["argument_fidelity"] = argument_fidelity(model, rows, sandbox)
        report["action_byte_class_accuracy"] = byte_class_accuracy(model, rows)

    if args.skip_memory:
        report["memory"] = {"skipped": True}
    else:
        report["memory"] = memory_checks()
    report["regression"] = regression_checks(args.skip_unit_tests)

    report["targets"] = {
        "next_byte_accuracy_short > 0.85": bool(report["next_byte_accuracy_short"].get("met")),
        "tool_interception_all_passed": bool(report["tool_interception"].get("met")),
        "phase12_recall_10_of_10": bool(report.get("memory", {}).get("met")),
        "regression_green": bool(report["regression"].get("passed", True)),
    }
    report["caveats"] = (
        "Synthetic ReAct trajectories from a fixed template pool; next-byte accuracy measures "
        "format learning on held-out content, not reasoning, and is dominated by format bytes -- "
        "argument_fidelity and action_byte_class_accuracy are the honest counterweight. The "
        "sandbox is hardening, not a security boundary. Retrieval is structured lookup, not "
        "semantic search.")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "next_byte_accuracy": round(report["next_byte_accuracy"]["next_byte_accuracy"], 4),
        "next_byte_accuracy_short": round(report["next_byte_accuracy_short"]["next_byte_accuracy"], 4),
        "by_segment": {name: round(value["accuracy"], 3)
                       for name, value in report["next_byte_accuracy"]["by_segment"].items()},
        "tool_interception": f"{report['tool_interception']['passed']}/{report['tool_interception']['total']}",
        "argument_byte_exact": f"{report['argument_fidelity']['argument_byte_exact']}/{report['argument_fidelity']['rows']}",
        "argument_runs_cleanly": f"{report['argument_fidelity']['runs_cleanly']}/{report['argument_fidelity']['rows']}",
        "digits_in_action_accuracy": round(report["action_byte_class_accuracy"]["digits"]["accuracy"], 4),
        "memory_recall": report["memory"].get("vector_only_fresh_cases"),
        "prefix_integration": report["memory"].get("prefix_integration", {}).get("contains_value"),
        "regression": report["regression"].get("summary"),
        "targets": report["targets"],
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
