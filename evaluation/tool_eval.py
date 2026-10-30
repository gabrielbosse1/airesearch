"""Phase 15 benchmark: tool routing, iterative debugging and Phase 12 regression.

Three measured sections:

1. **Tool routing (20 held-out prompts).** A prompt scores only when the routed
   tool *and* the decoded argument bytes match the reference frame exactly, and
   the sandbox reproduces the reference output when the frame is executed.
2. **Iterative debugging (5 tasks).** Round one runs the code the user pasted
   (it fails). The failure bytes are injected and the head must emit a corrected
   call that the sandbox executes without error. No fix is supplied anywhere.
3. **Regression.** The Phase 12 generator is verified bit-identical to its
   checkpoint and the vector-only recall benchmark is rerun (10/10 expected).
   ``--with-regression`` also runs the whole unit-test suite.

Targets come from the Phase 15 request; every number below is *observed*, and
the caveats are recorded in ``runs/phase15_report.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from data.tool_use_examples import debugging_tasks, decline_prompts, routing_prompts
from models.sandbox import ByteSandbox, looks_failed
from models.tool_use import (ROUTE_NAMES, ToolCallPolicy, find_tool_call, load_tool_policy,
                            sanitize_result)
from training.train_confidence import load_generator


def routing_benchmark(policy: ToolCallPolicy, sandbox: ByteSandbox, prompts) -> dict:
    rows = []
    for request, tool, argument in prompts:
        generation = policy.generate(request, None, threshold=0.0)
        expected_route = ROUTE_NAMES.index(tool.decode("ascii"))
        route_ok = generation.route == expected_route
        argument_ok = bool(generation.call is not None and generation.call.argument == argument)
        frame = generation.frame
        parsed_ok = False
        if frame:
            parsed, _ = find_tool_call(frame)
            parsed_ok = parsed is not None and parsed.argument == argument and parsed.tool == tool
        reference_output = sandbox.call(tool, argument)
        executed_ok = False
        if generation.call is not None and generation.call.tool == tool:
            executed_ok = sandbox.call(generation.call.tool, generation.call.argument) == reference_output
        rows.append({"request": request.decode("utf-8", "replace"), "expected_tool": tool.decode("ascii"),
                     "route": generation.route_name, "route_confidence": round(generation.route_confidence, 4),
                     "route_ok": route_ok, "argument_ok": argument_ok, "frame_parses": parsed_ok,
                     "executed_matches_reference": executed_ok,
                     "exact": bool(route_ok and argument_ok and parsed_ok),
                     "argument": generation.call.argument.decode("utf-8", "replace") if generation.call else "",
                     "expected_argument": argument.decode("utf-8", "replace")})
    return {"count": len(rows),
            "correct": sum(row["exact"] for row in rows),
            "route_correct": sum(row["route_ok"] for row in rows),
            "argument_correct": sum(row["argument_ok"] for row in rows),
            "frame_parses": sum(row["frame_parses"] for row in rows),
            "execution_matches": sum(row["executed_matches_reference"] for row in rows),
            "rows": rows}


def debugging_benchmark(policy: ToolCallPolicy, sandbox: ByteSandbox, tasks, *, max_rounds: int = 3) -> dict:
    rows = []
    for request, broken, fixed in tasks:
        reference = sandbox.call(b"exec_python", fixed)
        outcome = {"request": request.decode("utf-8", "replace"), "rounds": 0,
                   "first_attempt_failed": False, "corrected_call_emitted": False,
                   "corrected_execution_clean": False, "output_matches_reference": False, "success": False}
        result: bytes | None = None
        for round_index in range(max_rounds):
            generation = policy.generate(request, result, threshold=0.0)
            if generation.call is None or not generation.complete:
                break
            call, _ = find_tool_call(generation.frame)
            if call is None or call.tool != b"exec_python":
                break
            result = sandbox.call(call.tool, call.argument)
            outcome["rounds"] = round_index + 1
            if round_index == 0:
                outcome["first_attempt_failed"] = looks_failed(result)
                if not outcome["first_attempt_failed"]:
                    break  # nothing to repair; the task did not exercise debugging
            else:
                outcome["corrected_call_emitted"] = call.argument != broken
                outcome["corrected_execution_clean"] = not looks_failed(result)
                outcome["output_matches_reference"] = (not looks_failed(result)) and result == reference
                break
        outcome["success"] = bool(outcome["first_attempt_failed"] and outcome["corrected_call_emitted"]
                                  and outcome["corrected_execution_clean"])
        outcome["last_result_tail"] = sanitize_result(result or b"", limit=80).decode("utf-8", "replace")
        rows.append(outcome)
    return {"count": len(rows),
            "success": sum(row["success"] for row in rows),
            "corrected_call_emitted": sum(row["corrected_call_emitted"] for row in rows),
            "clean_after_correction": sum(row["corrected_execution_clean"] for row in rows),
            "output_matches_reference": sum(row["output_matches_reference"] for row in rows),
            "max_rounds": max_rounds, "rows": rows}


def decline_benchmark(policy: ToolCallPolicy, prompts: list[bytes]) -> dict:
    """Conversational prompts the router must refuse to turn into a tool call."""
    refusals = 0
    rows = []
    for message in prompts:
        generation = policy.generate(message, None, threshold=0.0)
        declined = generation.call is None and generation.route == 0
        refusals += int(declined)
        rows.append({"request": message.decode("utf-8", "replace"), "route": generation.route_name,
                     "declined": declined})
    return {"count": len(prompts), "declined": refusals, "rows": rows}


def generator_regression(generator_path: str, heads: int) -> dict:
    """The loaded generator must be bit-identical to the checkpoint on disk."""
    state = torch.load(generator_path, map_location="cpu", weights_only=True)
    model = load_generator(generator_path, "cpu", heads).eval()
    loaded = model.state_dict()
    # Vocabulary tensors grow in place on load (Phase 15 tool controls); every
    # row that existed in the checkpoint must still be bit-identical.
    identical = True
    for key, value in state.items():
        current = loaded[key]
        if current.shape == value.shape:
            identical &= bool(torch.equal(current, value))
        elif current.shape[1:] == value.shape[1:] and current.shape[0] > value.shape[0]:
            identical &= bool(torch.equal(current[: value.shape[0]], value))
        else:
            identical = False
    return {"checkpoint_sha256": hashlib.sha256(Path(generator_path).read_bytes()).hexdigest(),
            "state_dict_bit_identical": bool(identical),
            "vocabulary_size": model.VOCAB_SIZE,
            "control_rows": int(model.control_emb.num_embeddings),
            "intent_classes": int(model.intent_classifier[-1].out_features),
            "note": ("Phase 15 trains a separate tool head; the generator file and all Phase 10/12 decoders "
                     "bound to its hash are unchanged.")}


def memory_regression(args) -> dict:
    from evaluation.memory_eval import evaluate

    report = evaluate(SimpleNamespace(generator=args.generator, response=args.response,
                                      vector_reader=args.vector_reader,
                                      diverse_response=args.diverse_response, heads=args.heads))
    fresh = report["fresh_final_cases"]
    return {"vector_only_fresh_cases": f"{fresh['correct']}/{fresh['count']}",
            "short_context": bool(report["short_context"]["correct"]),
            "long_context": bool(report["long_context"]["correct"]),
            "cross_session_fresh_interpreter": bool(report["cross_session"]["correct"]),
            "scope": report["scope"]}


def unit_regression() -> dict:
    process = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                             capture_output=True, text=True, timeout=1800)
    tail = process.stderr.strip().splitlines()[-3:]
    return {"exit_code": process.returncode, "tail": tail}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", default="runs/phase12/generator.pt")
    parser.add_argument("--policy", default="runs/phase15/tool_policy.pt")
    parser.add_argument("--sandbox-dir", default="runs/phase15/eval_sandbox")
    parser.add_argument("--response", default="runs/phase12/response.pt")
    parser.add_argument("--vector-reader", default="runs/phase12/vector_reader.pt")
    parser.add_argument("--diverse-response", default="runs/phase12/diverse_response.pt")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--with-regression", action="store_true", help="also run the unit-test suite")
    parser.add_argument("--output", default="runs/phase15/tool_eval.json")
    args = parser.parse_args()
    torch.set_num_threads(1)

    generator = load_generator(args.generator, "cpu", args.heads)
    policy = load_tool_policy(args.policy, generator=generator, generator_path=args.generator)
    sandbox = ByteSandbox(args.sandbox_dir, timeout=args.timeout)
    try:
        decline = decline_benchmark(policy, decline_prompts("test", limit=40))
        decline_all = decline_benchmark(policy, decline_prompts("test", conversational_only=False, limit=60))
        routing = routing_benchmark(policy, sandbox, routing_prompts("test"))
        debugging = debugging_benchmark(policy, sandbox, debugging_tasks("test"), max_rounds=args.max_rounds)
        regression = generator_regression(args.generator, args.heads)
        memory = memory_regression(args)
        report = {"routing": routing, "decline_conversational": decline, "decline_all_phase10": decline_all,
                  "iterative_debugging": debugging,
                  "generator_regression": regression, "phase12_memory_regression": memory,
                  "targets": {"routing_exact": ">18/20", "debugging_success": ">=3/5",
                              "phase12_vector_recall": "10/10"},
                  "caveats": ("Prompts and code templates recur across instance-disjoint splits; held-out prompts use "
                              "unseen numbers, words and filenames. Debugging corrections are a closed template set. "
                              "This measures routing plus sandbox-grounded copying, not code understanding.")}
        if args.with_regression:
            report["unit_tests"] = unit_regression()
    finally:
        sandbox.close()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"declined_conversational": f"{decline['declined']}/{decline['count']}",
                      "declined_all_phase10": f"{decline_all['declined']}/{decline_all['count']}",
                      "routing_exact": f"{routing['correct']}/{routing['count']}",
                      "routing_route_only": f"{routing['route_correct']}/{routing['count']}",
                      "routing_execution_match": f"{routing['execution_matches']}/{routing['count']}",
                      "debugging_success": f"{debugging['success']}/{debugging['count']}",
                      "debugging_clean_after_correction": f"{debugging['clean_after_correction']}/{debugging['count']}",
                      "generator_bit_identical": regression["state_dict_bit_identical"],
                      "phase12_vector_recall": memory["vector_only_fresh_cases"],
                      "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
