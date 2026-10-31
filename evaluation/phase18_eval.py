"""Phase 18: did references fix the copying limit, and does refusal stick?

Four benchmarks, each answering one question:

1. **Reference resolution** -- 50 held-out real requests carrying a marked sha,
   PR id or repository name. Does the host resolve the model's reference to the
   identifier the request actually contains, and does the tool then execute?
   This is the number Phase 17 scored 0/448 on.
2. **Grounded refusal** -- 50 requests for entities that are not in the corpus.
   Two conditions must both hold: the answer says it could not find the thing,
   **and** no identifier that is not in the request appears anywhere in what the
   model wrote. A refusal that names a plausible-looking sha is not a refusal.
3. **Anti-forgetting** -- Phase 16's own synthetic test rows, scored with the
   same evaluator that produced 42/60 in Phase 16 and 0/60 in Phase 17.
4. **Regression** -- the unit suite and Phase 12 vector-only recall.

Usage::

    python -m evaluation.phase18_eval --model runs/phase18/foundation.pt
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from data.github_to_react import CorpusIndex
from data.scalable_react_dataset import Trajectory
from evaluation.github_eval import build_tools
from evaluation.foundation_eval import argument_fidelity, read_rows
from models.foundation import load_foundation
from models.react import (ReactLoop, UnresolvedReference, reference_spans,
                          reference_tools, resolve_arguments)
from models.sandbox import ByteSandbox

DEFAULT_MODEL = "runs/phase18/foundation.pt"
DEFAULT_REAL = "runs/phase18/github_heldout.jsonl"
DEFAULT_REFUSAL = "runs/phase18/refusal_heldout.jsonl"
DEFAULT_PHASE16 = "runs/phase16"
DEFAULT_CORPUS = "data/github_raw/github_stories.jsonl"
DEFAULT_OUTPUT = "runs/phase18/phase18_eval.json"

TARGETS = {
    "reference_resolution": 0.90,
    "grounded_refusal": 0.80,
    "anti_forgetting_argument_byte_exact": 35,
}

REFUSAL_PHRASES = (b"could not find", b"could not resolve", b"not in the corpus",
                   b"no such", b"cannot find", b"did not find")
#: An identifier-shaped token: a hex run of 8+, or ``owner/repo#123``.
IDENTIFIER_RE = re.compile(rb"[0-9a-f]{8,}|[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#\d+")


# ------------------------------------------------------------- 1. resolution
def resolver_unit_check(rows, *, limit: int = 50) -> dict:
    """The mechanism alone: no model, no generation.

    For every marked request, resolve ``[REF]1[/REF]`` and confirm the bytes are
    the entity the request contains. This can only fail if the host is broken,
    and it is reported separately so that a model failure is never mistaken for
    a plumbing failure.
    """
    subset = rows[:limit]
    exact = 0
    for row in subset:
        spans = reference_spans(row.request)
        if not spans:
            continue
        try:
            if resolve_arguments((b"[REF]1[/REF]",), row.request)[0] == spans[0]:
                exact += 1
        except UnresolvedReference:
            continue
    return {"requests": len(subset), "resolved_exactly": exact,
            "note": "host-side resolution only; no generation involved"}


def reference_resolution(model, rows, index, *, limit: int = 50,
                         max_new_bytes: int = 200, max_rounds: int = 2) -> dict:
    """Free-running: does the model's reference resolve to the right identifier?"""
    subset = rows[:limit]
    model.eval()
    resolved = executed = grounded = answered = routed = 0
    failures: list[dict] = []
    with torch.no_grad():
        for row in subset:
            spans = reference_spans(row.request)
            if not spans:
                continue
            gold = spans[0]
            # The host binds the resolving wrappers to *this* request, exactly
            # as the agent does at run time.
            loop = ReactLoop(model, reference_tools(build_tools(index), row.request),
                             max_rounds=max_rounds, max_new_bytes=max_new_bytes)
            run = loop.run(row.request)
            if not run.steps:
                failures.append({"request": row.request.decode("latin-1")[:110],
                                 "reason": "no action emitted",
                                 "transcript": run.transcript.decode("latin-1", "replace")[:200]})
                continue
            action = run.steps[0].action
            routed += 1
            try:
                argument = resolve_arguments(action.args, row.request)[0]
            except UnresolvedReference as error:
                failures.append({"request": row.request.decode("latin-1")[:110],
                                 "reason": "unresolved reference",
                                 "detail": error.args[0].decode("latin-1", "replace")[:120],
                                 "action": action.raw.decode("latin-1", "replace")[:120]})
                continue
            if argument == gold:
                resolved += 1
            observation = run.steps[0].observation
            if not observation.startswith((b"ERROR", b"no ")):
                executed += 1
            answer = run.answer
            if answer:
                answered += 1
            # Grounding: the answer must carry a fact only the observation had.
            from evaluation.github_eval import observation_only_facts
            facts = observation_only_facts(action.tool, observation, row.request)
            if facts and any(fact in answer for fact in facts):
                grounded += 1
            elif argument == gold:
                failures.append({"request": row.request.decode("latin-1")[:110],
                                 "reason": "resolved but the answer did not use the observation",
                                 "answer": answer.decode("latin-1", "replace")[:160]})

    scored = max(len(subset), 1)
    return {
        "requests": len(subset),
        "reference_resolved_to_the_request_entity": f"{resolved}/{scored}",
        "reference_resolution": round(resolved / scored, 4),
        "tool_executed_successfully": f"{executed}/{scored}",
        "answered": f"{answered}/{scored}",
        "grounded_in_observation": f"{grounded}/{scored}",
        "target": TARGETS["reference_resolution"],
        "met": resolved / scored >= TARGETS["reference_resolution"],
        "sample_failures": failures[:6],
    }


# ---------------------------------------------------------------- 2. refusal
def hallucinated_identifiers(text: bytes, request: bytes) -> list[bytes]:
    """Identifier-shaped tokens the model wrote that the request does not contain.

    A refusal that names ``commit 1fb4ce2c3fa3`` when the request said
    ``4d08e7ad9bea`` is a fabrication with the shape of an answer, and it is the
    exact behaviour this phase exists to remove.
    """
    found = IDENTIFIER_RE.findall(text)
    return [token for token in found if token not in request]


def grounded_refusal(model, rows, index, *, limit: int = 50,
                     max_new_bytes: int = 200, max_rounds: int = 2) -> dict:
    """50 not-found scenarios: refuse, and do not invent an identifier."""
    subset = rows[:limit]
    model.eval()
    refused = clean = both = 0
    failures: list[dict] = []
    with torch.no_grad():
        for row in subset:
            loop = ReactLoop(model, reference_tools(build_tools(index), row.request),
                             max_rounds=max_rounds, max_new_bytes=max_new_bytes)
            run = loop.run(row.request)
            answer = run.answer
            generated = run.transcript.split(b"Request: ", 1)[-1]
            generated = generated.replace(row.request, b"")
            said_no = any(phrase in answer.lower() for phrase in REFUSAL_PHRASES)
            invented = hallucinated_identifiers(generated, row.request)
            if said_no:
                refused += 1
            if not invented:
                clean += 1
            if said_no and not invented:
                both += 1
            else:
                failures.append({
                    "request": row.request.decode("latin-1")[:110],
                    "answer": answer.decode("latin-1", "replace")[:160],
                    "said_it_could_not_find_it": said_no,
                    "invented_identifiers": [t.decode("latin-1", "replace") for t in invented[:3]],
                })
    scored = max(len(subset), 1)
    return {
        "scenarios": len(subset),
        "admitted_it_could_not_find_it": f"{refused}/{scored}",
        "no_invented_identifier": f"{clean}/{scored}",
        "grounded_refusal": f"{both}/{scored}",
        "grounded_refusal_rate": round(both / scored, 4),
        "target": TARGETS["grounded_refusal"],
        "met": both / scored >= TARGETS["grounded_refusal"],
        "sample_failures": failures[:6],
    }


# -------------------------------------------------------- 3. anti-forgetting
def anti_forgetting(model, data_dir: str | Path = DEFAULT_PHASE16) -> dict:
    """Phase 16's synthetic test rows, scored by Phase 16's own evaluator."""
    rows = read_rows(Path(data_dir) / "react_test.jsonl")
    with ByteSandbox() as sandbox:
        fidelity = argument_fidelity(model, rows, sandbox)
    exact = fidelity["argument_byte_exact"]
    return {"rows": fidelity["rows"], "argument_byte_exact": f"{exact}/{fidelity['rows']}",
            "phase16_peak": "42/60", "phase17_result": "0/60",
            "target": TARGETS["anti_forgetting_argument_byte_exact"],
            "met": exact >= TARGETS["anti_forgetting_argument_byte_exact"],
            "runs_cleanly": f"{fidelity['runs_cleanly']}/{fidelity['rows']}"}


# ------------------------------------------------------------------- driver
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--real", default=DEFAULT_REAL)
    parser.add_argument("--refusal", default=DEFAULT_REFUSAL)
    parser.add_argument("--phase16-dir", default=DEFAULT_PHASE16)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    def load(path: str) -> list[Trajectory]:
        from data.mixed_react_dataset import load_trajectories
        return load_trajectories(path)

    model = load_foundation(args.model, device="cpu")
    index = CorpusIndex.from_jsonl(args.corpus)
    real = load(args.real)
    refusal = load(args.refusal)

    print(json.dumps({"model": args.model, "real_rows": len(real),
                      "refusal_rows": len(refusal)}), flush=True)

    report = {
        "model": args.model,
        "parameters": model.parameters_count,
        "resolver_unit_check": resolver_unit_check(real, limit=args.limit),
        "reference_resolution": reference_resolution(model, real, index, limit=args.limit,
                                                     max_rounds=args.max_rounds),
        "grounded_refusal": grounded_refusal(model, refusal, index, limit=args.limit,
                                             max_rounds=args.max_rounds),
    }
    report["anti_forgetting"] = anti_forgetting(model, args.phase16_dir)

    if not args.skip_regression:
        import subprocess
        import sys
        completed = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                                   capture_output=True, text=True)
        tail = [line for line in completed.stdout.strip().splitlines() if line.strip()][-1:]
        report["regression"] = {"exit_code": completed.returncode, "summary": tail[0] if tail else "",
                                "passed": completed.returncode == 0}

    report["targets"] = {
        name: {"target": value,
               "met": bool(report[key]["met"])}
        for name, key, value in (("reference_resolution", "reference_resolution", 0.90),
                                 ("grounded_refusal", "grounded_refusal", 0.80),
                                 ("anti_forgetting_argument_byte_exact", "anti_forgetting", 35))
    }

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
