"""Per-operation autonomous edit outcomes, thought steps, confidence and failure reasons."""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.ingest_real_traces import load_pairs, make_control_trace
from training.train_cognitive import task
from training.train_confidence import load_generator


def classify(done: bool, exact: bool, expected_kinds: set[str], actual_kinds: list[str]) -> str:
    if exact:
        return "success"
    if not done:
        return "step_limit_or_pending_edit"
    if not actual_kinds:
        return "done_without_edit"
    if not expected_kinds.intersection(actual_kinds):
        return "wrong_operation"
    return "wrong_position_payload_or_edit_count"


def summarize(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    subset = [r for r in rows if kind in r["expected_kinds"]]
    if not subset:
        return {"count": 0, "exact_fraction": None, "mean_think_steps": None, "mean_thought_confidence": None, "failures": {}}
    confidences = [r["confidence"] for r in subset if r["confidence"] is not None]
    failures: dict[str, int] = {}
    for item in subset:
        if item["failure"] != "success":
            failures[item["failure"]] = failures.get(item["failure"], 0) + 1
    return {"count": len(subset), "exact_fraction": sum(r["exact"] for r in subset)/len(subset),
            "mean_think_steps": sum(r["think_steps"] for r in subset)/len(subset),
            "mean_thought_confidence": sum(confidences)/len(confidences) if confidences else None,
            "failures": failures}


@torch.no_grad()
def synthetic_rows(g, payload_count: int) -> list[dict[str, Any]]:
    payloads = [bytes(x) for x in itertools.product(b"abcdefgh", repeat=3)]
    random.Random(23).shuffle(payloads)
    payloads = payloads[48:48+payload_count]
    if not payloads:
        raise ValueError("payload count must be positive")
    rows = []
    for payload in payloads:
        for cmd in b"IDRM":
            source = bytes([cmd]) + b":" + payload
            action, expected = task(g, source)
            result = g.cognitive_generate(source, max_steps=8)
            actual = [e.action.kind for e in result.events if e.action.token == g.EDIT_OP and e.action.kind]
            confidence = [e.confidence for e in result.events if e.action.token == g.THINK and e.confidence is not None]
            exact = result.done and result.canvas == expected
            rows.append({"id": source.hex(), "expected_kinds": [action.kind], "actual_kinds": actual,
                         "exact": exact, "done": result.done,
                         "think_steps": sum(e.thought_steps for e in result.events),
                         "confidence": sum(confidence)/len(confidence) if confidence else None,
                         "failure": classify(result.done, exact, {action.kind}, actual)})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generator", default="runs/dual_memory/generator.pt")
    p.add_argument("--real-pairs", default="data/real_edit_pairs.jsonl")
    p.add_argument("--real-metrics", default="runs/real_traces/metrics.json")
    p.add_argument("--output", default="runs/edit_operation_analysis.json")
    p.add_argument("--payloads", type=int, default=64)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    g = load_generator(args.generator, args.device, 4)
    synth = synthetic_rows(g, args.payloads)
    path = Path(args.real_metrics)
    if not path.exists():
        raise FileNotFoundError("run evaluation/test_real_traces.py before real edit analysis")
    real_report = json.loads(path.read_text())
    records = {r["id"]: r for r in load_pairs(args.real_pairs, split="test")}
    real = []
    for item in real_report["trained_test"]["examples"]:
        record = records[item["id"]]
        _, actions = make_control_trace(record["input"], record["output"])
        expected = {a.kind for a in actions if a.kind is not None}
        actual = [a["kind"] for a in item["actions"] if a.get("kind")]
        real.append({"id": item["id"], "expected_kinds": sorted(expected), "actual_kinds": actual,
                     "exact": item["exact"], "done": item["done"], "think_steps": item["think_steps"],
                     "confidence": item["thought_confidence"],
                     "failure": classify(item["done"], item["exact"], expected, actual)})
    kinds = ("insert", "delete", "replace", "move")
    report = {"protocol": {"synthetic": "held-out three-byte I/D/R/M tasks with original cognitive checkpoint",
                           "real": "disjoint-file repository excerpts; autonomous generator after trace co-training",
                           "confidence": "thought confidence is uncalibrated and must not be used as a quality guarantee",
                           "overlap": "Real examples with multiple optimal edit kinds contribute to each applicable row; MOVE is not a Levenshtein primitive."},
              "synthetic": {"count": len(synth), "by_operation": {k: summarize(synth, k) for k in kinds}, "examples": synth},
              "real_test": {"count": len(real), "by_operation": {k: summarize(real, k) for k in kinds}, "examples": real}}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"synthetic": report["synthetic"]["by_operation"], "real_test": report["real_test"]["by_operation"]}, indent=2))


if __name__ == "__main__":
    main()
