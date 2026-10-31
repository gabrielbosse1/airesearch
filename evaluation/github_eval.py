"""Phase 17: does the model use real tools on requests it has never seen?

Three questions, in increasing order of difficulty:

1. **Routing** -- given a held-out request, does it call the right tool?
2. **Argument fidelity** -- is the argument byte-identical to the one the
   reference trajectory used? A wrong key returns a wrong observation, so this
   is not cosmetic.
3. **Grounding** -- does the answer repeat a *fact that only the observation
   contains*? This is the hallucination test. Every fact below (a commit's
   author and date, a PR's state and comment count, the repository a search hit
   came from, a README's character count) is absent from the request, so an
   answer containing one cannot have been produced from the request alone.

The evaluation drives :class:`models.react.ReactLoop` with the four *real*
corpus tools from :mod:`data.github_to_react`, so the observations the model
sees at evaluation time are computed by executing code -- not replayed from the
stored trajectory. Step 3 additionally asserts that the bytes the loop received
match the bytes stored in the dataset, which is what makes "the tool really
ran" a measurement rather than a claim.

Usage::

    python -m evaluation.github_eval --model runs/phase17/foundation.pt
    python -m evaluation.github_eval --baseline runs/phase16/foundation.pt
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from data.github_to_react import (TOOLS, CorpusIndex, tool_explain_commit,
                                  tool_review_pr, tool_search_issues,
                                  tool_summarize_readme)
from data.mixed_react_dataset import GITHUB_KINDS, load_trajectories
from models.foundation import load_foundation
from models.react import ReactLoop, extract_answer, last_action

DEFAULT_MODEL = "runs/phase17/foundation.pt"
DEFAULT_HELD_OUT = "runs/phase17/github_heldout.jsonl"
DEFAULT_CORPUS = "data/github_raw/github_stories.jsonl"
DEFAULT_OUTPUT = "runs/phase17/github_eval.json"

# ``TOOLS`` is keyed by str; the stream carries bytes.
TOOL_NAMES = {name.encode("utf-8") for name in TOOLS}


# ------------------------------------------------------------------ grounding
_AUTHOR_DATE = re.compile(rb" by ([^:]{2,60}?) on (\d{4}-\d{2}-\d{2}):")
_PR_STATE = re.compile(rb" is (\w+) with (\d+) comments")
_SEARCH_HIT = re.compile(rb"#(\d+) \[([^\]]+)\]")
_README_STATS = re.compile(rb"(\d+) headings, (\d+) code blocks, (\d+) characters")

# A tool observation always begins with one of these. Seeing one *inside* a
# ``Thought:`` line means the model wrote it itself: the loop only ever appends
# observations on the ``Observation:`` channel.
_FABRICATED = re.compile(rb"^Thought: (commit [0-9a-f]{6,}|PR \S+ is|found \d+ matching)")


def grounding_facts(tool: bytes, observation: bytes) -> list[bytes]:
    """Facts carried by an observation that the request does not contain.

    Each pattern below matches a field the tool looked *up* -- the author of a
    commit, the state of a pull request, the repository a search hit came from,
    the size of a README. None of them appear in the request that triggered the
    call, so an answer containing one was assembled from what came back.
    """
    if tool == b"explain_commit":
        match = _AUTHOR_DATE.search(observation)
        return [fact for fact in (match.group(1), match.group(2))] if match else []
    if tool == b"review_pr":
        match = _PR_STATE.search(observation)
        return [match.group(0)] if match else []
    if tool == b"search_issues":
        return [b"[" + hit[1] + b"]" for hit in _SEARCH_HIT.findall(observation)]
    if tool == b"summarize_readme":
        match = _README_STATS.search(observation)
        return [match.group(3)] if match else []
    return []


def observation_only_facts(tool: bytes, observation: bytes, request: bytes) -> list[bytes]:
    """Grounding facts minus any that leak into the request anyway."""
    return [fact for fact in grounding_facts(tool, observation)
            if fact and fact not in request]


# ------------------------------------------------------------------ evaluation
def build_tools(index: "CorpusIndex") -> dict[bytes, object]:
    """Bind the four real corpus tools to one index (the loop takes bytes keys)."""
    return {
        b"search_issues": lambda query: tool_search_issues(index, query.decode("utf-8", "replace")),
        b"review_pr": lambda identifier: tool_review_pr(index, identifier.decode("utf-8", "replace")),
        b"explain_commit": lambda sha: tool_explain_commit(index, sha.decode("utf-8", "replace")),
        b"summarize_readme": lambda repo: tool_summarize_readme(index, repo.decode("utf-8", "replace")),
    }


def held_out_evaluation(model, rows, index, *, limit: int | None = None,
                        max_new_bytes: int = 200, max_rounds: int = 2) -> dict:
    """Drive the react loop over held-out requests and score what happens.

    ``max_rounds=2`` is the minimum that can measure grounding: round one emits
    the action and the host appends the observation, round two is the first
    chance the model has to write an answer that uses it. With one round the
    transcript always ends at the observation and "did the answer quote the
    tool's output" would be a measurement of nothing.
    """
    model.eval()
    loop = ReactLoop(model, build_tools(index), max_rounds=max_rounds,
                     max_new_bytes=max_new_bytes)
    subset = rows[:limit] if limit else rows

    counts = {"rows": len(subset), "routed": 0, "tool_correct": 0, "argument_exact": 0,
              "tool_executed": 0, "observation_matches_dataset": 0, "answered": 0,
              "grounded": 0, "answer_exact": 0, "valid_lookup_key": 0,
              "tool_reported_not_found": 0, "answered_despite_not_found": 0,
              "fabricated_observation_in_thought": 0}
    per_kind: dict[str, dict[str, int]] = {}
    failures: list[dict] = []

    with torch.no_grad():
        for row in subset:
            kind = row.kind
            bucket = per_kind.setdefault(kind, {"rows": 0, "routed": 0, "tool_correct": 0,
                                                 "argument_exact": 0, "grounded": 0,
                                                 "observation_matches_dataset": 0})
            bucket["rows"] += 1
            reference_action = last_action(row.text)
            reference_answer = extract_answer(row.text)
            run = loop.run(row.request)
            if not run.steps:
                # An unterminated or malformed call is refused by the loop, so
                # "nothing ran" and "nothing parsable was written" are different
                # failures and are reported as such.
                failures.append({"request": row.request.decode("latin-1")[:120],
                                 "reason": ("action emitted but not parseable"
                                            if b"Action:" in run.transcript
                                            else "no action emitted"),
                                 "transcript": run.transcript.decode("latin-1", "replace")[:240]})
                continue
            counts["routed"] += 1
            bucket["routed"] += 1
            step = run.steps[0]
            action = step.action

            if reference_action is not None and action.tool == reference_action.tool:
                counts["tool_correct"] += 1
                bucket["tool_correct"] += 1
            if reference_action is not None and action.args == reference_action.args:
                counts["argument_exact"] += 1
                bucket["argument_exact"] += 1
            if action.tool in TOOL_NAMES:
                counts["tool_executed"] += 1
            not_found = step.observation.startswith(b"no ")
            if not_found:
                counts["tool_reported_not_found"] += 1
            # A key the corpus does not contain is not a formatting slip: it is
            # an identifier the model invented. "no commit X in the corpus" is
            # what the tool actually says, so it can be counted directly.
            if not step.observation.startswith(b"no "):
                counts["valid_lookup_key"] += 1
            # The loop's observation must equal the one stored in the dataset.
            # Both come from executing the same tool, so a mismatch means the
            # evaluation is not measuring what it claims to.
            reference_observation = row.text.split(b"Observation: ", 1)[-1].split(b"\n", 1)[0]
            if step.observation == reference_observation.strip():
                counts["observation_matches_dataset"] += 1
                bucket["observation_matches_dataset"] += 1

            answer = run.answer
            if answer:
                counts["answered"] += 1
                if not_found:
                    # The tool said the key does not exist; answering anyway is
                    # the failure this architecture exists to prevent.
                    counts["answered_despite_not_found"] += 1
            # Anything that looks like a tool result but sits on a Thought line
            # was written by the model, not by a tool.
            if any(_FABRICATED.match(line.strip())
                   for line in run.transcript.split(b"\n")):
                counts["fabricated_observation_in_thought"] += 1
                if answer.strip() == reference_answer.strip():
                    counts["answer_exact"] += 1
            facts = observation_only_facts(action.tool, step.observation, row.request)
            if facts and any(fact in answer for fact in facts):
                counts["grounded"] += 1
                bucket["grounded"] += 1
            elif facts:
                failures.append({"request": row.request.decode("latin-1")[:120],
                                 "reason": "answer is not grounded in the observation",
                                 "answer": answer.decode("latin-1", "replace")[:160],
                                 "expected_fact": facts[0].decode("latin-1", "replace")})

    scored = max(counts["rows"], 1)
    # Only rows where a corpus tool actually ran can say anything about whether
    # the key existed; counting the rest would reward calling the wrong tool.
    # A model that never reaches these tools has no rate at all -- reporting
    # "277/1" would be worse than reporting nothing.
    lookups = counts["tool_executed"]
    report = {
        "rows": counts["rows"],
        "tool_routing": f"{counts['tool_correct']}/{scored}",
        "tool_routing_accuracy": round(counts["tool_correct"] / scored, 4),
        "argument_byte_exact": f"{counts['argument_exact']}/{scored}",
        "argument_byte_exact_accuracy": round(counts["argument_exact"] / scored, 4),
        "tools_executed": counts["tool_executed"],
        "valid_lookup_key": (f"{counts['valid_lookup_key']}/{counts['tool_executed']}"
                             if counts["tool_executed"] else "n/a"),
        "valid_lookup_key_rate": (round(counts["valid_lookup_key"] / lookups, 4)
                                  if lookups else None),
        "observation_reproduced_from_the_tool": f"{counts['observation_matches_dataset']}/{scored}",
        "answered": f"{counts['answered']}/{scored}",
        "grounded_in_observation": f"{counts['grounded']}/{scored}",
        "grounded_accuracy": round(counts["grounded"] / scored, 4),
        "answer_exact_match": f"{counts['answer_exact']}/{scored}",
        "tool_reported_not_found": counts["tool_reported_not_found"],
        "answered_despite_not_found": (str(counts["answered_despite_not_found"]) + "/"
                                       + str(counts["tool_reported_not_found"])),
        "fabricated_observation_in_thought": (str(counts["fabricated_observation_in_thought"])
                                              + "/" + str(scored)),
        "per_kind": {kind: {key: value for key, value in bucket.items()}
                     for kind, bucket in sorted(per_kind.items())},
        "note": ("facts used for grounding are absent from the request by "
                 "construction, so a grounded answer cannot be a copy of it"),
    }
    if failures:
        report["sample_failures"] = failures[:8]
    return report


def evaluate_model(path: str, rows, index, *, limit: int | None = None,
                   max_rounds: int = 2) -> dict:
    model = load_foundation(path, device="cpu")
    return {"model": path, "parameters": model.parameters_count,
            "max_seq_len": model.config.max_seq_len,
            "held_out": held_out_evaluation(model, rows, index, limit=limit,
                                            max_rounds=max_rounds)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--baseline", default=None,
                        help="a second checkpoint scored on the same held-out rows")
    parser.add_argument("--held-out", default=DEFAULT_HELD_OUT)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = [row for row in load_trajectories(args.held_out) if row.kind in GITHUB_KINDS]
    index = CorpusIndex.from_jsonl(args.corpus)
    print(json.dumps({"held_out_rows": len(rows), "corpus": str(args.corpus)}), flush=True)

    report = {"max_rounds": args.max_rounds,
              "evaluation": evaluate_model(args.model, rows, index, limit=args.limit,
                                           max_rounds=args.max_rounds)}
    if args.baseline:
        report["baseline"] = evaluate_model(args.baseline, rows, index, limit=args.limit,
                                            max_rounds=args.max_rounds)
        current = report["evaluation"]["held_out"]
        previous = report["baseline"]["held_out"]
        report["delta"] = {
            metric: (round(current[metric] - previous[metric], 4)
                     if current.get(metric) is not None and previous.get(metric) is not None
                     else None)
            for metric in ("tool_routing_accuracy", "argument_byte_exact_accuracy",
                           "grounded_accuracy", "valid_lookup_key_rate")
        }

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
