"""Phase 17 demo: the foundation answering questions about real repositories.

The four tools here are **corpus lookups over data collected from GitHub**, not
a live API client. That is deliberate: the observation a training trajectory
carried was produced by these same functions, so the loop the demo runs is
byte-identical to the loop the model was trained on -- only the request is new.

    python -m demo.github_agent --once "Explain commit 5c06b141f3b0: ..."
    python -m demo.github_agent --once "Review pull request denoland/deno#36798"
    python -m demo.github_agent --once "Summarize the project python/cpython."
    python -m demo.github_agent                       # interactive

Every line the model writes after ``Action:`` is discarded and replaced by the
bytes the tool really returned, so what you read below the action is not the
model's guess about the world -- it is the tool's answer.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch

from data.github_to_react import TOOLS, CorpusIndex, mark_entities
from evaluation.github_eval import build_tools, grounding_facts
from models.foundation import load_foundation
from models.react import ReactLoop, reference_tools, strip_references

DEFAULT_MODEL = "runs/phase18/foundation.pt"
DEFAULT_CORPUS = "data/github_raw/github_stories.jsonl"


@dataclass
class GitHubAgent:
    """Foundation model + the four real corpus tools, driven by the react loop.

    The tools are wrapped per request so that ``[REF]N[/REF]`` arguments resolve
    against what the user actually typed before the tool is called. That is the
    phase's central mechanism: the identifier the tool receives is a span of the
    request, never a span the model invented.
    """

    model: object
    index: CorpusIndex

    def __post_init__(self) -> None:
        self.loop = ReactLoop(self.model, {}, max_rounds=2, max_new_bytes=200)

    def respond(self, request: bytes):
        # The model was trained on requests whose identifiers are already
        # marked, so a raw request has to be marked before it is shown --
        # otherwise the model correctly emits ``[REF]1[/REF]`` into a request
        # that carries no span to resolve it against.
        marked = mark_entities(bytes(request))
        self.loop.tools = reference_tools(build_tools(self.index), marked)
        return self.loop.run(marked)


def render(run, *, show_grounding: bool = True) -> str:
    """Print one run, flagging which facts in the answer came from a tool."""
    # Markers are an internal addressing mechanism; show the bytes they refer to.
    lines = [strip_references(run.transcript).decode("latin-1", "replace")]
    if show_grounding and run.steps:
        facts = [fact for step in run.steps
                 for fact in grounding_facts(step.action.tool, step.observation)
                 if fact and fact not in run.request]
        if facts:
            found = [fact.decode("latin-1", "replace") for fact in facts
                     if fact in run.answer]
            lines.append("[grounded in the observation] "
                         + (", ".join(found) if found else "no observation-only fact "
                                                           "reached the answer"))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--once", default=None, help="run a single request and exit")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="emit one JSON object per request")
    args = parser.parse_args()

    model = load_foundation(args.model, device="cpu")
    index = CorpusIndex.from_jsonl(args.corpus)
    agent = GitHubAgent(model, index)
    agent.loop.max_rounds = args.max_rounds
    print(f"[model {args.model} | {model.parameters_count} parameters | "
          f"{len(index.issues)} issues, {len(index.prs)} pull requests, "
          f"{len(index.commits)} commits, {len(index.readmes)} READMEs]")

    if args.once:
        _handle(agent, args.once.encode("utf-8"), as_json=args.json)
        return

    print("Type a request, or /quit to leave. /tools lists what is available.")
    while True:
        try:
            raw = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        if raw in ("/quit", "/exit"):
            break
        if raw == "/tools":
            print("  " + "\n  ".join(sorted(TOOLS)))
            continue
        _handle(agent, raw.encode("utf-8"), as_json=args.json)


def _handle(agent: GitHubAgent, request: bytes, *, as_json: bool) -> None:
    torch.set_grad_enabled(False)
    run = agent.respond(request)
    if as_json:
        print(json.dumps({
            "request": run.request.decode("latin-1"),
            "tool_calls": [{"tool": step.action.tool.decode("latin-1"),
                            "args": [arg.decode("latin-1") for arg in step.action.args],
                            "observation": step.observation.decode("latin-1")}
                           for step in run.steps],
            "answer": run.answer.decode("latin-1"),
        }, ensure_ascii=False))
        return
    print(render(run))
    print()


if __name__ == "__main__":
    main()
