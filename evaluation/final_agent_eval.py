"""Phase 22 de-contaminated final agent evaluation.

Four held-out probes, none of which may appear in training data:

  1. Minecraft generalisation: the NOVEL goal "Build a cobblestone generator".
  2. Time-sensitive / obscure question that MUST be answered via search.
  3. A subtle logical trap fixable with EXACTLY 1-2 efficient ``[EDIT]`` tokens.
  4. Memory injection: a fact, a distractor, then a query -- the fact must win.

The harness is deliberately honest about two things:

  * ``--self-test`` uses a scripted ORACLE agent to prove the SCORER can pass.
    It is labelled ``oracle_mode=True`` and is NOT a model result.
  * Decontamination scans available training shards AND the repository for the
    exact probe strings. If no shard exists, the result is ``unverified`` --
    never "clean" by assumption.

Run on Colab after Phase 3::

    python -m evaluation.final_agent_eval --checkpoint runs/phase22/checkpoints/final_model.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from models.foundation import (CausalByteTransformer, decode_ids, encode_bytes)
from training.phase22_common import load_checkpoint
from training.phase22_reward import (DRAFT_CLOSE, DRAFT_OPEN, PAST_CLOSE,
                                     PAST_OPEN, WEIGHTS, encode_edit,
                                     parse_temporal_trajectory)

NOVEL_MINECRAFT_GOAL = b"Build a cobblestone generator"
TIME_SENSITIVE_QUESTION = b"Qual e o numero de downloads do dataset no_robots em 2026?"
LOGICAL_TRAP_DRAFT = b"agua + lava produzem obsidiana"
LOGICAL_TRAP_FIXED = b"agua + lava produzem pedregulho"
MEMORY_FACT = b"O codigo do cofre e 7419"
MEMORY_DISTRACTOR = b"O codigo da porta e 9082"
MEMORY_QUERY = b"Qual e o codigo do cofre?"


@dataclass(frozen=True)
class EvalCase:
    name: str
    category: str
    prompt: bytes
    expected_answer: bytes
    requires_search: bool
    required_edits: tuple[int, int] = (0, 0)
    must_contain: tuple[bytes, ...] = ()
    must_not_contain: tuple[bytes, ...] = ()
    initial_draft: bytes = b""
    expected_draft: bytes = b""
    observation: bytes = b""
    decontamination_phrases: tuple[bytes, ...] = ()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.prompt).hexdigest()


def build_cases() -> list[EvalCase]:
    edit_offset = LOGICAL_TRAP_DRAFT.find(b"obsidiana")
    fix = encode_edit(at=edit_offset, delete=9, insert=b"pedregulho")

    minecraft_prompt = (b"Goal: " + NOVEL_MINECRAFT_GOAL + b"\n"
                        b"You have a Minecraft world, a bucket of water and a lava source.\n"
                        b"Give the exact sequence of actions.\n")
    search_prompt = (b"Question: " + TIME_SENSITIVE_QUESTION + b"\n"
                     b"This value changes over time. Do not guess from memory.\n")
    # The prompt carries the immutable PAST block and the buggy draft as plain
    # text; the model must emit its OWN editable CURRENT_DRAFT block. Keeping
    # only one DRAFT block in the whole trajectory makes edit-site attribution
    # unambiguous (Phase 19's leak was exactly this kind of prompt reuse).
    trap_prompt = (b"Question: agua + lava produzem o que?\n" +
                   PAST_OPEN + b"\nAction: nenhuma acao executada\n" + PAST_CLOSE + b"\n" +
                   b"Initial draft: " + LOGICAL_TRAP_DRAFT + b"\n" +
                   b"Fix the draft with the minimum number of edits.\n")
    memory_prompt = (b"Memory injection:\n" + MEMORY_FACT + b"\n" +
                     MEMORY_DISTRACTOR + b"\n\n" +
                     b"Query: " + MEMORY_QUERY + b"\n")

    return [
        EvalCase(
            name="minecraft_novel_goal", category="generalisation",
            prompt=minecraft_prompt,
            expected_answer=b"agua e lava geram pedregulho continuamente",
            requires_search=False,
            must_contain=(b"lava", b"agua", b"pedregulho"),
            must_not_contain=(b"obsidiana",),
            decontamination_phrases=(NOVEL_MINECRAFT_GOAL, b"cobblestone generator"),
        ),
        EvalCase(
            name="search_required", category="search_over_guess",
            prompt=search_prompt,
            expected_answer=b"consulte a busca: no_robots downloads 2026",
            requires_search=True,
            must_contain=(b"search_ddg", b"Observation:", b"downloads"),
            must_not_contain=(b"sem busca",),
            observation=b"Observation: no_robots downloads 2026 = consulte a busca: no_robots downloads 2026",
            decontamination_phrases=(TIME_SENSITIVE_QUESTION, b"no_robots downloads 2026"),
        ),
        EvalCase(
            name="one_or_two_edits", category="efficient_edit",
            prompt=trap_prompt,
            expected_answer=b"pedregulho",
            requires_search=False, required_edits=(1, 2),
            must_contain=(b"pedregulho", b"[EDIT"),
            must_not_contain=(b"obsidiana",),
            initial_draft=LOGICAL_TRAP_DRAFT,
            expected_draft=LOGICAL_TRAP_FIXED,
            decontamination_phrases=(LOGICAL_TRAP_DRAFT, LOGICAL_TRAP_FIXED),
        ),
        EvalCase(
            name="memory_with_distractor", category="memory",
            prompt=memory_prompt,
            expected_answer=b"7419",
            requires_search=False,
            must_contain=(b"7419",),
            must_not_contain=(b"9082",),
            decontamination_phrases=(MEMORY_FACT, b"7419"),
        ),
    ]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class CaseResult:
    name: str
    passed: bool
    score: float
    checks: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed,
                "score": round(self.score, 4), "checks": self.checks,
                "detail": self.detail}


def score_case(case: EvalCase, trajectory: bytes) -> CaseResult:
    """Deterministic byte-level checks. No LLM judge, no fuzzy credit."""
    checks: dict[str, bool] = {}
    # The prompt legitimately contains the buggy draft and the distractor, so
    # content checks apply to the GENERATED continuation only. Temporal
    # markers are checked on the whole trajectory.
    continuation = trajectory[len(case.prompt):] if trajectory.startswith(case.prompt) else trajectory
    lowered = continuation.lower()

    checks["answer_present"] = b"Answer:" in continuation
    checks["expected_answer"] = case.expected_answer.lower() in lowered
    for phrase in case.must_contain:
        checks[f"contains:{phrase.decode('utf-8', 'replace')}"] = phrase.lower() in lowered
    for phrase in case.must_not_contain:
        checks[f"excludes:{phrase.decode('utf-8', 'replace')}"] = phrase.lower() not in lowered

    if case.requires_search:
        checks["used_search_tool"] = b"Action: search_ddg(" in trajectory
        checks["used_observation"] = b"Observation:" in trajectory
        parsed = parse_temporal_trajectory(trajectory)
        checks["search_result_used"] = parsed.used_search_result

    parsed = parse_temporal_trajectory(trajectory)
    edits = parsed.draft_edits
    low, high = case.required_edits
    if high:
        checks["edit_count_in_range"] = low <= len(edits) <= high
        checks["edits_only_in_current_draft"] = parsed.illegal_edits == 0
        checks["temporal_markers_present"] = (PAST_OPEN in trajectory and DRAFT_OPEN in trajectory)
        try:
            from training.phase22_reward import apply_edit_commands
            repaired = apply_edit_commands(case.initial_draft, edits)
            checks["verified_edit_fix"] = repaired == case.expected_draft
        except Exception as exc:
            checks["verified_edit_fix"] = False
            checks["edit_error"] = str(exc)[:200]

    score = sum(1 for value in checks.values() if value) / max(1, len(checks))
    detail = {
        "trajectory_bytes": len(trajectory),
        "edits": len(edits),
        "illegal_edits": parsed.illegal_edits,
        "searches": len(parsed.searches),
        "observations": len(parsed.observations),
        "trajectory_preview": trajectory[:400].decode("utf-8", errors="replace"),
    }
    return CaseResult(name=case.name, passed=all(checks.values()),
                      score=score, checks=checks, detail=detail)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
def oracle_agent(case: EvalCase) -> bytes:
    """Scripted IDEAL response -- proves the scorer, NOT model capability."""
    past = PAST_OPEN + b"\nAction: nenhuma acao executada\n" + PAST_CLOSE + b"\n"
    if case.name == "minecraft_novel_goal":
        steps = (b"Thought: explorar e localizar lava\n"
                 b"Action: explore lava source\n"
                 b"Observation: lava encontrada na caverna\n"
                 b"Thought: reunir agua com balde\n"
                 b"Action: gather water bucket\n"
                 b"Observation: balde com agua no inventario\n"
                 b"Thought: craftar o gerador\n"
                 b"Action: craft lava + agua em pedregulho\n"
                 b"Observation: pedregulho gerado\n"
                 b"Thought: construir a estrutura repetivel\n"
                 b"Action: build cobblestone generator\n"
                 b"Observation: gerador continuo pronto\n")
        return (case.prompt + past + steps +
                b"Answer: " + case.expected_answer + b"\n")
    if case.name == "search_required":
        return (case.prompt + past +
                b'Action: search_ddg("no_robots downloads 2026")\n' +
                case.observation + b"\n" +
                b"Answer: " + case.expected_answer + b"\n")
    if case.name == "one_or_two_edits":
        offset = case.initial_draft.find(b"obsidiana")
        fix = encode_edit(at=offset, delete=9, insert=b"pedregulho")
        return (case.prompt + DRAFT_OPEN + b"\n" + fix + b"\n" + DRAFT_CLOSE + b"\n" +
                b"Answer: " + case.expected_answer + b"\n")
    if case.name == "memory_with_distractor":
        return (case.prompt + past + b"Thought: usar apenas o fato injetado\n" +
                b"Action: recall cofre\n" + b"Observation: codigo do cofre = 7419\n" +
                b"Answer: 7419\n")
    raise KeyError(case.name)


def model_agent(model: CausalByteTransformer, case: EvalCase, *,
                max_new_bytes: int = 256, device: torch.device | str = "cpu") -> bytes:
    """Generate a continuation from a Phase 22 checkpoint."""
    model.eval()
    prefix = encode_bytes(case.prompt, bos=True)[-model.config.max_seq_len:].to(device)
    generated = model.generate(prefix, max_new_bytes=max_new_bytes, temperature=0.8)
    return case.prompt + decode_ids(generated)


# ---------------------------------------------------------------------------
# De-contamination
# ---------------------------------------------------------------------------
def scan_paths(paths: Iterable[Path], phrases: Iterable[bytes], *,
               max_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    """Look for exact probe byte strings in training shards / repo sources."""
    found: list[dict[str, str]] = []
    scanned_files = 0
    scanned_bytes = 0
    for path in paths:
        if not path.is_file() or path.stat().st_size > max_bytes:
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        scanned_files += 1
        scanned_bytes += len(raw)
        for phrase in phrases:
            if phrase and phrase in raw:
                found.append({"path": str(path),
                              "phrase": phrase.decode("utf-8", errors="replace")})
    return {"scanned_files": scanned_files, "scanned_bytes": scanned_bytes,
            "matches": found}


def decontamination_report(root: Path, cases: list[EvalCase], *,
                           shard_dirs: Iterable[str | Path] = ()) -> dict[str, Any]:
    phrases = {phrase for case in cases for phrase in case.decontamination_phrases}
    phrases.update(case.prompt for case in cases)
    # Only TRAINING CORPORA are scanned. Scanning this evaluator, the docs or
    # the notebooks would trivially match their own probe definitions and
    # produce a false "leaked" verdict.
    training_code = [root / "notebook" / "train.py",
                     root / "notebook" / "preprocess.py",
                     root / "debug" / "replay_training.py",
                     root / "data" / "synthetic.py"]
    repo = scan_paths([p for p in training_code if p.is_file()], phrases)
    shard_paths: list[Path] = []
    for folder in shard_dirs:
        path = Path(folder)
        if path.is_dir():
            shard_paths.extend(sorted(p for p in path.rglob("*") if p.is_file()))
    shards = scan_paths(shard_paths, phrases)
    # Phase 19/21 exact-prompt leakage check: those prompts were byte-identical
    # to Stage-4 RL rows. Reject any exact reuse here.
    legacy_files = [root / "notebook" / "train.py", root / "debug" / "replay_training.py"]
    legacy = scan_paths([p for p in legacy_files if p.is_file()], phrases)
    verified = shards["scanned_files"] > 0
    return {
        "status": ("clean" if verified and not (repo["matches"] or shards["matches"] or legacy["matches"])
                   else ("leaked" if (repo["matches"] or shards["matches"] or legacy["matches"])
                         else "unverified")),
        "shards_available": verified,
        "repo_training_corpora": repo, "training_shards": shards,
        "legacy_replay": legacy,
        "scanned_scope": "training corpora + shards only (evaluator/docs excluded)",
        "cases": [{"name": case.name, "sha256": case.sha256} for case in cases],
        "note": ("'unverified' means no Phase 22 shard was present locally to scan; "
                 "the Colab run must re-run this with --shard-dir data_cache_phase22. "
                 "Never report 'clean' on an assumption."),
    }


# ---------------------------------------------------------------------------
def evaluate(*, checkpoint: str | Path | None = None,
             root: str | Path = ".", shard_dirs: Iterable[str | Path] = (),
             device: str = "cpu", max_new_bytes: int = 256) -> dict[str, Any]:
    root = Path(root).resolve()
    cases = build_cases()
    decontam = decontamination_report(root, cases, shard_dirs=shard_dirs)
    oracle_mode = checkpoint is None or not Path(checkpoint).is_file()
    model = None
    if not oracle_mode:
        restored = load_checkpoint(checkpoint, device=device)
        model = restored["model"]
    results: list[dict[str, Any]] = []
    for case in cases:
        trajectory = (oracle_agent(case) if oracle_mode
                      else model_agent(model, case, max_new_bytes=max_new_bytes,
                                       device=device))
        result = score_case(case, trajectory)
        results.append(result.as_dict())
    passed = sum(1 for item in results if item["passed"])
    return {
        "schema": 1,
        "oracle_mode": oracle_mode,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "note": ("SCRIPTED ORACLE -- proves the scorer works, NOT model capability"
                 if oracle_mode else "real model generation from a Phase 22 checkpoint"),
        "cases_total": len(cases), "cases_passed": passed,
        "pass_rate": round(passed / len(cases), 4),
        "all_passed": passed == len(cases),
        "decontamination": decontam,
        "reward_weights": WEIGHTS.as_dict(),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--self-test", action="store_true",
                        help="run the scripted oracle (proves the scorer)")
    parser.add_argument("--shard-dir", action="append", default=[],
                        help="training shard directory to de-contaminate against")
    parser.add_argument("--root", default=".")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-bytes", type=int, default=256)
    parser.add_argument("--out", default="runs/phase22/final_eval.json")
    args = parser.parse_args()
    if not args.self_test and not args.checkpoint:
        parser.error("pass --self-test or --checkpoint <final_model.pt>")
    report = evaluate(checkpoint=args.checkpoint or None, root=args.root,
                      shard_dirs=args.shard_dir or ["data_cache_phase22"],
                      device=args.device, max_new_bytes=args.max_new_bytes)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({"oracle_mode": report["oracle_mode"],
                      "cases_passed": report["cases_passed"],
                      "cases_total": report["cases_total"],
                      "all_passed": report["all_passed"],
                      "decontamination": report["decontamination"]["status"],
                      "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
