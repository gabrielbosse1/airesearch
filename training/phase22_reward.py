"""Phase 3 sandbox RL reward: "Search > Guess" and "Efficient Edit > Thrashing".

Everything is byte-level (the model predicts raw bytes). Two behaviours are
encoded explicitly, with the exact numbers requested:

  * +1.0  correct final answer
  * -0.5  per ``[EDIT]`` token
  * +0.8  if 1-2 edits fix a VERIFIABLE error (precise minimal self-correction)
  * -2.0  heavy penalty for a wrong answer with NO search  (Search > Guess)
  * +1.5  heavy reward for ``Action: search_ddg("...")`` AND using its result

Temporal awareness is enforced with two blocks::

    [PAST_ACTION: executed] ... [/PAST_ACTION]        <- immutable history
    [CURRENT_DRAFT: editable] ... [/CURRENT_DRAFT]    <- the ONLY legal edit site

Edits inside the PAST block or outside the DRAFT block are rejected with an
extra penalty. An edit is a raw-byte command ``[EDIT at=N del=M ins_b64="..."]``
(base64 keeps arbitrary bytes lossless -- no escaping, no Unicode mangling).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

PAST_OPEN = b"[PAST_ACTION: executed]"
PAST_CLOSE = b"[/PAST_ACTION]"
DRAFT_OPEN = b"[CURRENT_DRAFT: editable]"
DRAFT_CLOSE = b"[/CURRENT_DRAFT]"
SEARCH_PREFIX = b"Action: search_ddg("
OBSERVATION_PREFIX = b"Observation:"
ANSWER_PREFIX = b"Answer:"
EDIT_MARKER = b"[EDIT"

EDIT_PATTERN = re.compile(rb"\[EDIT at=(\d+) del=(\d+) ins_b64=\"([A-Za-z0-9+/=]*)\"\]")
_PAST_PATTERN = re.compile(re.escape(PAST_OPEN) + rb"(.*?)" + re.escape(PAST_CLOSE), re.S)
_DRAFT_PATTERN = re.compile(re.escape(DRAFT_OPEN) + rb"(.*?)" + re.escape(DRAFT_CLOSE), re.S)


@dataclass(frozen=True)
class RewardWeights:
    """Exact requested values; change here, never inline in the loop."""

    correct_final: float = 1.0
    per_edit_token: float = -0.5
    verified_edit_fix: float = 0.8
    wrong_guess_without_search: float = -2.0
    lucky_guess_without_search: float = -0.5
    search_and_use: float = 1.5
    search_used_but_wrong: float = 0.5
    search_ignored: float = 0.1
    illegal_edit: float = -1.0
    thrashing_per_extra_edit: float = -0.75
    max_efficient_edits: int = 2

    def as_dict(self) -> dict[str, float | int]:
        return {
            "correct_final": self.correct_final,
            "per_edit_token": self.per_edit_token,
            "verified_edit_fix": self.verified_edit_fix,
            "wrong_guess_without_search": self.wrong_guess_without_search,
            "lucky_guess_without_search": self.lucky_guess_without_search,
            "search_and_use": self.search_and_use,
            "search_used_but_wrong": self.search_used_but_wrong,
            "search_ignored": self.search_ignored,
            "illegal_edit": self.illegal_edit,
            "thrashing_per_extra_edit": self.thrashing_per_extra_edit,
            "max_efficient_edits": self.max_efficient_edits,
        }


WEIGHTS = RewardWeights()


@dataclass(frozen=True)
class EditCommand:
    at: int
    delete: int
    insert: bytes
    site: str = "draft"          # draft | past | outside
    raw: bytes = b""

    def as_dict(self) -> dict[str, Any]:
        return {"at": self.at, "delete": self.delete,
                "insert": self.insert.decode("utf-8", errors="replace"),
                "site": self.site, "raw": self.raw.decode("ascii", errors="replace")}


class EditError(ValueError):
    pass


def encode_edit(at: int, delete: int, insert: bytes) -> bytes:
    """Lossless raw-byte edit command."""
    if at < 0 or delete < 0:
        raise ValueError("at/delete must be non-negative")
    payload = base64.b64encode(bytes(insert)).decode("ascii")
    return f'[EDIT at={at} del={delete} ins_b64="{payload}"]'.encode("ascii")


def parse_edits(trajectory: bytes, *, past: bytes = b"", draft: bytes = b"") -> list[EditCommand]:
    commands: list[EditCommand] = []
    for match in re.finditer(EDIT_PATTERN, trajectory):
        at, delete, payload = int(match.group(1)), int(match.group(2)), match.group(3)
        try:
            insert = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise EditError(f"invalid base64 in edit: {exc}") from exc
        raw = match.group(0)
        start = match.start()
        if past and PAST_OPEN in trajectory:
            past_start = trajectory.find(PAST_OPEN)
            if past_start <= start < past_start + len(PAST_OPEN) + len(past):
                site = "past"
            elif draft and DRAFT_OPEN in trajectory:
                draft_start = trajectory.find(DRAFT_OPEN)
                site = "draft" if draft_start <= start < draft_start + len(DRAFT_OPEN) + len(draft) else "outside"
            else:
                site = "outside"
        elif draft and DRAFT_OPEN in trajectory:
            draft_start = trajectory.find(DRAFT_OPEN)
            site = "draft" if draft_start <= start < draft_start + len(DRAFT_OPEN) + len(draft) else "outside"
        else:
            site = "outside"
        commands.append(EditCommand(at=at, delete=delete, insert=insert,
                                    site=site, raw=raw))
    return commands


def apply_edit_commands(draft: bytes, commands: Iterable[EditCommand]) -> bytes:
    """Apply edits right-to-left so earlier offsets stay valid. Lossless."""
    ordered = sorted(commands, key=lambda item: item.at, reverse=True)
    result = bytearray(draft)
    for command in ordered:
        if command.at + command.delete > len(result):
            raise EditError(f"edit out of bounds: at={command.at} del={command.delete} len={len(result)}")
        result[command.at:command.at + command.delete] = command.insert
    return bytes(result)


@dataclass(frozen=True)
class TemporalTrajectory:
    past: bytes
    draft: bytes
    edits: tuple[EditCommand, ...]
    searches: tuple[bytes, ...]
    observations: tuple[bytes, ...]
    answer: bytes
    evidence_used: bool
    illegal_edits: int

    @property
    def draft_edits(self) -> tuple[EditCommand, ...]:
        return tuple(edit for edit in self.edits if edit.site == "draft")

    @property
    def used_search_result(self) -> bool:
        return bool(self.searches) and bool(self.observations) and self.evidence_used


def _line_after(text: bytes, prefix: bytes) -> bytes:
    """Return the last labelled line's payload."""
    found = b""
    for line in text.split(b"\n"):
        stripped = line.strip()
        if stripped.startswith(prefix):
            found = stripped[len(prefix):].strip()
    return found


def _tokens(value: bytes) -> set[bytes]:
    words = re.findall(rb"[a-z0-9]{4,}", value.lower())
    stop = {b"that", b"this", b"with", b"from", b"about", b"search", b"result",
            b"answer", b"observation", b"action", b"source", b"https", b"http"}
    return {word for word in words if word not in stop}


def parse_temporal_trajectory(trajectory: bytes) -> TemporalTrajectory:
    """Parse immutable PAST vs editable CURRENT_DRAFT and search evidence."""
    if not isinstance(trajectory, (bytes, bytearray)):
        raise TypeError("trajectory must be bytes")
    trajectory = bytes(trajectory)
    past_match = _PAST_PATTERN.search(trajectory)
    draft_match = _DRAFT_PATTERN.search(trajectory)
    past = past_match.group(1).strip() if past_match else b""
    draft = draft_match.group(1).strip() if draft_match else b""
    edits = parse_edits(trajectory, past=past, draft=draft)
    searches = tuple(line.strip() for line in trajectory.split(b"\n")
                     if line.strip().startswith(SEARCH_PREFIX))
    observations = tuple(line.strip()[len(OBSERVATION_PREFIX):].strip()
                         for line in trajectory.split(b"\n")
                         if line.strip().startswith(OBSERVATION_PREFIX))
    answer = _line_after(trajectory, ANSWER_PREFIX)
    evidence = set().union(*(_tokens(item) for item in observations)) if observations else set()
    evidence_used = bool(_tokens(answer) & evidence)
    return TemporalTrajectory(
        past=past, draft=draft, edits=tuple(edits), searches=searches,
        observations=observations, answer=answer, evidence_used=evidence_used,
        illegal_edits=sum(1 for edit in edits if edit.site != "draft"),
    )


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    components: dict[str, float]
    flags: dict[str, bool]
    edits: int
    verified_fix: bool

    def as_dict(self) -> dict[str, Any]:
        return {"total": round(self.total, 4),
                "components": {key: round(value, 4) for key, value in self.components.items()},
                "flags": dict(self.flags), "edits": self.edits,
                "verified_fix": self.verified_fix}


def verifier(answer: bytes, expected: bytes) -> bool:
    """Exact byte match after stripping whitespace; no fuzzy credit."""
    return answer.strip() == expected.strip()


def compute_reward(trajectory: bytes, case: Mapping[str, Any],
                   weights: RewardWeights = WEIGHTS) -> RewardBreakdown:
    """Compute the complete Phase 3 reward for one rollout.

    ``case`` may provide ``expected_answer``, ``initial_draft``,
    ``expected_draft``, and ``search_required``. A verified edit fix is only
    awarded when applying the 1-2 draft edits to ``initial_draft`` produces
    ``expected_draft`` byte-for-byte.
    """
    parsed = parse_temporal_trajectory(trajectory)
    expected = case.get("expected_answer", b"")
    if isinstance(expected, str):
        expected = expected.encode("utf-8")
    correct = verifier(parsed.answer, expected)
    required = bool(case.get("search_required", False))
    components: dict[str, float] = {}
    flags = {
        "correct_final": correct,
        "used_search": bool(parsed.searches),
        "used_search_result": parsed.used_search_result,
        "guessed_without_search": not parsed.searches and required,
        "efficient_edit": False,
        "thrashing": False,
        "temporal_violation": parsed.illegal_edits > 0,
    }
    if correct:
        components["correct_final"] = weights.correct_final

    # ---- Search > Guess -------------------------------------------------
    if not parsed.searches:
        if required and not correct:
            components["wrong_guess_without_search"] = weights.wrong_guess_without_search
        elif required and correct:
            components["lucky_guess_without_search"] = weights.lucky_guess_without_search
    else:
        if parsed.used_search_result and correct:
            components["search_and_use"] = weights.search_and_use
        elif parsed.used_search_result:
            components["search_used_but_wrong"] = weights.search_used_but_wrong
        else:
            components["search_ignored"] = weights.search_ignored

    # ---- Efficient Edit > Thrashing -------------------------------------
    draft_edits = parsed.draft_edits
    count = len(draft_edits)
    components["per_edit_token"] = weights.per_edit_token * count
    verified = False
    initial = case.get("initial_draft", b"")
    target = case.get("expected_draft", b"")
    if isinstance(initial, str):
        initial = initial.encode("utf-8")
    if isinstance(target, str):
        target = target.encode("utf-8")
    if 1 <= count <= weights.max_efficient_edits and initial and target:
        try:
            repaired = apply_edit_commands(initial, draft_edits)
            verified = (repaired == target)
        except EditError:
            verified = False
        if verified:
            components["verified_edit_fix"] = weights.verified_edit_fix
            flags["efficient_edit"] = True
    if count > weights.max_efficient_edits:
        extra = count - weights.max_efficient_edits
        components["thrashing"] = weights.thrashing_per_extra_edit * extra
        flags["thrashing"] = True
    if parsed.illegal_edits:
        components["illegal_edit"] = weights.illegal_edit * parsed.illegal_edits
    return RewardBreakdown(total=sum(components.values()), components=components,
                           flags=flags, edits=count, verified_fix=verified)


class RunningBaseline:
    """EMA baseline for REINFORCE advantage (variance reduction)."""

    def __init__(self, decay: float = 0.9) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self.value = 0.0
        self.count = 0

    def update(self, reward: float) -> None:
        self.count += 1
        self.value = (reward if self.count == 1
                      else self.decay * self.value + (1 - self.decay) * reward)

    def advantage(self, reward: float) -> float:
        """Advantage against the PRIOR baseline (unbiased), then update it."""
        prior = self.value if self.count else reward
        advantage = reward - prior
        self.update(reward)
        return advantage


def dominance_report(weights: RewardWeights = WEIGHTS) -> dict[str, Any]:
    """Prove the two ordering constraints with concrete rollouts."""
    case = {
        "expected_answer": b"Obsidiana e obtida com agua sobre lava",
        "initial_draft": b"Obsidiana e obtida com gelo sobre lava",
        "expected_draft": b"Obsidiana e obtida com agua sobre lava",
        "search_required": True,
    }
    search_prefix = (b'Action: search_ddg("obsidiana agua lava")\n'
                     b'Observation: obsidiana e obtida com agua sobre lava\n')
    fix = encode_edit(at=27, delete=4, insert=b"agua")  # replaces "gelo"
    # Verify the offset actually repairs the draft (self-check, not a claim).
    repaired = apply_edit_commands(case["initial_draft"],
                                   parse_edits(DRAFT_OPEN + fix + DRAFT_CLOSE))
    if repaired != case["expected_draft"]:
        # Deterministically locate the wrong token instead of guessing offsets.
        position = case["initial_draft"].find(b"gelo")
        fix = encode_edit(at=position, delete=4, insert=b"agua")
        repaired = apply_edit_commands(case["initial_draft"],
                                       parse_edits(DRAFT_OPEN + fix + DRAFT_CLOSE))
    assert repaired == case["expected_draft"], "edit verifier fixture is broken"

    answer = b"Answer: " + case["expected_answer"] + b"\n"
    draft = DRAFT_OPEN + b"\n" + fix + b"\n" + DRAFT_CLOSE + b"\n"

    search_and_use = compute_reward(search_prefix + PAST_OPEN + b"\nAction: previous step\n" +
                                    PAST_CLOSE + b"\n" + draft + answer, case, weights)
    guess_wrong = compute_reward(PAST_OPEN + b"Action: none" + PAST_CLOSE +
                                 b"\nAnswer: uma resposta inventada\n", case, weights)
    guess_lucky = compute_reward(b"Answer: " + case["expected_answer"] + b"\n", case, weights)
    efficient = compute_reward(search_prefix + draft + answer, case, weights)
    thrash = compute_reward(search_prefix + DRAFT_OPEN + b"\n" + (fix + b"\n") * 5 +
                            DRAFT_CLOSE + b"\n" + answer, case, weights)
    past_edit = compute_reward(search_prefix + PAST_OPEN + fix + PAST_CLOSE +
                               b"\n" + draft + answer, case, weights)
    ordering = {
        "search_gt_guess": search_and_use.total > guess_wrong.total,
        "search_gt_lucky_guess": search_and_use.total > guess_lucky.total,
        "efficient_edit_gt_thrashing": efficient.total > thrash.total,
        "past_action_is_immutable": past_edit.total < efficient.total,
    }
    return {
        "ordering": ordering, "all_satisfied": all(ordering.values()),
        "weights": weights.as_dict(),
        "rewards": {
            "search_and_use": search_and_use.as_dict(),
            "wrong_guess": guess_wrong.as_dict(),
            "lucky_guess": guess_lucky.as_dict(),
            "efficient_edit": efficient.as_dict(),
            "thrashing": thrash.as_dict(),
            "edit_in_past_block": past_edit.as_dict(),
        },
    }


def build_rl_case(*, question: bytes, expected_answer: bytes,
                  initial_draft: bytes, expected_draft: bytes,
                  past_actions: bytes = b"", search_required: bool = True) -> dict[str, Any]:
    """Build a Phase 3 case with explicit temporal markers."""
    prompt = (b"Question: " + question + b"\n" +
              PAST_OPEN + b"\n" + past_actions + b"\n" + PAST_CLOSE + b"\n" +
              DRAFT_OPEN + b"\n" + initial_draft + b"\n" + DRAFT_CLOSE + b"\n" +
              b"Rules: search before answering; edit ONLY the current draft.\n")
    return {"prompt": prompt, "expected_answer": expected_answer,
            "initial_draft": initial_draft, "expected_draft": expected_draft,
            "search_required": search_required}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(dominance_report(), indent=2))
    else:
        print(json.dumps({"weights": WEIGHTS.as_dict(),
                          "past_marker": PAST_OPEN.decode(),
                          "draft_marker": DRAFT_OPEN.decode()}, indent=2))


if __name__ == "__main__":
    main()
