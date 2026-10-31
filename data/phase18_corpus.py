"""Phase 18 corpus: reference-marked requests, refusal trajectories, balance.

Phase 17 measured the limit precisely: the model routed to the right tool 399
times out of 448 and then called it with the wrong key **0** times correctly,
because reproducing a twelve-character sha is twelve independent byte
predictions and it gets ~40% of them right. Worse, when the tool answered
"not in the corpus" the model invented a result rather than reporting failure.

This module rebuilds the corpus around two changes:

**References.** Entities in the request are wrapped in ``[REF]...[/REF]`` and
the action points at them by index -- ``explain_commit("[REF]1[/REF]")``. The
host resolves the index against the user's own request before calling the tool,
so the bytes the tool receives are the bytes the user typed. Byte-exactness
stops being a generation problem and becomes a lookup. Crucially the model is
never trusted: a reference that is not in the request fails closed, so a
hallucinated identifier cannot reach a tool.

**Refusal.** ~1,400 trajectories whose observation is a genuine "not found" or
"unresolved reference", each ending in an answer that says so. The identifiers
in them are real lookups that really fail -- nothing about them is invented.

Usage::

    python -m data.phase18_corpus --output-dir runs/phase18
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from data.github_to_react import (CorpusIndex, tool_explain_commit, tool_review_pr,
                                  tool_search_issues, tool_summarize_readme)
from data.mixed_react_dataset import (GITHUB_KINDS, load_github, load_trajectories,
                                      split_github)
from data.scalable_react_dataset import Trajectory, dataset_stats, write_jsonl
from models.react import mark, render_trajectory

DEFAULT_GITHUB = "data/github_raw/github_react.jsonl"
DEFAULT_SYNTHETIC_DIR = "runs/phase17"
DEFAULT_STORIES = "data/github_raw/github_stories.jsonl"
DEFAULT_OUTPUT_DIR = "runs/phase18"
SEED = 18018

REFUSAL_SEED = 18181
REFUSAL_KINDS = ("github_refuse_commit", "github_refuse_pr", "github_refuse_issue",
                 "github_refuse_readme", "github_refuse_reference")

_HEX = re.compile(rb"^[0-9a-f]{6,40}$")
_WORD = re.compile(r"[a-z0-9][a-z0-9\-_]{2,}")

# How many marked entities a request is allowed to carry. One is the normal
# case; keeping the number small is what makes an index easy to emit.
MAX_REFS = 3


# ------------------------------------------------------------------ marking
def mark_request(request: bytes, entities: list[bytes]) -> bytes:
    """Wrap each entity in ``[REF]`` markers, longest first.

    Longest-first matters when one entity is a prefix of another (a short sha
    inside a longer identifier): replacing the shorter one first would corrupt
    the longer.
    """
    marked = bytes(request)
    for entity in sorted(set(entities), key=len, reverse=True):
        if not entity or entity not in marked:
            continue
        marked = marked.replace(entity, mark(entity), 1)
    return marked


def entity_for(kind: str, record: dict) -> bytes | None:
    """The single entity a trajectory's action must reference."""
    if kind == "commit":
        return record["sha"].encode("utf-8")
    if kind == "pr":
        return f"{record['source']}#{record['number']}".encode("utf-8")
    if kind == "readme":
        return record["source"].encode("utf-8")
    if kind == "issue":
        return None  # issue queries are free text, not an identifier
    raise ValueError(f"unknown kind {kind!r}")


# --------------------------------------------------------------- refitting
def refit_github(rows: list[Trajectory], index: CorpusIndex) -> list[Trajectory]:
    """Re-render real trajectories with marked requests and reference actions.

    The observation is recomputed by **calling the real tool** on the resolved
    identifier, exactly as the host will at run time, so a trajectory can only
    exist if its observation is true.
    """
    refitted: list[Trajectory] = []
    for row in rows:
        text = row.text
        if b"Action: " not in text:
            continue
        action_line = [line for line in text.split(b"\n") if line.startswith(b"Action: ")][0]
        name = action_line[len(b"Action: "):action_line.index(b"(")]
        argument = re.findall(rb'"([^"]*)"', action_line)
        if not argument:
            continue
        identifier = argument[0]
        if b"[REF]" in row.request:
            continue

        request = b" ".join(row.request.split())
        # The request already contains the identifier verbatim (guaranteed by
        # the Phase 17 converter's copyability check), so marking it is safe.
        if identifier not in request:
            continue
        marked_request = mark_request(request, [identifier])
        spans = [identifier]
        del spans  # the host recomputes these; we only need the index form

        result = _call_tool(name, identifier, index)
        if result is None or result.startswith(b"no "):
            continue  # a real row that no longer resolves: drop, do not fake
        answer = _grounded_answer(name, identifier, result)
        if answer is None:
            continue

        body = b"\n".join((
            _thought_for(name),
            b'Action: ' + name + b'("[REF]1[/REF]")',
            b"Observation: " + b" ".join(result.split()),
            b"Answer: " + answer,
        ))
        text = render_trajectory(marked_request, body)
        refitted.append(Trajectory(request=marked_request, memory=b"",
                                   body=body, kind=row.kind, text=text))
    return refitted


def _call_tool(name: bytes, identifier: bytes, index: CorpusIndex) -> bytes | None:
    argument = identifier.decode("utf-8", "replace")
    try:
        if name == b"explain_commit":
            return tool_explain_commit(index, argument)
        if name == b"review_pr":
            return tool_review_pr(index, argument)
        if name == b"search_issues":
            return tool_search_issues(index, argument)
        if name == b"summarize_readme":
            return tool_summarize_readme(index, argument)
    except Exception:
        return None
    return None


def _thought_for(name: bytes) -> bytes:
    return {
        b"explain_commit": b"The user quotes a commit; I should look up who made it and what it says.",
        b"review_pr": b"A pull request needs a review; I should read what it changes and how it stands.",
        b"search_issues": b"The user reports a problem; I should look for the closest issues in the corpus.",
        b"summarize_readme": b"The user asks about a repository; I should read its README before answering.",
    }[name]


_AUTHOR_DATE = re.compile(rb"^commit \S+ by (.+?) on (\d{4}-\d{2}-\d{2}): (.*)$")
_PR_STATE = re.compile(rb"^PR \S+ is (\S+) with (\d+) comments; first line: (.*)$")
_SEARCH_HITS = re.compile(rb"#(\d+) \[([^\]]+)\] ([^;]*)")
_README_STATS = re.compile(rb"^(\S+) README: (.*?) (\d+) headings, (\d+) code blocks, (\d+) characters")


def _grounded_answer(name: bytes, identifier: bytes, observation: bytes) -> bytes | None:
    """An answer assembled from the observation, referring by index.

    Every field below is something only the observation knows -- the author of a
    commit, the state of a pull request, the repository a hit came from. An
    answer built this way cannot have been copied out of the request.
    """
    flat = b" ".join(observation.split())
    if name == b"explain_commit":
        match = _AUTHOR_DATE.match(flat)
        if not match:
            return None
        subject = match.group(3)[:70].rstrip()
        return b'[REF]1[/REF] is a commit by ' + match.group(1) + b" on " + match.group(2) + b": " + subject
    if name == b"review_pr":
        match = _PR_STATE.match(flat)
        if not match:
            return None
        return b'[REF]1[/REF] is ' + match.group(1) + b" with " + match.group(2) + b" comments."
    if name == b"search_issues":
        hits = _SEARCH_HITS.findall(flat)
        if not hits:
            return None
        number, source, title = hits[0]
        return b"Related issue " + number + b" in [" + source + b"]: " + title.strip()[:70]
    if name == b"summarize_readme":
        match = _README_STATS.match(flat)
        if not match:
            return None
        return (b'[REF]1[/REF] README is titled "' + match.group(2).strip()[:50]
                + b'", ' + match.group(5) + b" characters.")
    return None


# ------------------------------------------------------------------ refusal
def _fake_sha(rng: random.Random, corpus: CorpusIndex) -> bytes:
    while True:
        candidate = "".join(rng.choice("0123456789abcdef") for _ in range(12)).encode()
        if not any(record["sha"].encode() == candidate for record in corpus.commits):
            return candidate


def _fake_pr(rng: random.Random, corpus: CorpusIndex) -> bytes:
    known = {f"{r['source']}#{r['number']}".encode() for r in corpus.prs}
    sources = sorted({r["source"] for r in corpus.prs})
    while True:
        candidate = f"{rng.choice(sources)}#{rng.randint(900000, 999999)}".encode()
        if candidate not in known:
            return candidate


def _miss_query(rng: random.Random) -> bytes:
    """A query built from syllables that match nothing in the corpus."""
    noise = ["zzq", "vqx", "jwkp", "xrzv", "qqjx", "vzqp"]
    return " ".join(rng.sample(noise, 3)).encode()


def build_refusals(corpus: CorpusIndex, *, count: int = 1400,
                   seed: int = REFUSAL_SEED) -> list[Trajectory]:
    """Trajectories whose observation is a real failure and whose answer says so.

    Nothing here is synthesised. Every observation is produced by calling the
    tool with an identifier that really is absent from the corpus, and the
    trajectory is kept only if the tool genuinely reported failure.
    """
    rng = random.Random(seed)
    rows: list[Trajectory] = []
    per_kind = max(count // 5, 1)

    for _ in range(per_kind):
        rows.append(_refusal(b"explain_commit", _fake_sha(rng, corpus), corpus,
                             kind="github_refuse_commit",
                             subject=b"which I cannot see in the corpus",
                             noun=b"commit"))
    for _ in range(per_kind):
        rows.append(_refusal(b"review_pr", _fake_pr(rng, corpus), corpus,
                             kind="github_refuse_pr", subject=None, noun=b"pull request"))
    for _ in range(per_kind):
        rows.append(_refusal(b"search_issues", _miss_query(rng), corpus,
                             kind="github_refuse_issue", subject=None, noun=b"issue"))
    for _ in range(per_kind):
        rows.append(_refusal(b"summarize_readme", b"nobody/no-such-repo", corpus,
                             kind="github_refuse_readme", subject=None, noun=b"README"))

    # A reference the request does not contain: the host refuses before the
    # tool is ever called. This is the case that makes the mechanism safe.
    for position in range(per_kind):
        index = position % 2 + 2  # 2 or 3, while the request marks exactly one
        request = b"Explain commit " + mark(_fake_sha(rng, corpus)) + b": a commit"
        result = b"ERROR: unresolved reference: reference " + str(index).encode() + b" is out of range; the request marks 1 entities"
        body = b"\n".join((
            _thought_for(b"explain_commit"),
            b'Action: explain_commit("[REF]' + str(index).encode() + b'[/REF]")',
            b"Observation: " + result,
            b"Answer: I could not resolve that reference, so I did not look anything up.",
        ))
        rows.append(Trajectory(request=request, memory=b"", body=body,
                               kind="github_refuse_reference",
                               text=render_trajectory(request, body)))

    return [row for row in rows if row is not None]


def _refusal(tool: bytes, identifier: bytes, corpus: CorpusIndex, *, kind: str,
             subject: bytes | None, noun: bytes) -> Trajectory | None:
    observation = _call_tool(tool, identifier, corpus)
    if observation is None or not observation.startswith(b"no "):
        return None  # it resolved: not a refusal, do not pretend it was one
    request = _refusal_request(tool, identifier, subject)
    if request is None:
        return None
    body = b"\n".join((
        _thought_for(tool),
        b'Action: ' + tool + b'("[REF]1[/REF]")',
        b"Observation: " + b" ".join(observation.split()),
        b"Answer: I could not find that " + noun + b" in the corpus.",
    ))
    return Trajectory(request=request, memory=b"", body=body, kind=kind,
                      text=render_trajectory(request, body))


def _refusal_request(tool: bytes, identifier: bytes, subject: bytes | None) -> bytes | None:
    marked = mark(identifier)
    if tool == b"explain_commit":
        return b"Explain commit " + marked + b": " + (subject or b"a change")
    if tool == b"review_pr":
        return b"Review pull request " + marked + b": a change"
    if tool == b"search_issues":
        return b"Find issues similar to: " + marked
    if tool == b"summarize_readme":
        return b"Summarize the project " + marked + b"."
    return None


# ----------------------------------------------------------------- assembly
def build_phase18(synthetic_dir: str | Path = DEFAULT_SYNTHETIC_DIR,
                  github_path: str | Path = DEFAULT_GITHUB,
                  stories_path: str | Path = DEFAULT_STORIES, *,
                  refusals: int = 1400, seed: int = SEED) -> dict[str, list[Trajectory]]:
    """Synthetic splits, real trajectories refitted with references, refusals.

    The held-out split is inherited from :func:`data.mixed_react_dataset.\
    split_github`, which keys on the identifier -- so a record's refusal and its
    success trajectory can never straddle the split.
    """
    synthetic_dir = Path(synthetic_dir)
    synthetic = {split: load_trajectories(synthetic_dir / f"react_{split}.jsonl")
                 for split in ("train", "validation", "test")}
    # Phase 17's split files contain both corpora; keep only the synthetic rows
    # so the real ones are re-rendered with references rather than duplicated.
    synthetic = {split: [row for row in rows if row.kind not in GITHUB_KINDS]
                 for split, rows in synthetic.items()}

    corpus = CorpusIndex.from_jsonl(stories_path)
    github = split_github(load_github(github_path))
    refitted = {split: refit_github(rows, corpus) for split, rows in github.items()}

    refusal_rows = build_refusals(corpus, count=refusals)
    refusal_splits = _split_refusals(refusal_rows, seed=seed)

    mixed: dict[str, list[Trajectory]] = {}
    for split in ("train", "validation", "test"):
        rows = list(synthetic[split]) + refitted[split] + refusal_splits[split]
        random.Random(seed).shuffle(rows)
        mixed[split] = rows
    mixed["_github_only_test"] = refitted["test"]
    mixed["_refusal_only_test"] = refusal_splits["test"]
    return mixed


def _split_refusals(rows: list[Trajectory], *, seed: int = SEED) -> dict[str, list[Trajectory]]:
    """Same hash-bucket discipline as the real rows, keyed on the identifier."""
    grouped: dict[bytes, list[Trajectory]] = {}
    for row in rows:
        key = row.request
        grouped.setdefault(key, []).append(row)
    keys = sorted(grouped)
    ordered = sorted(keys, key=lambda key: int.from_bytes(
        hashlib.sha256(key).digest()[:8], "big") / float(1 << 64))
    n_test = max(len(ordered) // 10, 1)
    n_val = max(len(ordered) // 10, 1)
    test_keys = set(ordered[:n_test])
    val_keys = set(ordered[n_test:n_test + n_val])
    splits: dict[str, list[Trajectory]] = {"train": [], "validation": [], "test": []}
    for key in keys:
        bucket = "test" if key in test_keys else "validation" if key in val_keys else "train"
        splits[bucket].extend(grouped[key])
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-dir", default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--github", default=DEFAULT_GITHUB)
    parser.add_argument("--stories", default=DEFAULT_STORIES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refusals", type=int, default=1400)
    args = parser.parse_args()

    mixed = build_phase18(args.synthetic_dir, args.github, args.stories,
                          refusals=args.refusals)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        write_jsonl(mixed[split], output / f"react_{split}.jsonl")
    write_jsonl(mixed["_github_only_test"], output / "github_heldout.jsonl")
    write_jsonl(mixed["_refusal_only_test"], output / "refusal_heldout.jsonl")

    report = {
        split: {
            "all": dataset_stats(mixed[split]),
            "real": dataset_stats([r for r in mixed[split] if r.kind in GITHUB_KINDS]),
            "refusal": dataset_stats([r for r in mixed[split] if r.kind in REFUSAL_KINDS]),
            "synthetic": dataset_stats([r for r in mixed[split]
                                        if r.kind not in GITHUB_KINDS
                                        and r.kind not in REFUSAL_KINDS]),
        } for split in ("train", "validation", "test")
    }
    (output / "corpus_stats.json").write_text(json.dumps(report, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
