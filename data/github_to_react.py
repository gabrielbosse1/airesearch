"""Turn collected GitHub data into ReAct training sequences with **real** observations.

The sketch for this module synthesised its observations ("Found 3 similar
issues", "PR modifies 5 files", "README describes a project with N
characters"). Those numbers were invented, and training on them would teach the
model to hallucinate tool results -- the one failure this architecture cannot
tolerate, because the observation is the only channel that reports what the
world actually did.

So every ``Observation:`` below is the **return value of a real function over
the collected corpus**, computed at conversion time:

=======================  ==========================================================
``search_issues(q)``     lexical search over real issue titles/bodies -> real numbers + titles
``review_pr(id)``        real state / merged flag / comment count / first body line
``explain_commit(id)``   real sha, author, date and first message line
``summarize_readme(r)``  real heading, section count, code-block count, character count
=======================  ==========================================================

The tools are corpus lookups, not a hosted GitHub search: they are honest about
what they can answer, and they say so when they find nothing.

Output is JSONL with a ``text`` field. Bytes are recovered exactly with
``text.encode("utf-8")``; storing them as JSON integer arrays (the original
sketch) would quadruple the file size for no information.

Usage::

    python -m data.github_to_react --input data/github_raw/github_stories.jsonl
    python -m data.github_to_react --verify 50
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from models.react import (ANSWER, OBSERVATION, format_action,
                          format_observation, last_action, render_trajectory)

DEFAULT_INPUT = "data/github_raw/github_stories.jsonl"
DEFAULT_OUTPUT = "data/github_raw/github_react.jsonl"
DEFAULT_LIMIT = 200  # observation bytes, matching the runtime's own cap

_WORD = re.compile(r"[a-z0-9][a-z0-9\-_]{2,}")
_STOP = frozenset(("the", "and", "for", "with", "that", "this", "from", "when", "not", "are",
                   "but", "has", "have", "was", "were", "you", "your", "its", "all", "can",
                   "use", "using", "into", "out", "get", "got", "how", "why", "what", "should"))


# --------------------------------------------------------------------- index
@dataclass
class CorpusIndex:
    """Searchable view over the collected stories."""

    issues: list[dict] = field(default_factory=list)
    prs: list[dict] = field(default_factory=list)
    commits: list[dict] = field(default_factory=list)
    readmes: dict[str, dict] = field(default_factory=dict)
    _tokens: dict[int, list[str]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "CorpusIndex":
        index = cls()
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                kind = record.get("type")
                if kind == "issue":
                    index.issues.append(record)
                elif kind == "pr":
                    index.prs.append(record)
                elif kind == "commit":
                    index.commits.append(record)
                elif kind == "readme":
                    index.readmes[record["source"]] = record
        return index

    def _tokenize(self, key: int, text: str) -> list[str]:
        cached = self._tokens.get(key)
        if cached is None:
            cached = [token for token in _WORD.findall(text.lower()) if token not in _STOP]
            self._tokens[key] = cached
        return cached

    def search(self, records: Sequence[dict], query: str, k: int = 3) -> list[tuple[float, dict]]:
        """Score by how many query tokens appear in the record (real lexical match)."""
        wanted = [token for token in _WORD.findall(query.lower()) if token not in _STOP][:8]
        if not wanted:
            return []
        scored: list[tuple[float, dict]] = []
        for position, record in enumerate(records):
            tokens = self._tokenize(id(record) ^ position, f"{record.get('title', '')} {record.get('body', '')}")
            if not tokens:
                continue
            present = set(tokens)
            score = sum(1 for token in wanted if token in present) / len(wanted)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:k]

    def search_issues(self, query: str, k: int = 3) -> list[tuple[float, dict]]:
        return self.search(self.issues, query, k)

    def search_prs(self, query: str, k: int = 3) -> list[tuple[float, dict]]:
        return self.search(self.prs, query, k)

    def find_pr(self, identifier: str) -> dict | None:
        for record in self.prs:
            if f"{record['source']}#{record['number']}" == identifier:
                return record
        return None

    def find_commit(self, sha: str) -> dict | None:
        for record in self.commits:
            if record.get("sha") == sha:
                return record
        return None


# --------------------------------------------------------------------- tools
def _first_line(text: str, limit: int = 100) -> str:
    line = (text or "").strip().split("\n", 1)[0].strip()
    return line[:limit]


def tool_search_issues(index: CorpusIndex, query: str, k: int = 3) -> bytes:
    """Real lexical search over the collected issues. Reports what it found."""
    hits = index.search_issues(query, k)
    if not hits:
        return b"no issue in the corpus matches: " + query[:60].encode("utf-8")
    rendered = []
    for score, record in hits:
        rendered.append(f"#{record['number']} [{record['source']}] {_first_line(record['title'], 60)}")
    return ("found " + str(len(hits)) + " matching issues: " + "; ".join(rendered)).encode("utf-8")


def tool_review_pr(index: CorpusIndex, identifier: str) -> bytes:
    """Real PR metadata: state, merge status, comment count, first body line."""
    record = index.find_pr(identifier)
    if record is None:
        return b"no pull request " + identifier.encode("utf-8") + b" in the corpus"
    state = "merged" if record.get("merged") else record.get("state", "unknown")
    detail = (f"PR {identifier} is {state} with {record.get('comments', 0)} comments; "
              f"first line: {_first_line(record.get('body', ''), 80)}")
    return detail.encode("utf-8")


def tool_explain_commit(index: CorpusIndex, sha: str) -> bytes:
    """Real commit facts from the API payload: sha, author, date, subject."""
    record = index.find_commit(sha)
    if record is None:
        return b"no commit " + sha.encode("utf-8") + b" in the corpus"
    date = (record.get("date") or "")[:10]
    detail = f"commit {record['sha']} by {record.get('author', 'unknown')} on {date}: {_first_line(record['message'], 80)}"
    return detail.encode("utf-8")


def tool_summarize_readme(index: CorpusIndex, repo: str) -> bytes:
    """Real README statistics computed from the actual document."""
    record = index.readmes.get(repo)
    if record is None:
        return b"no README collected for " + repo.encode("utf-8")
    content = record.get("content", "")
    headings = [line for line in content.split("\n") if line.startswith("#")]
    code_blocks = content.count("```") // 2
    heading = _first_line(headings[0].lstrip("# ").strip() if headings else (record.get("description") or repo), 60)
    detail = (f"{repo} README: {heading}. {len(headings)} headings, {code_blocks} code blocks, "
              f"{len(content)} characters, licence {record.get('license', 'NOASSERTION')}")
    return detail.encode("utf-8")


TOOLS = {
    "search_issues": tool_search_issues,
    "review_pr": tool_review_pr,
    "explain_commit": tool_explain_commit,
    "summarize_readme": tool_summarize_readme,
}


# --------------------------------------------------------------- converters
def _trajectory(request: str, thought: str, tool: str, args: tuple[str, ...],
                observation: bytes, answer: str, limit: int = DEFAULT_LIMIT) -> dict:
    body = b"\n".join((
        thought.encode("utf-8"),
        format_action(tool.encode("utf-8"), tuple(arg.encode("utf-8") for arg in args)),
        OBSERVATION + b" " + format_observation(observation, limit),
        ANSWER + b" " + " ".join(answer.split()).encode("utf-8"),
    ))
    request_bytes = " ".join(request.split()).encode("utf-8")
    text = render_trajectory(request_bytes, body)
    # latin-1, not utf-8: the training corpus stores byte strings this way so
    # that ``text.encode("latin-1")`` recovers the exact bytes the model will
    # see. Multi-byte utf-8 sequences survive as byte sequences -- which is the
    # point of a byte-level model -- instead of being collapsed to U+FFFD.
    return {"text": text.decode("latin-1"), "tool": tool, "bytes": len(text)}


def _quote_safe(text: str) -> str:
    """Arguments are re-parsed from the rendered text, so they cannot contain quotes."""
    return " ".join(text.replace('"', "'").split())


def _issue_example(index: CorpusIndex, record: dict, limit: int) -> dict | None:
    title = _quote_safe(_first_line(record["title"], 90))
    query = title
    observation = tool_search_issues(index, query)
    if observation.startswith(b"no issue"):
        return None  # no verifiable observation -> no training example
    return _trajectory(
        request=f"Find issues similar to: {title}",
        thought="The user reports a problem; I should look for the closest issues in the corpus.",
        tool="search_issues", args=(query,),
        observation=observation,
        answer=f"Related issue in {record['source']}: {_first_line(record['title'], 80)}",
        limit=limit)


def _pr_example(index: CorpusIndex, record: dict, limit: int) -> dict | None:
    identifier = f"{record['source']}#{record['number']}"
    observation = tool_review_pr(index, identifier)
    if observation.startswith(b"no pull request"):
        return None
    state = "merged" if record.get("merged") else record.get("state", "open")
    return _trajectory(
        request=f"Review pull request {identifier}: {_quote_safe(_first_line(record['title'], 80))}",
        thought="A pull request needs a review; I should read what it changes and how it stands.",
        tool="review_pr", args=(identifier,),
        observation=observation,
        answer=f"{identifier} is {state} with {record.get('comments', 0)} comments.",
        limit=limit)


def _commit_example(index: CorpusIndex, record: dict, limit: int) -> dict | None:
    observation = tool_explain_commit(index, record["sha"])
    if observation.startswith(b"no commit"):
        return None
    subject = _quote_safe(_first_line(record["message"], 80))
    return _trajectory(
        request=f"Explain commit {record['sha']}: {subject}",
        thought="The user quotes a commit; I should look up who made it and what it says.",
        tool="explain_commit", args=(record["sha"],),
        observation=observation,
        answer=f"{record['sha']}: {subject}",
        limit=limit)


def _readme_example(index: CorpusIndex, record: dict, limit: int) -> dict | None:
    observation = tool_summarize_readme(index, record["source"])
    if observation.startswith(b"no README"):
        return None
    description = record.get("description") or record["source"]
    return _trajectory(
        request=f'Summarize the project {record["source"]}.',
        thought="The user asks about a repository; I should read its README before answering.",
        tool="summarize_readme", args=(record["source"],),
        observation=observation,
        answer=f"{record['source']}: {_first_line(description, 90)}",
        limit=limit)


BUILDERS = {"issue": _issue_example, "pr": _pr_example,
            "commit": _commit_example, "readme": _readme_example}


def copyable_report(examples: Sequence[dict]) -> dict:
    """How many action arguments appear verbatim in the request.

    A tool argument that cannot be recovered from the request is unlearnable
    from the request alone: the model would have to invent it. Reporting the
    fraction that *is* recoverable states what the task actually asks for.
    """
    checked = total = 0
    for example in examples:
        raw = example["text"].encode("latin-1")
        request = raw.split(b"\n", 1)[0]
        call = last_action(raw)
        if call is None:
            continue
        for argument in call.args:
            total += 1
            checked += int(argument in request)
    return {"arguments": total, "recoverable_from_request": checked,
            "fraction": round(checked / total, 4) if total else 0.0}


def github_to_react(github_data_file: str = DEFAULT_INPUT, output_file: str = DEFAULT_OUTPUT, *,
                    observation_limit: int = DEFAULT_LIMIT, max_request_bytes: int = 300,
                    max_text_bytes: int = 1200) -> list[dict]:
    """Convert collected stories into ReAct sequences whose observations are real."""
    index = CorpusIndex.from_jsonl(github_data_file)
    examples: list[dict] = []
    skipped = 0
    for kind, records in (("issue", index.issues), ("pr", index.prs),
                          ("commit", index.commits), ("readme", list(index.readmes.values()))):
        for record in records:
            example = BUILDERS[kind](index, record, observation_limit)
            if example is None:
                skipped += 1
                continue
            text = example["text"]
            if len(text.encode("utf-8")) > max_text_bytes:
                skipped += 1
                continue
            example["kind"] = kind
            example["source"] = record.get("source", "")
            examples.append(example)

    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    stats = {
        "input": str(github_data_file),
        "output": str(output_file),
        "examples": len(examples),
        "skipped_no_real_observation_or_too_long": skipped,
        "by_kind": {kind: sum(1 for e in examples if e["kind"] == kind) for kind in BUILDERS},
        "total_bytes": sum(example["bytes"] for example in examples),
        "median_bytes": sorted(e["bytes"] for e in examples)[len(examples) // 2] if examples else 0,
        "longest_bytes": max((e["bytes"] for e in examples), default=0),
        "tools": sorted(TOOLS),
        "copyable_arguments": copyable_report(examples),
        "observation_source": ("every Observation is the return value of a corpus tool executed "
                               "at conversion time; nothing is synthesised"),
    }
    target.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return examples


def verify(examples: Sequence[dict], index: CorpusIndex, count: int = 50) -> dict:
    """Re-execute the tool for a sample and require the stored observation to match."""
    import random

    sample = list(examples)
    random.Random(17017).shuffle(sample)
    sample = sample[:count]
    mismatches = 0
    for example in sample:
        text = example["text"].encode("latin-1")
        call = last_action(text)
        if call is None:
            mismatches += 1
            continue
        name, args = call.tool, call.args
        tool = TOOLS.get(name.decode("utf-8", "replace"))
        if tool is None:
            mismatches += 1
            continue
        expected = format_observation(tool(index, *(arg.decode("utf-8") for arg in args)), DEFAULT_LIMIT)
        if OBSERVATION + b" " + expected not in text:
            mismatches += 1
    if mismatches:
        raise ValueError(f"{mismatches}/{len(sample)} observations are not reproducible from the corpus")
    return {"checked": len(sample), "mismatches": mismatches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--observation-limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-text-bytes", type=int, default=1200)
    parser.add_argument("--verify", type=int, default=50, help="re-run N tools and require a match")
    args = parser.parse_args()

    examples = github_to_react(args.input, args.output,
                               observation_limit=args.observation_limit,
                               max_text_bytes=args.max_text_bytes)
    if examples and args.verify:
        index = CorpusIndex.from_jsonl(args.input)
        print(json.dumps(verify(examples, index, args.verify)))


if __name__ == "__main__":
    main()
