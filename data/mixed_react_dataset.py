"""Mix the synthetic ReAct corpus with the real GitHub trajectories (Phase 17).

Two things are being combined, and they are combined deliberately:

* **Phase 16 synthetic trajectories** -- generated with a real sandbox, with
  guarded disjoint pools. They teach the *shape* of a ReAct exchange: thought,
  action, observation, answer -- and they do it for tools whose behaviour is
  fully known (``exec_python``, ``read_file``, ``write_file``).

* **Phase 17 real trajectories** -- built by :mod:`data.github_to_react` from
  4,476 records pulled from ten public repositories. Every observation in them
  is the return value of a corpus tool that was really executed, so they teach
  something the synthetic data cannot: that an argument is a *lookup key* into
  a world the model does not control, and that the answer must be grounded in
  what came back.

The GitHub rows are split **by record, with a held-out tail**, so the held-out
set contains requests the model has never trained on. That is the only split
which tests anything: an identical-distribution split of a 2 MB corpus would
be answered by memorisation.

Usage::

    python -m data.mixed_react_dataset --output-dir runs/phase17
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from data.scalable_react_dataset import Trajectory, dataset_stats, write_jsonl

DEFAULT_SYNTHETIC_DIR = "runs/phase16"
DEFAULT_GITHUB = "data/github_raw/github_react.jsonl"
DEFAULT_OUTPUT_DIR = "runs/phase17"
SEED = 17017

# GitHub rows are tagged with their own kind so the report can separate them
# from the synthetic corpus instead of averaging the two together.
GITHUB_KINDS = ("github_issue", "github_pr", "github_commit", "github_readme")
_KIND_FOR = {"issue": "github_issue", "pr": "github_pr",
             "commit": "github_commit", "readme": "github_readme"}


def load_trajectories(path: str | Path) -> list[Trajectory]:
    """Read a ``text``-format JSONL file written by :func:`write_jsonl`."""
    rows: list[Trajectory] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            text = payload["text"].encode("latin-1")
            rows.append(Trajectory(
                request=payload.get("request", "").encode("latin-1"),
                memory=payload.get("memory", "").encode("latin-1"),
                body=payload.get("body", "").encode("latin-1"),
                kind=payload.get("kind", "unknown"),
                text=text,
            ))
    return rows


def load_github(path: str | Path = DEFAULT_GITHUB) -> list[Trajectory]:
    """Read the converter's output, splitting ``text`` back into prompt + body.

    ``github_react.jsonl`` stores only the rendered ``text``. The trainer needs
    ``request``/``body`` too (the evaluator scores the body), so they are
    recovered here by splitting on the ``Thought:`` line -- the same boundary
    :func:`models.react.render_trajectory` built them from.
    """
    rows: list[Trajectory] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            text = payload["text"]
            head, _, body = text.partition("Thought:")
            # ``text`` already carries the ``Request:`` marker; the trainer and
            # the agent re-add it, so store the bare request.
            request = head.strip()
            if request.startswith("Request:"):
                request = request[len("Request:"):].strip()
            kind = _KIND_FOR.get(payload.get("kind", ""), "github")
            rows.append(Trajectory(
                request=request.encode("latin-1"),
                memory=b"",
                body=("Thought:" + body).strip().encode("latin-1"),
                kind=kind,
                text=text.encode("latin-1"),
            ))
    return rows


def _bucket(key: str) -> float:
    """Stable pseudo-random value in ``[0, 1)`` from a record identifier.

    A hash rather than a shuffle: the same record lands in the same split on
    every run, in every process, without persisting a shuffled index.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def split_github(rows: list[Trajectory], *, validation_fraction: float = 0.10,
                 test_fraction: float = 0.10) -> dict[str, list[Trajectory]]:
    """Record-disjoint split of the real corpus.

    The key is the action's argument, which identifies the underlying record
    (a sha, an ``owner/repo#number``, a query). Grouping by it keeps every
    trajectory for one record on the same side of the split -- otherwise the
    model could be trained on a commit's observation and tested on its request.
    """
    from models.react import last_action

    grouped: dict[str, list[Trajectory]] = {}
    for row in rows:
        call = last_action(row.text)
        key = call.args[0].decode("latin-1") if call and call.args else row.request.decode("latin-1")
        grouped.setdefault(key, []).append(row)

    keys = sorted(grouped)
    ordered = sorted(keys, key=_bucket)
    n_test = int(round(len(ordered) * test_fraction))
    n_val = int(round(len(ordered) * validation_fraction))
    test_keys = set(ordered[:n_test])
    val_keys = set(ordered[n_test:n_test + n_val])

    splits: dict[str, list[Trajectory]] = {"train": [], "validation": [], "test": []}
    for key in keys:
        bucket = "test" if key in test_keys else "validation" if key in val_keys else "train"
        splits[bucket].extend(grouped[key])
    return splits


def build_mixed(synthetic_dir: str | Path = DEFAULT_SYNTHETIC_DIR,
                github_path: str | Path = DEFAULT_GITHUB, *,
                seed: int = SEED, github_repeat: int = 1) -> dict[str, list[Trajectory]]:
    """Synthetic splits plus the record-disjoint real corpus.

    ``github_repeat`` duplicates the real rows inside the training split only.
    At one real row per four synthetic rows the mixture is dominated by the
    synthetic distribution; repeating restores the balance without touching the
    held-out sets, which stay single-copy.
    """
    synthetic_dir = Path(synthetic_dir)
    synthetic = {split: load_trajectories(synthetic_dir / f"react_{split}.jsonl")
                 for split in ("train", "validation", "test")}
    github = split_github(load_github(github_path))

    mixed: dict[str, list[Trajectory]] = {}
    for split in ("train", "validation", "test"):
        rows = list(synthetic[split])
        real = github[split]
        rows.extend(real * (github_repeat if split == "train" else 1))
        random.Random(seed).shuffle(rows)
        mixed[split] = rows

    mixed["_github_only_test"] = github["test"]  # kept for the held-out report
    return mixed


def mixture_report(mixed: dict[str, list[Trajectory]]) -> dict:
    """Per-split statistics, reported separately for real and synthetic rows."""
    report: dict = {}
    for split in ("train", "validation", "test"):
        rows = mixed[split]
        real = [row for row in rows if row.kind in GITHUB_KINDS]
        report[split] = {
            "all": dataset_stats(rows),
            "github": dataset_stats(real),
            "synthetic": dataset_stats([row for row in rows if row.kind not in GITHUB_KINDS]),
            "github_fraction_of_rows": round(len(real) / len(rows), 4) if rows else 0.0,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-dir", default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--github", default=DEFAULT_GITHUB)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--github-repeat", type=int, default=1)
    args = parser.parse_args()

    mixed = build_mixed(args.synthetic_dir, args.github, github_repeat=args.github_repeat)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        write_jsonl(mixed[split], output / f"react_{split}.jsonl")
    # The held-out real rows alone, so the evaluator can score them in isolation.
    write_jsonl(mixed["_github_only_test"], output / "github_heldout.jsonl")

    report = mixture_report(mixed)
    (output / "mixture_stats.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
