"""Where exactly does copying fail? A teacher-forced vs free-running split.

A free-running argument score of zero has two very different explanations, and
which one it is decides whether more training can fix it:

* **The model cannot copy** -- even with the gold prefix in front of it, it does
  not know which byte comes next. Teacher-forced accuracy on the argument bytes
  is low. More training or more capacity might help.
* **The model will not copy when it is on its own** -- teacher-forced accuracy
  is high but free-running is not, i.e. errors compound once the model has to
  live with its own output. That is an exposure problem, not a capacity one.

This script measures the first directly: it takes held-out trajectories, feeds
the *reference* bytes up to each position, and scores only the bytes that lie
between the quotes of the ``Action:`` argument. It does that separately for the
synthetic rows and the real GitHub rows, because their arguments are not the
same difficulty -- `print(6 * 7)` is not a twelve-character hex sha.

Usage::

    python -m evaluation.copy_diagnostic --model runs/phase17/foundation.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.mixed_react_dataset import GITHUB_KINDS, load_trajectories
from models.foundation import encode_bytes, load_foundation

DEFAULT_MODEL = "runs/phase17/foundation.pt"
DEFAULT_ROWS = "runs/phase17/react_test.jsonl"
DEFAULT_OUTPUT = "runs/phase17/copy_diagnostic.json"


def argument_span(text: bytes) -> tuple[int, int] | None:
    """Byte range of the quoted argument on the ``Action:`` line, or ``None``."""
    line = None
    for candidate in text.split(b"\n"):
        if candidate.startswith(b"Action: "):
            line = candidate
    if line is None:
        return None
    start = line.find(b'("')
    if start == -1:
        return None
    start += 2
    end = line.find(b'")', start)
    if end == -1:
        return None
    return start, end


def argument_offset(text: bytes, span: tuple[int, int]) -> int:
    """Where the argument begins inside the whole trajectory."""
    line = None
    for candidate in text.split(b"\n"):
        if candidate.startswith(b"Action: "):
            line = candidate
    return text.index(line) + span[0]


def measure(model, rows, *, label: str) -> dict:
    model.eval()
    hits = total = 0
    digits_hit = digits_total = 0
    exact = considered = 0
    with torch.no_grad():
        for row in rows:
            span = argument_span(row.text)
            if span is None:
                continue
            considered += 1
            begin = argument_offset(row.text, span)
            end = begin + (span[1] - span[0])
            ids = encode_bytes(row.text, bos=True, eos=True)[: model.config.max_seq_len]
            if end + 1 > ids.shape[0]:
                continue
            logits = model(ids[:-1].unsqueeze(0))[0]
            predictions = logits.argmax(dim=-1)
            targets = ids[1:]
            # ids are shifted by one (BOS), and the target for position i is
            # ids[i + 1], so the byte at text offset k is predicted at k.
            window = slice(begin, end)
            chosen = predictions[window]
            wanted = targets[window]
            hits += int((chosen == wanted).sum().item())
            total += int(wanted.shape[0])
            digits = (wanted >= 48) & (wanted <= 57)
            hexish = digits | ((wanted >= 97) & (wanted <= 102))
            digits_hit += int(((chosen == wanted) & hexish).sum().item())
            digits_total += int(hexish.sum().item())
            exact += int(bool((chosen == wanted).all().item()))
    return {
        "label": label,
        "rows_with_an_argument": considered,
        "argument_bytes": total,
        "teacher_forced_argument_byte_accuracy": round(hits / total, 4) if total else None,
        "teacher_forced_hex_or_digit_accuracy": round(digits_hit / digits_total, 4) if digits_total else None,
        "whole_argument_predicted_exactly": f"{exact}/{considered}",
        "note": ("the gold prefix is fed for every position, so this is what the "
                 "model can copy, not what it does copy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rows", default=DEFAULT_ROWS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model = load_foundation(args.model, device="cpu")
    rows = load_trajectories(args.rows)
    report = {
        "model": args.model,
        "rows": args.rows,
        "synthetic": measure(model, [row for row in rows if row.kind not in GITHUB_KINDS],
                             label="Phase 16 synthetic trajectories"),
        "github": measure(model, [row for row in rows if row.kind in GITHUB_KINDS],
                          label="real GitHub trajectories"),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
