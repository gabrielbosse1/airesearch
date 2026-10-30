"""Collect known raw-byte validation/test records for memory guards."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def synthetic_holdouts(processed_dir: str | Path, *, max_length: int | None = None,
                       require: bool = True) -> list[dict[str, Any]]:
    """Read validation/test bytes; fail if expected files are missing by default.

    This registry is only as complete as its inputs. Exact or substring hashes
    cannot detect paraphrases, unknown datasets or model-weight memorization.
    """
    result = []
    for split in ("val", "test"):
        path = Path(processed_dir) / f"{split}.jsonl"
        if not path.exists():
            if require:
                raise FileNotFoundError(f"missing held-out dataset: {path}")
            continue
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    source = bytes(row["bytes"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid raw bytes in {path}:{line_number}") from exc
                result.append({"input": source})
                # The data loader may truncate a long sequence before inference.
                if max_length is not None and len(source) > max_length:
                    result.append({"input": source[:max_length]})
    return result
