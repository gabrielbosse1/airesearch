"""Strict opt-in configuration for task-scoped adaptive computation."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .epistemic import ConfidenceHead


def configured_threshold(config: Mapping[str, Any] | None, head: ConfidenceHead, scope: str) -> float | None:
    """Resolve a configured threshold; refuse missing or mismatched calibration.

    This check prevents a known scope mismatch, not distribution shift or a
    false claim of safety. The caller must independently validate the policy
    for the checkpoint, input distribution, inference mode, and quality metric.
    Explicit threshold arguments remain an experimental, warning-only API.
    """
    if config is None:
        return None
    if not isinstance(config, Mapping) or not isinstance(config.get("enabled", False), bool):
        raise ValueError("adaptive configuration requires a boolean enabled flag")
    if not config.get("enabled", False):
        return None
    threshold = config.get("confidence_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("enabled adaptive policy requires a threshold in [0, 1]")
    if scope not in head.SCOPES or not bool(head.calibrated.item()) or int(head.calibration_scope.item()) != head.SCOPES[scope]:
        raise ValueError(f"enabled adaptive policy requires confidence calibrated for {scope}")
    return float(threshold)
