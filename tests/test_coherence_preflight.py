"""Phase 25.2 pre-flight contract tests (offline, no 500-step run in pytest)."""
from __future__ import annotations

import copy

import pytest

from coherence_preflight import (CODE_PROMPT, PTBR_PROMPT, evaluate_snapshots,
                                 offline_slice, preflight_config)


def _snapshots() -> dict:
    pt_good = " sou um modelo de linguagem em português do Brasil e posso ajudar com texto e código."
    code_good = "\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n"
    pt_bad = "\x10" * 24
    code_bad = "pppp\x10\x10"
    def snap(loss: float, pt: str, code: str) -> dict:
        return {"loss": loss,
                "ptbr": {"prompt": PTBR_PROMPT, "continuation": pt, "full_text": PTBR_PROMPT + pt},
                "code": {"prompt": CODE_PROMPT, "continuation": code, "full_text": CODE_PROMPT + code}}
    return {0: snap(5.5, pt_bad, code_bad), 250: snap(0.2, pt_good, code_good),
            500: snap(0.05, pt_good, code_good)}


def test_offline_slice_is_1000_ptbr_and_code_examples():
    examples = offline_slice(1000)
    assert len(examples) == 1000
    groups = {group for group, _ in examples}
    assert groups == {"ptbr", "code"}
    assert all(sample["kind"] == "text" for _, sample in examples)
    assert all(isinstance(sample["data"], bytes) for _, sample in examples)
    joined = b"\n".join(sample["data"] for _, sample in examples)
    assert PTBR_PROMPT.encode("utf8") in joined
    assert CODE_PROMPT.encode("utf8") in joined


def test_preflight_config_keeps_phase251_attention_contract():
    config = preflight_config(steps=500)
    model = config["model"]
    assert model["attention_type"] == "sliding_window"
    assert model["window_size"] == 1024
    assert model["memory_cross_attention"] is True
    assert model["n_layers"] == 8
    assert model["gradient_checkpointing"] is True
    assert config["training"]["max_steps"] == 500
    assert config["encoders"] == {"image": None, "audio": None, "video": None}


def test_evaluate_snapshots_passes_when_learning_and_coherent():
    metrics = evaluate_snapshots(_snapshots(), strict=True)
    assert metrics["loss_step_500"] < metrics["loss_step_0"]
    assert metrics["ptbr_markers_step_500"] >= 2
    assert metrics["code_compiles_step_500"] is True
    assert metrics["step0_gibberish"] is True


def test_evaluate_snapshots_rejects_no_learning():
    snapshots = _snapshots()
    snapshots[500]["loss"] = snapshots[0]["loss"] + 1.0
    with pytest.raises(AssertionError, match="did not learn"):
        evaluate_snapshots(snapshots, strict=True)


def test_evaluate_snapshots_rejects_incoherent_ptbr():
    snapshots = _snapshots()
    snapshots[500]["ptbr"]["continuation"] = "\x00\x01\x02" * 10
    snapshots[500]["ptbr"]["full_text"] = PTBR_PROMPT + snapshots[500]["ptbr"]["continuation"]
    with pytest.raises(AssertionError):
        evaluate_snapshots(snapshots, strict=True)


def test_evaluate_snapshots_rejects_invalid_python():
    snapshots = _snapshots()
    snapshots[500]["code"]["continuation"] = "\n    if n < 2\n        return n\n"
    snapshots[500]["code"]["full_text"] = CODE_PROMPT + snapshots[500]["code"]["continuation"]
    with pytest.raises(AssertionError, match="valid Python syntax"):
        evaluate_snapshots(snapshots, strict=True)
