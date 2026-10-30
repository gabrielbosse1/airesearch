"""Run regression checks and write a compact Phase 10 verification artifact."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from models.dialogue_response import load_dialogue_decoder


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                           env=env, capture_output=True, text=True, timeout=300)
    if tests.returncode:
        print(tests.stdout + tests.stderr)
        raise SystemExit(tests.returncode)
    match = re.search(r"Ran (\d+) tests? in", tests.stderr)
    if not match or "\nOK\n" not in tests.stderr:
        raise RuntimeError("could not verify the unittest result")
    test_count = int(match.group(1))
    subprocess.run(["git", "diff", "--check"], check=True, env=env)
    subprocess.run([sys.executable, "-m", "compileall", "-q", "models", "training", "data", "evaluation", "demo", "tests"], check=True, env=env)
    bad = []
    for directory in ("models", "training", "data", "evaluation", "demo"):
        for path in Path(directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "tik" + "token" in text or "Auto" + "Tokenizer" in text:
                bad.append(str(path))
    if bad:
        raise RuntimeError(f"unexpected external tokenizer references: {bad}")
    source_path = Path("runs/phase9/structured_generator.pt")
    generator_path = Path("runs/phase10/generator.pt")
    decoder_path = Path("runs/phase10/response.pt")
    old = torch.load(source_path, map_location="cpu", weights_only=True)
    new = torch.load(generator_path, map_location="cpu", weights_only=True)
    changed = [key for key, value in old.items() if key not in new or not torch.equal(value, new[key])]
    if changed:
        raise RuntimeError(f"Phase 9 generator tensors changed: {changed}")
    if not bool(new["intent_trained"]):
        raise RuntimeError("intent checkpoint is not marked trained")
    load_dialogue_decoder(decoder_path, generator_path=generator_path)
    regression = json.loads(Path("runs/phase10/phase9_regression.json").read_text(encoding="utf-8"))
    if not all(regression[key]["exact"] for key in ("diffusion_native", "cognitive_natural_only", "cognitive_explicit_request")):
        raise RuntimeError("Phase 9 training-example diagnostic regressed")
    if regression["memory"]["existing_memory"]["found"]:
        raise RuntimeError("the real memory unexpectedly contains the diagnostic target")
    benchmark = json.loads(Path("runs/phase10/benchmark.json").read_text(encoding="utf-8"))
    if benchmark["generator_sha256"] != digest(generator_path) or benchmark["decoder_sha256"] != digest(decoder_path):
        raise RuntimeError("benchmark and current checkpoints do not match")
    old_hashes = {
        "runs/model.pt": "7ed9dd3cb5faed61f8e38fe35f5e4943849c55825298100e7b81e4a93c0aebd2",
        "runs/dual_memory/generator.pt": "2122e69c0a272fe3cc5b6d252799f31128df094b44d6b02012e531d5990995d6",
        "runs/dual_memory/memory.pt": "52c1d939758a60ddafad738e3b0d29f7687680c4f5f10726385f3b9befb8995f",
    }
    for path, expected in old_hashes.items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"historical checkpoint changed: {path}")
    training = json.loads(Path("runs/phase10/training_metrics.json").read_text(encoding="utf-8"))
    if training["source_checkpoint_sha256"] != digest(source_path):
        raise RuntimeError("Phase 9 structured source checkpoint changed")
    with tempfile.TemporaryDirectory(prefix="airesearch-cli-") as tmp:
        path = str(Path(tmp) / "memory.pt")
        # The default demo now uses Phase 11. Pin this historical smoke test
        # to its original checkpoints and explicit-memory behavior.
        cmd = [sys.executable, "-m", "demo.conversation", "--legacy-phase10", "--memory", path, "--user", "verification"]
        first = subprocess.run(cmd, input="Remember my hometown is Recife.\n/quit\n", text=True,
                               capture_output=True, env=env, check=True, timeout=60)
        second = subprocess.run(cmd, input="Can you recall my hometown.\n/quit\n", text=True,
                                capture_output=True, env=env, check=True, timeout=60)
        if "I'll remember your hometown" not in first.stdout or "Your hometown is Recife" not in second.stdout:
            raise RuntimeError("cross-process interactive CLI smoke failed")
    report = {
        "tests_total": test_count, "tests_passed": test_count,
        "test_command": "python -m unittest discover -s tests -v",
        "phase9_source_tensors_unchanged": len(old), "phase9_hello_diagnostic": "all three paths exact; example in training",
        "historical_checkpoint_sha256": old_hashes,
        "phase9_structured_generator_sha256": digest(source_path),
        "phase10_generator_sha256": digest(generator_path), "phase10_response_sha256": digest(decoder_path),
        "matching_decoder_strict_load": True, "cli_memory_cross_process_smoke": True,
        "compileall": True, "git_diff_check": True, "external_tokenizer_reference_scan": True,
        "benchmark_matches_checkpoint_hashes": True,
    }
    Path("runs/phase10/verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
