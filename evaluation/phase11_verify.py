"""Run regressions, checkpoint parity, CLI privacy and persistence checks."""
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

from models.contextual_response import load_contextual_decoder


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = Path("runs/phase11"); out.mkdir(parents=True, exist_ok=True)
    original_paths = ["runs/model.pt", "runs/dual_memory/generator.pt", "runs/dual_memory/memory.pt",
                      "runs/phase9/structured_generator.pt", "runs/phase9/generator.pt", "runs/phase9/diffusion/model.pt",
                      "runs/phase10/generator.pt", "runs/phase10/response.pt", "runs/real_traces/memory.pt"]
    before = {path: digest(path) for path in original_paths}
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                           env=env, capture_output=True, text=True, timeout=300)
    (out / "unittest.log").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    if tests.returncode:
        print(tests.stdout + tests.stderr)
        raise SystemExit(tests.returncode)
    match = re.search(r"Ran (\d+) tests? in", tests.stderr)
    if not match or "\nOK\n" not in tests.stderr:
        raise RuntimeError("could not verify unittest result")
    test_count = int(match.group(1))
    if test_count < 120:
        raise RuntimeError("regression suite has fewer than the required 120 tests")
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
    phase9 = torch.load("runs/phase9/structured_generator.pt", map_location="cpu", weights_only=True)
    phase11 = torch.load("runs/phase11/generator.pt", map_location="cpu", weights_only=True)
    changed = [name for name, value in phase9.items() if name not in phase11 or not torch.equal(value, phase11[name])]
    if changed:
        raise RuntimeError(f"Phase 9 tensors changed: {changed}")
    load_contextual_decoder("runs/phase11/response.pt", generator_path="runs/phase11/generator.pt")
    subprocess.run([sys.executable, "-m", "evaluation.phase9_diagnostic", "--structured-generator", "runs/phase11/generator.pt",
                    "--output", str(out / "phase9_regression.json")], env=env, capture_output=True, text=True,
                   check=True, timeout=180)
    regression = json.loads((out / "phase9_regression.json").read_text(encoding="utf-8"))
    if not all(regression[name]["exact"] for name in ("diffusion_native", "cognitive_natural_only", "cognitive_explicit_request")):
        raise RuntimeError("Phase 9 diagnostic failed")
    if regression["memory"]["existing_memory"]["found"]:
        raise RuntimeError("the real memory unexpectedly contains the diagnostic target")
    benchmark = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    if benchmark["generator_sha256"] != digest("runs/phase11/generator.pt") or benchmark["decoder_sha256"] != digest("runs/phase11/response.pt"):
        raise RuntimeError("benchmark was produced with different checkpoints")
    with tempfile.TemporaryDirectory(prefix="phase11-cli-") as directory:
        path = str(Path(directory) / "memory.pt")
        cmd = [sys.executable, "-m", "demo.conversation", "--legacy-phase11", "--memory", path, "--user", "alice"]
        first = subprocess.run(cmd, input="My favorite color is blue.\n/quit\n", text=True, capture_output=True,
                               env=env, check=True, timeout=60)
        second = subprocess.run(cmd, input="What is my favorite color?\n/quit\n", text=True, capture_output=True,
                                env=env, check=True, timeout=60)
        other = subprocess.run([sys.executable, "-m", "demo.conversation", "--legacy-phase11", "--memory", path, "--user", "bob"],
                               input="What is my favorite color?\n/quit\n", text=True, capture_output=True,
                               env=env, check=True, timeout=60)
        if "PRIVACY" not in first.stdout or "UNENCRYPTED" not in first.stdout or "blue" not in second.stdout.lower():
            raise RuntimeError("Phase 11 CLI persistence/consent notice smoke failed")
        if "blue" in other.stdout.lower():
            raise RuntimeError("Phase 11 CLI user isolation smoke failed")
        disabled = str(Path(directory) / "disabled.pt")
        subprocess.run([sys.executable, "-m", "demo.conversation", "--legacy-phase11", "--memory", disabled, "--no-episode-storage"],
                       input="My favorite color is violet.\n/quit\n", text=True, capture_output=True,
                       env=env, check=True, timeout=60)
        if Path(disabled).exists():
            raise RuntimeError("episode opt-out created a persistent memory file")
        private = str(Path(directory) / "private.pt")
        subprocess.run([sys.executable, "-m", "demo.conversation", "--legacy-phase11", "--memory", private],
                       input="Please do not save my password is secret.\n/quit\n", text=True, capture_output=True,
                       env=env, check=True, timeout=60)
        if Path(private).exists():
            raise RuntimeError("explicit no-storage turn created a memory file")
        legacy = str(Path(directory) / "legacy.pt")
        legacy_cmd = [sys.executable, "-m", "demo.conversation", "--legacy-phase10", "--memory", legacy]
        old_first = subprocess.run(legacy_cmd, input="Remember my hometown is Recife.\n/quit\n", text=True,
                                   capture_output=True, env=env, check=True, timeout=60)
        old_second = subprocess.run(legacy_cmd, input="Can you recall my hometown.\n/quit\n", text=True,
                                    capture_output=True, env=env, check=True, timeout=60)
        if "I'll remember your hometown" not in old_first.stdout or "Your hometown is Recife" not in old_second.stdout:
            raise RuntimeError("legacy Phase 10 CLI smoke failed")
    after = {path: digest(path) for path in original_paths}
    if before != after:
        raise RuntimeError("an earlier checkpoint or real memory changed during verification")
    report = {"tests_total": test_count, "tests_passed": test_count,
              "test_command": f"{sys.executable} -m unittest discover -s tests -v", "test_log": str(out / "unittest.log"),
              "phase9_source_tensors_unchanged": len(phase9), "phase9_three_way_diagnostic_exact": True,
              "phase9_diagnostic_caveat": "the example was in training; explicit-request and stored-memory variants are not general autonomy",
              "historical_checkpoint_sha256": after, "historical_checkpoints_unchanged_during_verification": True,
              "response_checkpoint_binding_verified": True, "benchmark_matches_checkpoints": True,
              "cli_cross_process_retention": True, "cli_privacy_notice": True, "cli_user_isolation": True,
              "cli_no_episode_opt_out": True, "cli_no_storage_turn": True, "phase10_legacy_cli": True,
              "compileall": True, "git_diff_check": True, "external_tokenizer_reference_scan": True}
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
