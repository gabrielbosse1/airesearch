"""Full regressions, frozen-backbone parity, checkpoint binding and CLI privacy."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import torch

from models.contextual_response import load_contextual_decoder
from models.diverse_response import load_diverse_decoder
from models.vector_memory import load_vector_reader
from training.finalize_replay import final_guard
from training.replay import ReplayBuffer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = Path("runs/phase12"); out.mkdir(parents=True, exist_ok=True)
    paths = ["runs/model.pt", "runs/dual_memory/generator.pt", "runs/dual_memory/memory.pt",
             "runs/phase9/structured_generator.pt", "runs/phase9/generator.pt", "runs/phase9/diffusion/model.pt",
             "runs/phase10/generator.pt", "runs/phase10/response.pt", "runs/phase11/generator.pt",
             "runs/phase11/response.pt", "runs/real_traces/memory.pt"]
    before = {p: digest(p) for p in paths}
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                           env=env, capture_output=True, text=True, timeout=300)
    (out / "unittest.log").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    if tests.returncode:
        print(tests.stdout + tests.stderr); raise SystemExit(tests.returncode)
    match = re.search(r"Ran (\d+) tests? in", tests.stderr)
    if not match or "\nOK\n" not in tests.stderr:
        raise RuntimeError("could not verify unittest result")
    count = int(match.group(1))
    if count < 147:
        raise RuntimeError("regression suite has fewer than 147 tests")
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
    baseline = torch.load("runs/phase9/structured_generator.pt", map_location="cpu", weights_only=True)
    current = torch.load("runs/phase12/generator.pt", map_location="cpu", weights_only=True)
    changed = [name for name, value in baseline.items() if name not in current or not torch.equal(value, current[name])]
    if changed:
        raise RuntimeError(f"Phase 9 generator tensors changed: {changed}")
    load_contextual_decoder(out / "response.pt", generator_path=out / "generator.pt")
    load_vector_reader(out / "vector_reader.pt", generator_path=out / "generator.pt")
    load_diverse_decoder(out / "diverse_response.pt", generator_path=out / "generator.pt")
    subprocess.run([sys.executable, "-m", "evaluation.phase9_diagnostic", "--structured-generator", str(out / "generator.pt"),
                    "--output", str(out / "phase9_regression.json")], env=env, capture_output=True,
                   text=True, check=True, timeout=180)
    regression = json.loads((out / "phase9_regression.json").read_text(encoding="utf-8"))
    if not all(regression[name]["exact"] for name in ("diffusion_native", "cognitive_natural_only", "cognitive_explicit_request")):
        raise RuntimeError("Phase 9 diagnostic failed")
    if regression["memory"]["existing_memory"]["found"]:
        raise RuntimeError("the real memory unexpectedly contains the diagnostic target")
    benchmark = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    if benchmark["generator_sha256"] != digest(out / "generator.pt"):
        raise RuntimeError("benchmark was produced with a different generator")
    guard = final_guard(Namespace(pairs="data/real_edit_pairs.jsonl", processed_dir="data/processed"))
    replay = ReplayBuffer.load(out / "replay_multitask_phase12.pt", guard=guard)
    if len(replay) != 1000 or replay.capacity != 1000:
        raise RuntimeError("multitask replay checkpoint has unexpected capacity or size")
    with tempfile.TemporaryDirectory(prefix="phase12-cli-") as directory:
        path = str(Path(directory) / "memory.pt")
        cmd = [sys.executable, "-m", "demo.conversation", "--memory", path, "--user", "alice", "--show-intent"]
        first = subprocess.run(cmd, input="My favorite color is blue.\n/quit\n", text=True, capture_output=True,
                               env=env, check=True, timeout=60)
        second = subprocess.run(cmd, input="What is my favorite color?\n/quit\n", text=True, capture_output=True,
                                env=env, check=True, timeout=60)
        other = subprocess.run([sys.executable, "-m", "demo.conversation", "--memory", path, "--user", "bob"],
                               input="What is my favorite color?\n/quit\n", text=True, capture_output=True,
                               env=env, check=True, timeout=60)
        if ("PRIVACY" not in first.stdout or "UNENCRYPTED" not in first.stdout or "Assistant: blue" not in second.stdout
                or "source=vector_reader" not in second.stdout or "vector_only=True" not in second.stdout):
            raise RuntimeError("Phase 12 CLI vector persistence/consent smoke failed")
        if "Assistant: blue" in other.stdout:
            raise RuntimeError("Phase 12 CLI user isolation failed")
        disabled = str(Path(directory) / "disabled.pt")
        subprocess.run([sys.executable, "-m", "demo.conversation", "--memory", disabled, "--no-episode-storage"],
                       input="My favorite color is violet.\n/quit\n", text=True, capture_output=True,
                       env=env, check=True, timeout=60)
        if Path(disabled).exists():
            raise RuntimeError("episode opt-out created a persistent memory file")
        private = str(Path(directory) / "private.pt")
        subprocess.run([sys.executable, "-m", "demo.conversation", "--memory", private],
                       input="Please do not save my password is secret.\n/quit\n", text=True, capture_output=True,
                       env=env, check=True, timeout=60)
        if Path(private).exists():
            raise RuntimeError("explicit no-storage turn created a memory file")
        legacy11 = str(Path(directory) / "phase11.pt")
        cmd11 = [sys.executable, "-m", "demo.conversation", "--legacy-phase11", "--memory", legacy11]
        subprocess.run(cmd11, input="My favorite color is blue.\n/quit\n", text=True, capture_output=True,
                       env=env, check=True, timeout=60)
        old11 = subprocess.run(cmd11, input="What is my favorite color?\n/quit\n", text=True, capture_output=True,
                               env=env, check=True, timeout=60)
        if "blue" not in old11.stdout.lower():
            raise RuntimeError("Phase 11 legacy CLI failed")
        legacy10 = str(Path(directory) / "phase10.pt")
        cmd10 = [sys.executable, "-m", "demo.conversation", "--legacy-phase10", "--memory", legacy10]
        subprocess.run(cmd10, input="Remember my hometown is Recife.\n/quit\n", text=True, capture_output=True,
                       env=env, check=True, timeout=60)
        old10 = subprocess.run(cmd10, input="Can you recall my hometown.\n/quit\n", text=True, capture_output=True,
                               env=env, check=True, timeout=60)
        if "Your hometown is Recife" not in old10.stdout:
            raise RuntimeError("Phase 10 legacy CLI failed")
    after = {p: digest(p) for p in paths}
    if after != before:
        raise RuntimeError("historical checkpoints or real memory changed during verification")
    result = {"tests_total": count, "tests_passed": count, "test_log": str(out / "unittest.log"),
              "phase9_source_tensors_unchanged": len(baseline), "phase9_three_way_diagnostic_exact": True,
              "historical_checkpoint_sha256": after, "historical_files_unchanged_during_verification": True,
              "all_response_checkpoint_bindings_verified": True, "guarded_replay_loaded_and_audited": len(replay),
              "cli_vector_cross_process": True, "cli_unencrypted_warning": True, "cli_user_isolation": True,
              "cli_no_episode_opt_out": True, "cli_no_storage_turn": True, "legacy_phase10_cli": True,
              "legacy_phase11_cli": True, "compileall": True, "git_diff_check": True,
              "external_tokenizer_reference_scan": True}
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
