"""Build the three Phase 22 Colab notebooks with Python file I/O.

Run ``python -m notebook.build_phase22_notebooks`` after editing any cell.
The notebooks are plain nbformat-4 JSON; no IPython magics are used so every
code cell can be ``compile()``d in the offline test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def markdown(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(keepends=True)}


def notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    for index, cell in enumerate(cells):
        cell["id"] = f"phase22-cell-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = r'''
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(os.environ.get("AIRESEARCH_REPO", "/content/airesearch"))
if not (REPO / ".git").exists():
    subprocess.check_call(["git", "clone", "https://github.com/gabrielbosse1/airesearch.git", str(REPO)])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "datasets>=3,<5", "huggingface_hub>=0.25,<2", "pyyaml", "tqdm"])
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
print("repo", REPO)
'''

GPU = r'''
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print("GPU", torch.cuda.get_device_name(DEVICE),
          "capability", torch.cuda.get_device_capability(DEVICE),
          "bf16_supported", torch.cuda.is_bf16_supported())
else:
    print("WARNING: no CUDA device. This notebook must run on a Colab T4 GPU for real training.")
'''

CONFIG = r'''
from models.foundation import CausalByteTransformer
from training.phase22_common import (estimate_vram, load_config, model_config,
                                     resolve_precision, vram_fill_ladder, write_report)

CONFIG_PATH = "configs/t4_scaled.yaml"
cfg = load_config(CONFIG_PATH)
with torch.device("meta"):
    PARAMETERS = CausalByteTransformer(model_config(cfg)).parameters_count
print("parameters", PARAMETERS, "(target 150-200M)")
assert 150_000_000 <= PARAMETERS <= 200_000_000
est = estimate_vram(cfg, batch_size=4, seq_len=1024)
print("analytic VRAM estimate (GB)", est["estimated_total_gb"])
print("formula", est["formula"])
'''

PRECISION = r'''
plan = resolve_precision(DEVICE, cfg["training"]["requested_precision"],
                         cfg["training"]["precision"])
print(json.dumps(plan.as_dict(), indent=2))
# T4 = Turing sm_75: CUDA bf16 is NOT supported. `auto` falls back to fp16 +
# GradScaler. Both are 2-byte formats, so the VRAM budget is unchanged.
model = CausalByteTransformer(model_config(cfg)).to(DEVICE)
ladder = vram_fill_ladder(cfg, device=DEVICE, model=model, plan=plan)
print(json.dumps({"target_band_gb": ladder["target_band_gb"],
                  "default_estimate_gb": ladder["default_estimate_gb"],
                  "selected": ladder["selected"],
                  "target_met": ladder["target_met"],
                  "measured": ladder["measured"]}, indent=2))
write_report("runs/phase22/vram_fill_ladder.json", ladder)
if not ladder["measured"]:
    print("VRAM is UNMEASURED here; Colab will probe every candidate.")
print(json.dumps([{k: t[k] for k in ("micro_batch_size", "seq_len",
                                     "gradient_checkpointing", "estimated_gb",
                                     "peak_allocated_gb")} for t in ladder["trials"]], indent=2))
# Free the probe model before Phase22Trainer allocates its own copy.
del model
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
'''

HF_LOGIN = r'''
from huggingface_hub import login

# Set these in Colab secrets / environment. The checkpoint repo is REQUIRED for
# robust resume when the Colab VM is recycled.
HF_REPO = os.environ.get("HF_REPO", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN)
cfg["checkpoints"]["hf_repo"] = HF_REPO or None
cfg["checkpoints"]["resume_from_hf"] = bool(HF_REPO)
print("HF checkpoint repo", cfg["checkpoints"]["hf_repo"] or "(not set - local resume only)")
'''

DISCOVER = r'''
from data.phase22_sources import (HF_SEARCH_QUERIES, SourceRegistry, discover,
                                  stream_phase1)
from training.phase22_common import write_report

findings = {}
for query in HF_SEARCH_QUERIES[:8]:
    try:
        findings[query] = discover(query, limit=5)
    except Exception as exc:
        findings[query] = {"error": f"{type(exc).__name__}: {exc}"}
write_report("runs/phase22/hf_discovery.json", findings)
print("searched", len(findings), "queries")

registry = SourceRegistry()
sample = []
for source_id, raw in stream_phase1(cfg, limit_rows_per_source=2, registry=registry):
    sample.append({"source": source_id, "bytes": len(raw),
                   "preview": raw[:160].decode("utf-8", errors="replace")})
    if len(sample) >= 12:
        break
manifest = registry.manifest(cfg["data"]["license_manifest"],
                             extra={"phase": 1, "note": "streaming probe"})
print(json.dumps(sample, indent=2, ensure_ascii=False))
print(json.dumps({"sources": [r.as_dict() for r in registry.records]}, indent=2))
'''

PHASE1_TRAIN = r'''
from training.phase22_train import Phase22Trainer

MAX_STEPS = int(os.environ.get("PHASE1_MAX_STEPS", "0")) or None
trainer = Phase22Trainer(cfg, phase=1, device=DEVICE, data_mode="hub",
                         max_steps=MAX_STEPS)
report = trainer.run()
print(json.dumps({"steps": report["steps_completed"],
                  "resumed": report["resumed"],
                  "precision": report["precision"],
                  "parameters": report["parameters"],
                  "vram_measured": report["vram_measured"],
                  "checkpoint": report["checkpoint"]["path"],
                  "hf_upload": report["checkpoint"]["hf"],
                  "last_loss": report["last_loss"]}, indent=2))
'''

PHASE1_VERIFY = r'''
from training.phase22_common import hf_upload, save_deploy_copy

ck = cfg["checkpoints"]
path = trainer._checkpoint_path()
assert path.is_file(), f"missing {path}"
print("checkpoint_phase1.pt bytes", path.stat().st_size)
deploy = save_deploy_copy(path.with_name("checkpoint_phase1_fp16.pt"), path)
print("deploy copy", deploy)
if ck.get("hf_repo"):
    print("upload", hf_upload(path, ck["hf_repo"],
                              f"{ck['hf_path_in_repo']}/checkpoint_phase1.pt"))
print("To RESUME: re-run this notebook. Phase22Trainer restores weights, AdamW,")
print("GradScaler, RNG and the lossless byte-packer carry from local disk or the Hub.")
'''

PHASE2_PREP = r'''
from training.phase22_common import hf_download, load_config, load_checkpoint

cfg = load_config(CONFIG_PATH)
cfg["checkpoints"]["hf_repo"] = HF_REPO or None
ck = cfg["checkpoints"]
phase1_path = pathlib.Path(ck["local_dir"]) / ck["phase1_name"]
print("fetch phase 1", hf_download(ck.get("hf_repo"),
                                   f"{ck['hf_path_in_repo']}/{ck['phase1_name']}",
                                   phase1_path))
assert phase1_path.is_file(), "Run notebook 01 first, or set HF_REPO to resume from the Hub."
'''

PHASE2_DATA = r'''
from data.phase22_sources import (SourceRegistry, minecraft_phase_coverage,
                                  stream_phase2)

registry = SourceRegistry()
transcripts = []
for transcript, prefix in stream_phase2(cfg, limit_rows_per_source=8, registry=registry):
    transcripts.append(transcript)
print("phase2 samples", len(transcripts))
print("ReAct labels", all(b"Thought:" in t and b"Action:" in t and
                          b"Observation:" in t and b"Answer:" in t
                          for t in transcripts if b"Provenance:" in t and b"Ladder:" in t))
print("phase ladder coverage", minecraft_phase_coverage(transcripts))
registry.manifest(cfg["data"]["license_manifest"],
                  extra={"phase": 2, "note": "alignment sources + ReAct conversion"})
'''

PHASE2_TRAIN = r'''
from training.phase22_train import Phase22Trainer

MAX_STEPS = int(os.environ.get("PHASE2_MAX_STEPS", "0")) or None
trainer = Phase22Trainer(cfg, phase=2, device=DEVICE, data_mode="hub",
                         max_steps=MAX_STEPS, init_from_phase=1,
                         strict_phase_order=True)
report = trainer.run()
print(json.dumps({"steps": report["steps_completed"],
                  "initialisation": report["initialisation"],
                  "resumed": report["resumed"],
                  "checkpoint": report["checkpoint"]["path"],
                  "last_loss": report["last_loss"]}, indent=2))
'''

PHASE2_VERIFY = r'''
path = trainer._checkpoint_path()
assert path.is_file(), f"missing {path}"
print("checkpoint_phase2.pt bytes", path.stat().st_size)
ck = cfg["checkpoints"]
if ck.get("hf_repo"):
    from training.phase22_common import hf_upload
    print("upload", hf_upload(path, ck["hf_repo"],
                              f"{ck['hf_path_in_repo']}/checkpoint_phase2.pt"))
print("To RESUME: re-run notebook 02. Phase 1 weights are loaded first, then any")
print("checkpoint_phase2.pt state overrides them.")
'''

PHASE3_PREP = r'''
from training.phase22_common import hf_download, load_config

cfg = load_config(CONFIG_PATH)
cfg["checkpoints"]["hf_repo"] = HF_REPO or None
ck = cfg["checkpoints"]
phase2_path = pathlib.Path(ck["local_dir"]) / ck["phase2_name"]
print("fetch phase 2", hf_download(ck.get("hf_repo"),
                                   f"{ck['hf_path_in_repo']}/{ck['phase2_name']}",
                                   phase2_path))
assert phase2_path.is_file(), "Run notebook 02 first, or set HF_REPO to resume from the Hub."
'''

PHASE3_REWARD = r'''
from training.phase22_reward import dominance_report

dom = dominance_report()
print(json.dumps({"ordering": dom["ordering"],
                  "all_satisfied": dom["all_satisfied"],
                  "weights": dom["weights"]}, indent=2))
assert dom["all_satisfied"], "Search>Guess / EfficientEdit>Thrashing must hold"
'''

PHASE3_TRAIN = r'''
from training.phase22_train import Phase22Trainer

MAX_STEPS = int(os.environ.get("PHASE3_MAX_STEPS", "0")) or None
trainer = Phase22Trainer(cfg, phase=3, device=DEVICE, data_mode="hub",
                         max_steps=MAX_STEPS, init_from_phase=2,
                         strict_phase_order=True)
report = trainer.run()
print(json.dumps({"steps": report["steps_completed"],
                  "initialisation": report["initialisation"],
                  "mean_reward": report.get("mean_reward"),
                  "dominance": report.get("dominance", {}).get("ordering"),
                  "checkpoint": report["checkpoint"]["path"]}, indent=2))
'''

PHASE3_FINAL = r'''
path = trainer._checkpoint_path()
assert path.is_file(), f"missing {path}"
print("final_model.pt bytes", path.stat().st_size)
ck = cfg["checkpoints"]
if ck.get("hf_repo"):
    from training.phase22_common import hf_upload
    print("upload", hf_upload(path, ck["hf_repo"],
                              f"{ck['hf_path_in_repo']}/final_model.pt"))

# De-contaminated generalisation evaluation against the real model.
result = subprocess.run(
    [sys.executable, "-m", "evaluation.final_agent_eval",
     "--checkpoint", str(path), "--shard-dir", cfg["data"]["cache_dir"],
     "--root", ".", "--out", "runs/phase22/final_eval.json"],
    capture_output=True, text=True)
print(result.stdout)
print(result.stderr[-2000:])
eval_report = json.loads(pathlib.Path("runs/phase22/final_eval.json").read_text(encoding="utf8"))
print(json.dumps({"oracle_mode": eval_report["oracle_mode"],
                  "pass_rate": eval_report["pass_rate"],
                  "decontamination": eval_report["decontamination"]["status"],
                  "case_names": [r["name"] for r in eval_report["results"]]}, indent=2))
'''


def build() -> list[str]:
    specs = [
        ("01_setup_and_pretrain.ipynb", [
            markdown("""# Phase 22 / Notebook 01 -- Setup, VRAM scaling, Phase 1 pretraining

**Goal:** train a 183,830,791-parameter byte-level foundation on REAL Hugging Face
corpora (raw next-byte causal LM, no prompts, no ReAct), on a Colab T4.

Pipeline: `01` (this notebook) -> `02_distill_and_align.ipynb` -> `03_rl_and_agent.ipynb`.
Checkpoints: `checkpoint_phase1.pt`, `checkpoint_phase2.pt`, `final_model.pt`
(saved locally AND to the Hub when `HF_REPO` is set).

**T4 precision warning:** `torch.bfloat16` CUDA autocast is **not** supported on
Turing/sm_75. `precision: auto` resolves to bfloat16 on Ampere+ and to float16 +
`GradScaler` on T4. The analytic budget shows ~3.97 GB at B=4/L=1024 with full
gradient checkpointing, so a measured fill ladder grows token throughput toward
the 10-12 GB target instead of faking utilisation.

**Resume:** every cell is safe to re-run. `Phase22Trainer` restores weights,
AdamW, GradScaler, RNG and the lossless byte-packer carry; Hub checkpoints are
downloaded automatically if the Colab VM was recycled."""),
            code(SETUP), code(GPU), markdown("## Architecture and VRAM budget"),
            code(CONFIG), code(PRECISION), markdown("## Hugging Face access + dataset discovery"),
            code(HF_LOGIN), code(DISCOVER), markdown("## Phase 1: raw next-byte causal LM"),
            code(PHASE1_TRAIN), code(PHASE1_VERIFY),
            markdown("""## Checkpoint / resume contract

`runs/phase22/checkpoints/checkpoint_phase1.pt` contains `schema=3` state:
fp32 master weights, AdamW moments, GradScaler, Python/Torch/CUDA RNG state, the
`BytePacker` carry (so no UTF-8 codepoint is split or lost across a restart) and
the dataset provenance manifest. `checkpoint_phase1_fp16.pt` is the 2-byte
inference-only copy. Set `HF_REPO` and re-run the notebook to resume after a
Colab disconnect."""),
        ]),
        ("02_distill_and_align.ipynb", [
            markdown("""# Phase 22 / Notebook 02 -- Phase 2 alignment distillation

Loads `checkpoint_phase1.pt`, then distils high-quality conversation and REAL
Minecraft trajectory data into `Thought -> Action -> Observation -> Answer`
byte transcripts, and saves `checkpoint_phase2.pt` locally + on the Hub.

Sources: `HuggingFaceH4/no_robots` (CC BY-NC 4.0 -- non-commercial, attribute),
`TucanoBR/Tucano-SFT` (PT-BR), `m-a-p/CodeFeedback-Filtered-Instruction`,
`TESS-Computer/minecraft-vla-stage3` and `vanyacohen/MET-Bench-Minecraft`.
There is **no verified HF dataset of human-authored textual Minecraft ReAct
trajectories**; the converter preserves real source actions/states and emits an
explicit `[source: no textual observation]` marker rather than inventing one.

Prompt/header bytes are masked from the loss (only the assistant / ReAct
continuation is trained), and the exact UTF-8 packing carry is checkpointed."""),
            code(SETUP), code(GPU), markdown("## Load the Phase 1 weights (strict phase order)"),
            code(HF_LOGIN), code(PHASE2_PREP), markdown("## Verify the alignment sources"),
            code(PHASE2_DATA), markdown("## Phase 2 training"),
            code(PHASE2_TRAIN), code(PHASE2_VERIFY),
            markdown("""## Resume

Re-running this notebook loads `checkpoint_phase1.pt` first (strict phase order),
then restores `checkpoint_phase2.pt` if it exists and skips completed steps.
If the VM is recycled, set `HF_REPO` and the previous checkpoints are fetched
from the Hub automatically."""),
        ]),
        ("03_rl_and_agent.ipynb", [
            markdown("""# Phase 22 / Notebook 03 -- Phase 3 sandbox RL + final agent

Loads `checkpoint_phase2.pt`, runs REINFORCE / PPO-lite with a running EMA
baseline and a deterministic `search_ddg` tool sandbox, saves `final_model.pt`,
then runs `evaluation/final_agent_eval.py` with de-contamination checks.

Reward: `+1` correct final answer, `-0.5` per `[EDIT]` token, `+0.8` for a
verified 1-2 edit fix, `-2.0` for a wrong answer with no search, `+1.5` for
`Action: search_ddg("...")` **and** using its `Observation:` result.
`[PAST_ACTION: executed]` is immutable; only `[CURRENT_DRAFT: editable]` may
contain edits. This encodes **Search > Guess** and **Efficient Edit > Thrashing**."""),
            code(SETUP), code(GPU), markdown("## Load Phase 2 weights + verify the reward ordering"),
            code(HF_LOGIN), code(PHASE3_PREP), code(PHASE3_REWARD),
            markdown("## Phase 3 RL"), code(PHASE3_TRAIN),
            markdown("## Final model + de-contaminated evaluation"), code(PHASE3_FINAL),
            markdown("""## Resume + honest reporting

Re-running loads `checkpoint_phase2.pt` first, then restores `final_model.pt`
state and skips completed RL steps. The evaluation report records
`oracle_mode` (scripted scorer proof) separately from real model generation,
and decontamination reports `unverified` unless Phase 22 training shards are
present to scan. Neither VRAM utilisation nor model quality is claimed in this
sandbox -- both must be measured here on the T4."""),
        ]),
    ]
    written: list[str] = []
    for filename, cells in specs:
        path = ROOT / filename
        path.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n",
                        encoding="utf8")
        written.append(str(path))
    return written


if __name__ == "__main__":
    for item in build():
        print(item)
