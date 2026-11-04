# Phase 21 — collapse forensics toolkit

This directory diagnoses the Phase 19 evaluation collapse (all probes FAIL
except `edit`) and tests one causal fix. See **`root_cause_analysis.md`** for the
report and **`../runs/phase21/`** for the generated evidence.

## The single most important caveat

The Phase 19 Colab runtime is gone. `data_cache/`, `checkpoints/`,
`final_model/`, `final_model_fp16.pt` and the tqdm-only training log are
**gitignored** and are not in this checkout. Every script therefore does two
things:

1. works against real Colab artifacts when you supply them, and
2. otherwise runs a **deterministic CPU replay** of the exact in-repo pipeline
   and labels the source explicitly as `LOCAL REPLAY — NOT the Colab run`.

No number in `runs/phase21/` is claimed to be a measurement of the 4-hour T4
run. Source-level facts (mixture ratios, prompt equality, freeze counts,
encoding) are exact; quality measurements are mechanistic evidence.

## Run everything

```bash
# from the repository root, with torch/numpy/matplotlib/pytest available
python -m debug.run_all                 # ~4 min: full A/B replay + all probes
python -m debug.run_all --from-existing # reuse existing replay results

python -m pytest tests -q              # full suite (360 passed, 1 skipped)
```

## Individual steps

| Task | Command | Main output |
|---|---|---|
| 21.1 log autopsy | `python -m debug.analyze_training_logs [--logs colab.log]` | `training_logs*.json`, `loss_curves_*.png`, `.md` |
| 21.2 stage checkpoints | `python -m debug.check_stage_checkpoints [--checkpoints-dir DIR\|--repo-id user/repo]` | `stage_checkpoints.json/.md` |
| 21.3 data corruption | `python -m debug.verify_training_data [--samples data_cache/samples.json]` | `training_data.json/.md` |
| 21.4 encoding | `python -m debug.test_byte_encoding` | `byte_encoding.json` (27/27 checks) |
| 21.5 weights/gradients | `python -m debug.analyze_model_weights [--checkpoint PATH]` | `model_weights.json/.md` |
| 21.6 activations | `python -m debug.probe_activations [--checkpoint PATH]` | `activations.json/.md` |
| 21.7 edit mystery | `python -m debug.investigate_edit_success` | `edit_mystery.json/.md` |
| 21.9 recovery A/B | `python -m debug.quick_recovery [--from-existing]` | `quick_recovery.json/.md` |
| replay engine | `python -m debug.replay_training --variant baseline\|fixed --steps 300` | `{variant}_metrics.jsonl`, `{variant}_stage{1..4}.pt` |
| docs sync | `python -m debug.update_docs` | appends the Phase 21 section to `AGENTS.md`, `QWEN_CHAT_PROMPT.md`, `README.md` |

`--repo-id` needs `HF_TOKEN` in the environment and `huggingface_hub`; the token
is never printed or stored. `torch.load` uses `weights_only=False` because
resume snapshots carry RNG state — **only load checkpoints you trust**.

## The fix

`notebook/train.py::stage_mixture()` (default) replaces the raw stage weights:

* multiply each group's weight by `min(1, unique_rows / 64)`, renormalise;
* reserve 40% of Stage 3/4 updates for a Stage-1 `pt`/`en`/`code` rehearsal.

`train(..., legacy_mix=True)` reproduces the broken Phase 19 curriculum for A/B,
and `train(..., metrics_path=...)` persists per-step loss/LR/grad-norm/NaN so the
next run does not need a forensic replay. `debug/replay_training.py` calls the
real `training.draw`, `preprocess.batch`, `evaluate.CASES` and `tools.reward`.

## Results at a glance

* Encoding: **27/27** checks pass; `"Olá" ↔ [79,108,195,161]`. Not the cause.
* Data: 7,939 rows, **0** hard-corrupted, **0** pipeline failures (85 low-density
  source-level mojibake flags, 1.1%). Not the cause.
* Weights/activations: no NaN/Inf, no dead neurons, valid softmax. Not the cause.
* **Collapse stage 3**: held-out Portuguese loss 2.713 → 4.507 (+66.2%) while
  training loss falls to 1.58 (template memorisation).
* Stage 3/4 give 30%/50% of updates to groups of **4 unique rows** (667×/588×
  exposure imbalance), with 8/12 blocks frozen and the tied byte embedding
  fully trainable.
* `edit` PASS = verbatim reproduction of a 45-character training suffix; the
  Stage 4 REINFORCE prompt is byte-identical to the eval prompt.
* Fix A/B: worst held-out loss **−39.3%** (4.507 → 2.736), final held-out loss
  −11.3%, forgetting ratio 1.802 → 1.233, coherence +8.7%, Portuguese word hits
  1 → 7, and no Stage-3 spike. Full retrain still required; the surrogate is not
  fluent and the strict gibberish flag is unchanged in this run.
