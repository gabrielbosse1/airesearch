# Phase 21 — Root Cause Analysis: the ByteCore language collapse

**Date:** 2026-11-03 · **Branch:** `arena/9fca27b9-airesearch`
**Verdict:** the gibberish is a **distributional collapse caused by the Stage 3/4
curriculum**, not a numerical or encoding failure. The collapse begins at
**Stage 3** (tool-use / ReAct + search); Stage 4 cements it while leaving one
memorised `edit` template as the only passing probe.

> **Evidence boundary — read this first.** The Phase 19 Colab logs, stage
> checkpoints and `data_cache/` are gitignored and were never uploaded to this
> checkout (`artifact_status()`: `any_colab_artifact: false`). There is no Colab
> log to autopsy and no Colab checkpoint to load. All measurements below come
> from (a) **source-level forensics** of the exact code that ran, and (b) a
> **deterministic CPU replay** of that code at reduced scale
> (`debug/replay_training.py`: 300 optimizer steps/stage, lr 3e-3, accumulation 6,
> seed 19, same `MIXES`, `FREEZE`, cosine LR, AdamW, clipping, prompts and
> rewards; the replay corpus is pinned so it cannot drift as files are added).
> Absolute quality numbers do not transfer to the 4-hour T4 run; the failure
> *mechanism* does. Where real artifacts are re-supplied, every script accepts
> them: `--logs`, `--checkpoints-dir`, `--repo-id`, `--samples`, `--checkpoint`.

---

## 1. Timeline of the failure

| Stage | Mixture (`notebook/train.py`) | Frozen | Held-out PT loss (replay) | 100 bytes from `Olá, como` (greedy) | Probes |
|---|---|---|---|---|---|
| 1 — bilingual foundation | 50/30/20 pt/en/code | 0/12 | 2.945 | spaces / initial byte salad | 0/5 |
| 2 — code + Mineflayer | 40/40/20 code/mineflayer/pt | 6/12 | **2.713 (best)** | valid UTF-8, Portuguese-like fragments | 0/5 |
| **3 — tool use + search** | **40/30/30 react/search/code** | **8/12** | **4.507 (+66.2% vs Stage 2)** | `'dteasis te\nTho lou\nTho tho pe po teu…'` — nonsense | 0/5 |
| 4 — RL + self-edit | **50/20/15/15 edit/pt/code/mineflayer** | 8/12 | 2.501 (partial metric recovery) | repetitive `'tá á co co co … vo vo vo'` | **1/5, only `edit`** |

**Collapse stage: 3.** `debug.check_stage_checkpoints.locate_collapse()`:
`{'stage': 3, 'signal': 'heldout_pt_loss', 'best_prior_stage': 2,
'best_prior_loss': 2.71273, 'collapse_loss': 4.50746, 'ratio': 1.662}`.
The log autopsy independently reports `collapse_stage: 3`,
`worst_over_first_ratio: 1.53`, `signs_of_catastrophic_forgetting: true` for the
baseline and `false` for the fixed run.

The Stage-3 signature is textbook overfitting-plus-forgetting: **training loss
falls to 1.581 while held-out Portuguese loss rises to 4.507**. Stage 4 drives
training loss to 0.886 (four memorised edit templates) while held-out loss
recovers only to 2.501, and the final eval reproduces the reported
"everything FAIL except `edit`".

*Measurement caveat:* the heuristic `coherence` score rises at Stage 3 (0.419 →
0.570) because it rewards UTF-8/printable byte shape, not meaning. The
teacher-forced **held-out Portuguese loss** is the reliable forgetting signal;
we report both and do not treat coherence as a quality benchmark.

---

## 2. Ranked hypotheses

| Rank | Hypothesis | Status | Evidence |
|---|---|---|---|
| **1** | **Dataset mixing ratio wrong → one task dominated** | **[X] PRIMARY CAUSE** | `MIXES[2]` = 30% `search`, `MIXES[3]` = 50% `edit`. Both groups contain **exactly 4 unique rows** (`preprocess.synthetic()`) versus 3,587 real `react` rows, 703 `code` rows and 521 Mineflayer rows. Effective exposures per unique row per 32-sequence optimizer step: Stage 3 `search` **2.40** vs `react` **0.0036** → **666.7× imbalance**; Stage 4 `edit` **4.00** vs `code` **0.0069** → **588.2×**. At ~1,000+ steps per stage, each of those 4 rows is replayed thousands of times. |
| **2** | **Catastrophic forgetting → Stages 3–4 wiped Stages 1–2** | **[X] CO-PRIMARY (the mechanism)** | `FREEZE = [0, 6, 8, 8]` freezes 8 of 12 blocks in Stages 3/4 but leaves `token`, `pos`, `norm` and the **tied output head** trainable. Because `PhoneLM.head.weight is PhoneLM.token.weight`, every late-stage gradient from the 4-row templates rewrites the **shared byte codebook** — the object defining the global distribution over all 256 bytes. Stage 3 has no `pt`/`en` group at all; Stage 4 keeps only 20% `pt`. Held-out PT loss 2.713 → 4.507. |
| **3** | Evaluation contamination / over-trained on edit operations | **[X] CONFIRMED — explains the `edit` PASS** | `RL_CASES` in `notebook/train.py` contains prompts **byte-identical** to 3 of 5 eval cases (`pt-br`, `code`, `edit`). The `edit` RL case uses the same prompt *and* the same `reward('edit', …, 'voce', 'você')` as `notebook/evaluate.py`. The replayed Stage-4 model emits the **45-character training suffix verbatim** (`[EDIT pos=3 delete=1 bytes=c3aa]\nAnswer: você`) and **FAILS** a held-out typo (`obrigadoo`). Stage 4 did dominate with edit data — but with 4 templates, not a diverse edit curriculum. |
| **4** | Insufficient training / random-init byte decoder + greedy decode | **[~] CONTRIBUTING** | The notebook starts from **random weights** (Phase 19.6 warns that four time-boxed stages on a free T4 "cannot guarantee fluency"). A random `PhoneLM` already produces the reported symptom: `'r_8 0[3 0L Æ|=l'` (measured in `debug/test_byte_encoding.py`). `evaluate()` uses greedy decoding (`sample=False`) everywhere, which amplifies repetition. This sets a low baseline; it does not explain the Stage-3 *jump*. |
| 5 | Learning rate too high → gradient explosion | **[ ] RULED OUT** | `lr=3e-4` default, cosine decay to 0 within each stage, AdamW, `clip_grad_norm_(…, 1.0)`. Replay log (1,204 records/variant): **0 loss spikes**, **0 NaN/Inf**, LR schedule **0 intra-stage increases**, grad-norm outliers **1/1200 (0.08%)**, median 0.734 — and those are *pre-clip* values, clipped to 1.0 before the step. Stage boundaries do restart LR with a fresh optimizer (a real but secondary concern). |
| 6 | Data corruption → garbage in, garbage out | **[ ] RULED OUT as primary** | 7,939 rows reconstructed and scanned: **0** invalid UTF-8, **0** NUL, **0** lone surrogates, **0** split codepoints through `encode`/`batch`, **0** malformed ReAct trajectories, **0** Mineflayer rows with static JS errors. 85 rows (1.1%) carry low-density mojibake markers or U+FFFD — GitHub issues *discussing* encoding, upstream double-encoded product names (`CasÃ©ta`, `WattWÃ¤chter`), and our own mojibake regex. A real data-quality issue, far too small for total collapse. The Hub-streamed portion of the Colab corpus is unavailable offline. |
| 7 | Vocabulary mismatch → wrong byte encoding | **[ ] RULED OUT as primary; one latent defect** | **27/27 encoding checks pass.** `"Olá" ↔ [79, 108, 195, 161]`; `encode("Olá") == [256, 79, 108, 195, 161, 257]`; all 256 byte rows reachable; no OOB ids; tied head/embedding correct. **Latent bug:** BOS=256, PAD=258 and reserved 259–267 are **never training targets**, yet `generate()` masks only `logits[PAD:]`, leaving BOS reachable. If BOS wins argmax, `bytes([256])` raises `ValueError`. That would crash the eval cell, not produce gibberish — so it is not the reported failure, but must be fixed. |
| 8 | Batch size issues → unstable gradients | **[ ] RULED OUT** | `micro_batch=1 × accumulation=32` gives an effective batch of 32 sequences; the replay used 6 and showed no instability. No NaN/Inf, no loss spikes. |
| 9 | Mixed precision (bf16) bugs → numerical instability | **[ ] RULED OUT** | `configs/phone_100m.yaml` specifies **fp16**, not bf16 ("T4 does not support native bf16"); `train.py` uses `torch.float16` autocast with `torch.amp.GradScaler` and grad clipping. Stage-4 checkpoint: **0 NaN, 0 Inf**, near-zero fraction `4e-06`, byte-codebook mean \|cos\| `0.636`, live grad norm `16.38`, **no dead/saturated layers**, valid final softmax. The network is computationally healthy — the *content* of the distribution is wrong (81.5% mass on byte 32 = space, entropy 0.952 nats). |

**Most likely cause, in one sentence:** Stages 3 and 4 allocated 30% and 50% of
their gradient updates to groups of four unique rows; with 8/12 blocks frozen and
a fully trainable tied byte embedding/output head, those repetitions rewrote the
global byte distribution, while Stage-1 language data was no longer rehearsed.
The one surviving `edit` PASS is an eval-identical memorised template, not
capability.

---

## 3. Evidence per hypothesis (files)

| Task | Script → artifact | Observed result |
|---|---|---|
| 21.1 loss spikes / NaN / LR / forgetting | `debug/analyze_training_logs.py` → `training_logs{,_baseline,_fixed}.json`, `loss_curves_*.png` | baseline: no spikes, no NaN/Inf, LR OK, **forgetting = true, collapse stage 3**; fixed: forgetting = false. `source_is_colab_log: false`. |
| 21.2 which checkpoint degrades | `debug/check_stage_checkpoints.py` → `stage_checkpoints.json/.md` | 8 stage checkpoints replayed; Stage 2 best (2.713), Stage 3 collapse (4.507), Stage 4 partial metric recovery but degenerate output. |
| 21.3 corrupted data? | `debug/verify_training_data.py` → `training_data.json/.md` | clean pipeline; exposure ratios 666.7× / 588.2×; 3 contaminated eval prompts. |
| 21.4 encoding? | `debug/test_byte_encoding.py` → `byte_encoding.json` | 27/27 pass; BOS/reserved rows never trained. |
| 21.5 weights / gradients? | `debug/analyze_model_weights.py` → `model_weights.json/.md` | no exploded/vanished/dead weights. |
| 21.6 activations? | `debug/probe_activations.py` → `activations.json/.md` | all layers finite and transforming; valid but **degenerate** softmax. |
| 21.7 edit mystery | `debug/investigate_edit_success.py` → `edit_mystery.json/.md` | verbatim 45-char training suffix; held-out edit FAIL. |
| 21.9 recovery A/B | `debug/quick_recovery.py` → `quick_recovery.json/.md` | improved (see §5). |
| consolidated | `debug/run_all.py` → `runs/phase21/forensic_report.json` | `collapse_stage: 3`, `numerical_pathology: false`, `data_corruption_found: false`, `quick_fix_improved: true`. |

---

## 4. Recommended fix

**Primary fix — implemented and tested (one change to the curriculum sampler in
`notebook/train.py`):** `stage_mixture()` replaces the raw stage weights:

1. `balanced_weights()` multiplies each group's weight by
   `min(1, unique_rows / 64)` and renormalises. Stage 1/2 are essentially
   unchanged; Stage 4's `edit` group drops from 50% to ~4%.
2. Stages 3/4 reserve `REPLAY_FRACTION = 0.4` of their updates for a Stage-1
   `pt`/`en`/`code` rehearsal, so the byte codebook is not rewritten by tool
   templates alone.

It is the default (`train(..., legacy_mix=False)`); `legacy_mix=True` reproduces
the broken curriculum for A/B. No architecture change, no new data, no extra
GPU time. `train()` also gained `metrics_path=` so the next run persists one
JSON line per step (loss, LR, grad norm, NaN flags) — the missing Phase 19 log
was the reason this autopsy needed a replay at all.

**Follow-up fixes, in priority order (not part of the single quick fix):**

1. **De-contaminate evaluation.** `RL_CASES` must not contain any `CASES` prompt.
   Replace the 5 smoke probes with held-out prompts, including held-out edit
   typos; report memorisation separately from capability.
2. **Grow tiny groups.** `search` and `edit` have 4 rows each. Generate ≥200
   unique rows (or add a real corpus) and add a preflight that rejects a group
   with `< MIN_DIVERSE_ROWS` unique rows carrying >10% of a stage's weight.
3. **Protect the byte codebook.** Either freeze `token.weight`/`head` in Stages
   3–4, or use a much lower late-stage LR; the current `FREEZE=8` funnels all
   late pressure into it.
4. **Checkpoint selection.** Save a stage-end checkpoint plus held-out
   validation; select the final model by held-out byte loss, not by "last step of
   Stage 4". Stage 2 was the best model in the replay.
5. **Mask untrained control ids.** In `phone_model.generate`, mask `logits[256:]`
   (or at least BOS) so a never-trained logit cannot be emitted; either drop
   reserved rows 259–267 from the head or give them an explicit target.
6. **Full retrain.** Do not try to repair the final weights. Retrain from the
   Stage 2 checkpoint (or from scratch) with the corrected mixture, expanded
   fixtures, clean eval and persisted metrics. A 46M-parameter byte decoder
   trained from random init in ~3 hours is still not expected to be fluent;
   the fix removes the collapse, it does not create fluency.

---

## 5. Quick recovery test (21.9) — measured, not a target

`python -m debug.quick_recovery` (300 steps/stage × 4 × 2 variants, seed 19,
**222.6 s CPU**). Same RNG stream for both variants; **only the mixture differs**.

| Metric (direction) | Baseline (`legacy_mix=True`) | Fixed (default) | Δ |
|---|---|---|---|
| Final held-out Portuguese loss (↓) | 2.50072 | **2.21858** | **−0.28214 (−11.3%)** |
| Worst held-out Portuguese loss (↓) | 4.50746 | **2.73610** | **−1.77136 (−39.3%)** |
| Forgetting ratio max/min (↓) | 1.802 | **1.233** | **−0.569 (−31.6%)** |
| Final 100-byte coherence (↑) | 0.5517 | **0.5997** | **+0.0480 (+8.7%)** |
| Final Portuguese word hits (↑) | 1 | **7** | +6 |
| Final output valid UTF-8 | `True` | `True` | — |
| Final output flagged gibberish | `False` | `False` | no change |
| Contaminated probes passed | 1 (`edit`) | 0 | expected |
| Final **training** loss (informational) | 0.8859 | 2.18337 | +1.29747 |

**Result: IMPROVED.** Per-stage held-out Portuguese loss goes from
`2.945 → 2.713 → 4.507 → 2.501` (Stage-3 spike) to
`2.736 → 2.480 → 2.423 → 2.219` (monotone improvement, no forgetting).

Interpretation caveats:
- The `1 → 0` probe count is **not** a regression: that one pass was
  memorisation of an eval-identical training template (§2, rank 3). A proper
  held-out benchmark is required to measure edit ability.
- Fixed **training loss is higher** because the model no longer over-fits four
  repeated rows. Held-out loss, forgetting ratio and byte coherence are the
  meaningful metrics.
- The strict gibberish heuristic is `False` for both final outputs in this run;
  the improvement is the **absence of the Stage-3 spike**, 7× more Portuguese
  word hits and valid UTF-8 throughout — not a change in that single flag.
- The surrogate still does not produce fluent Portuguese. The demonstrated
  claim is that **Stage-3 forgetting and byte-salad collapse are prevented**,
  not that language quality is solved. A full retrain is still required.

---

## 6. Success criteria — assessment

| Criterion | Met? | Notes |
|---|---|---|
| Identify the exact stage where collapse occurred | **Yes (Stage 3)**, with one caveat | Directly measured in the deterministic replay; inferred for the real run from the static Stage 3/4 mixture evidence. Real Colab stage checkpoints are unavailable; re-run `debug/check_stage_checkpoints.py --checkpoints-dir …` to verify against them. |
| Identify the root cause with evidence | **Yes** | diversity-capped mixture + missing language rehearsal + tied byte codebook + 4-row groups; numerically and structurally ruled-out alternatives. |
| Provide a clear, actionable fix | **Yes** | `notebook/train.py::stage_mixture`, default on, plus an ordered follow-up list. |
| Demonstrate the fix with a small test | **Yes** | 222.6 s A/B above; `improved: true` in `quick_recovery.json`. Full retrain still needed. |

## 7. What would change this conclusion

Real artifacts could still reveal a second failure (e.g. an fp16 overflow event
in the Colab run that the CPU replay cannot reproduce). To settle it:
`python -m debug.analyze_training_logs --logs <colab.log>`,
`python -m debug.check_stage_checkpoints --repo-id <user/repo>` (needs `HF_TOKEN`),
`python -m debug.verify_training_data --samples data_cache/samples.json`,
`python -m debug.analyze_model_weights --checkpoint final_model_fp16.pt`.
Until then, every LR / NaN / weight statement about the **real** run is an
inference from source inspection plus replay, not a direct measurement.
