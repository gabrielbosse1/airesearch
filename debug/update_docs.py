"""Idempotently append the Phase 21 section to the project docs via Python I/O.

AGENTS.md mandates that architecture/diagnosis work update AGENTS.md and
QWEN_CHAT_PROMPT.md through Python I/O. This script does that (plus README.md),
so the documentation change is reproducible and re-runnable.

Run: python -m debug.update_docs
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Generic marker: README.md uses a slightly different Phase 21 heading.
MARKER = '## Phase 21:'

AGENTS_SECTION = """
## Phase 21: forensic diagnosis of the Phase 19 collapse

**Observed (reported by the user).** After the four-stage Colab run, all probes
FAIL except `edit`; pt-br, code, search, memory and all Mineflayer tasks emit
byte salad. **Not reproducible from the Colab run:** its logs, `checkpoints/`,
`data_cache/` and `final_model_fp16.pt` are gitignored and were never uploaded to
this checkout (`debug.verify_training_data.artifact_status()` returns
`any_colab_artifact: false`). Phase 21 therefore combines exact source-level
forensics with a deterministic CPU replay of the same pipeline
(`debug/replay_training.py`, 300 steps/stage, seed 19); every report labels
whether a result is a measurement, a replay or an inference.

- **21.1 logs.** `debug/analyze_training_logs.py` parses supplied
  .jsonl/.json/.log/.ipynb logs or the labelled replay metrics, plots per-stage
  loss/LR curves and detects spikes, near-zero collapse, NaN/Inf, LR failure,
  gradient explosion and forgetting. Baseline replay: no spikes, no NaN/Inf, LR
  schedule OK, **forgetting at Stage 3**. `notebook/train.py` gained
  `metrics_path=` so the next run persists one JSON line per step; Phase 19 kept
  loss only in a transient tqdm bar.
- **21.2 checkpoints.** `debug/check_stage_checkpoints.py` accepts a local
  `checkpoints/` dir, a private HF repo (`--repo-id`, `HF_TOKEN`) or the replay
  snapshots. Held-out Portuguese loss by stage: 2.945 / 2.713 / **4.507** /
  2.501. **Collapse located at Stage 3** (1.662x the best prior stage); Stage 4
  cements the degenerate distribution while leaving `edit` as the only pass.
- **21.3 data.** 7,939 rows scanned: 0 invalid UTF-8, 0 NUL, 0 split codepoints
  through `encode`/`batch`, 0 malformed ReAct rows, 0 Mineflayer static-JS
  errors. 85 rows (1.1%) carry low-density source-level mojibake markers or
  U+FFFD (GitHub issues discussing encoding, upstream double-encoded product
  names, our own regex). A real data-quality issue, not a total-collapse cause.
  The Hub-streamed part of the Colab corpus is unavailable offline.
- **21.4 encoding.** 27/27 checks pass: `"Olá" <-> [79,108,195,161]`,
  `encode("Olá") == [256,79,108,195,161,257]`, all 256 byte rows reachable, no
  OOB ids, tied head/embedding correct. **Latent defect:** BOS=256, PAD=258 and
  reserved 259-267 are never training targets while `generate()` masks only
  `logits[258:]`, leaving BOS reachable; `bytes([256])` would raise ValueError.
  Not the observed failure (that would crash, not emit gibberish).
- **21.5/21.6 weights and activations.** No NaN/Inf, near-zero fraction 8e-06,
  no dead/saturated layers, differentiated byte codebook (mean |cos| 0.636),
  live grad norm 16.38, valid final softmax. The network is computationally
  healthy; the *content* of the distribution is wrong (81.5% mass on byte 32, entropy 0.952 nats, in the replayed Stage-4
  checkpoint).
- **21.7 edit mystery.** `RL_CASES` contains prompts byte-identical to 3 of 5
  eval cases (`pt-br`, `code`, `edit`); the edit RL case also shares the exact
  reward. The replayed Stage-4 model reproduces the **45-character training
  suffix verbatim** and FAILS a held-out typo. The lone PASS is memorisation of
  one test item, not an edit skill.
- **Root cause.** `MIXES[2]` gives 30% to `search` and `MIXES[3]` gives 50% to
  `edit`; both groups contain **4 unique rows** while `react`/`code` have
  3,587/701. Exposure imbalance **667x / 588x** per unique row per 32-sequence
  step. `FREEZE = [0,6,8,8]` leaves only 4 blocks plus the **tied byte
  embedding/output head** trainable, so those repetitions rewrite the global
  byte distribution with no Stage-1 language rehearsal. Ranked-out alternatives:
  LR/gradient explosion, data corruption, vocabulary mismatch, batch size, bf16
  (Phase 19 uses fp16 + GradScaler + clipping). Random-init undertraining and
  greedy decoding are contributing, not causal.
- **21.9 fix (implemented, ONE curriculum change).** `stage_mixture()` in
  `notebook/train.py` scales group weights by `min(1, unique_rows/64)` and
  reserves 40% of Stage 3/4 updates for `pt`/`en`/`code` rehearsal. Default on;
  `legacy_mix=True` restores the broken curriculum for A/B; no architecture or
  compute change. Full retrain is still required.
- **Measured A/B (replay, ~170 s CPU).** Worst held-out PT loss 4.50746 ->
  2.73610 (-39.3%); final held-out PT loss 2.50072 -> 2.21858 (-11.3%); final
  byte coherence 0.5517 -> 0.5997 (+8.7%); forgetting ratio 1.802 -> 1.233
  (-31.6%); Portuguese word hits 1 -> 7. Final *training* loss rises 0.886 ->
  2.183 because the model no longer over-fits four rows. The contaminated probe count drops 1 -> 0,
  which is expected, not a regression. The surrogate is still not fluent
  Portuguese; no fluency claim is made.
- **Regression evidence.** `python -m pytest tests -q`: **360 passed, 1 skipped,
  39 subtests passed**. `debug/run_all.py` produces
  `runs/phase21/forensic_report.json`; `debug/root_cause_analysis.md` is the
  report. Full caveat: real Colab artifacts could still reveal a second failure
  (e.g. an fp16 overflow) that a CPU replay cannot reproduce; re-run the scripts
  with `--logs`, `--checkpoints-dir`, `--samples` or `--checkpoint` to verify.

Reproduce:

```bash
python -m debug.run_all --from-existing
python -m debug.quick_recovery
python -m debug.analyze_training_logs --logs /path/to/colab.log
python -m debug.check_stage_checkpoints --checkpoints-dir /path/to/checkpoints
python -m pytest tests/test_phase21_debug.py -q
```
"""

QWEN_SECTION = """
## Phase 21: forensic diagnosis of the Phase 19 collapse

**Reported:** after the four-stage Colab run every probe emits byte salad except
`edit`. The Colab logs and checkpoints are gitignored and unavailable, so Phase
21 uses exact source forensics plus a labelled deterministic CPU replay; no
replay number is claimed as a measurement of the 4-hour T4 run.

1. **Where.** Held-out Portuguese loss per stage (replay): 2.945, 2.713,
   **4.507**, 2.501. Training loss falls to 1.58 at Stage 3 while held-out loss
   jumps +66.2% — overfitting plus forgetting. Collapse stage = **3**; Stage 4
   cements it and produces the single memorised `edit` pass.
2. **Why.** Stage 3 gives 30% of updates to `search` and Stage 4 gives 50% to
   `edit`; each group has **4 unique rows** (vs 3,587 real react rows), a 667x /
   588x per-row exposure imbalance. `FREEZE=8` leaves only 4 blocks plus the
   tied byte embedding/output head trainable, so four repeated templates rewrite
   the global byte distribution. No Stage-1 language rehearsal remains.
3. **Ruled out.** 27/27 encoding checks (`"Olá" <-> [79,108,195,161]`); 0
   invalid UTF-8 / split codepoints / malformed ReAct / Mineflayer JS errors in
   7,939 rows; no NaN/Inf, no dead neurons, no gradient explosion, LR schedule
   OK; fp16 + GradScaler + clip, not bf16. A latent defect remains: BOS=256 and
   reserved ids are never training targets but `generate()` leaves BOS reachable.
4. **Edit PASS.** Three eval prompts are byte-identical to Stage-4 REINFORCE
   prompts. The `edit` case shares its reward too; the model emits the 45-char
   training suffix verbatim and fails a held-out typo. Memorisation, not skill.
5. **Fix.** `notebook/train.py::stage_mixture()`: diversity-capped weights
   (`min(1, unique_rows/64)`) + 40% Stage-1 `pt`/`en`/`code` rehearsal in Stages
   3/4. Default on; `legacy_mix=True` for A/B; `metrics_path=` persists logs.
   A/B: worst held-out loss -39.3%, coherence +8.7%, and the Stage-3 spike
   disappears (4.507 -> 2.423). Full retrain still required; the tiny surrogate
   is not fluent.
6. **Limits.** `debug/root_cause_analysis.md`, `runs/phase21/forensic_report.json`.
   Full suite: 360 passed, 1 skipped. Real artifacts may reveal a second failure;
   re-run with `--logs`/`--checkpoints-dir`/`--samples`/`--checkpoint`.
"""

README_SECTION = """
## Phase 21: diagnosing the Phase 19 collapse

The four-stage Colab run ended with every probe FAIL except `edit`. The Colab
logs and checkpoints are gitignored and were never uploaded, so `debug/` combines
exact source forensics with a deterministic CPU replay of the same pipeline
(`MIXES`, `FREEZE`, cosine LR, prompts, rewards) and labels every measurement as
a replay, not a T4 result. See `debug/root_cause_analysis.md`.

**Finding.** Collapse starts at **Stage 3**. Stage 3 gives 30% of updates to
`search` and Stage 4 gives 50% to `edit`, while each of those groups has only
**4 unique rows** (a 667x / 588x per-row exposure imbalance). With 8 of 12
blocks frozen, the tied byte embedding/output head absorbs those repetitions and
the global byte distribution is rewritten; no Stage-1 language data is rehearsed.
Held-out Portuguese loss goes 2.713 -> 4.507 at Stage 3 while training loss
falls to 1.58. Encoding (27/27 checks), data (0 invalid UTF-8 in 7,939 rows),
weights (no NaN/Inf, no dead neurons) and the LR schedule are all healthy. The
one `edit` PASS is a **verbatim 45-character training suffix**: the Stage 4
REINFORCE prompt is byte-identical to the eval prompt, and a held-out typo fails.

**Fix, implemented and tested.** `notebook/train.py::stage_mixture()` scales
weights by unique-row diversity and reserves 40% of Stages 3/4 for `pt`/`en`/
`code` rehearsal (`legacy_mix=True` reproduces the broken curriculum). In the
3-minute A/B replay: worst held-out loss -39.3%, byte coherence +8.7%, and
the Stage-3 spike disappears (4.507 -> 2.423). Training loss rises because the model stops
over-fitting four rows. This prevents the collapse; it does not create fluency,
and a full retrain is still required.

```bash
python -m debug.run_all --from-existing   # or without the flag to replay both variants
python -m debug.analyze_training_logs --logs /path/to/colab.log
python -m debug.check_stage_checkpoints --checkpoints-dir /path/to/checkpoints
```

Evidence: `runs/phase21/` (JSON, Markdown, loss curves). Tests:
`python -m pytest tests -q` — 360 passed, 1 skipped, 39 subtests passed.
"""


def sync(path: str, section: str) -> bool:
    """Replace the trailing Phase 21 section so numbers can be refreshed."""
    target = ROOT / path
    text = target.read_text(encoding='utf8')
    index = text.find(MARKER)
    action = 'appended' if index < 0 else 'refreshed'
    if index >= 0:
        text = text[:index].rstrip()
    target.write_text(text + '\n' + section, encoding='utf8')
    print(f'{path}: {action} Phase 21 section ({len(section)} chars)')
    return True


def main(argv=None):
    changed = [sync('AGENTS.md', AGENTS_SECTION),
               sync('QWEN_CHAT_PROMPT.md', QWEN_SECTION),
               sync('README.md', README_SECTION)]
    print('docs updated:', sum(changed), 'file(s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
