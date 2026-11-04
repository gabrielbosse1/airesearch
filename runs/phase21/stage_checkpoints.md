# 21.2 — Checkpoint degradation

**Checkpoint source:** `LOCAL REPLAY — not the Colab checkpoints`

Phase 19 Colab artifacts are gitignored (.gitignore: data_cache/, checkpoints/, final_model/, final_model_fp16.pt). They live only in the ephemeral Colab runtime and are NOT in this checkout. Any conclusion about the 4-hour run must be derived from source forensics plus a deterministic local replay, or from artifacts the user re-uploads.

Prompt: `Olá, como` (greedy, 100 bytes, faithful to `phone_model.generate`)

| stage | origin | coherence | gibberish | valid UTF-8 | PT word hits | held-out PT loss | text (first 60 chars) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LOCAL REPLAY (baseline) | 0.3796 | True | True | 0 | 2.94543 | '                                                            ' |
| 2 | LOCAL REPLAY (baseline) | 0.4194 | True | True | 0 | 2.71273 | ' e to to to a to  a  a  a tu  a  a a a  a a tu a tu a a to a' |
| 3 | LOCAL REPLAY (baseline) | 0.5702 | True | True | 0 | 4.50746 | ' dteasis te\nTho lou\nTho tho pe po teu ve teu\nTharou ghas par' |
| 4 | LOCAL REPLAY (baseline) | 0.5517 | False | True | 1 | 2.50072 | ' tá á co co co co vo vo bo vo vo vo vo vo vo vo vo vo ve ves' |
| 1 | LOCAL REPLAY (fixed) | 0.3713 | True | True | 0 | 2.7361 | ' e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e' |
| 2 | LOCAL REPLAY (fixed) | 0.3975 | True | True | 0 | 2.48041 | ' a a a a a aro a a a a a a a a a a a a a a a a a a a a a a a' |
| 3 | LOCAL REPLAY (fixed) | 0.456 | True | True | 0 | 2.4231 | ' a a to o to o o ra o rare a rento anto andanto a anto a ant' |
| 4 | LOCAL REPLAY (fixed) | 0.5997 | False | True | 7 | 2.21858 | ' co co com com como com como com com ar ada a a a a a a a a ' |

**Collapse located at stage 3** (signal: heldout_pt_loss)

## Baseline vs fixed (local replay)

| variant | final coherence | final gibberish | final held-out PT loss | probes passed |
| --- | --- | --- | --- | --- |
| baseline | 0.5517 | False | 2.50072 | 1 |
| fixed | 0.5997 | False | 2.21858 | 0 |
