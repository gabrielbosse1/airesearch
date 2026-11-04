# 21.9 — Quick recovery A/B

**Fix:** stage_mixture(): diversity-capped group weights + 40% Stage-1 language rehearsal in Stages 3/4 (notebook/train.py, default on)

**Result: IMPROVED**

Experiment: 300 steps/stage × 4 stages × 2 variants, lr=0.003, seed=19, 222.58s wall clock.

| metric (better direction) | baseline | fixed | delta |
| --- | --- | --- | --- |
| held-out PT loss (↓) | 2.50072 | 2.21858 | -0.28214 |
| worst held-out PT loss (↓) | 4.50746 | 2.7361 | -1.77136 |
| forgetting ratio max/min (↓) | 1.802 | 1.233 | -0.569 |
| final byte coherence (↑) | 0.5517 | 0.5997 | 0.048 |
| final training loss (informational) | 0.8859 | 2.18337 | 1.29747 |
| final output gibberish? | False | False | — |
| final output valid UTF-8? | True | True | — |
| contaminated probes passed | 1 | 0 | — |

## Per-stage held-out Portuguese loss (teacher-forced)

| stage | baseline | fixed |
| --- | --- | --- |
| 1 | 2.94543 | 2.7361 |
| 2 | 2.71273 | 2.48041 |
| 3 | 4.50746 | 2.4231 |
| 4 | 2.50072 | 2.21858 |

## Stage 3/4 mixture weights

| variant | Stage 3 | Stage 4 |
| --- | --- | --- |
| baseline | {'react': 40, 'search': 30, 'code': 30} | {'edit': 50, 'pt': 20, 'code': 15, 'mineflayer': 15} |
| fixed | {'react': 34.054054054054056, 'search': 1.6216216216216215, 'code': 37.65765765765766, 'pt': 13.333333333333334, 'en': 13.333333333333334} | {'edit': 3.592814371257485, 'pt': 36.327345309381236, 'code': 29.500998003992017, 'mineflayer': 17.245508982035926, 'en': 13.333333333333334} |

## Final 100-byte generations from `Olá, como`

- baseline: ` tá á co co co co vo vo bo vo vo vo vo vo vo vo vo vo ve vese ve são s ves ve vo `
- fixed:    ` co co com com como com como com com ar ada a a a a a a a a a a a a a a ta a te te t`

## Caveats

- The single "probes passed" count drops because that one pass was a memorised, eval-identical [EDIT] template; it is not a capability loss and is not a valid success metric (see 21.7).
- Fixed final TRAINING loss is higher: the fixed model is no longer over-fitting to four repeated rows. Held-out Portuguese loss and byte coherence are the meaningful metrics.
- The strict gibberish heuristic may remain True for the tiny fixed surrogate even when all held-out metrics improve; that flag rewards Portuguese word hits, which a 245k-parameter model rarely produces. The claim is prevention of forgetting/invalid bytes, not fluency.
- This is a small-scale mechanistic demonstration. Restoring the real Phase 19 model still requires a full retrain (or fine-tuning from the Stage 2 checkpoint) with the corrected mixture and a clean eval set.
- The tiny surrogate still does not produce fluent Portuguese; the claim is that forgetting and byte-salad collapse are prevented, not that language quality is solved.
