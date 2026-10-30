# Phase 5 Failure Modes

## Autonomous real edit failures

Held-out repository excerpts: **8**. No host actions or target outputs were passed to the generator.

| Classification | Cases |
|---|---:|
| wrong operation | 3 |
| wrong position payload or edit count | 5 |

## Diffusion recovery

At 10% fixed-length insertion/deletion corruption, compared to the clean sequence. `shift dominated proxy` means position accuracy < 50% while Levenshtein similarity >= 75%; this heuristic does not prove causation.

| Checkpoint | Mode | Samples | Exact | Shift-dominated proxy | Other inexact |
|---|---|---:|---:|---:|---:|
| baseline | clean | 300 | 267 | 0 | 33 |
| baseline | insert | 300 | 12 | 169 | 119 |
| baseline | delete | 300 | 2 | 158 | 140 |
| soft_dtw_retrained | clean | 300 | 248 | 0 | 52 |
| soft_dtw_retrained | insert | 300 | 12 | 169 | 119 |
| soft_dtw_retrained | delete | 300 | 2 | 157 | 141 |

## Overconfident byte predictions

Scores >= 0.90 on the separate calibration test; bytes from the same sample/noise conditions are correlated.

| Model and calibration scope | High-confidence bytes | Wrong among them | Error fraction |
|---|---:|---:|---:|
| diffusion | 56215 | 833 | 1.482% |
| generator | 0 | 0 | n/a |

## Retrieval

On 64 unique-prefix validation queries, 44 top-1 memories had a different concept label. This is a metadata-label proxy, not a causal diagnosis.

## Interpretation and next checks

- Pointer/payload and action-count errors limit autonomous real edits; collect broader paired edits with explicit task instructions.
- The shift proxy suggests testing length-aware, aligned decoders and corruption-specific supervision; do not equate high edit similarity with reconstruction of missing bytes.
- Refit confidence after training and validate independently for each use; next-byte/denoising calibration does not certify THINK or ancestral stopping.
- Retrieve/evict slots under task-grounded utility tests before trusting raw importance scores.
