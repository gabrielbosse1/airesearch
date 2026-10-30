# Phase 12 — Force Memory Usage + Prevent Forgetting

## Executive result

The short, long, and fresh-process demonstrations use a **trained neural reader over actual retrieved vectors**, with the earlier episode removed from the model's byte context. The representation is a narrowly parsed, reversible bit codec; it is not learned semantic compression. Replay improves an inspected historical comparison but does not prove forgetting is prevented. The diversity decoder meets the direct-decoder count, while the complete session **misses** it on the protected prompt because the classifier misroutes the greeting.

| Acceptance check | Measurement | Status / scope |
|---|---:|---|
| Cleared-context favorite-color example | `blue` | PASS; actual vector reader, no answer fallback |
| Fresh structured vector facts | 10/10 | PASS; small synthetic statements |
| 500 filler turns | 500/500 recorded; answer `blue` | PASS; episodic capacity 96 |
| New process / Alice works at ByteCorp | `ByteCorp` | PASS; real checkpoint reload |
| Phase 10 accuracy >70% | 298/384 (77.6%) | PASS; previously inspected development set |
| Fresh overall intent >85% | 62/70 (88.6%) | PASS; balanced synthetic prompts |
| >10 distinct responses / 20 | direct 14/20; routed 1/20 | Direct decoder passes; complete session **does not** on this prompt |
| Malformed <10% | direct 0/20 conservative proxy failures | Narrow template/structure proxy only; not a general quality test |
| Regression tests ≥147 | 168/168 | PASS |

## What forced memory actually means

1. The observed statement is parsed by a narrow host grammar into a key and value. The value is encoded, at write time, into one to five 64-dimensional vectors: signed bits for up to eight bytes per slot, multiplied by a fixed orthogonal matrix. It is lossless, finite-capacity, **not encrypted**, and deliberately structured; this is not an opaque semantic summary.
2. The observed episode is written into the existing episodic key-value bank. On a supported question, an exact owner/topic filter retrieves the vector slots. The generator gets only `U:` plus the current question and these vectors as `memory_context`; the separate learned reader gets the question features and the same vectors. The answer label is never passed into inference. The reader emits raw bytes and an internal EOS; a failed read abstains.
3. Evaluation patches recent-transcript retrieval, explicit procedural fact lookup, the legacy episode-fact helper, and the contextual decoder to fail if called during the positive vector-only route. It also checks that every generator byte input is exactly the current question. No positive evidence-grounding or answer oracle corrects a wrong reader output.

- Original vector → **blue**. Zeroed vector → **I couldn't read that vector fact reliably. Could you clarify?**. Another user's manually substituted vector → **green**. Restoring the vector → **blue**. A normal unrelated user did not retrieve the first user's fact. The swap is an intentional counterfactual test, not authorized cross-user retrieval.
- In the 500-turn run, the stored owner transcripts contained the target word before the final question: **False**; another user's vector remained unchanged: **True**. Filler messages went through `respond()` and were recorded, rather than merely incrementing a counter.
- In the fresh-process workplace case, the intent classifier predicted `clarify` and the exact read-only grammar performed the explicitly labelled `exact_vector_read_only_recall` override. The expected value was used only for scoring after the child returned.
- The public blue case has been inspected. The ten new cases are protected by source/target hash before Phase 12 training, but related question wordings and structures recur across prior data. Ten examples cannot establish generalization, and exact hash guards cannot detect paraphrases, hidden exposure or false provenance.

## Raw-byte scope and limitations

- The codec can represent every byte 0–255. An exploratory direct-reader probe decoded **256/256 single-byte values**. However, it corrupted some multi-byte binary sequences:
  - `00010203040506` → `00010203040506`: exact.
  - `0001020304050607` → `0001820304050607`: incorrect.
  - `000102030405060708090a0b0c0d0e0f` → `000182030405060708098a8b0c0d0e0f`: incorrect.
  - `000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f2021222324252627` → `000182030405060708098a8b0c0d0e0f101112131415161718191a1b1c1d1e1f2021222324252627`: incorrect.
- This is not evidence of arbitrary binary fidelity. The conversational grammar accepts visible UTF-8 text; long values over 40 bytes are not silently truncated and do not enter vector recall. The fixed codec fingerprint and generator SHA-256 bind the selected reader; incompatible checkpoints fail to load.

## Rehearsal and forgetting

- The intent trainer preflight accepted **1876 old** and **522 new** synthetic examples, blocked 6, and registered 4076 holdout hashes. Every optimizer batch had **90 new / 38 old** rows (29.69% old) plus teacher distillation.
- Selected epoch 4 using validation only. The feasibility threshold was raised from .82 to .85 after the first validation-only run, before inspecting the Phase 12 final test; this is validation tuning, not a fully preregistered selection procedure.
- The old, previously inspected Phase 10 development set went from **287/384** (Phase 10) to **229/384** (Phase 11) to **298/384** (Phase 12). FORGET is **23/32**, still below the original **30/32**. The previously inspected Phase 11 boundary is 17/20 (vague 7/10); Phase 11 was 19/20 and 9/10 respectively. Rehearsal is not a guarantee against regressions.
- `replay_multitask_phase12.pt` has **1000/1000** guarded synthetic training records: 300 old intent, 300 new intent, 160 vector, 160 diverse-response, 80 contextual-response. It samples the training data because the buffer is bounded, and the live conversation path never automatically inserts user transcripts. Guarded task labels prevent accidental mixing of incompatible objectives. The 70/30 optimizer mix applies to intent training; the newly introduced reader/response tasks were **not all** trained with 30% replay.

## Response diversity and quality

- The learned style-conditioned autoregressive byte decoder has 135,873 parameters. It is trained with cross-entropy plus a cosine penalty on **predicted probability distributions**, not nondifferentiable sampled byte IDs. Thirty-two human-authored synthetic response styles recur in training.
- Protected prompt: `Good to see you again.`. Direct CHAT-conditioned sampling, with randomly sampled style and bytes at temperature 0.15, gave **14/20 distinct outputs**, 20/20 complete UTF-8, and 0/20 failing a deliberately strict structural/known-template proxy. This proxy rewards memorization; it is not a general grammar or semantic-quality score.
- The actual classifier labelled the same prompt `clarify` and the session produced **1/20 distinct responses**, with **0** diverse-decoder replies. Thus the end-to-end diversity target fails for the protected prompt. An illustrative previously used `Hello there` development prompt, selected after observing the failure, produced 14/20 distinct system replies; it is not a fresh test.
- Direct samples (repetitions preserved):
  - `It's good to hear from you. What's up?`
  - `Thanks for telling me. How can I help?`
  - `I'm glad you told me. What's next?`
  - `Feel free to tell me more.`
  - `Thanks for telling me. How can I help?`
  - `Would it help to talk this through?`
  - `You can share as much as you like.`
  - `Can you tell me a little more?`
  - `I appreciate you sharing that.`
  - `That makes sense. What would help now?`
  - `All right. What are you thinking about?`
  - `It's good to hear from you. What's up?`
  - `Thanks for opening up. What's on your mind?`
  - `I'm listening. What would be useful?`
  - `How has the day been for you?`
  - `How has the day been for you?`
  - `How has the day been for you?`
  - `What would you like to talk about?`
  - `I'm listening. What would be useful?`
  - `How has the day been for you?`

## Safeguards, compatibility and costs

- 5/5 constructed safety checks passed: user isolation, no-storage wording, negated edit, underspecified edit, and forget/recall. These are not a comprehensive security evaluation. The English privacy pattern and intent confidence are limited; neither replaces host consent or authentication.
- By default storage is ephemeral. Supplying `--memory PATH` automatically persists full user/assistant transcripts and **reversible** fact vectors in an **unencrypted** file. Hashes and rotation are not encryption; user IDs are namespaces, not identity verification. Use `--no-episode-storage` to disable implicit transcript/vector storage and retrieval. Explicit old-style remembered facts and local process history remain available in that mode.
- `VectorConversationMemory` recycles only the same owner's ordinary episodes. It does not evict vector facts or other users' entries; full banks can refuse writes. Updates and forgetting are scoped and rolled back on ordinary save failure. `Forget my KEY` removes that fact; `Forget Alice's workplace` removes the corresponding workplace vector. Both purge the user's live transcript but not backups, other facts, old checkpoints or disk remnants. Process-local locks do not synchronize independent writers.
- Parameter counts: generator **390,463**, contextual response decoder **106,498**, vector reader **8,306**, diverse decoder **135,873**. This adds separate models and storage; it does not replace the raw-byte backbone with a tokenized LLM. All 155 original Phase 9 generator tensors remain unchanged, and the three-way edit diagnostic remains exact. Older Phase 10/11 paths are available via `--legacy-phase10` / `--legacy-phase11`.
- Vector-reader training preflight accepted 1714 and blocked 486 synthetic examples; validation-only selected epoch 35 (240/240 exact). Diverse response epoch 54 had 64/64 teacher-forced validation sequences. The separate compatible contextual decoder selected epoch 20 with 30/40 validation copy cases; it is not part of the vector-only positive answer.
- Reader/response checkpoints are SHA-256-bound to their generator; vector slots record a codec fingerprint. Memory provenance checks require truthful raw sources and targets, but cannot audit opaque tensor contents. Training/validation data are synthetic and small, and prior real uninstructed editing remains 0/8.

## Reproduce

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_replay_intents
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_vector_memory
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_diverse_responses
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_contextual_responses --generator runs/phase12/generator.pt --phase12-guard --output-dir runs/phase12 --epochs 48 --seed 122012
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.finalize_replay
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.memory_eval
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase12_benchmark
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase12_verify
.venv/bin/python -m demo.conversation --show-intent
```

Evidence: `runs/phase12/{memory_eval,benchmark,verification,intent_metrics,vector_reader_metrics,diverse_response_metrics,response_metrics,replay_metrics}.json`, plus selected checkpoints and bounded replay snapshots in the same directory. `AGENTS.md`, `QWEN_CHAT_PROMPT.md`, and `README.md` describe the architecture, trade-offs and privacy implications.
