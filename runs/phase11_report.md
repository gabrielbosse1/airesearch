# Phase 11 — Conversational Refinement & Multi-Turn Memory

## Results

| Measure | Result | Scope |
|---|---:|---|
| Vague classification | 9/10 (90%) | 10 of 20 boundary prompts; >80% target |
| Specific classification | 10/10 (100%) | Other half of the same boundary set |
| Mixed boundary accuracy | 19/20 (95%) | Includes vague and specific prompts |
| Overall intent accuracy | 62/70 (88.6%) | 70 synthetic prompts, seven balanced classes; >85% target |
| Historical Phase 10 development set | 229/384 (59.6%) | Down from 287/384 (74.7%) before fine-tuning; not a pristine test |
| End-to-end two-turn retention | 10/10 (100%) | Independent ephemeral user memories; >70% system target |
| Learned decoder within retention | 5/10 | Exact whole-value check; other 5 successes used the grounded host fallback |
| Same-prompt response-byte diversity | 5/20 distinct | Direct decoder, CHAT intent, temperature 0.65; 20/20 complete UTF-8, not a quality guarantee |
| Full unittest regressions | 147/147 | Includes Phase 9, Phase 10 and Phase 11; >=120 target |

**The system meets the numerical retention target because it includes an exact-evidence fallback; the neural decoder alone does not meet 70%.** Classification and retention fixtures are small, synthetic, English-language examples. This is a bounded prototype, not demonstrated open-domain conversational competence.

**Important regression:** on the older Phase 10 development set, REMEMBER fell from 188/256 to 129/256, and FORGET fell from 30/32 to 18/32; CLARIFY improved from 4/12 to 11/12. The set is template-heavy and was used during Phase 10 development. Accuracy cannot be assumed to improve across all intents.

### Public blue example

- User: `My favorite color is blue.` → Assistant: `Got it, thanks for letting me know.`
- User: `What is my favorite color?` → Assistant: `You said blue.` (source: `contextual_byte_decoder`, retrieved episodes: 1).
- This particular example was inspected during development and is not counted in the ten fresh retention cases.

### What the memory ablation actually shows

- With both retrieved vectors and raw episode bytes, the direct decoder produced the exact value in **5/10** cases. With raw episode bytes but **no** vectors it produced **5/10**; vectors alone: **0/10**; neither: **0/10**.
- Removing the vectors while retaining episode bytes changed **0/10** greedy outputs. The retrieved vectors are injected and change generator features (unit tested), but this ablation does **not** demonstrate that they drive these responses; byte-level history plus the host fallback accounts for observed retention.
- The ablation uses gold RECALL intent to isolate the decoder. The end-to-end score instead uses the classifier and labels every exact read-only routing override and fallback.

### Response diversity

- Same input, same intent: 5 distinct byte sequences from 20 samples; byte entropy 4.119 bits, distinct-1 0.0386, distinct-2 0.1254. These statistics are length- and sample-dependent.
- Greedy decoding across eight related CHAT inputs produced 3 unique outputs. The small model mostly reproduces familiar training-style greetings and sometimes emits malformed wording; valid UTF-8 is not semantic quality.
- Distinct sampled outputs:
  - `Hi there, what is on your mind?`
  - `Hello! What would you like to talk about?`
  - `Good to see you. What shall we discuss?`
  - `Hello! What would you like to ta like to talk about?`
  - `Hi there, what insten is on your mind?`

## Architecture and safeguards

- The generator has 390,463 parameters; the separate autoregressive pointer/generator decoder has 106,498 (Phase 10 decoder: 62,273). Inputs, histories and outputs are raw UTF-8 bytes; EOS and segment identifiers are internal metadata, not output bytes. There is no external tokenizer.
- The current-turn learned classifier runs before action selection. The session retrieves up to five latest scoped episodes using `MemoryModel.retrieve()` and passes their values as `memory_context` to the generator. It also passes actual current/episode bytes to the response decoder. Exact recall and narrowly anchored affirmative remember/forget commands may override an incorrect or low-confidence intent; results label those routes. The command overrides were added after the legacy development regression was inspected, not independently validated as a model improvement.
- Destructive or mutating actions require narrow affirmative grammar; requested edits must be verified by the existing Phase 9 cognitive mechanism. For memory recall the generated value is checked against exact retrieved evidence; on mismatch or truncation, `grounded_fallback` returns the actual value. It is not neural generation and can miss paraphrases or facts outside the last five turns.
- The constructed checks passed 6/6: blue example, user isolation, explicit no-storage, negated edit, underspecified edit and forgetting. These cases are not a comprehensive safety test. A negated edit can still be classified EDIT at high confidence; the anchored side-effect grammar blocks it.

## Privacy and compatibility

- By default the demo keeps episodes in process only. **Adding `--memory PATH` now automatically writes full conversation turns to an unencrypted file.** The CLI prints a warning; `--no-episode-storage` disables implicit episode storage/retrieval but retains in-process history and permits explicitly remembered facts. A recognized no-storage turn is not recorded; session-wide no-storage also disables future session writes and local history. The English detector is limited; it is not a universal consent or privacy guarantee.
- User IDs are namespaces, not authentication. Hosts must supply consent, access control, encryption, and cross-process synchronization. Atomic replacement rolls back ordinary failures; it is not crash-proof transactional storage or forensic erasure. When the episodic bank is full it recycles only the same user's oldest episode, never another user's.
- Forgetting an exact fact also purges that user's entire live stored transcript to avoid retaining the detail in previous turns. It cannot erase backups, earlier checkpoints or filesystem remnants. Episodes are not promoted to semantic memory. The legacy demo remains available through `--legacy-phase10`.
- All 155 Phase 9 source generator tensors remain identical. The three-way one-byte diagnostic remains exact: diffusion `Hello Word`, natural cognition `Hello Word`, and explicit-request cognition `Hello Word`. This example was in training; explicit request and stored replay are not autonomous generalization. Historical checkpoint hashes and real memory stayed unchanged during verification.

## Training, held-out discipline and limitations

- Intent training preflight accepted 2,374 examples and blocked 6; selected epoch 35 using only the predeclared boundary, overall and prior validation sets. The final intent checkpoint was not selected on test.
- Response training preflight accepted 1,805 rows and blocked 275. It used varied synthetic response styles, synthetic copying examples and training-only byte-alignment labels. Epoch 48 was selected by validation copying (32/40) and then validation NLL (0.1545); held-out answers were never passed to inference.
- The first decoder's ten retention cases were examined during development; they were replaced by ten new values protected **before** the copy-alignment run. The new test shares generic question wording and response styles with training, so it tests value retention in a narrow format, not fully novel dialogue. The first decoder and its inspected fixtures remain in `runs/phase11_initial/` as developmental evidence.
- Exact and long-substring holdout guards cannot establish semantic novelty, audit hidden pretrained exposure, or serve as a privacy boundary. The generator's 0.55 threshold is not calibrated for deployment, auto-generated replies may be incorrect or malformed, the response decoder is deliberately small, and no evidence here overturns earlier 0/8 real uninstructed edit results.

## Reproduction and artifacts

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.conversational_eval_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase11_verify
.venv/bin/python -m demo.conversation --show-intent
```

Training commands and privacy details are in `AGENTS.md` and `README.md`. Detailed evidence: `runs/phase11/intent_metrics.json`, `runs/phase11/response_metrics.json`, `runs/phase11/benchmark.json`, `runs/phase11/phase9_regression.json`, `runs/phase11/verification.json`, and `runs/phase11/unittest.log`. Selected checkpoints: `runs/phase11/generator.pt` and `runs/phase11/response.pt`.
