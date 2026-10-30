# Phase 10 — Conversational Foundation

## Outcome

Implemented and trained a raw-byte intent classifier, bounded social response decoder, and a conservative session layer with optional persistent user-scoped memory. This is an early research prototype, **not** a general conversational agent: it cannot reliably understand arbitrary requests, answer open-domain questions, or infer edits from unconstrained language.

## What was built

- `GeneratorConditioned.intent_classifier` is `Linear(d_model,d_model) → SiLU → Linear(d_model,7)`. The stable labels are chat, question, edit, remember, recall, forget and clarify. It combines frozen causal byte features with a new trainable raw-byte convolutional encoder; the current-user span is supplied separately from history. Explicit lengths preserve byte 0 and mask padding.
- `ByteDialogueDecoder` generates social replies autoregressively over literal bytes with an internal EOS. The social training targets are three recurring canned strings; outputs are not evidence of general knowledge. When classified as a question, four exact reference questions are answered from a host table and other questions abstain; misclassification can choose another path. The decoder checkpoint is bound to the generator SHA-256 and expected attention-head count.
- `ConversationSession` classifies before action selection. It keeps bounded, in-process recent turns; validates explicit commands before memory writes/deletes or edits; and asks for clarification when unsupported. Only explicit single-byte numeric replacements are routed through the Phase 9 learned cognitive policy. The host verifies that the model actually completed exactly that edit before returning a changed document; it does not overwrite the predicted bytes.
- `ConversationMemory` stores explicitly requested facts as length-framed, exact raw-byte procedural key/value records. It supports separate user namespaces, cross-process reload, atomic file replacement and rollback on ordinary save errors, and deletion of every matching live version. It rejects a new fact when the bank is full rather than evicting another user's record through this API. Holdout guards audit the full explicit write request.

## Training and independent checks

The final run used 1876 training, 384 validation and 384 test messages; 0 training messages were blocked by the byte holdout guard. Inputs and wording templates were split; prompts are synthetic and English-only. The classifier and encoder selected epoch 7 by validation macro accuracy and then validation NLL; the decoder selected by validation response loss.

| Evaluation | Result | Scope |
|---|---:|---|
| Validation, mixed with/without a short history | 553/768 (72.0%); macro 63.1% | Epoch selection |
| Synthetic template test, no history | 287/384 (74.7%); macro 70.9% | Development holdout; head-only baseline test was inspected before changing architecture |
| Later, manually written challenge | 23/35 (65.7%); macro 65.7% | Post-selection; small, not representative |
| Challenge: vague/clarify class | 0/5 | Important unmet classification capability |
| Action on those vague requests | 5/5 asked | Fallbacks, not correct classification |
| Social decoder on new test inputs | 10/10 ended; 10/10 valid UTF-8; 5/10 exact seen template | Outputs/targets are not novel |
| Side-effect checks | 5/5 | Five specifically constructed negative/malformed cases, not a security certification |

The original head-only experiment reached 36.6% macro accuracy on the template test; that result informed the addition of the byte encoder, so this test must not be presented as an untouched final architecture holdout. The later 35-case challenge was authored after model selection and was retrospectively checked for raw-byte overlap with training inputs, but may contain semantic similarities and is too small for deployment claims.

At the uncalibrated 0.55 abstention threshold, challenge coverage was 80.0% and covered accuracy 71.4%. High confidence does not imply correctness: `Do not remember my name is Sol.` was classified as remember with approximately 0.995 confidence, but the explicit-command check prevented a write.

## Multi-turn behavior

A single fresh synthetic session greeted the user, explicitly stored a hometown, reloaded an isolated memory file in a new session, recalled it for the original namespace, did not return it to a different namespace, asked about an ambiguous edit, completed a specified one-byte edit, and deleted the fact. This is one end-to-end scenario, not general proof of memory comprehension. The temporary test memory was discarded.

- Greeting: `Hello! What would you like to talk about?` (byte_decoder).
- After reload: `Your hometown is Recife.`
- Edit events: `[[259], [261], [262, 3, 2, 113], [260], [263]]`; exact output was verified, and the requested fields were explicitly parsed rather than inferred freely.
- Known reference question answered: True; unknown question abstained: True.

## Parameters, compatibility and verification

The Phase 9 generator has 369,368 parameters; the new encoder adds 16,480 and head 4,615, for 390,463. The separate decoder adds 62,273 (452,736 combined). All 155 Phase 9 checkpoint tensors were bit-identical. Entirely missing legacy intent components initialize deterministically and remain marked untrained; partially missing new checkpoints fail strict loading. Older software cannot strictly load the new weights; rebuild optimizer state and revalidate calibration after architectural changes.

The full regression suite passed **120/120**. The Phase 9 `Hello Worl → Hello Word` diagnostic still succeeded in natural, explicitly conditioned and masked paths, but the example was in training and does not establish generalization. Warm one-thread CPU intent-only latency over 30 samples: median 2.00 ms, p95 2.84 ms; excludes loading, memory I/O, generation and editing.

## Privacy and operational limits

Memory is ephemeral unless a path is supplied; saved checkpoint contents are **unencrypted**. Namespaces are not authentication, and the host must authenticate users, restrict file access, secure backups and protect sensitive data. Locks cover sessions sharing one wrapper only, not multiple processes; atomic replacement is not a power-loss durability guarantee. Deletion clears the current in-memory bank and new checkpoint, not old backups, disk remnants or OS caches. Only explicit `remember my ... is ...` facts persist—not full conversations or inferred personal details. Memory capacity is finite; there is no semantic recall of arbitrary conversations.

The current context limit is 128 bytes, and only recent whole turns fitting that limit are included. Natural-language inputs must be UTF-8; edit documents may be arbitrary raw bytes. The classifier threshold is not calibrated for deployment, the learned models are narrow, and the bounded command grammars do not cover ordinary free-form tasks. The holdout guard cannot detect paraphrases, unknown datasets, opaque embeddings or private tensor mutation.

## Reproduce

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_conversation --intent-epochs 60 --response-epochs 100 --batch-size 128 --feature-batch-size 32 --intent-lr 0.003 --response-lr 0.003 --seed 101010 --threads 1 --output-dir runs/phase10
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase10_benchmark --samples 30 --threads 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase9_diagnostic --structured-generator runs/phase10/generator.pt --output runs/phase10/phase9_regression.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase10_verify
.venv/bin/python -m evaluation.phase10_report
.venv/bin/python -m demo.conversation --show-intent
```

Detailed evidence: `runs/phase10/training_metrics.json`, `runs/phase10/benchmark.json`, `runs/phase10/phase9_regression.json`, and `runs/phase10/verification.json`. The development head-only metrics are in `runs/phase10_baseline/training_metrics.json`.
