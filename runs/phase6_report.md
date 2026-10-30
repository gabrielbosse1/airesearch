# Phase 6 — Fix the Fundamentals

## Outcome

- **Clean reconstruction:** 99.73% across real (non-padding) bytes; 99.87% on 361 test rows not exactly present in train. The original checkpoint was evaluated, not retrained.
- **Guarded, train-only replay:** 10/10 stored training procedures; 12/12 held-out write attempts rejected. Lookup is not model editing.
- **Autonomous real edits:** 0/8 test excerpts without memory; 0/8 with guarded train memory. The target is not passed to the model.
- **Adaptive halting:** implemented and config-wired, but left disabled: no validated quality-preserving diffusion policy was selected; thought confidence remains uncalibrated. This requested target was not met.
- **Architectural dispatch:** caller-declared structural operations use the cognitive loop; fixed-length denoising uses diffusion. This does not establish instruction following or success.

## 1. Memory contamination audit

The previous `runs/real_traces/memory.pt` contained 12 exact validation/test pairs among 22 procedures. It was replaced with the audited guarded train-only snapshot. Historical Phase 5 22/22 numbers describe the old verification run, not the replacement.
The separate legacy `runs/dual_memory/memory.pt` has 67 used slots with missing, unverifiable provenance; that is not proof all slots are contaminated, but it cannot be certified clean and was not used as evaluation memory.
The new checkpoint has 10 audited slots and 0 failures. Its registry covers the 12 real held-out pairs plus 1000 available synthetic validation/test records. Test-time retrieval runs read-only, including access statistics; the persisted snapshot is saved before test evaluation.

The guard uses domain-separated SHA-256, length-framed pair identities, exact input/output matching, and registered substrings of at least 16 bytes. It checks nested metadata and decodes `*_hex` fields; explicit test/validation split labels are blocked even without a registry. Whole ingestion batches, intermediate symbolic canvases and consecutive trace payload runs are checked before mutation. Guarded checkpoints persist the policy, reject unproven legacy slots and reject direct public bank/state loading.

**Limits:** it is opt-in for unlabeled embedding writes, depends on a complete caller-supplied registry and honest raw provenance, and cannot establish what is inside opaque vectors or pretrained weights. Near duplicates, short embedded substrings, alternate encodings and direct tensor mutation can evade it. It is not a security boundary or a privacy guarantee.

## 2. Autonomous edits and routing

Repository fixture: 10 train, 4 validation, 8 test; split by source file, not an external benchmark. The current run uses a fixed 128-action budget, with no target, oracle action, target-derived budget or host controller during generation. The earlier Phase 5 budget was target-derived, an additional evaluation caveat.
The real-trained checkpoint produces 5/10 train and 0/8 test examples exactly; train-memory conditioning gives 0/8, and priming the generic provenance explanation gives 0/8. These excerpts lack actionable edit instructions, so this is not a fair test of general instruction following.

Explicit route dispatch passed 7/7 declared cases. This is a deterministic dispatch check, **not** inferred routing accuracy or edit success:

| Synthetic command | Engine | Exact edits |
|---|---|---:|
| I | cognitive | 64/64 |
| D | cognitive | 64/64 |
| R | diffusion | 0/64 |
| M | cognitive | 64/64 |

For 100 exact-byte-novel synthetic inputs with 10% substitutions, the corrupted input itself matched 90.59% of original bytes; the diffusion output matched 90.55%. The denoiser mostly copies its input; it did not learn targeted replacements. Instruction-directed diffusion substitution raises `NotImplementedError` rather than silently ignoring the instruction. Same-length synthetic R tasks therefore fail under the requested route. Cognitive instruction priming is experimental and was not trained as general instruction following.

## 3. Diffusion reconstruction and training defaults

`evaluation/evaluate.py`: clean 99.73%; 139/500 synthetic test rows also occur exactly in train. The 361-row subset excludes those exact duplicates, but not near duplicates. 10% position-wise original recovery: substitution 91.28%, insertion 36.93%, deletion 38.12%. CPU one-pass inference: 1.624 ms/sample.
Soft-DTW weight and edit-corruption probability now both default to zero; the optional differentiable loss remains available. This changes future training, not the performance of any already-trained checkpoint. The earlier matched experiment does not establish that frozen target embeddings blocked gradients; backbone gradients were observed.

## 4. Paired halting evidence

Diffusion: calibration, selection and test each used 48 exact-byte-distinct synthetic sequences, with training-width padding, a last-timestep q-sampled input, the same initial state and matched per-step RNG. A temperature fit on validation yielded held-out per-byte ECE 0.362; byte/timestep observations are correlated, and this is not unconditional generation. The validation grid selected no policy under the strict no-per-sample-byte-loss criterion.
Test full schedule: 13.16% original-byte accuracy, 0/48 exact, 50.0 mean steps. An aggressive exploratory threshold used 1.0 step, 18.06% aggregate byte accuracy, but 13/48 samples lost correct bytes; both policies produced zero exact sequences. Paired production-loop behavior was checked against the trajectory simulator.
THINK: on 128 narrow synthetic tasks, the uncalibrated exploratory threshold 0.05 achieved 128/128 with 1.0 steps versus 128/128 at 4.0 steps. The existing default already uses 1.0 step and scores 128/128; there is no demonstrated saving over that default.
Config-driven halting requires a matching calibration scope and an explicit threshold, but that check does not certify quality or guard against distribution shift. Both default config switches remain false; explicit threshold arguments are warning-only experimental overrides.

## 5. Compatibility, cost and verification

No learned parameters were added: diffusion **195,905**, included generator **344,408**, memory at capacity 256 **83,565**. Hashing/substring scans and metadata increase CPU/storage; alignment remains quadratic when explicitly enabled. Guarded hierarchical memory saves schema 4; unguarded saves remain schema 3. Guarded concept checkpoints use a wrapper so old loaders fail rather than silently discard the policy. New code still loads legacy unguarded checkpoints, but guarded workflows reject old slots without provenance. Optimizers are not migrated across architecture changes. Concept-bank slot counting/FIFO was fixed, changing previously broken multi-concept writes.
Verification: **70 unit tests**, compilation, whitespace checks, tokenizer-import check, guarded ingestion and checkpoint reloads. See [verification.json](phase6/verification.json).

## Artifacts

- [Clean real-edit metrics](phase6/real_traces/metrics.json) and [guarded memory](phase6/real_traces/memory.pt).
- [Historical memory audit](phase6/legacy_audit.json).
- [Full diffusion report](phase6/evaluation/report.md) and [metrics](phase6/evaluation/evaluation.json).
- [Routing measurements](phase6/routing.json) and [paired halting measurements](phase6/halting.json).
- [Historical Phase 5 diagnostics](diagnostic_report.md).
