# Phase 13 feasibility report

## Decision

**History-dependent routing: NO-GO for full-scale training under the requested criteria.** Five-epoch train loss fell 45.7%, but final validation accuracy was 45% (binary balanced chance 50%); the required >60% was not achieved. Gradients were finite and nonzero.

**Chunked generation: NO-GO for full-scale training under the requested criteria.** Training loss fell 70.5%, but validation mean normalized byte similarity was 26.7%, with 0/5 structurally valid long Python outputs. The required >40% and coherent chunks were not achieved.

These are go/no-go recommendations for the specified five-epoch probes, not proof that the architecture cannot learn. The chunk experiment did not train a cognitive THINK policy or verifier, so it cannot independently establish feasibility of cognitive verification.

## Stage 1 consolidation comparison

| Metric | Phase 12 baseline | Experimental candidate | Requested target |
|---|---:|---:|---:|
| Context greetings | 12/20 | 17/20 | >18/20 |
| Session intents | 43/50 | 41/50 | >42/50 |
| Routed response distinctness | 15/20 | 13/20 | >18/20 |
| Direct decoder distinctness | 15/20 | 16/20 | diagnostic |
| Phase 10 development (no history) | 298/384 | 248/384 | retain performance |
| Candidate vector-only recall | — | 10/10 | 10/10 |
| Regression suite | — | 177/177 | >=174 |

The intent candidate used 88 preflighted synthetic greeting rows and 1200 replay contextual/no-context rows; selected epoch 1 by small greeting validation plus replay **training** accuracy. The response candidate was trained from scratch for 20 epochs with differentiable diversity weight 0.1 on 32 CHAT response templates. For the routed candidate, 16/20 used the diverse decoder; its direct generations were complete and valid text 16/20. Fixed-seed distinctness measures surface bytes, not appropriateness.

Ten response variations for confirmation, question-answer, and clarification are provided as fixture text only, **not used to train those intents**: the diverse decoder is CHAT-only, while the session handles the other intents through separate safety/grounding paths. No general per-intent diversity claim is supported.

The candidate changed intent encoder/head tensors. The other generator tensors were checked bit-for-bit, and three response/reader checkpoint copies were explicitly rebound to the new generator hash for this experiment. This does not guarantee calibration: recall was rerun end to end (including 500 filler turns and fresh interpreter), but Phase 10 and session regressions remain. Original Phase 12 checkpoints remain unchanged; do not promote the candidate.

## History-dependent routing

Twenty-five distinct prompts each have benign/unsafe-history pairs: 50 training rows. Ten disjoint validation prompts yield 20 rows, with held-out histories. The labels are synthetic and subjective; a classifier cannot be treated as a safety guard. Frozen causal backbone features carry history; the convolutional intent encoder focuses only on the current message. Only the intent encoder and seven-way classifier are trained; the balanced two-label chance baseline is 50%.

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Gradient norm mean (min–max, pre-clip) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.7599 | 52.0% | 1.8837 | 45.0% | — |
| 1 | 1.3544 | 52.0% | 1.2094 | 60.0% | 5.730 (2.803–8.584) |
| 2 | 1.2526 | 50.0% | 1.1763 | 50.0% | 8.040 (2.465–17.883) |
| 3 | 1.0996 | 56.0% | 1.0024 | 60.0% | 8.162 (2.300–21.929) |
| 4 | 1.0246 | 54.0% | 0.9340 | 60.0% | 5.940 (2.795–9.264) |
| 5 | 0.9553 | 52.0% | 0.9291 | 45.0% | 5.768 (2.974–10.706) |

Gradient norms are aggregate, measured before clipping to 5; all were finite and nonzero, although some exceeded the clipping threshold. Lower cross-entropy did not translate to consistent accuracy: validation varied between 45% and 60%. A plausible bottleneck is that only the frozen backbone conveys history while the trainable n-gram encoder is restricted to the prompt; insufficient data, optimization, class imbalance relative to the original seven-way task, and the labels themselves are also viable explanations. This experiment does not isolate cause or model capacity.

## Chunk generation

Twenty train and five disjoint validation examples each contain 120 target bytes as four 30-byte chunks. Every target is valid Python with a synthetic repeated scale-function template and padded comment. Five epochs teacher-force the existing style-conditioned autoregressive decoder on raw bytes and EOS, with a fixed style; targets are not passed to generation. The existing adapter emits chunk/THINK metadata and uses an accept-all callback. There is no trained THINK action, halt policy, or verifier; controls are not predicted. No probe weights are saved.

| Epoch | Train NLL | Validation NLL | Gradient norm mean (min–max, pre-clip) |
|---:|---:|---:|---:|
| 0 | 6.8431 | 6.8954 | — |
| 1 | 4.4771 | 4.5337 | 5.436 (3.982–8.155) |
| 2 | 3.2603 | 3.3162 | 2.473 (2.129–2.792) |
| 3 | 2.6119 | 2.6623 | 1.450 (1.103–1.892) |
| 4 | 2.2818 | 2.3419 | 1.139 (1.024–1.195) |
| 5 | 2.0165 | 2.0772 | 0.915 (0.854–0.981) |

Free-running train: 21.2% mean byte similarity, 0/20 structural proxy. Validation: 26.7% mean similarity, 0/5 structural proxy, 5/5 printable UTF-8. Similarity is 1 − Levenshtein/max(output length, reference length); it penalizes valid alternatives. Structural proxy requires at least 100 bytes and a parseable top-level Python function; it does not check correctness. Printability is not chunk coherence. The resulting outputs do not satisfy the coherence criterion.

A rapid decline in teacher-forced NLL alongside poor free-running generation is consistent with exposure bias, insufficient training/data, social-decoder pretraining mismatch, or inadequate conditioning. The synthetic template and five validation rows are too small to diagnose capacity or predict open-domain code performance.

## Limitations and next decision

- Both probes start from the experimental candidate, not the untouched Phase 12 checkpoints. Five epochs, one seed, tiny synthetic sets, a single optimizer, and no independent blind evaluation limit conclusions. Existing Phase 10/12 evaluation fixtures were previously inspected; the consolidation tests are development comparisons, not pristine holdouts.
- An initial routing probe was inspected before a holdout preflight revealed overlaps with historical protected bytes. Conflicting training phrases/history were replaced and the probe rerun without changing hyperparameters; validation was therefore observed previously and is developmental. The final script preflights both probe datasets against the historical registry and the new validation records. Exact hash guards do not prevent paraphrase leakage or audit pretrained weights.
- The 50 session test cases are selected across seven Phase 12 intent classes, but special vector recall/forget routes deliberately use current-only features. Accuracy reports model-predicted intent, not policy-action correctness.
- Recommend retaining Phase 12, not the experimental checkpoint. Before any full-scale Phase 13 training, seek a user decision. If authorized to continue research, next isolate a trainable history path with balanced contextual data and a genuinely held-out multi-seed test; compare teacher-forced and free-running baselines, and design an actual learned verification/stop interface with semantic tests. Do not promise target accuracy from these probes.

## Reproduction

Install requirements, then:

```sh
python data/generate.py  # deterministic ignored holdout registry
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m training.consolidate_sessions
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m training.train_diverse_responses --generator runs/phase12/generator.pt --output-dir runs/phase12_consolidation/diverse_training --epochs 20 --steps-per-epoch 15 --diversity-weight 0.1
python -m training.rebind_consolidation
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m evaluation.consolidation_eval
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m evaluation.memory_eval --generator runs/phase12_consolidation/generator.pt --response runs/phase12_consolidation/response.pt --vector-reader runs/phase12_consolidation/vector_reader.pt --diverse-response runs/phase12_consolidation/diverse_response.pt --output runs/phase12_consolidation/memory_eval.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m training.phase13_feasibility
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests
```

Machine-readable outputs: `runs/phase12_consolidation/{intent_training,benchmark,baseline,memory_eval}.json`, `runs/phase12_consolidation/diverse_training/diverse_response_metrics.json`, and `runs/phase13_feasibility.json`.
