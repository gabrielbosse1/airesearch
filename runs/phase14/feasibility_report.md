# Phase 14 redesigned training: feasibility report

## Decision: NO-GO for promotion or full-scale commitment

The prescribed GO requires validation >60%, Phase 10 >280/384, and healthy gradients. At the fixed tenth epoch routing validation was **65.5%**, but Phase 10 was **277/384**, below GO. This is between the specified GO and NO-GO retention cutoffs (not below 270), so the numerical result is **inconclusive under the literal thresholds**; the operational decision is **NO-GO for promotion** because the GO conjunction and generation targets are unmet. No test-based epoch was selected. Epochs 6 and 8 did reach 281/384 and >60% validation, but those development numbers were monitored; selecting them after inspection would require an untouched evaluation.

The learned verifier classified synthetic shuffled negatives accurately, but **0/10** verified generated outputs met the long-Python structural criterion and the verifier did not reject any of the generated chunks. The verified and ungated outputs were identical. The response distinctness target >18/20 was met for one of seven direct intent tests, not for all intents. These are not grounds for a capacity-limit claim.

## Stable baseline and data

The Phase 12 generator (SHA-256 `96b53fb369c20a552781d098906ddcf3e3a63050d66487a6762b77a8bca807e1`) was the starting checkpoint for all three experiments; Phase 12 weights were not overwritten. Routing: 800 training / 200 validation rows (80/20), with 25 identical prompt wordings each paired across safe/risky prior histories. History topics are entity-disjoint, but grammatical templates and current prompts are shared across splits. Code: 120 train / 30 validation examples (80/20), disjoint function identifiers with shared six arithmetic templates. The verifier produced 1,200 train / 300 validation labeled chunks from those outputs. Both routing and verifier training fixtures passed the historical holdout guard; exact hashes do not ensure semantic independence. For verifier validation, 129/300 candidate chunk byte strings also occurred in training even though complete (prefix, chunk) pairs were disjoint. Neither synthetic labels nor shuffled-byte negatives define general safety or semantic coherence.

## History routing: 10-epoch replay + EWC probe

The existing intent encoder/head were trainable; the generator backbone was frozen. Every optimization batch had 50% new rows and 50% Phase 12 historical intent replay, with round-robin class stratification and no repeated old row within a batch. Diagonal EWC used per-example squared gradients from 70 class-balanced replay examples; λ=20.0, plus teacher-logit KL weight 0.5. This is empirical Fisher on replay, not a guarantee of protected capabilities. There is no no-EWC/no-replay ablation, so any benefit cannot be attributed to EWC. Phase 10 test accuracy was monitored each epoch but never used to select a checkpoint.

| Epoch | Train loss | Validation loss | Validation accuracy | Phase 10 correct / 384 | Mean grad (min–max), pre-clip |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.0359 | 2.1640 | 38.5% | 298 | — |
| 1 | 1.3058 | 1.3038 | 52.0% | 226 | 3.664 (1.518–7.638) |
| 2 | 1.1949 | 1.1805 | 53.0% | 237 | 2.435 (0.997–7.152) |
| 3 | 1.1017 | 1.0660 | 56.0% | 254 | 2.260 (0.676–5.307) |
| 4 | 1.0287 | 0.9998 | 56.5% | 271 | 2.316 (0.949–6.459) |
| 5 | 0.9628 | 0.9186 | 61.5% | 275 | 2.420 (1.005–5.295) |
| 6 | 0.9157 | 0.8908 | 63.0% | 281 | 1.850 (0.626–3.870) |
| 7 | 0.8660 | 0.8245 | 63.5% | 274 | 2.136 (0.540–4.500) |
| 8 | 0.8429 | 0.7996 | 63.5% | 281 | 2.400 (0.628–5.174) |
| 9 | 0.8037 | 0.7757 | 65.5% | 273 | 2.650 (0.804–5.807) |
| 10 | 0.7807 | 0.7475 | 65.5% | 277 | 1.954 (0.601–6.114) |

Routing gradients were finite and nonzero; the largest pre-clip norm was 7.64 and clipping was at 5. Baseline Phase 10 development accuracy was 298/384. Final EWC penalty was small (mean unweighted 0.0000521 in epoch 10), so this implementation did not prevent the observed 21-case drop. The Phase 10 set has been inspected in prior phases; it is not a pristine test.

## Intent-conditioned response variations

A separate seven-intent decoder was trained from scratch on 12 curated responses per intent, eight training prompts, and an intent embedding, with NLL plus differentiable probability-distribution diversity weight 0.1. Two different prompts provide an 80/20 prompt split; response targets are shared, so this tests reuse of seen variations, not generalization to unseen responses. Validation teacher-forced exactness reached 46/84. Each result below is 20 stochastic samples at temperature 0.3 for a single held-out prompt and supplied gold intent, not routed session output. `_visible_text` and EOS measure only basic validity, not correctness, grounding, grammar, or safe effects.

| Intent | Distinct / 20 | Complete and visible / 20 | Exact seen variants / 20 |
|---|---:|---:|---:|
| chat | 15 | 20 | 6 |
| question | 19 | 20 | 4 |
| edit | 17 | 20 | 7 |
| remember | 14 | 20 | 5 |
| recall | 17 | 20 | 8 |
| forget | 14 | 20 | 14 |
| clarify | 18 | 20 | 8 |

This experimental decoder is not wired into session actions. In particular, generated text is never authority to assert that a memory, edit, or deletion happened. This preserves the Phase 12 safety/grounding paths.

## Learned verification and chunk generation

The experimental `VerifiedChunkGenerator` wraps the byte decoder with a separately trained convolutional binary head over the bounded accepted-prefix suffix and candidate bytes. At each chunk boundary, a score below the fixed 0.5 threshold rejects the candidate and stops; the raw output never contains control tags. It does not use the generator scratchpad, the input prompt, semantic labels, a learned DONE policy, or a repair/retry action. The head was trained on original vs shuffled candidate chunks, with the same byte multiset. High classification accuracy is limited to this synthetic discrimination task.

| Epoch | Decoder train NLL | Decoder val NLL | Verifier train BCE | Verifier val accuracy | Mean decoder grad (max) | Mean verifier grad (max) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.6490 | 8.6657 | — | 50.0% | — | — |
| 1 | 4.7088 | 4.7074 | 0.6325 | 96.3% | 45.058 (562.993) | 0.323 (0.501) |
| 2 | 2.5635 | 2.5817 | 0.3750 | 99.7% | 3.195 (4.049) | 0.718 (1.129) |
| 3 | 1.5037 | 1.5225 | 0.1205 | 99.7% | 2.158 (5.206) | 0.595 (0.998) |
| 4 | 1.0004 | 1.0211 | 0.0375 | 99.7% | 1.333 (2.487) | 0.310 (0.706) |
| 5 | 0.7529 | 0.7759 | 0.0163 | 99.7% | 0.569 (0.778) | 0.133 (0.233) |
| 6 | 0.5997 | 0.6235 | 0.0098 | 99.7% | 0.406 (0.494) | 0.099 (0.172) |
| 7 | 0.4845 | 0.5097 | 0.0067 | 100.0% | 0.286 (0.376) | 0.064 (0.108) |
| 8 | 0.3935 | 0.4197 | 0.0051 | 99.7% | 0.295 (0.401) | 0.063 (0.157) |
| 9 | 0.3230 | 0.3510 | 0.0040 | 99.7% | 0.218 (0.309) | 0.054 (0.103) |
| 10 | 0.2727 | 0.3001 | 0.0031 | 99.7% | 0.181 (0.226) | 0.035 (0.070) |

The first decoder epoch had a **562.99 pre-clip gradient norm spike** (clipped at 5); later norms were much smaller and finite. It would be misleading to call all gradients healthy without qualification. At epoch 10 the verifier classified 150/150 positive and 149/150 shuffled-negative validation chunks correctly. However, free-running generated outputs for the first ten validation prompts were all malformed: verified **0/10** vs ungated **0/10** met the >=100 byte + Python AST function proxy. All candidate chunks were accepted, and verified and ungated byte streams matched. Mean reference byte similarity was **82.0%** on a repetitive shared-template task, showing why edit distance alone cannot validate code. AST parsing does not check function behavior. The ten generation prompts are from the same validation partition used for the verifier/NLL metrics, not an independent final test.

## Regression and operational cautions

- Stable Phase 12 vector-only recall reran **10/10**, including 500 filler turns and a fresh interpreter. This is a baseline check only: vector recall was **not** revalidated with the Phase 14 routing checkpoint. Changing intent features may require revalidating and explicitly rebinding paired checkpoints.
- Repository regression suite: **182/182 passed**. Unit tests cover 80/20 splits, paired labels, differentiable EWC, class-balanced replay, verifier padding, and that the learned-head score can reject a chunk.
- Only one seed, one optimizer configuration, synthetic templates, and small held-out entity groups were tested. The validation accuracy is not a safety or open-domain capability metric. Because risk labels and intent interpretation depend on context, semantic errors and ambiguous labels remain possible.
- Phase 14 checkpoint artifacts are experiments and have not been integrated with or promoted into the demo. `runs/phase14/routing_generator.pt` is not paired with the Phase 12 response/reader checkpoint hashes. Do not deploy it.

## Reproduction

Install requirements. `python data/generate.py` creates the deterministic ignored historical holdout registry if missing. Then:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m training.phase14_routing
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m training.phase14_responses
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m training.phase14_chunks
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m evaluation.memory_eval --output runs/phase14/memory_baseline.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests
```

Full per-epoch metrics and outputs are in `runs/phase14/{routing_metrics,response_metrics,chunk_metrics}.json`; stable recall is in `memory_baseline.json`.

## Next steps, if authorized

Use a genuinely independent semantic verification set with realistic model-generated negative chunks, include the prompt and accumulated output, and compare against deterministic syntax checks and an ungated baseline. For routing, compare replay-only vs replay+EWC, estimate Fisher on more diverse historical tasks, and select on separate validation retention rather than the inspected Phase 10 test. Evaluate multiple seeds, independent prompts/templates, and end-to-end side effects before considering promotion.
