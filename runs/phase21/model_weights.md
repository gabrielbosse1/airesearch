# 21.5 — Weight & gradient health

Checkpoint: `/home/user/airesearch/runs/phase21/checkpoints/baseline_stage4.pt` · 245,088 parameters

| check | result |
| --- | --- |
| weights_exploded_to_inf_or_nan | False |
| weights_vanished | False |
| dead_neurons | False |
| embedding_codebook_collapsed | False |
| vanishing_gradients | False |
| exploding_gradients | False |

| metric | value |
| --- | --- |
| NaN values | 0 |
| Inf values | 0 |
| near-zero fraction | 4e-06 |
| dead weight rows | 0/4396 |
| byte row norm (mean ± std) | 2.025386 ± 0.768367 |
| mean |cos| between byte rows | 0.636272 |
| live global grad norm | 16.382529 |
| max activation zero fraction | 0.0 |

**Interpretation:** No numerical pathology: weights are finite, diverse and gradients are healthy. The collapse is DISTRIBUTIONAL (the output distribution was rewritten by a tiny, repeated task corpus), not a weight-level blow-up.
