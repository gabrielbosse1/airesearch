# 21.6 — Activation probe

Checkpoint: `/home/user/airesearch/runs/phase21/checkpoints/baseline_stage4.pt` · input `Olá, como você`

| flag | result |
| --- | --- |
| early_layers_produce_reasonable_vectors | True |
| middle_layers_transform_activations | True |
| final_distribution_valid | True |
| garbage_propagating | False |
| dead_layers | 0 |

## Layer statistics

| layer | std | mean | max |x| | zero frac | nonfinite |
| --- | --- | --- | --- | --- | --- |
| block_01 | 1.892971 | -0.057553 | 6.306652 | 0.0 | 0 |
| block_02 | 2.041028 | -0.079387 | 6.814827 | 0.0 | 0 |
| block_03 | 2.137637 | -0.092072 | 7.258292 | 0.0 | 0 |
| block_04 | 2.208399 | -0.09287 | 7.426204 | 0.0 | 0 |
| block_05 | 2.278913 | -0.095143 | 7.530763 | 0.0 | 0 |
| block_06 | 2.334721 | -0.091388 | 7.72535 | 0.0 | 0 |
| … | … | … | … | … | … |
| final_norm | 1.640425 | -0.015239 | 5.217826 | 0.0 | 0 |
| logits | 4.622903 | -5.766261 | 11.744143 | 0.0 | 0 |
| positional | 0.092517 | -0.001799 | 0.353656 | 0.0 | 0 |

## Final softmax

| metric | value |
| --- | --- |
| valid distribution | True |
| softmax sum | 1.0 |
| entropy / max entropy | 0.9516 / 5.591 |
| uniformity | 0.1702 |
| argmax | 32 ' ' |

| probability mass | value |
| --- | --- |
| raw_bytes_0_255 | 0.99972 |
| BOS_256_untrained_target | 1e-05 |
| EOS_257 | 0.000268 |
| PAD_and_reserved_258_267 | 2e-06 |
| ascii_letters | 0.061109 |
| space | 0.815451 |
| utf8_continuation_0x80_0xbf | 0.00054 |
| ascii_control_0_31 | 0.000175 |

Top 10 ids: 32:' '=0.8155, 44:','=0.0714, 195:'Ã'=0.0171, 46:'.'=0.0143, 63:'?'=0.0090, 117:'u'=0.0089, 115:'s'=0.0087, 109:'m'=0.0065, 114:'r'=0.0050, 97:'a'=0.0044

**Interpretation:** Activations are finite and transform across layers: the network is computationally healthy. The failure is in WHAT the output distribution encodes, not in garbage propagating through dead or exploding layers.
