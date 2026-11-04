# 21.3 — Training data verification

Source: `local reconstruction (notebook.dataset_discovery._local_sources)`

## Group sizes

| group | rows |
| --- | --- |
| code | 703 |
| edit | 4 |
| en | 3099 |
| minecraft | 21 |
| mineflayer | 521 |
| pt | 0 |
| react | 3587 |
| search | 4 |

## Checks

| check | result |
| --- | --- |
| rows scanned | 7939 |
| hard-corrupted text rows | 0 |
| rows flagged for review | 85 |
| encode/batch round-trip failures | 0 |
| max token id seen | 257 |
| malformed ReAct rows | 0 |
| Mineflayer rows with static issues | 0 |
| PT-BR suspicious rows | 0 |

## Effective exposure per unique row, per optimizer step (32 sequences)

### Stage 1 (freeze 0 blocks)

| group | weight % | unique rows | exposures/row/step |
| --- | --- | --- | --- |
| code | 20.0 | 703 | 0.0091 |
| en | 30.0 | 3099 | 0.0031 |
| pt | 50.0 | 200 | 0.08 |

Max/min exposure ratio: **25.8** (estimated offline: ['pt']); balanced weights: `{'pt': 50.0, 'en': 30.0, 'code': 20.0}`

### Stage 2 (freeze 6 blocks)

| group | weight % | unique rows | exposures/row/step |
| --- | --- | --- | --- |
| code | 40.0 | 703 | 0.0182 |
| mineflayer | 40.0 | 514 | 0.0249 |
| pt | 20.0 | 200 | 0.032 |

Max/min exposure ratio: **1.8** (estimated offline: ['pt']); balanced weights: `{'code': 40.0, 'mineflayer': 40.0, 'pt': 20.0}`

### Stage 3 (freeze 8 blocks)

| group | weight % | unique rows | exposures/row/step |
| --- | --- | --- | --- |
| code | 30.0 | 703 | 0.0137 |
| react | 40.0 | 3587 | 0.0036 |
| search | 30.0 | 4 | 2.4 |

Max/min exposure ratio: **666.7** (estimated offline: []); balanced weights: `{'react': 55.65, 'search': 2.61, 'code': 41.74}`

### Stage 4 (freeze 8 blocks)

| group | weight % | unique rows | exposures/row/step |
| --- | --- | --- | --- |
| code | 15.0 | 703 | 0.0068 |
| edit | 50.0 | 4 | 4.0 |
| mineflayer | 15.0 | 514 | 0.0093 |
| pt | 20.0 | 200 | 0.032 |

Max/min exposure ratio: **588.2** (estimated offline: ['pt']); balanced weights: `{'edit': 5.88, 'pt': 37.65, 'code': 28.24, 'mineflayer': 28.24}`

## Evaluation contamination

| case | contaminated | RL matches |
| --- | --- | --- |
| pt-br | True | pt |
| code | True | code |
| search | False | — |
| edit | True | edit |
| memory | False | — |

**Verdict:** The byte pipeline is clean (0 invalid UTF-8, 0 split codepoints, 0 malformed ReAct, 0 Mineflayer syntax errors) apart from ~1% low-density source mojibake. Stage 3 and Stage 4 give 30% and 50% of every optimizer step to groups containing only FOUR unique rows (exposure ratios 667x and 588x). Stage 4 additionally RL-trains on three evaluation prompts verbatim.
