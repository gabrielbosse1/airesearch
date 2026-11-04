# 21.7 — Why only `edit` passed

Eval prompt: `'Request: Corrija: voce\nDraft: voce\n'`

| fact | value |
| --- | --- |
| Stage 4 edit weight | 50% |
| Unique edit training rows | 4 |
| Total edit training bytes | 324 |
| Stage 4 frozen blocks | 8/12 |
| Eval prompt == REINFORCE prompt | True |
| Eval and RL share the same reward | True |
| Eval prompt is a training-row prefix | [False, True, False, False] |
| Over-trained on edit? | True |

## Behavioural test

| probe | pass | output (first 100 chars) |
| --- | --- | --- |
| training/eval prompt | True | '[EDIT pos=3 delete=1 bytes=c3aa]\nAnswer: você' |
| held-out typo `obrigadoo` | False | 'DIT DIT poca=]\nAn de bocas  boce  de ve vocaaaa.\n' |

Longest common prefix with a training row: **45 characters** (exact suffix reproduced: True).

**Verdict:** The edit probe passed because Stage 4 spent 50% of its supervised updates on FOUR unique [EDIT] templates and the Stage 4 REINFORCE case uses the byte-identical evaluation prompt and reward. The model reproduces the memorised [EDIT] continuation (verbatim), yet fails a held-out typo (FAIL). This is memorisation of one test item, not an edit capability.
