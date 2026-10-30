# Phase 17: training on real GitHub data

Phase 16 proved the architecture on **synthetic** trajectories: templated
requests, a sandbox the curriculum itself defined. Every observation in that
corpus was produced by code, but every request was invented. This phase asks
what changes when the requests are real.

Short answer: the protocol transfers, the copying transfers, and the part that
does **not** transfer is visible in the numbers instead of being averaged away.

## 17.1 Collection -- 4476 real records

`data/github_collector.py` pulls from the GitHub REST API with a token read from
`GITHUB_TOKEN` (never printed, never persisted) using only the standard library:

| | |
| --- | --- |
| repositories | 10 |
| records | 4476 |
| by type | {'readme': 10, 'issue': 496, 'pr': 1970, 'commit': 2000} |
| characters | 7,666,483 |
| API requests | 80 |
| authenticated | True (limit 5000/h) |

Every record carries the SPDX licence of the repository it came from
(denoland/deno: MIT, facebook/react: MIT, home-assistant/core: Apache-2.0, huggingface/transformers: Apache-2.0, microsoft/vscode: MIT, python/cpython: NOASSERTION, pytorch/pytorch: NOASSERTION, rust-lang/rust: Apache-2.0, scikit-learn/scikit-learn: BSD-3-Clause, tensorflow/tensorflow: Apache-2.0).
Content is scrubbed heuristically -- e-mail addresses, URLs and long opaque
blobs become placeholders. That reduces accidental personal data; it does not
guarantee its absence, and the manifest says so in those words. Commit shas are
abbreviated to 12 characters, matching git's default abbreviation, to keep
sequences short.

**Commit-list economics.** One `/commits` page returns 100 messages, while
`/commits/{sha}` costs one request each. Commit observations are therefore built
from the list payload (sha, author, date, subject) and individual commits are
never fetched.

## 17.2 Conversion -- every observation is a real tool result

`data/github_to_react.py` renders 4475 ReAct trajectories
({'issue': 495, 'pr': 1970, 'commit': 2000, 'readme': 10}) over four corpus tools:
`explain_commit`, `review_pr`, `search_issues`, `summarize_readme`.

The sketch for this module synthesised its observations -- "Found 3 similar
issues", "PR modifies 5 files". Those numbers would have been invented, and
training on invented observations teaches the one failure this architecture
cannot tolerate, because the observation is the only channel that reports what
the world actually did. So:

* every `Observation:` in `github_react.jsonl` is the **return value of a real
  function over the collected corpus**, executed at conversion time;
* `python -m data.github_to_react --verify N` re-executes a random sample and
  **raises** unless every observation reproduces byte for byte;
* records with no verifiable observation are **skipped**, not filled in
  (1 skipped).

**Every action argument is recoverable from its request**
(4475/
4475). This is a
property of the data, not a hope: an argument that cannot be read out of the
request would have to be invented, so the requests were rewritten to carry the
identifier the tool needs (`Explain commit <sha>: <subject>`,
`Review pull request <owner/repo#number>: <title>`).

## 17.3 Mixture and a held-out split that means something

`data/mixed_react_dataset.py` combines the Phase 16 synthetic corpus with the
real trajectories, splitting the real rows **by record** (a stable hash of the
tool argument), so no record appears on both sides. An identical-distribution
split of a 2 MB corpus would be answered by memorisation.

| split | rows | bytes | real rows | real share of rows | longest |
| --- | --- | --- | --- | --- | --- |
| train | 14780 | 4,437,093 | 3580 | 24.22% | 690 |
| validation | 927 | 324,220 | 447 | 48.22% | 690 |
| test | 928 | 322,648 | 448 | 48.28% | 678 |

Real trajectories are longer than synthetic ones (median
441 bytes vs
190), so
`max_seq_len` moves from 512 to 768 -- the `github` preset in
`configs/default.yaml`, which changes nothing else. The model grows from
266,243 to **282,627** parameters,
all of it positional embedding.

## 17.4 Training

_training history is unavailable_

Objective: plain next-byte cross-entropy over every byte of every trajectory.
Optimiser AdamW, warmup then cosine decay, gradient clipping at 5.0, batches
bounded by a 4096-token budget. Loss
0.0000 -> 0.0000; peak gradient norm
0.00.

## 17.5 Held-out evaluation: real tools on unseen requests

`evaluation/github_eval.py` drives `ReactLoop` with the four **real** corpus
tools over the n/a held-out rows. The loop executes code; nothing is
replayed from the dataset. Three questions, in increasing order of difficulty.

| metric | Phase 16 model (never saw this data) | Phase 17 model | delta |
| --- | --- | --- | --- |
| routed to the right tool | 0/448 (0.00%) | **n/a** (n/a) | n/a |
| argument byte-exact | 0/448 (0.00%) | **n/a** (n/a) | n/a |
| answer grounded in the observation | 0/448 (0.00%) | **n/a** (n/a) | n/a |
| answered at all | 1/448 | **n/a** | |
| answer matches the reference exactly | 0/448 | **n/a** | |

**Grounding is the hallucination test.** For each tool, `grounding_facts`
extracts the fields the tool looked *up* -- a commit's author and date, a pull
request's state and comment count, the repository a search hit came from, a
README's character count. None of them appear in the request, and the
evaluation keeps only the facts that are genuinely absent from it
(`observation_only_facts`), so an answer containing one cannot have been
produced by copying the request. A test asserts this holds for **all**
448 held-out rows.

**Integrity check.** The observation the loop received is compared with the one
stored in the dataset: n/a. Both come from
executing the same tool, so a mismatch would mean the evaluation is not
measuring what it claims to.

_per-kind numbers are unavailable (the evaluation has not run)_

## 17.6 Targets

| target | measured | status |
| --- | --- | --- |
| held_out_tool_routing >= 0.90 | not measured | NOT MET |
| held_out_argument_byte_exact >= 0.80 | not measured | NOT MET |
| held_out_grounded_in_observation >= 0.60 | not measured | NOT MET |
| every_scored_observation_reproduced_by_the_tool | see below | pending |
| phase12_vector_only_recall == 10/10 | see below | pending |
| regression_suite_green | see below | pending |

## 17.7 Limits, stated plainly

* **The corpus is one afternoon of ten repositories** -- 7,666,483 characters. That
  is enough to ask whether the protocol transfers to real text. It is not
  enough to claim competence on GitHub, or on any repository outside these ten.
* **`search_issues` is lexical**, not semantic: token overlap over titles and
  bodies, stopwords removed. A "relevant" hit means "shares words", nothing more.
* **Observations are capped at 200 bytes**, so long pull-request bodies and
  READMEs are truncated mid-sentence. The model sees a truncated world and was
  trained on a truncated world; a longer cap changes the task.
* **Grounding is a substring test**, not entailment. An answer that quotes the
  author's name is grounded even if the rest of it is wrong. It is a floor on
  honesty, not a ceiling on correctness.
* **Scrubbing is heuristic.** E-mail addresses, URLs and long blobs are
  replaced; personal data in prose is not detected.
* **Scale is unchanged**: d_model 64, 4 layers. Everything
  here is a property of a 0-parameter model on 2 CPU cores.
  Nothing about the code path would differ at the production preset.

## Reproduce

```bash
export GITHUB_TOKEN="$(gh auth token)"
python -m data.github_collector --repos pytorch/pytorch,tensorflow/tensorflow \
    --max-items-per-kind 200 --output-dir data/github_raw
python -m data.github_to_react --verify 60
python -m data.mixed_react_dataset --output-dir runs/phase17
OMP_NUM_THREADS=2 python -m training.train_foundation --preset github \
    --data-dir runs/phase17 --output-dir runs/phase17 --epochs 8 --lr 0.003
python -m evaluation.github_eval --model runs/phase17/foundation.pt \
    --baseline runs/phase16/foundation.pt
python -m demo.github_agent --once "Summarize the project python/cpython."
python -m pytest tests/ -q
```
