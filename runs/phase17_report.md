# Phase 17: training on real GitHub data

Phase 16 proved the architecture on **synthetic** trajectories: templated
requests, a sandbox the curriculum itself defined. Every observation in that
corpus was produced by code, but every request was invented. This phase asks
what changes when the requests are real.

Short answer: **routing transfers and copying does not.** On 448 held-out real
requests the model reaches the right tool 399/448 times and writes the
trained thought almost verbatim -- then produces the correct lookup key only
0/448 times, because a twelve-character sha is twelve
independent byte predictions and it gets 39.6% of them right even with
the gold prefix in front of it. Nothing below averages that away: the failure is
measured three different ways.

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

**Every action argument is recoverable from its request** (4475/4475). This is a
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

Real trajectories are longer than synthetic ones (median 441 bytes vs 190), so
`max_seq_len` moves from 512 to 768 -- the `github` preset in
`configs/default.yaml`, which changes nothing else. The model grows from
266,243 to **282,627** parameters,
all of it positional embedding.

## 17.4 Training -- three stages, because eight epochs was not enough

| epoch (stage 3) | lr | train loss | train next-byte acc | val loss | val next-byte acc | wall s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8.00e-04 | 0.6137 | 81.71% | 0.7974 | 76.44% | 437.0 |
| 2 | 7.70e-04 | 0.6049 | 81.95% | 0.7900 | 76.72% | 877.2 |
| 3 | 6.83e-04 | 0.5957 | 82.20% | 0.7798 | 76.99% | 1319.7 |
| 4 | 5.53e-04 | 0.5855 | 82.51% | 0.7695 | 77.14% | 1760.5 |
| 5 | 4.00e-04 | 0.5739 | 82.83% | 0.7577 | 77.63% | 2202.1 |
| 6 | 2.47e-04 | 0.5647 | 83.09% | 0.7522 | 77.74% | 2660.0 |
| 7 | 1.17e-04 | 0.5581 | 83.29% | 0.7444 | 77.96% | 3106.8 |
| 8 | 3.04e-05 | 0.5540 | 83.40% | 0.7413 | 78.02% | 3556.9 |

The first two stages followed Phase 16's recipe (8 epochs at 3e-3, then 8 at
1.2e-3). Val loss was still falling when stage 2 ended, so a third stage ran.

| | final train loss | val next-byte acc | test next-byte acc |
| --- | --- | --- | --- |
| stage 1 (8 epochs @ 3e-3, from scratch) | 0.6555 | 75.06% | not measured |
| stage 2 (+8 @ 1.2e-3) | 0.5872 | 76.91% | not measured |
| stage 3 (+8 @ 8e-4) | 0.5540 | 78.02% | 78.25% |

**Checkpoint selection is the fixed final epoch of stage 3**, not a choice made
after looking at the held-out score. The stage-1 and stage-2 checkpoints are
kept (`runs/phase17/foundation_stage1.pt`, `foundation_stage2.pt`) so the
comparison above can be re-run.

Objective: plain next-byte cross-entropy over every byte of every trajectory.
Optimiser AdamW, warmup then cosine decay, gradient clipping at 5.0, batches
bounded by a 4096-token budget. Loss
0.6137 -> 0.5540; peak gradient norm
0.61.

## 17.5 Held-out evaluation: real tools on unseen requests

`evaluation/github_eval.py` drives `ReactLoop` with the four **real** corpus
tools over the 448 held-out rows. The loop executes code; nothing is
replayed from the dataset. Three questions, in increasing order of difficulty.

| metric | Phase 16 model (never saw this data) | Phase 17 model | delta |
| --- | --- | --- | --- |
| routed to the right tool | 0/448 (0.00%) | **399/448** (89.06%) | 89.1% |
| argument byte-exact | 0/448 (0.00%) | **0/448** (0.00%) | 0.0% |
| answer grounded in the observation | 0/448 (0.00%) | **0/448** (0.00%) | 0.0% |
| answered at all | 1/448 | **183/448** | |
| answer matches the reference exactly | 0/448 | **0/448** | |
| lookup key the corpus actually contains | n/a | **77/399** | |

The baseline column is what a model that has never seen this data does: it
routes 0/448 of 448 requests to a real corpus tool, because the only tools it
knows are the sandbox ones. Its `valid_lookup_key` number is not comparable --
it is measured only over rows where a corpus tool ran.

**Grounding is the hallucination test.** For each tool, `grounding_facts`
extracts the fields the tool looked *up* -- a commit's author and date, a pull
request's state and comment count, the repository a search hit came from, a
README's character count. None of them appear in the request, and the
evaluation keeps only the facts that are genuinely absent from it
(`observation_only_facts`), so an answer containing one cannot have been
produced by copying the request. A test asserts this holds for **all**
448 held-out rows.

**Integrity check.** The observation the loop received is compared with the one
stored in the dataset: 0/448. Both come from
executing the same tool, so a mismatch would mean the evaluation is not
measuring what it claims to.

| kind | rows | routed | right tool | argument byte-exact | grounded |
| --- | --- | --- | --- | --- | --- |
| `github_commit` | 211 | 209 | 209 | 0 | 0 |
| `github_issue` | 46 | 1 | 1 | 0 | 0 |
| `github_pr` | 189 | 189 | 189 | 0 | 0 |
| `github_readme` | 2 | 2 | 0 | 0 | 0 |

### The anatomy of the failure

The aggregate numbers above are all-or-nothing because grounding depends on a
correct lookup key, and no key was copied correctly. Three finer measurements
say exactly where it breaks:

| | Phase 17 checkpoint |
| --- | --- |
| requests where a corpus tool actually ran | 399 / 448 |
| requests where the tool found the key | 77/399 |
| requests where the tool reported "not in the corpus" | 324 |
| ...and the model answered anyway | 182/324 |
| ...and the model wrote a tool-shaped result on a `Thought:` line | 132/448 |

The last row is the important one and it is not a formatting slip. The model
routes correctly (399/448), writes the trained thought
verbatim, and then emits an argument of the right *shape* with the wrong
*content* -- `explain_commit("cfe5ccfc3a2")` for a commit whose real sha is
`00c3ce0a55e5`, `review_pr("denoland/deno#3696")` for a request that names
`#36764`. The tool answers `no commit cfe5ccfc3a2 in the corpus`. The model then
writes, on a `Thought:` line, `commit a9cfec7cfc9ef by An Brouwer on 2026-10-29: ...`
-- a complete observation it invented -- and finishes with an answer that states
facts about a pull request that does not exist.

### Why: the arithmetic of copying a sha

`evaluation/copy_diagnostic.py` settles whether this is a capacity failure or an
exposure failure. It feeds the **gold** prefix for every position and scores only
the bytes between the quotes of the `Action:` argument -- what the model *can*
copy, not what it *does*.

| teacher-forced, gold prefix | synthetic arguments | real GitHub arguments |
| --- | --- | --- |
| rows with an argument | 440 | 448 |
| per-byte accuracy | 87.1% | 51.9% |
| per-byte accuracy on hex digits and digits | 63.4% | 39.6% |
| whole argument predicted exactly | 17/440 | 0/448 |

Both failure modes are present, and this is what separates them. With the gold
prefix, 17/440 synthetic arguments come out exactly right while
0/448 real ones do. So exposure -- having to live with your own
output -- costs something, and on the easy arguments the model can still pay
it. On the real ones it cannot: even with the answer in front of it the model
predicts an argument byte correctly only
51.9% of the time, and
39.6% on hex digits and digits. A twelve-character sha is twelve
independent such predictions, so the chance of copying one exactly is around
1 in 68,272 -- which is what 0/448 looks like from the inside. Phase 16's synthetic
arguments survived the same filter because they are short and templated
(`print(6 * 7)`), not because copying had been solved.

Two things follow, and both are honest limits rather than bugs:

1. **The loop's guarantee is narrower than "the model never fabricates".** It
   guarantees that every line on the `Observation:` channel came from a tool,
   and that anything the model wrote after an `Action:` is discarded. It does
   not stop the model from writing a tool-shaped result inside a `Thought:`.
   That is measured above rather than assumed away.
2. **A negative observation is not yet understood.** When the tool says the key
   does not exist, the model should say so. It does not; it answers as if the
   lookup had succeeded.

Phase 16 saw the same weakness in a milder form and named it: inside `Action:`
lines the model scored 98.8% on punctuation and 96.7% on letters but 89.7% on
digits, "the only bytes that must be copied out of the request". Real
identifiers are almost nothing *but* those bytes -- twelve hex digits, or a
number after a slash -- so the weakness that cost 18 of 60 arguments there
costs all 448 here.

### Did it forget the synthetic task?

Adding a harder, longer, noisier distribution can cost a small model what it
knew. The only fair way to see that is to score the Phase 17 checkpoint on
Phase 16's own test rows -- the same 480 rows, the same evaluator.

| metric (Phase 16's 480 synthetic test rows) | Phase 16 | Phase 17 |
| --- | --- | --- |
| next-byte accuracy | 94.63% | **92.03%** |
| next-byte accuracy, short sequences | 95.03% | **93.59%** |
| digits inside `Action:` | 89.7% | **32.3%** |
| letters inside `Action:` | 96.7% | **91.0%** |
| free-running argument byte-exact | 42/60 | **0/60** |

## 17.6 Targets

| target | measured | status |
| --- | --- | --- |
| held_out_tool_routing >= 0.90 | 89.06% | NOT met |
| held_out_argument_byte_exact >= 0.80 | 0.00% | NOT met |
| held_out_grounded_in_observation >= 0.60 | 0.00% | NOT met |
| observations_come_from_executing_the_tool | 399 rows ran a corpus tool; a unit test asserts the loop's observation equals a fresh call to the same function | met |
| phase12_vector_only_recall == 10/10 | 10/10 through the prefix integration | met |
| regression_suite_green | 281 tests pass (166.88 s), 23 of them new in Phase 17 | met |

Three of the six targets are met. **Routing, argument copying and grounding are
reported as unmet** rather than re-scored against a looser definition: the
thresholds above were fixed before the run, and the gap is not marginal
(0.00% against 0.80 and 0.60).

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
  here is a property of a 282,627-parameter model on 2 CPU cores.
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
