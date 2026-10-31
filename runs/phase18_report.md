# Phase 18: reference-based arguments and grounded refusal

Phase 17 measured three failures precisely, and this phase addresses each of
them with a mechanism rather than with more epochs.

| Phase 17 result | Phase 18 response |
| --- | --- |
| argument byte-exact **0/448** | the model points at an entity in the request instead of spelling it |
| answered anyway after "not in the corpus" **182/324** | 1,164 refusal trajectories, weighted 2.0x |
| synthetic digits inside `Action:` fell to **32.3%** | every batch is fixed at 50% synthetic / 50% real |

## 18.1 Reference-based arguments

Phase 17's diagnosis was arithmetic, not hand-waving: with the **gold prefix**
in front of it, the model predicted a real argument byte correctly 51.9% of the
time and 39.6% on hex digits, so a twelve-character sha sat at about 1 in 68,272.
No amount of the same training fixes a per-byte rate raised to the twelfth
power -- the task has to stop asking the model to spell the identifier.

**The mechanism.** Entities in the request are wrapped in markers and the action
points at them by index:

```
Request: Explain commit [REF]00c3ce0a55e5[/REF]: Update coverage to 7.17.0
Thought: The user quotes a commit; I should look up who made it and what it says.
Action: explain_commit("[REF]1[/REF]")
Observation: commit 00c3ce0a55e5 by renovate[bot] on 2026-10-30: Update coverage to 7.17.0
Answer: [REF]1[/REF] is a commit by renovate[bot] on 2026-10-30: Update coverage to 7.17.0
```

Two control ids were added to the foundation: `REF_START = 259`, `REF_END = 260`
(vocabulary 259 -> 261). The host resolves `[REF]N[/REF]` against the request
**before** calling the tool, so the bytes a tool receives are bytes the user
typed.

**The safety rule is the point.** A reference resolves only if it names an
entity the request actually contains; anything else fails closed with
`ERROR: unresolved reference: ...` and the tool is never called. The model is
therefore incapable of passing a hallucinated identifier to a tool -- not
because it learned not to, but because there is no path for it to do so. This
is a stronger guarantee than "the model usually copies correctly", and unlike a
learned behaviour it cannot decay.

The resolution lives in `models.react.reference_tools`, a wrapper applied at the
point where actions become tool calls. `ReactLoop` is untouched -- it still owns
no policy -- and the wrapper works for the sandbox tools as well as the GitHub
ones.

## 18.2 Grounded refusal

1,164 refusal trajectories in the training split
(128 validation, 108 test), covering five
kinds of failure: a commit sha that is not in the corpus, a pull request that
does not exist, a query that matches nothing, a repository with no README, and
an out-of-range reference.

**Nothing in them is synthesised.** Each identifier is a real lookup that
really fails, and the trajectory is kept only if the tool genuinely reported
absence. The generator verifies this at build time and discards anything that
resolves.

```
Request: Explain commit [REF]4d08e7ad9bea[/REF]: which I cannot see in the corpus
Thought: The user quotes a commit; I should look up who made it and what it says.
Action: explain_commit("[REF]1[/REF]")
Observation: no commit 4d08e7ad9bea in the corpus
Answer: I could not find that commit in the corpus.
```

**Weighting.** The bytes of the `Answer:` line carry weight
**2.0** whenever the observation reports a failure
(an `ERROR`, or a "not found" / "no commit" / "no pull request" / "no issue" /
"no README" / "unresolved reference" marker). Refusals are ~9% of the corpus;
without the extra weight the model averages them away against thousands of
successful lookups. The marker is matched against the observation line only, so
a commit subject containing the word "error" cannot trigger it.

## 18.3 Balanced batches against forgetting

Phase 17 mixed the real corpus into the synthetic one and the model lost the
easy task: digits inside `Action:` fell from 89.7% to 32.3% and free-running
arguments from 42/60 to 0/60. Phase 18 fixes the composition **per batch**, not
per epoch:

```
--balance "exec,file,chat,repair,memory:0.5;github:0.5"
```

Each step draws from both groups in a fixed proportion. One implementation
detail matters: the groups are kept as **separate sub-batches** and their losses
summed, rather than concatenated into one tensor. Concatenation pads every short
synthetic trajectory out to the length of the longest real one and measured at
roughly twice the compute per epoch for no benefit.

## 18.4 Benchmarks

| split | rows | bytes | synthetic | real | refusal |
| --- | --- | --- | --- | --- | --- |
| train | 15944 | 4,875,591 | 11200 | 3580 | 1164 |
| validation | 1055 | 374,594 | 480 | 447 | 128 |
| test | 1035 | 366,864 | 480 | 447 | 108 |

### Reference resolution

| | |
| --- | --- |
| host resolves `[REF]1[/REF]` to the marked entity (no model involved) | n/a/n/a |
| free-running: model's reference resolved to the right identifier | **n/a** |
| ...and the tool then executed successfully | n/a |
| ...and the answer used a fact only the observation carried | n/a |
| target | 0.9 -- **not measured** |

For comparison, Phase 17's free-running argument byte-exactness on the same kind
of request was **0/448**.

### Grounded refusal

Two conditions, and both must hold: the answer says it could not find the thing,
**and** no identifier-shaped token that is absent from the request appears
anywhere in what the model wrote. A refusal that names a plausible-looking sha
is a fabrication with the shape of an answer, so it does not count.

| | |
| --- | --- |
| admitted it could not find the item | n/a |
| invented no identifier | n/a |
| **both** | **n/a** (n/a) |
| target | 0.8 -- **not measured** |

Phase 17's equivalent number: the model wrote a tool-shaped result on a
`Thought:` line in **132/448** rows and answered as
though the lookup had succeeded in **182/324**.

### Anti-forgetting

Scored on Phase 16's own 480 synthetic test rows with Phase 16's own evaluator.

| | Phase 16 | Phase 17 | Phase 18 |
| --- | --- | --- | --- |
| free-running argument byte-exact | 42/60 | 0/60 | **n/a** |
| target | | | >= 35/60 -- **not measured** |

### Regression

`python -m pytest tests/ -q`: **not run**; Phase 12 vector-only recall
**10/10** through the prefix integration.

## 18.5 Targets

| target | measured | status |
| --- | --- | --- |
| reference_resolution >= 0.90 | n/a | **not measured** |
| grounded_refusal >= 0.80 | n/a | **not measured** |
| anti_forgetting >= 35/60 | n/a (Phase 16 , Phase 17 ) | **not measured** |
| regression suite green | not run | NOT met |
| Phase 12 vector-only recall 10/10 | 10/10 | met |

## 18.6 Limits, stated plainly

* **A reference only names something already in the request.** It cannot express
  "the commit *before* this one" or any identifier the user did not type. This
  solves identifier copying, not identifier reasoning.
* **The model still has to emit the right index.** With one marked entity that
  is trivial; with several it is a genuine choice, and the corpus here marks one
  entity per request. Multi-entity requests are untested.
* **Refusal is detected by phrase and by identifier shape**, not by meaning. A
  refusal that omits every expected phrase but is clearly a refusal is counted
  as a failure; an answer that contains a real identifier from the request is
  not an invention even if it is wrong.
* **The refusal corpus is synthetic in one respect**: the identifiers in it were
  generated to be absent, then verified to be absent. Real "not found" traffic
  has different shapes -- typos, deleted records, permission errors -- and none
  of that is represented.
* **Scale is unchanged**: d_model 64, 4 layers, 282,885 parameters,
  two CPU cores. Nothing here is evidence about a larger model.

## Reproduce

```bash
python -m data.phase18_corpus --output-dir runs/phase18
OMP_NUM_THREADS=2 python -m training.train_foundation --preset github \
    --data-dir runs/phase18 --output-dir runs/phase18 --epochs 8 --lr 0.003 \
    --refusal-weight 2.0 \
    --balance "exec,file,chat,repair,memory:0.5;github:0.5"
python -m evaluation.phase18_eval --model runs/phase18/foundation.pt
python -m demo.github_agent --once "Explain commit 00c3ce0a55e5: Update coverage to 7.17.0"
```

## Training

| epoch | lr | train loss | train next-byte acc | val loss | val next-byte acc | wall s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.00e-03 | 1.3801 | 61.51% | 2.0771 | 52.18% | 618.4 |
