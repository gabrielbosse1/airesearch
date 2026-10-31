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

**The fix did not work, and a distribution bug was the reason it looked worse
than it was.** The first eight epochs produced a model that resolved every
reference and then produced no answer at all: on a ten-row probe,
`answered 0/10` and `grounded_refusal 0/10`. The cause was not capacity. `ReactLoop.run` appends a
`Thought:` line after every observation before generating again, but the corpus
went straight from `Observation:` to `Answer:`. The model was therefore
off-distribution at exactly the position where the answer is produced, and its
transcripts showed a second `Thought:` full of garbage where the answer should
have been. Inserting a bridging thought -- `Thought: The tool answered; I will
report what it said.` after a success, `Thought: The tool found nothing; I
should say so.` after a failure -- into all three trajectory builders took
`answered` from 0/50 to **50/50** on the held-out set. Two tests hold that invariant: every
`github_*` row has a `Thought:` between `Observation:` and `Answer:`, and fewer
than 0.1% of rows exceed the 768-byte context window.

Training therefore ran in **two stages**: eight balanced epochs from scratch,
then three plain continuation epochs (no `--balance`) at lr 0.0015 on the same
corpus, whose natural 70/22/8 synthetic/real/refusal split further favours the
synthetic task. That recovered `runs_cleanly` from 23/60 to **60/60** and lifted
held-out next-byte accuracy from 77.60% to
80.99%, but the byte-exact argument
score stayed at 0/60.

## 18.4 Benchmarks

| split | rows | bytes | synthetic | real | refusal |
| --- | --- | --- | --- | --- | --- |
| train | 15944 | 5,134,271 | 11200 | 3580 | 1164 |
| validation | 1055 | 406,026 | 480 | 447 | 128 |
| test | 1035 | 397,296 | 480 | 447 | 108 |

### Reference resolution

| | |
| --- | --- |
| host resolves `[REF]1[/REF]` to the marked entity (no model involved) | 50/50 |
| free-running: model's reference resolved to the right identifier | **50/50** |
| ...and the tool then executed successfully | 50/50 |
| ...and the answer used a fact only the observation carried | 9/50 |
| target | 0.9 -- **met** |

For comparison, Phase 17's free-running argument byte-exactness on the same kind
of request was **0/448**.

### Grounded refusal

Two conditions, and both must hold: the answer says it could not find the thing,
**and** no identifier-shaped token that is absent from the request appears
anywhere in what the model wrote. A refusal that names a plausible-looking sha
is a fabrication with the shape of an answer, so it does not count.

| | |
| --- | --- |
| admitted it could not find the item | 50/50 |
| invented no identifier | 50/50 |
| **both** | **50/50** (100.00%) |
| target | 0.8 -- **met** |

Phase 17's equivalent number: the model wrote a tool-shaped result on a
`Thought:` line in **132/448** rows and answered as
though the lookup had succeeded in **182/324**.

### Anti-forgetting

Scored on Phase 16's own 480 synthetic test rows with Phase 16's own evaluator.

| | Phase 16 | Phase 17 | Phase 18 |
| --- | --- | --- | --- |
| free-running argument byte-exact (Phase 16 rows) | 42/60 | 0/60 | **0/60** |
| ...of which the emitted program executes | 60/60 | 2/60 | **60/60** |
| teacher-forced hex/digit bytes, Phase 17 rows | not run | 63.38% | **49.39%** |
| teacher-forced hex/digit bytes, Phase 18 rows | not run | not run | 55.25% |
| target | | | >= 35/60 -- **NOT met** |

**This target is not met, and the split between the two rows is the finding.**
The model still writes a *correct program* -- 60/60 of the emitted
arguments execute without error -- but the *literals* inside it are wrong: it
writes `print(sorted([12, 22, 23]))` where the request said `27, 4 and 9`. A
whole argument needs every byte right, and at a per-digit rate of
49.39% a ten-byte argument is a few in ten thousand.

### Two copy tables, and why both are needed

Teacher-forced, so these measure what the model *can* copy, not what it does
copy. **A: the task the model is now given** -- held-out Phase 18 rows, requests
marked, arguments are references.

| held-out phase-18 rows | synthetic | real GitHub |
| --- | --- | --- |
| argument bytes | 81.94% | **99.81%** |
| hex / digit bytes | 55.25% | **99.48%** |
| whole argument exact | 108/548 | **438/447** |

**B: the same rows the Phase 17 report used, where the request is unmarked and
the argument is a literal identifier.** Same 440 + 448 rows, same script, Phase
17 model against Phase 18 model.

| identical phase-17 rows | Phase 17 model | Phase 18 model |
| --- | --- | --- |
| synthetic argument bytes | 87.06% | 79.07% |
| synthetic hex / digit bytes | 63.38% | 49.39% |
| synthetic whole argument exact | 17/440 | 0/440 |
| real argument bytes | 51.95% | 33.04% |
| real hex / digit bytes | 39.55% | 22.63% |
| real whole argument exact | 0/448 | 0/448 |

Read together, the two tables say the thing the phase has to be honest about.
Table A's real-GitHub column -- **438/447** whole arguments exact
at a per-hex-digit rate of 99.48% -- is not the model getting
better at spelling. It is the model being asked to emit `[REF]1[/REF]` instead
of twelve hex characters, so the copying difficulty is gone from the task.
Table B is the cost: measured on identical rows doing the *old* task, literal
copying got worse, not better -- synthetic digits
63.38% -> 49.39%, real digits
39.55% -> 22.63%.

That is the trade the phase makes deliberately: a learned skill that decays
(literal copying, and the 0/60 anti-forgetting score is the bill) is exchanged
for a structural guarantee that cannot (a reference either names something in
the request or the tool is never called). It also means **an unmarked request
is now a bug, not a slow path** -- the model will emit a reference the host
cannot resolve, which is why `data.github_to_react.mark_entities` marks the
request at run time in `demo/github_agent.py` and why the resolver fails closed
with `ERROR: unresolved reference` instead of guessing.

### Regression

`python -m pytest tests/ -q`: **320 passed, 1 warning, 39 subtests passed in 185.54s (0:03:05)**; Phase 12 vector-only recall
**10/10** through the prefix integration.

## 18.5 Targets

| target | measured | status |
| --- | --- | --- |
| reference_resolution >= 0.90 | 50/50 | **met** |
| grounded_refusal >= 0.80 | 50/50 | **met** |
| anti_forgetting >= 35/60 | 0/60 (Phase 16 42/60, Phase 17 0/60) | **NOT met** |
| regression suite green | 320 passed, 1 warning, 39 subtests passed in 185.54s (0:03:05) | met |
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
* **Literal copying got worse, and the phase accepts that.** Measured on
  identical rows, synthetic teacher-forced digit accuracy fell
  63.38% -> 49.39% and real
  39.55% -> 22.63%. The anti-forgetting
  target (>= 35/60) is **not met** at 0/60. A 282,885-parameter
  model apparently cannot hold both behaviours at once, and this phase chose
  the structural one.
* **Requests must be marked at run time or the mechanism fails closed.** The
  model emits references unconditionally; a request with no marked spans yields
  `ERROR: unresolved reference: reference 1 is out of range; the request marks 0
  entities`. `data.github_to_react.mark_entities` does this for the demo. Any
  new surface -- a server, a different tool set -- has to call it too.
* **Scale is unchanged**: d_model 64, 4 layers, 282,885 parameters,
  two CPU cores. Nothing here is evidence about a larger model.

## Reproduce

```bash
python -m data.phase18_corpus --output-dir runs/phase18

# stage 1: eight balanced epochs from scratch
OMP_NUM_THREADS=2 python -m training.train_foundation --preset github \
    --data-dir runs/phase18 --output-dir runs/phase18 --epochs 8 --lr 0.003 \
    --seed 18018 --refusal-weight 2.0 \
    --balance "exec,file,chat,repair,memory:0.5;github:0.5"

# stage 2: three plain continuation epochs, favouring the synthetic task
cp runs/phase18/foundation.pt runs/phase18/foundation_stage1.pt
OMP_NUM_THREADS=2 python -m training.train_foundation --preset github \
    --data-dir runs/phase18 --output-dir runs/phase18 \
    --init-from runs/phase18/foundation_stage1.pt \
    --epochs 3 --lr 0.0015 --seed 18020 --refusal-weight 2.0

python -m evaluation.phase18_eval --model runs/phase18/foundation.pt
python -m evaluation.copy_diagnostic --model runs/phase18/foundation.pt \
    --rows runs/phase18/react_test.jsonl --output runs/phase18/copy_diagnostic.json
python -m demo.github_agent --once "Explain commit 5c06b141f3b0: Fix the async close"
```

## Training

Stage 1 -- eight balanced epochs from scratch, lr 0.003, fixed 50/50 batch
composition:

| epoch | lr | train loss | train next-byte acc | val loss | val next-byte acc | wall s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.00e-03 | 1.3402 | 62.42% | 2.3652 | 47.04% | 676.2 |
| 2 | 2.89e-03 | 0.8996 | 74.40% | 1.7985 | 59.58% | 1357.5 |
| 3 | 2.56e-03 | 0.7336 | 79.00% | 1.4336 | 65.44% | 2083.2 |
| 4 | 2.07e-03 | 0.6401 | 81.47% | 1.2674 | 68.36% | 2798.7 |
| 5 | 1.50e-03 | 0.5865 | 82.87% | 1.1044 | 71.25% | 3409.3 |
| 6 | 9.26e-04 | 0.5541 | 83.70% | 1.0110 | 73.68% | 4054.7 |
| 7 | 4.39e-04 | 0.5363 | 84.16% | 0.9005 | 75.83% | 4717.3 |
| 8 | 1.14e-04 | 0.5396 | 84.03% | 0.8051 | 77.80% | 5313.7 |

Stage 2 -- three plain continuation epochs from the stage-1 checkpoint, lr
0.0015, no `--balance` (the corpus's own 70/22/8 split favours the synthetic
task). Held-out next-byte accuracy 77.60% -> 80.99%:

| epoch | lr | train loss | train next-byte acc | val loss | val next-byte acc | wall s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.50e-03 | 0.6111 | 82.05% | 0.6960 | 79.70% | 637.2 |
| 2 | 1.13e-03 | 0.5664 | 83.30% | 0.6672 | 80.45% | 1256.4 |
| 3 | 3.75e-04 | 0.5337 | 84.18% | 0.6394 | 81.20% | 1891.2 |

Both stages run at 282,885 parameters on two CPU cores; stage 1 took
5313.7 s and stage 2 a further
1891.2 s.
