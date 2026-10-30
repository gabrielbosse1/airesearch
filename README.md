# ByteCore-DualMemory

An experimental PyTorch architecture for multimodal **raw bytes**, persistent
memory, and bounded latent computation. Text is UTF-8 bytes, images are
8-bit grayscale bytes, and audio is 8-bit PCM bytes. There is **no external
tokenizer**. This is a small research system, not a generally reliable language
model, fact checker, or autonomous agent.

## Architecture

| Upgrade | Implementation |
|---|---|
| Hierarchical memory | Time-decaying episodic entries, semantic mixtures of distinct exemplars, and procedural traces. `ImportanceScorer` combines surprise, access frequency, and recency. Writes and consolidation do not require gradients. |
| RETRO-style injection | Top-K memory vectors are injected into selected transformer layers by gated cross-attention and/or prefix tokens. Live chunked retrieval uses only the preceding chunk; memory and generator can be co-trained. |
| Cognitive loop | A session-local latent scratchpad reads byte spans; `THINK` updates latents without emitting bytes; `EDIT_OP` stages insert/delete/replace/move edits; `SAVE_SPAN` commits them; `DONE` finishes. |
| Epistemic state | A parallel causal thought stream and supervised confidence heads. Adaptive computation is opt-in and bounded; calibration is tracked by task. |
| Executable and graph memory | Procedures store exact input/output byte patterns, ordered latent traces, and explanation bytes for safe replay. Explicit structured facts support bounded, temporally valid multi-hop evidence paths. |

The generator vocabulary is exactly 264 IDs: raw bytes **0–255**, function
controls `FUNC_CALL=256`, `FUNC_ARG=257`, `FUNC_END=258`, and cognitive controls
`LOAD_SPAN=259`, `SAVE_SPAN=260`, `THINK=261`, `EDIT_OP=262`, `DONE=263`.
Controls are internal IDs, never canvas bytes. Regular `generate()` masks the
cognitive controls; function controls remain host-intercepted.

## Install and run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python data/generate.py                 # creates ignored data/processed/
OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 .venv/bin/python evaluation/evaluate.py
OMP_NUM_THREADS=1 .venv/bin/python evaluation/evaluate_upgrades.py
# Uses a separate directory; the demo writes its grown checkpoint there.
OMP_NUM_THREADS=1 .venv/bin/python demo/dual_memory_demo.py --runs-dir /tmp/bytecore-demo
```

Omitting the demo's `--runs-dir` reads and **updates** `runs/dual_memory/`.
The evaluation scripts write reports and metrics into `runs/`; the generated
JSONL splits are ignored by Git. `runs/report.md` measures diffusion and
`runs/upgrade_report.md` measures the upgrade mechanisms.

## Cognitive editing

The included generator checkpoint has a **narrow synthetic** policy trained on
`I:`, `D:`, `R:`, and `M:` followed by three letters in `a`–`h`. This proves
that learned control/pointer heads can drive the entire loop; it is not general
instruction-following training.

```python
import torch
from models import GeneratorConditioned

generator = GeneratorConditioned(d_model=64, n_layers=2, n_heads=4, d_ff=128)
generator.load_state_dict(torch.load("runs/dual_memory/generator.pt", weights_only=True))
generator.eval()
result = generator.cognitive_generate(b"R:abc", max_steps=8)
assert result.done and result.canvas == b"R:?bc"
print([event.action.token for event in result.events])  # LOAD, THINK, EDIT, SAVE, DONE
```

For deterministic workflows or supervision, supply a host controller that
returns `CognitiveAction` objects. LOAD ranges are half-open canvas indices;
EDIT indices are relative to the loaded workspace. MOVE's destination is in
the original workspace coordinates. Edits are staged: neither a subsequent
LOAD nor DONE can silently discard a dirty span. At the step limit, the result
contains the unchanged committed canvas plus `pending_workspace` and
`pending_span`. The scratchpad is cloned per call, never shared or saved in a
checkpoint. Both `max_steps` and `max_canvas_bytes` are bounded.

```python
from models import CognitiveAction

plan = iter([
    CognitiveAction(generator.LOAD_SPAN, start=0, end=3),
    CognitiveAction(generator.THINK, think_steps=2),
    CognitiveAction(generator.EDIT_OP, kind="insert", start=1, data=b"\x00\xff"),
    CognitiveAction(generator.SAVE_SPAN),
    CognitiveAction(generator.DONE),
])
result = generator.cognitive_generate(b"abc", controller=lambda observation: next(plan))
assert result.canvas == b"a\x00\xffbc" and result.done
```

`cognitive_policy` and `training.cognitive_train.cognitive_supervision_loss`
expose differentiable action, span, kind, and byte logits for labelled traces.
The learned policy emits one payload byte per insert/replace operation; host
controllers can supply arbitrary-length byte payloads within the canvas limit.
`training/train_cognitive.py` reproduces the narrow synthetic training run.

## Long-term memory and safe procedure replay

`MemoryModel.write()` and `consolidate()` directly update persistent banks;
`retrieve()` returns a `(K, d_model)` matrix, while
`retrieve_differentiable()` carries gradients through top-K selection using a
straight-through estimator. `pool=True` is available for legacy single-vector
callers. An empty or fully masked retrieval cannot leak other entities.

```python
import torch
from models import MemoryModel, CognitiveAction

memory = MemoryModel(d_model=64, capacity=256)
slot = memory.write_procedure(
    b"\x00input\xff", b"\xffoutput\x00", torch.randn(2, 64),
    b"exact stored example", metadata={"name": "example"},
)
execution = memory.execute_procedure(b"\x00input\xff")
assert execution.slot == slot and execution.output == b"\xffoutput\x00"
# An explicit host can replay saved latent steps into a fresh scratchpad:
result = generator.cognitive_generate(
    execution.output, initial_trace=execution.latent_trace,
    controller=lambda _: CognitiveAction(generator.DONE),
)
assert result.canvas == execution.output
memory.save("memory.pt")  # includes metadata and the session clock
```

A procedure is **data**, not Python code. Replay requires an exact input match
and never evaluates stored bytes; an optional explicitly supplied host callback
can consume latent steps. It does not generalize an example to unseen inputs.
Newer matching entries win. Slots can be evicted or reused, so they are not
permanent IDs. Oversized structured patterns/explanations/traces are rejected,
not truncated. Defaults are 512 input/output bytes, 1,024 explanation bytes,
and four latent steps; configure them when constructing `MemoryModel`.

`ingest_examples()` also writes exact replayable procedures and co-trains the
models. Explicit masks distinguish padding from valid byte `0`. Co-training
updates weights and invalidates prior confidence calibration.

## Contradiction evidence, not lie detection

Use `Fact` / `write_fact` for structured assertions. Identifiers are raw bytes
(or UTF-8 strings converted to bytes). The graph follows positive `same_as`
links in either direction and chains a predicate only when explicitly marked
`transitive=True`. A different positive object is a conflict only when the
relation is explicitly `exclusive=True`; opposite polarity of the same fact
also conflicts. Validity intervals are half-open `[valid_from, valid_to)` and all
provided numeric times must use the same scale. `as_of` checks a specific time.

```python
from models import Fact

memory.write_fact(Fact(b"alice", b"same_as", b"alice-id"))
memory.write_fact(Fact(b"alice-id", b"language", b"python",
                       exclusive=True, valid_from=10, valid_to=20))
claim = Fact(b"alice", b"language", b"rust")
report = memory.explain_contradiction(claim, as_of=15, max_hops=3)
assert report.status == "conflict" and len(report.evidence[0].path_slots) == 2
assert memory.explain_contradiction(claim, as_of=25).status == "unknown"
```

The report distinguishes `support`, `conflict`, `mixed`, and `unknown`, includes
actual paths and interval intersections, and flags a truncated search. The
trainable temporal message-passing score is reported separately from explicit
logical evidence. `training.graph_train.graph_consistency_loss` can train the
reasoner on trusted labels. The compatibility method `check_contradiction()`
returns `1.0` when no evidence is found; **that does not establish truth**. For
unstructured embeddings it is only a similarity heuristic. Neither method can
infer intent or decide whether someone lied. Graph edges are rebuilt from live
memory slots so eviction cannot leave stale edges.

## Confidence and adaptive halting

`GeneratorConditioned.forward(..., return_state=True)` exposes per-position
thought, confidence logits, and probabilities. `DiffusionCore` exposes
confidence logits with `return_confidence=True`. Training uses detached exact
byte-correctness targets and binary cross entropy; validation can fit a
scalar temperature with `ConfidenceHead.calibrate(logits, targets, scope=...)`.
A single generator head shares its temperature across uses, so recalibrating
one scope replaces the previous calibration. Calling `.train()` invalidates it.

`think_scratchpad(..., confidence_threshold=...)` and
`ByteCoreDiffusion.generate(..., confidence_threshold=..., return_stats=True)`
stop work per sample once a threshold is reached, subject to minimum and maximum
steps. By default adaptive halting is **off**. The included diffusion head was
calibrated on **q-sampled denoising**, not ancestral generation; the included
generator head was calibrated on next-byte prediction, not scratchpad
thinking. A scope-mismatch warning is emitted for those adaptive uses. Fit and
validate calibration for the actual application and distribution before relying
on a threshold; high confidence is not a correctness guarantee.

The reproducible `training/train_confidence.py` trains only the heads, fits
validation temperatures, and writes new checkpoints to a separate output
directory; it does not overwrite the input checkpoints. The included checkpoint
heads were trained and assessed this way. On the included held-out multimodal
next-byte set, the generator's correctness was about **1%** without memory;
well-calibrated low confidence does not mean good generation.

## Compatibility and limits

- Phase-1 259-ID generator checkpoints load strictly into the 264-ID model;
  existing byte/function embedding and output rows are copied exactly. Older
  single-vector generator checkpoints migrate their projection to the first
  RETRO adapter, but memory-conditioned behavior is not identical.
- Memory schema 3 loads v1 flat and v2 hierarchical checkpoints, preserving
  the original pair scorer and initializing new graph/procedure fields. Guarded
  memories use schema 4 and reject unverifiable legacy slots. Use `save()` /
  `load()` for metadata, guard policy and session time, not bare `state_dict()`.
- Legacy diffusion checkpoints load without changing their denoising tensors;
  a missing confidence head is initialized and is **uncalibrated**. Checkpoint
  dimensions/capacities must match constructors. Old optimizer state is not
  resized automatically; recreate optimizers after a vocabulary upgrade. New
  264-ID or guarded schema-4 checkpoints cannot be expected to load in older code.
- At `d_model=64`, three layers, `d_ff=256`, memory capacity 512: memory has
  100,205 parameters (Phase 1: 83,499); the generator has 523,420 with injection
  every layer (Phase 1: 329,932), or 477,337 every two layers, before Phase 9
  adds 16,640 structured-request parameters to width-64 generators. Diffusion
  now has 200,065 parameters (195,905 before Phase 9). Memory byte buffers and
  per-layer attention also add storage/compute.
- The current diffusion report explicitly separates reconstructing an already
  corrupted input from recovering the original. At 10% corruption, measured
  real-byte recovery is 0.9128 for substitutions, 0.3693 for insertions, and
  0.3812 for deletions; insertion/deletion comparisons are position-wise without
  realignment. The small models and synthetic demonstrations are not production
  quality without substantial task-specific training and independent testing.

## Phase 5: Diagnostic Attribution

Phase 5 adds held-out confidence calibration, optimal raw-byte edit traces,
differentiable sequence alignment, and intervention-based diagnostics. The
complete measured results and limitations are in
[`runs/diagnostic_report.md`](runs/diagnostic_report.md) and
[`runs/failure_modes.md`](runs/failure_modes.md).

### Calibration and thresholds

`evaluation/calibrate_confidence.py` fits a temperature on even validation
examples, selects thresholds on odd validation examples with balanced noise
conditions, and reports ECE/MCE and error on the test split. Diffusion records
contain per-byte embedding MSE; generator records contain next-byte
cross-entropy. Confidence predicts correctness; it is not the MSE or loss.
`runs/calibration_predictions.npz` retains the underlying observations and
`runs/calibration.png` visualizes test reliability and observed error. The
calibrated `state_dict` copies are under `runs/phase5_calibrated/`; they do not
replace the original checkpoints.

No 90%, 95%, or 99% threshold was supported in this experiment. Those config
values are `null`, and `phase5_calibration` is a diagnostic record, **not** an
instruction to enable inference halting. The task-specific selection does not
validate THINK or ancestral sampling. Refit calibration after training.

### Edit traces and co-training

`data/ingest_real_traces.py` accepts JSONL records containing `input_hex`,
`output_hex`, optional `id`, `split`, `kind`, `source` and UTF-8 `explanation`.
Hex encoding preserves arbitrary raw bytes including NUL and 255. It uses
optimal unit-cost Levenshtein insertion/deletion/replacement (MOVE is not a
primitive in that optimality claim), bounded quadratic dynamic programming,
and generates internal LOAD/THINK/EDIT/SAVE/DONE actions. Procedural memory
stores exact inputs and outputs, sampled actual scratchpad states, serialized
actions and explanation bytes. No stored code is executed.

`training.co_train.ingest_examples(..., cognitive_trace_weight=...)` can
jointly train the generator and memory using validated teacher-forced actions;
it includes conditioned and unconditioned cognitive-policy losses. The model
emits one byte per learned insert/replace action. Exact stored-answer replay is
not model-generated editing. The fixture contains short real before/after
repository excerpts split by source file; the 80-epoch isolated experiment
historically reproduced 22/22 stored procedures, 5/10 training edits autonomously
and 0/8 held-out edits autonomously. The old verification memory contained
held-out targets added after model evaluation and has since been replaced with
guarded train-only memory. The historical model budget was target-derived;
Phase 6 uses a fixed budget. This small fixture is not an external benchmark.

```bash
OMP_NUM_THREADS=1 .venv/bin/python data/ingest_real_traces.py \
  --pairs data/real_edit_pairs.jsonl --split train --epochs 0 --output-dir runs/phase6/ingestion
OMP_NUM_THREADS=1 .venv/bin/python evaluation/test_real_traces.py --generator runs/real_traces/generator.pt --train-epochs 0
```

### Soft-DTW alignment

`models.alignment.soft_dtw_alignment()` aligns embedding sequences with masks;
valid byte positions need not be contiguous. Soft-DTW is entropy-smoothed and
may be negative. `ByteCoreDiffusion.compute_loss()` accepts `corrupted_emb`,
`alignment_loss_weight` and `alignment_gamma`; its direct API defaults to the
legacy weight zero. Config-driven `training/train.py` now also defaults to
zero alignment weight and zero paired edit-corruption probability; either can
be enabled explicitly for a new training experiment. Alignment adds **zero
learned parameters**, but its training cost is
quadratic in sequence length. The fixed-length model cannot infer missing output
lengths from this loss alone.

The matched retraining test at 10% corruption measured roughly 39% insertion
and 40% deletion original-position accuracy (target >70% unmet); clean
reconstruction fell by about 0.7 percentage points. Edit-aligned similarity is
a separate metric and does not show that deleted bytes were recovered. The
original checkpoints were not replaced.

### Reproduce diagnostics

Run from the repository root after generating `data/processed/`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python evaluation/calibrate_confidence.py --write-config
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python evaluation/test_real_traces.py --generator runs/real_traces/generator.pt --train-epochs 0 --output-dir runs/phase6/real_traces
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python evaluation/evaluate_alignment.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python evaluation/ablation_study.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python evaluation/head_attribution.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python evaluation/importance_validation.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python evaluation/edit_operation_analysis.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python evaluation/learning_curve.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python evaluation/failure_detection.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python evaluation/evaluate.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python evaluation/diagnostic_report.py
```

The head heatmap measures output-projection ablation of RETRO cross-attention
heads, not self-attention or prefix ablation. Importance/utility correlations
use a small set of slot removals and may be dominated by unselected slots.
Current clean trace experiments write under `runs/phase6/real_traces/`, and
alignment experiments under `runs/alignment/`, leaving the main checkpoints
intact. The default generator is not a reliable natural-language or code editing
model.

## Phase 6: clean evaluation, guarded memory and explicit routing

The measured results and unmet targets are in [runs/phase6_report.md](runs/phase6_report.md).
The prior verification memory contained 12 held-out pairs and was replaced by a
guarded train-only snapshot. The separate legacy dual-memory checkpoint has 67
slots without verifiable provenance; guarded evaluation rejects such slots, not
because every one is known contaminated, but because cleanliness cannot be proved.
The new real-trace run saves before test, evaluates read-only with a fixed budget,
and gets 10/10 stored train replay, 5/10 autonomous train and 0/8 autonomous test.
The fixture lacks actionable edit instructions; lookup is not learned generation.

### Attach a holdout registry before writing

```python
from data.holdout_registry import synthetic_holdouts
from data.ingest_real_traces import load_pairs
from models.holdout import HoldoutGuard
from models.memory_model import MemoryModel

held_out = (load_pairs("data/real_edit_pairs.jsonl", split="validation")
            + load_pairs("data/real_edit_pairs.jsonl", split="test")
            + synthetic_holdouts("data/processed", max_length=128))
guard = HoldoutGuard.from_records(held_out)
memory = MemoryModel(d_model=64, capacity=256)
memory.set_holdout_guard(guard)        # attach before any writes or loads
# Guarded generic writes need raw_input=... and, if known, raw_output=...
with memory.read_only():              # no writes or access-stat updates
    entries, vectors = memory.retrieve(query_embedding)
assert memory.audit_holdouts()["failed_slots"] == 0
```

`query_embedding` above is an application-provided tensor. Guarded concepts use
`model.bank.set_holdout_guard(guard)` and `model.write_concept(name, raw_bytes)`.
A guarded checkpoint preserves its registry; old slots without provenance fail
closed. Metadata with a `test`/`validation` split is rejected even unguarded.
Hash checks cover exact content and registered substrings at least 16 bytes;
co-training also scans symbolic intermediate canvases and consecutive trace
payload runs before mutation. They do not catch semantic/near duplicates, short substrings, hidden encodings, opaque vectors,
pretrained weights, or direct tensor tampering. The registry must include every
relevant held-out dataset and the caller must supply truthful provenance; this
is not a security or privacy boundary.

### Explicit editing dispatch

```python
from models.edit_router import EditRequest, EditRouter
# diffusion and generator are loaded models; 128 is this checkpoint's training width.
router = EditRouter(diffusion, generator, diffusion_pad_to=128)
result = router.execute(EditRequest(b"I:abc", "insert"), max_steps=8)
print(result.engine, result.output, result.completed)  # DONE is not correctness
```

The operation is declared by the caller, not inferred from a target. Insert,
delete and move use the cognitive model; fixed-length denoising uses diffusion;
`replace` needs an explicit `length_preserving` flag. Free-form diffusion
instructions are still rejected; Phase 9 additionally accepts a trained model
and explicit binary mask and replacement bytes. Cognitive natural-language
instruction priming remains experimental. The historical Phase 6 toy benchmark gets
64/64 each for insert/delete/move but 0/64 targeted same-length replacements
when routed to diffusion. On 100 synthetic 10% substitution inputs the denoiser
mostly copied the already-corrupted bytes rather than recovering them.

### Halting policy and reproduction

`configs/default.yaml` provides disabled `adaptive_halting.diffusion_sampling`
and `.thought` entries. Pass the relevant mapping as `adaptive_config=` to
`ByteCoreDiffusion.generate`/`p_sample_loop` or `cognitive_generate`. Enabling
requires an explicit threshold and matching calibration scope; it does not prove
the policy is safe on a new distribution. Explicit `confidence_threshold=` is
an experimental warning-only override. Phase 6's paired sampling test found no
policy meeting its strict validation no-loss criterion; the THINK experiment
saved work versus a four-step comparison but not the existing one-step default.
Do not enable either by default based on those results.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python evaluation/phase6_routing.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python evaluation/phase6_halting.py
.venv/bin/python evaluation/phase6_report.py
```

## Phase 8: reasoning supervision and memory-free policy gradients

See [the measured Phase 8 report](runs/phase8_rl_report.md). The final-epoch
policy, the validation-selected policy and their starting checkpoint all scored
**0/8 exact** on the real held-outs; the requested >4/8 was not met. The selected
model also scored **0/4** real validation (only two episodes completed) and
**0/10** instructed synthetic validation. Real held-out records do not contain
an actionable edit request, so the desired output is generally underdetermined.
The byte guards cannot certify the contents of pretrained weights or semantic
novelty. The checkout did not contain verifiable Phase 7 mask-conditioning code;
this phase does not claim it was validated.

`GeneratorConditioned.reasoning_predictor` predicts a detached, order-sensitive
but lossy sketch of `target - current` from the mean post-THINK scratchpad. The
target is a training label, never a policy input. With cognitive traces,
`co_train_step` adds `0.5 * reasoning_loss` by default. The head adds 8,320
parameters to the width-64 Phase 8 model (352,728 total before Phase 9). An entirely absent head is
initialized during strict legacy load; a partially missing one remains an error.
Older code cannot strictly load a checkpoint containing the new head. Existing
optimizer state cannot automatically acquire the new parameters, and confidence
calibration is invalidated by further training.

The selected reasoning head is worse than an all-zero predictor on 5/6 training groups; the sketch is especially small for line/move examples. See `runs/phase8/reasoning_train_diagnostic.json`. This is a training-only diagnostic, not held-out evidence.

`training.rl_editing` samples the same bounded cognitive loop used at inference,
without memory or controllers, and then replays the sampled sequence with
gradients. It includes actual masked control and argument probabilities and
accounts for destination remapping. The self-critical greedy baseline is
independent of the sampled actions. The requested byte-Levenshtein reward minus
0.01 per THINK iteration or EDIT action can favor no-ops on long documents;
reward, completion and exact success are different metrics. The replay shares
an attention code path with sampling; the PyTorch process-global MHA fast-path
flag is temporarily disabled under a lock and restored.

The opt-in `ingest_examples(..., curriculum=True)` and the RL trainer schedule
five 20%-wide bands: single-byte substitutions, single-word replacements,
single-line edits, multiline insert/delete, and code-block MOVE. The supervised
and RL mixture anneals from 80/20 to 20/80. Only train examples and train traces
are used in loss/reward; all examples and symbolic intermediate canvases are
checked against registered holdouts before optimizer updates. Validation selects
only final-stage checkpoints, then the real test is scored with a fixed
128-action budget, no memory, stored trace, target input or controller.

```bash
.venv/bin/python data/generate.py  # rebuild ignored processed holdout registry
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.rl_editing \
  --epochs 20 --examples-per-stage 8 --real-anchors 2 --rollout-budget 16 \
  --eval-budget 128 --temperature 0.8 --lr 0.0003 --seed 881 --threads 1
.venv/bin/python -m evaluation.phase8_reasoning
.venv/bin/python -m evaluation.phase8_report
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

`runs/phase8/generator.pt` is validation-selected; `last_epoch.pt` is the last
training epoch. `metrics.json` records the reward curve, choices and per-task
outputs; `reward_curve.png` plots it. Training was tested with PyTorch
2.5.1+cu124 on CPU, NumPy 1.26.4 and Matplotlib 3.11.2. Stage-to-stage reward
changes are not a controlled comparison because the tasks change.


## Phase 9: explicit edit arguments and native masked diffusion

See [the Phase 9 report](runs/phase9_report.md) for training, selection, tests,
raw traces and limitations. The requested `Hello Worl` → `Hello Word` case now
passes both native diffusion and the no-memory/no-controller cognitive loop.
The cognitive case was **included and oversampled in training**: the separate
natural-instruction test scored **0/68**, so the example is not evidence of
general instruction following. The guarded memory did not already contain the
example; a copy reproduced it only after the exact answer was explicitly stored.

### Cognitive argument frames

`models.cognitive_arguments.encode_edit_arguments` and
`decode_edit_arguments` serialize/validate the learned policy's EDIT fields
before they are applied. `CognitiveEvent.argument_bytes` contains literal data
bytes following the internal `EDIT_OP` control; `event.control_and_arguments`
exposes their internal control-plus-byte sequence. The heads predict fields jointly;
this is not autoregressive argument-byte generation. Controls never become
canvas bytes, and edits remain staged until `SAVE_SPAN`.

The header is `[position_low, opcode_and_flags, payload_byte]`. Unflagged
opcodes are insert=0, delete=1, replace=2, move=3; delete/move use zero as a
payload placeholder. Single-byte replace at position 9 with byte 100 is
`[9,2,100]`. Position >255 sets `0x80` and appends the high bits as canonical
unsigned LEB128 (position 300: `[44,130,100,1]`). Flag `0x40` adds a non-unit
span length; `0x20` adds a non-unit payload length and remaining bytes; MOVE
then appends its original-workspace destination. Extensions occur in that order.
The default maximum coordinate is 4096; malformed or extra fields are rejected.
Host actions can carry multi-byte payloads; autonomous insert/replace predicts
one payload byte per action.

Optional `structured_request` supplies the desired position, kind and byte
explicitly to learned argument heads. It is **not** a host controller and does
not force the edit, but because the input already specifies the action, its
results measure structured copying rather than natural-language understanding.
Only one-byte replacement conditioning is trained in the bundled checkpoint;
other kinds and non-unit spans require an explicit untrained override.

```python
from training.train_confidence import load_generator

model = load_generator("runs/phase9/structured_generator.pt", "cpu", 4).eval()
source = b"Hello Worl"
instruction = b"Replace the byte at zero-based position 9 from 108 ('l') to 100 ('d')."
result = model.cognitive_generate(source, instruction_bytes=instruction,
                                  structured_request=bytes([9, 2, 100]), max_steps=12)
assert result.done and result.canvas == b"Hello Word"
assert [e.argument_bytes for e in result.events if e.action.token == model.EDIT_OP] == [bytes([9, 2, 100])]
```

Distinct structured test cases scored **254/264**, including **7/8** with
positions >255. Natural-only held-out cases scored 0/68; the eight real
held-outs without actionable instructions remain 0/8. These are small synthetic
experiments, not a deployment accuracy claim.

### Native diffusion masking

`p_sample_loop_with_mask(shape, edit_mask, original_emb=...,
requested_byte_ids=...)` and `p_sample_loop(..., edit_mask=...)` preserve
unmasked source embeddings at every DDPM step. `masked_edit` returns bytes and
copies **only unmasked originals**; selected values are decoded from model
predictions. A mask alone describes where, not what: supply the requested
values independently. Untrained mask checkpoints fail closed by default, and
masked adaptive confidence thresholds are not supported.

```python
import torch
from training.train_masked_diffusion import load_model

model = load_model("configs/default.yaml", "runs/phase9/diffusion").eval()
ids = torch.tensor([[72,101,108,108,111,32,87,111,114,108]])
mask = torch.tensor([[0,0,0,0,0,0,0,0,0,1]], dtype=torch.bool)
requested = torch.zeros_like(ids); requested[0, 9] = 100
torch.manual_seed(42)
assert model.masked_edit(ids, mask, requested).tolist() == [[72,101,108,108,111,32,87,111,114,100]]
```

`compute_loss` accepts `edit_mask` or `random_edit_mask_probability`, optional
`source_emb` and requested IDs; it supervises selected valid bytes only. On
separate synthetic sets, diffusion achieved **78/80** exact multi-mask rows
and **235/256** all-byte requests; all unmasked bytes were preserved. The
original core has no padding-attention mask, so changing sequence width changes
context. `EditRouter` accepts binary `edit_mask` and `replacement_bytes` fields
for explicit fixed-length edits, but still rejects free-form instructions.

Width-64 Phase 9 adds 16,640 structured-request generator parameters and 4,160
diffusion parameters. Absent legacy heads initialize to zero and are marked
untrained; partial newer heads remain strict-load errors. Older code cannot
strictly load new checkpoints, and optimizer/calibration state needs rebuilding.
The trained flags are metadata, not accuracy guarantees. Historical checkpoints
were not overwritten.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_positional_edits
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_structured_requests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_masked_diffusion
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase9_diagnostic
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase9_benchmark
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m evaluation.phase9_report
```

## Phase 10: conversational foundation (legacy mode)

The commands and explicit-memory-only behavior in this section require `--legacy-phase10`; the Phase 12 default automatically stores complete transcripts and reversible vectors when `--memory` is supplied.

The Phase 10 path is a small, **bounded research prototype**, not a general
chatbot. `GeneratorConditioned` now has a learned seven-way intent classifier
(`Linear → SiLU → Linear`) and a trainable byte-convolution encoder. The
frozen causal generator provides context without changing Phase 9 editing
weights. A separate autoregressive byte decoder generates limited social
responses. Its internal EOS is not a raw output byte. Four exact factual
questions use a small host-supplied reference table; other inputs classified as
questions abstain instead of inventing an answer. Misclassification can route a
question to a different path.

Run an ephemeral interactive session (no persistent memory):

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m demo.conversation --legacy-phase10 --show-intent
```

Opt in to a local, **unencrypted** memory file, then restart the command to
recall explicitly saved facts:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m demo.conversation --legacy-phase10 --memory runs/user_memory/local.pt --user local --show-intent
# Try: Remember my hometown is Recife.
# On a later run: Can you recall my hometown.
# To delete from the current bank: Forget my hometown.
```

Memory commands intentionally have narrow grammar: `remember my KEY is VALUE`,
`what is my KEY?`, `can you recall my KEY`, and `forget my KEY` (plus documented
minor variants). The classifier can still misunderstand them. Memories are
explicit, exact raw-byte lookup—not automatic recording or general semantic
recall. Recent turns fit only within the 128-byte context; longer current
messages receive a request to shorten them. Keys in the chat grammar are ASCII;
values are UTF-8. The programmatic memory API can store arbitrary bytes, but
non-text values are not emitted as conversational text.

For a narrowly specified edit, `/document TEXT` or `/document-hex HEX` loads
raw bytes in the demo; `replace byte 3 with 113` then supplies a **zero-based
byte** position and replacement. The host parses those explicit fields, sends a
structured request to the learned cognitive policy, and returns the changed
document only after verifying the model completed exactly that edit. It does not
force the output. Other edits and vague requests require clarification. The
classifier and decoder are not evidence of free-form instruction following.

`ConversationMemory` stores scoped facts in existing procedural key/value
memory, with opt-in atomic checkpoint replacement and rollback for ordinary
write failures. A full bank rejects new facts through this API rather than
silently evicting another user's record; updates replace the same key. User IDs
are **not authentication**. The host must authenticate users, restrict access,
protect backups and supply encryption where needed. Locks are shared only
through one wrapper, not between processes. Forgetting clears current bank
entries and the latest saved file, not prior backups, filesystem remnants or
caches. `runs/user_memory/` is ignored by Git; avoid storing private data in
other tracked paths.

Width-64 parameter counts: Phase 9 generator 369,368; intent encoder +16,480;
intent head +4,615; Phase 10 generator 390,463; separate social decoder
+62,273. Legacy generators load with deterministically initialized, untrained
intent components; a session refuses to use them until trained. Partial new
checkpoints fail strict loading, older code cannot strictly load Phase 10
checkpoints, and optimizer/calibration state must be rebuilt or revalidated.
The response checkpoint is bound to its exact generator SHA-256 and expected attention-head count.

On the post-selection, manually authored 35-message challenge, intent accuracy
was **23/35**, and vague/clarification intents were **0/5**. The policy still
asked for clarification on **5/5** of those vague prompts via fallbacks. A
five-case negative/malformed-command safety check passed, but is not security
evidence.
The 0.55 confidence threshold is uncalibrated; even an incorrect negated memory
request received ~0.995 confidence, and the explicit grammar—not confidence—
prevented the write. A single synthetic cross-session memory/edit scenario
passed. The head-only baseline's template-test result informed the encoder
design, so that template test is a development holdout, not untouched final
evidence. Social response targets are repeated training templates. See
`runs/phase10_report.md`, `runs/phase10/training_metrics.json`, and
`runs/phase10/benchmark.json` for details and limitations.

Reproduce from this checkout after installing requirements and generating
`data/processed/` if absent (new training writes into `runs/phase10`):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_conversation --intent-epochs 60 --response-epochs 100 --batch-size 128 --feature-batch-size 32 --intent-lr 0.003 --response-lr 0.003 --seed 101010 --threads 1 --output-dir runs/phase10
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase10_benchmark --samples 30 --threads 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

## Phase 11: contextual byte conversation

Phase 11 adds hard-negative intent training, user-scoped recent episodic retrieval and an autoregressive byte pointer/generator. Run this historical path explicitly:

```bash
.venv/bin/python -m demo.conversation --legacy-phase11 --show-intent
# My favorite color is blue.
# What is my favorite color?
```

**Privacy change:** adding `--memory runs/user_memory/local.pt` automatically writes complete user and assistant turns to an **unencrypted** checkpoint. User names are not authentication. Run with `--no-episode-storage` to disable implicit transcript storage/retrieval; local history and explicitly requested facts still work. A per-turn "do not save" request is not stored; a session-wide "don't store anything" also disables future storage and local history for that session, without deleting older records. `Forget my <key>` removes that exact explicit fact and clears the user's *entire* stored conversation transcript, but not backups or disk remnants. Atomic saves do not make concurrent processes safe. Supply host-level consent, authentication, permissions and encryption.

The response network sees raw input bytes, intent and up to five retrieved episodes, and the generator receives retrieved memory vectors. A deterministic, exact-evidence fallback may answer when neural copying fails; benchmark output distinguishes these paths. Known factual questions, grammar-limited side-effect safeguards and tiny synthetic English datasets do not provide open-domain language competence or calibrated safety. Use `--legacy-phase10` for the original explicit-memory-only prototype. See `runs/phase11_report.md`, `AGENTS.md`, and `QWEN_CHAT_PROMPT.md` for reproducibility, held-out caveats, regression results and parameter counts.

Measured on small synthetic sets: vague intent **9/10**, overall intent **62/70**, and two-turn system retention **10/10** (only **5/10** from the learned decoder; five required the grounded fallback). Same-input sampled responses had **5 distinct byte sequences in 20 samples**; some wording was malformed. Injected memory vectors changed none of the ten greedy decoder outputs when raw episode bytes were retained. The full regression suite passed **147/147**. On the older Phase 10 development set, intent accuracy fell from **287/384 to 229/384**, particularly for remember/forget. Anchored explicit memory commands can override the classifier after it runs, and the result exposes the route. These are not evidence of general dialogue or privacy safety.

## Phase 12: vector-only memory, bounded replay and diverse byte responses

The default demo adds a narrow vector-only recall path:

```bash
.venv/bin/python -m demo.conversation --show-intent
# My favorite color is blue.
# What is my favorite color?
# Assistant: blue   [source=vector_reader, vector_only=True]
```

The system parses a limited affirmative fact grammar and stores 1–40 UTF-8 value bytes in one to five reversible width-64 bit-vector slots. For supported recall questions it supplies **only the question and retrieved vectors** to a trained neural readout, not the earlier transcript or a host-provided answer. An exact owner/topic parser selects the slots. This demonstrates vector dependence, **not learned semantic compression or general dialogue**. Values over 40 bytes are not supported by this path. With `--no-episode-storage`, automatic transcript/vector storage and retrieval are disabled; old explicit facts and in-process history remain available through earlier behavior.

**Privacy:** with `--memory PATH`, complete conversations **and reversible fact vectors** are stored unencrypted. Hashed owner/topic keys and the orthogonal rotation are not encryption; user IDs are not authentication. Opt in only with appropriate host-level consent, access controls and encryption. Forgetting a key purges its vectors and the user's stored transcript, not backups, disk remnants, other facts or unsynchronized writers. `Forget Alice's workplace` removes the corresponding workplace vector and transcript for that user. The default mode is ephemeral. `--legacy-phase11` and `--legacy-phase10` retain the older prototypes.

The capacity-1,000 guarded replay snapshot contains synthetic training examples only and is never populated automatically from user conversations. Intent training used 90 new and 38 old examples per batch; the new decoder objectives did not all use a 70/30 mix. Response diversity is trained with a differentiable loss over predicted byte distributions plus byte cross-entropy. Styles are synthetic and may be memorized.

Measured on small synthetic tests: **10/10** vector-only structured fact recalls; the public blue answer changed when vectors were zeroed or swapped; retention passed after **500 actual filler messages** and after a fresh-process save/reload. Fresh intent was **62/70 (88.6%)**. The inspected old Phase 10 development set improved to **298/384** from Phase 11's **229/384**, but this does not guarantee no forgetting. Direct decoding on one protected prompt yielded **14 distinct/20** and no structural/template-proxy failures; the complete session misrouted that prompt and produced only **1 distinct/20**. An exploratory raw-byte probe decoded 256/256 individual byte values but failed some multi-byte binary sequences. The full suite passed **168/168**. See `runs/phase12_report.md` for all limitations, ablations and reproduction commands.
