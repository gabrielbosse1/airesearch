# Phase 16 report: the scalable foundation (ByteCore v2)

Generated from `runs/phase16/foundation_metrics.json` and `runs/phase16/foundation_eval.json`.
Every number here comes from those artifacts; nothing is transcribed by hand.

## 1. Why this phase exists

The prototype hit a ceiling it could not engineer its way past. Tool selection was solved
(20/20) but only 17/20 frames carried byte-exact arguments, and `evaluation/copy_probe.py`
measured verbatim copying at 18/60 on unseen snippets. The pointer-generator copy head was
compensating for a decoder that could not reproduce bytes: it memorised training snippets
(92/96) instead of learning to copy (2/6 held out), and its copy distribution is tied to one
request's positions, so it does not scale. The rigid control-byte frames had the same problem
in a different form -- one wrong byte cost the whole turn.

Phase 16 deletes the compensation and keeps the thing that scales: a standard causal LM.

## 2. What was built

| File | Role |
| --- | --- |
| `models/foundation.py` | standard causal byte Transformer, 259 ids, config-driven scale |
| `models/react.py` | ReAct byte format, one interception regex, the host loop |
| `models/foundation_memory.py` | top-K retrieval rendered as `[MEM]` prefix bytes |
| `data/scalable_react_dataset.py` | 12,160 ReAct trajectories with sandbox-verified observations |
| `training/train_foundation.py` | plain next-byte cross-entropy, warmup + cosine, per-epoch checkpoints |
| `evaluation/foundation_eval.py` | next-byte accuracy, interception, memory recall, regression |
| `demo/tool_agent.py` | interactive ReAct agent (legacy frames behind `--legacy-frames`) |
| `tests/test_phase16.py` | 49 structural tests |

## 3. Architecture

`259` ids (bytes 0-255 + `BOS` 256 + `EOS` 257 + `PAD` 258), learned token and absolute
positional embeddings, 4 pre-norm blocks (causal multi-head self-attention + GELU FFN),
LayerNorm, one `Linear` head. **266,243 parameters** at `d_model=64`,
`n_heads=4`, `d_ff=256`, `max_seq_len=512`.

Three structural guarantees are enforced by tests, not by convention:

* no module, parameter or buffer in the model has a name containing `copy`, `pointer`, `gate`,
  `frame`, `tool_call`, `retro` or `verifier`;
* `models/foundation.py` does not import the legacy frame stack;
* the production preset (~87M parameters) is instantiated on the meta device, and a 512-wide
  stack runs a real forward pass on CPU -- the scaling claim is exercised.

## 4. Data

| Kind | Train | Validation | Test |
| --- | --- | --- | --- |
| chat | 1,120 | 40 | 40 |
| exec | 5,040 | 220 | 220 |
| file | 1,680 | 70 | 70 |
| memory | 1,120 | 50 | 50 |
| repair | 2,240 | 100 | 100 |

**12,160** trajectories, 2,820,960 training bytes, longest 469 bytes.
Splits are instance-disjoint (verified in `tests/test_phase16.py`). Every observation is produced
by the real `ByteSandbox` -- one temporary file per run, 5 s timeout -- and reduced by
`sandbox_observation`, the same function the runtime calls, so a training observation and a live
observation are byte-identical. Repair labels are asserted to be genuine failures.

## 5. Training

8 epochs at lr 0.003 from scratch, then 8 epochs at lr 0.0012 continuing from that checkpoint (runs/phase16/foundation_8epochs.pt); this report is the final epoch of the second stage.

Batch 64 with token-budget batching (4096 tokens), AdamW
(weight decay 0.01, 1 warmup epoch, cosine decay),
seed 16001, gradient clipping at 5.0, fused scaled-dot-product attention. The table below is
the second stage; its first row continues from the stage-one checkpoint.

| Epoch | LR | Train loss | Train acc % | Val loss | Val acc % | Grad norm | Elapsed s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.00120 | 0.33808 | 89.10 | 0.27359 | 91.14 | 0.50 | 337 |
| 2 | 0.00115 | 0.31708 | 89.75 | 0.24296 | 92.09 | 0.56 | 666 |
| 3 | 0.00102 | 0.28809 | 90.62 | 0.22083 | 92.67 | 0.45 | 992 |
| 4 | 0.00083 | 0.26054 | 91.45 | 0.19569 | 93.47 | 0.72 | 1312 |
| 5 | 0.00060 | 0.23345 | 92.29 | 0.17987 | 93.97 | 0.42 | 1629 |
| 6 | 0.00037 | 0.21335 | 92.93 | 0.16403 | 94.43 | 0.62 | 1955 |
| 7 | 0.00018 | 0.19991 | 93.39 | 0.15518 | 94.79 | 0.46 | 2273 |
| 8 | 0.00005 | 0.19319 | 93.59 | 0.15177 | 94.90 | 0.43 | 2593 |

## 6. Evaluation

### 6.1 Next-byte accuracy (held-out test split)

Overall **94.63%** over 480 trajectories; **95.03%** on the
243 short sequences (target >85%: met).

| Segment | Accuracy % | Bytes |
| --- | --- | --- |
| action | 97.06 | 26,307 |
| answer | 92.48 | 14,110 |
| observation | 94.21 | 18,856 |
| preamble | 90.16 | 34,794 |
| thought | 99.25 | 28,068 |

The per-segment split is the honest reading of the headline number: `Thought:` and `Answer:`
boilerplate is nearly free, while `Observation:` bytes -- the values the sandbox actually returned --
are the part that requires the model to have read the tool result.

### 6.2 Tool interception

| Check | Result | Detail |
| --- | --- | --- |
| exec_python intercepted and executed | PASS | `observation=42` |
| Observation injected after the Action | PASS | `Request: Run this. | Thought: compute it | Action: exec_python("print(6 * 7)") | Observati` |
| two-argument action parses both arguments | PASS | `(b'eval_note.txt', b'hello sandbox')` |
| unknown tool fails closed | PASS | `ERROR: unknown tool: delete_everything` |
| unterminated action is not executed | PASS | `parse_action returned None` |
| prose mentioning Action is not a call | PASS | `parse_action returned None` |
| sandbox error surfaces as the observation | PASS | `NameError: name 'pritn' is not defined. Did you mean: 'print'?` |
| round budget stops the loop | PASS | `rounds=2 steps=2` |

### 6.3 Memory recall

| Check | Result |
| --- | --- |
| Phase 12 vector-only recall | **10/10** |
| Short / long / cross-session context | True / True / None |
| Prefix contains the retrieved value | **True** |

Prefix produced by the live integration: `[MEM] my:favorite_color = blue`

### 6.4 Argument fidelity (the counterweight to the headline number)

Free-running greedy generation on 60 held-out `exec` trajectories:

| Measure | Result |
| --- | --- |
| Argument byte-identical to the reference | **42/60** |
| Generated code that runs cleanly in the sandbox | **58/60** |

Teacher-forced accuracy inside the `Action:` segment, split by byte class:

| Byte class | Accuracy | Bytes |
| --- | --- | --- |
| digits (must be copied from the request) | **89.7%** | 923 |
| letters | 96.7% | 17,603 |
| punctuation | 98.8% | 7,781 |

**This is the most informative result in the phase.** At the end of stage one (8 epochs) the same
measurement (saved in `runs/phase16/copy_fidelity.json`, measured with the stage-one checkpoint
`runs/phase16/foundation_8epochs.pt`) was **0/60** byte-exact with **27.3%** digit accuracy: the model had learned the
protocol perfectly and was inventing every literal. Eight more epochs took it to **42/60**
byte-exact and **89.7%** on digits -- with no copy head, no pointer network and no
copy-specific objective. Literal copying is not missing from a standard causal LM; it arrives with
training, which is precisely the argument for deleting the hand-built mechanism.

The residual gap is honest: 18/60 arguments still drift (usually one
operand), and the failures are visible rather than structural -- the sandbox runs the code and
reports what it actually produced, so a wrong literal is a wrong *answer*, not a crashed turn.

### 6.5 Regression

258 tests in 164.051s, exit code 0.

## 7. Targets

next_byte_accuracy_short > 0.85=met, tool_interception_all_passed=met, phase12_recall_10_of_10=met, regression_green=met

## 8. Limits and what was not shown

- **Format learning, not competence.** The trajectories come from a fixed template pool. A high
  next-byte score says the model learned the ReAct format and can copy held-out content; it says
  nothing about reasoning, planning or code understanding. The per-segment table exists so this
  cannot be hidden behind boilerplate.
- **Scale was not trained.** 266,243 parameters on 11,200 synthetic trajectories
  is a structural proof, not a scaled result. The production configuration is instantiated and its
  parameter count measured, but not trained.
- **Learned absolute positions** cap context at `max_seq_len`. This is the standard GPT-style
  baseline; moving to RoPE or relative positions is a separate change.
- **The sandbox is hardening, not a security boundary.** The ReAct loop executes whatever appears
  between the quotes, which is *easier* to trigger than the old frame parser. The round budget and
  the root-containment check on file tools are mitigations, not guarantees.
- **Retrieval is structured lookup**, not semantic search: `top_k` bounds how many stored facts can
  reach the prefix, and the store returns exact topic matches first.

## 9. Reproduce

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m training.train_foundation --epochs 8 --batch-size 64 --lr 0.0012
python -m evaluation.foundation_eval
python -m demo.tool_agent --once "Calculate 6 * 7."
python -m unittest discover -s tests
```
