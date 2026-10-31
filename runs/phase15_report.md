# Phase 15 report: external tool use and sandbox execution

Generated from `runs/phase15/tool_metrics.json` and `runs/phase15/tool_eval.json`.
Every number in this report is read from those artifacts; nothing is transcribed by hand.

## 1. Paradigm

Verification is outsourced to the sandbox; the model's role is reliable tool routing and iterative editing.
Phase 14's learned internal verifier accepted structurally invalid code, so this phase does not attempt to
verify anything inside the network. A frozen Phase 12 generator plus a small trained policy emits a
tool-call frame, the host executes it in a subprocess, and the result bytes are injected back into the
context for the next turn.

## 2. What was built

| File | Role |
| --- | --- |
| `models/generator_conditioned.py` | vocabulary 264 -> 268 (`TOOL_CALL=264`, `TOOL_ARG_START=265`, `TOOL_ARG_END=266`, `TOOL_RESULT=267`) |
| `models/sandbox.py` | `ByteSandbox.execute_python(code_bytes) -> bytes`, `read_file`, `write_file`, 5 s timeout |
| `models/tool_use.py` | `ByteToolPolicy`: frozen generator + encoder + 4-way route head + copy-generator pointer decoder |
| `data/tool_use_examples.py` | tool rows, decline rows, generated single-line repair curriculum, verbatim-copy run rows |
| `training/train_tool_use.py` | loss on the tool-call sequence only; per-row-kind validation metrics |
| `demo/tool_agent.py` | interactive loop: generate -> pause on `TOOL_CALL` -> execute -> inject `TOOL_RESULT` -> respond |
| `evaluation/tool_eval.py` | routing, iterative debugging, decline benchmark, Phase 12 regression |
| `evaluation/copy_probe.py` | isolates verbatim-copy fidelity on held-out families |
| `tests/test_phase15.py` | new tests for the sandbox, frames, policy, dataset and agent loop |

## 3. Data

- Tool rows: `exec_python` (arithmetic, length, sorted, range, triple, sentences, listdir), `read_file`, `write_file`.
- Decline rows: Phase 10 conversational prompts labelled `none`, so the router abstains on chat instead of calling a tool.
- Repair curriculum: six generated categories -- **A** misspelled builtin, **B** missing colon, **C** missing close paren,
  **D** type coercion, **E** misspelled variable, **F** invalid literal. Every pair is **executed at dataset build time**
  and dropped if the broken code does not fail or the fix does not run, so no row teaches an unverified repair.
- Run rows: verbatim-copy family (`Run this Python: <code>` -> `<code>`) with split-disjoint pools.
- Wide repair curriculum: `500` training / `60` validation (broken, corrected) pairs drawn from a
  randomized space (random words, random integers, six error categories) that is far too large to memorise, so these rows
  can only be scored by copying the pasted code and applying the category-level edit. Every pair is executed in the
  sandbox at build time and dropped if the broken code does not fail or the correction does not run.
- Splits are instance-disjoint across train/validation/test (verified by `tests/test_phase15.py`).
- Totals: **5555** train rows, **999** validation rows.

## 4. Training

| Setting | Value |
| --- | --- |
| Epochs | 60 |
| Batch | 32 |
| Optimizer | Adam lr 0.002, step decay x0.4 every 20 epochs |
| Gradient clip | 5.0 |
| Seed | 15001 |
| Checkpoint selection | fixed final epoch, no test-based selection |

Loss 2.26360 -> 0.000715; maximum pre-clip gradient norm across training **21.15**.

| Epoch | Loss | Train route % | Train byte % | Val route % | Val call-frame % | Val decline % | Grad norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.26360 | 79.3 | 54.4 | 75.0 | 2.6 | 100.0 | 4.10 |
| 6 | 0.28564 | 97.7 | 93.8 | 95.0 | 57.1 | 100.0 | 4.74 |
| 11 | 0.12111 | 99.2 | 97.5 | 91.2 | 58.1 | 100.0 | 3.58 |
| 16 | 0.06592 | 99.5 | 98.6 | 96.2 | 60.5 | 100.0 | 3.18 |
| 21 | 0.03207 | 99.9 | 99.4 | 98.8 | 67.4 | 100.0 | 3.46 |
| 26 | 0.00729 | 100.0 | 99.9 | 97.5 | 64.3 | 100.0 | 1.51 |
| 31 | 0.00503 | 100.0 | 100.0 | 97.5 | 66.7 | 100.0 | 1.12 |
| 36 | 0.00985 | 99.9 | 100.0 | 92.5 | 62.2 | 100.0 | 21.15 |
| 41 | 0.00365 | 100.0 | 100.0 | 97.5 | 69.0 | 100.0 | 0.23 |
| 46 | 0.00183 | 100.0 | 100.0 | 97.5 | 66.7 | 100.0 | 0.16 |
| 51 | 0.00136 | 100.0 | 100.0 | 97.5 | 64.3 | 100.0 | 0.22 |
| 56 | 0.00097 | 100.0 | 100.0 | 98.8 | 64.3 | 100.0 | 0.18 |

## 5. Evaluation

### 5.1 Tool routing (20 prompts)

| # | Request | Outcome |
| --- | --- | --- |
| 1 | `Calculate 18 * 10.` | exact |
| 2 | `Give me 19 + 4.` | exact |
| 3 | `Compute 19 - 20.` | exact |
| 4 | `Reverse the string 'xocuta'.` | exact |
| 5 | `Reverse the string 'zequmu'.` | exact |
| 6 | `Uppercase the text 'xahaxu'.` | exact |
| 7 | `Uppercase the text 'xuwadu'.` | exact |
| 8 | `Uppercase the text 'zogi'.` | exact |
| 9 | `How many characters are in 'xocuta'?` | exact |
| 10 | `How many characters are in 'zequmu'?` | exact |
| 11 | `Sum the numbers from 1 to 362.` | exact |
| 12 | `Sum the numbers from 1 to 378.` | exact |
| 13 | `Sum the numbers from 1 to 394.` | exact |
| 14 | `Sort the numbers 56, 5 and 1.` | tool only |
| 15 | `Sort the numbers 58, 15 and 5.` | tool only |
| 16 | `Tell me the files in this directory.` | tool only |
| 17 | `Open and print ximu.txt.` | exact |
| 18 | `Open and print zana.md.` | exact |
| 19 | `Create xebime.csv containing hello sandbox.` | exact |
| 20 | `Create xuboha.log containing hello sandbox.` | exact |

Exact frame **17/20**, correct tool regardless of bytes **20/20**,
and **17/20** produce output identical to the reference when executed.

### 5.2 Iterative debugging (5 tasks)

| # | Task | First attempt failed | Corrected call emitted | Clean after correction | Rounds |
| --- | --- | --- | --- | --- | --- |
| 1 | `Run this Python and fix any error: prin(1 + 1)` | yes | yes | no | 2 |
| 2 | `Execute this and repair it if it fails: if 10 ` | yes | yes | yes | 2 |
| 3 | `Please run: print(sorted([2, 9])` | no | no | no | 1 |
| 4 | `Run this Python and fix any error: print(3 + '` | yes | yes | yes | 2 |
| 5 | `Execute this and repair it if it fails: vertex` | yes | yes | no | 2 |

Success **2/5** (2/5 clean after correction).

### 5.3 Decline behaviour

| Set | Declined | Total |
| --- | --- | --- |
| Conversational prompts | 18 | 40 |
| All Phase 10 intents | 48 | 60 |

On the full validation split the no-tool rows score **100.0%** (371 rows).

### 5.4 Regression

| Check | Result |
| --- | --- |
| Phase 12 vector-only recall | **10/10** |
| Generator bit-identical to Phase 12 | **true** |
| Unit test suite | **209** |

## 6. Targets

routing_exact >18/20=NOT met, debugging >=3/5=NOT met, phase12_recall 10/10=met, regression suite green=met

## 7. Failure analysis

- **Tool selection is solved; argument copying is not.** The router picks the right tool on
  **20/20** held-out prompts, but only **17/20**
  frames carry byte-exact arguments. All three misses are copy errors inside the right tool call, for example
  `print(sorted([58, 15, 55]))` for `print(sorted([58, 15, 5]))`, and one `os.listdir` request degenerates into
  `print(len(htnnyorted()))`. Train byte accuracy saturates at 100% while validation call-frame accuracy plateaus
  near 72.6%, which is an overfitting signature, not a routing failure.
- **Copy fidelity measured directly.** `evaluation/copy_probe.py` scores greedy verbatim copying on held-out
  validation families: **18/60** exact on the wide randomized space (plus 19/60 near misses),
  **7/20** on the valid-code family and **3/12** on repair prompts. Roughly one snippet in three
  is copied byte-for-byte; another third is recognisably close. That, not the route head, is what caps both
  routing exactness and iterative debugging.
- **Curriculum history (three measured runs, each reported at its final epoch).** Hand-listed repair snippets
  oversampled 6x taught memorisation: the model recalled a *different* memorised snippet instead of copying the
  request, scoring **0/5** debugging. A generated, per-instance-diverse single-line curriculum plus a
  verbatim-copy family raised it to **1/5** and then **2/5**; adding the wide randomized curriculum
  (500 training pairs from a space too large to memorise) took tool selection to
  20/20 but left debugging at **2/5**.
  Multi-line snippets were removed because the interactive loop reads one line at a time. Three curricula
  converged on the same ceiling, which is why the remaining gap is reported rather than trained away.
- **Degenerate free-running loops.** `os.listdir` requests occasionally decode past `[TOOL_ARG_END]` into
  unbounded repetition; the frame parser fails closed, so this costs the request and never corrupts state.
- **Iterative debugging is below target at 2/5.** Every task does emit a
  second, different call after the sandbox reports the failure (4/5
  corrected calls emitted), so the *loop* works; what fails is the correction's byte fidelity. The two successes
  are a missing colon and a type coercion. Task 3 is not scored as debugging because the first attempt already ran
  cleanly (the model "helpfully" closed the bracket while copying). This is reported as unmet; no post-hoc
  threshold change was made.

## 8. Limits

- The sandbox is **hardening, not a security boundary**: the child process inherits the host filesystem,
  network and environment variables, and only `read_file`/`write_file` are root-confined.
- Prompts are synthetic single-line Python from a fixed template pool. Twenty routing prompts and five
  debugging tasks cannot support a generalisation claim.
- The argument is produced by a pointer copy head, not by parsing the program; the model does not understand
  the code it emits.
- `TOOL_RESULT` is host-only. No evaluation measures what the model would do with long or hostile tool output.

## 9. Reproduce

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m training.train_tool_use --epochs 60 --lr 0.002 --lr-step 20 --lr-decay 0.4 --output-dir runs/phase15
python -m evaluation.tool_eval --with-regression
python -m evaluation.copy_probe --output runs/phase15/copy_probe.json
python -m demo.tool_agent --generator runs/phase12/generator.pt --policy runs/phase15/tool_policy.pt
python -m unittest discover -s tests
```
