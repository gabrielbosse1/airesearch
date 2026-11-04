# QWEN_CHAT_PROMPT.md — working prompt for the ByteCore-DualMemory repository

You are working inside a cleaned, single-architecture PyTorch repository:
a byte-level causal Transformer plus an external hierarchical memory store. The
normative description of the code is `ARCHITECTURE_AUDIT.md`; operational rules
are in `AGENTS.md`. Both are shorter than they look and both are accurate — read
them before proposing anything.

## What you may assume (verified facts)

- **Vocabulary:** 261 ids. Bytes `0..255` are their own ids; `BOS=256`,
  `EOS=257`, `PAD=258`, `REF_START=259`, `REF_END=260`. No tokenizer, ever.
- **Model:** `models/foundation.py::CausalByteTransformer`. Production preset
  `configs/t4_scaled.yaml`: `d_model=1024`, `n_layers=12`, `n_heads=8`,
  `n_kv_heads=2`, `d_ff=4096`, `max_seq_len=2048`, RoPE, RMSNorm, SwiGLU,
  `refinement_steps=2`, `refinement_rank=256` — **183,830,791 parameters**
  (embedding 267,264 · 12 blocks × 15,208,960 · refinement 787,458 · final norm
  1,024 · head 267,525).
- **Blocks are pre-norm:** `x ← x + Attn(RMSNorm(x))`, then
  `x ← x + SwiGLU(RMSNorm(x))`. RoPE rotates `q`/`k` in pairs `(i, i+64)`;
  `n_kv_heads=2` means four query heads share one KV head.
- **Refinement** is `h ← h + α_t·Cell(RMSNorm(h))`, twice, with a shared
  `1024→256→1024` gated cell and `α` initialised to zero (identity at init).
  It is applied to the whole `(B,L,1024)` tensor before the byte head. It is
  **not** a thought-token stream and there is no halting head.
- **Memory** (`models/memory_model.py`) is a tensor store in its own `d_model`
  space: episodic `(256,64)`, semantic exemplars `(128,4,64)` with a mixture
  LogSumExp readout, procedural `(128,64)` plus `(128,4,64)` traces. Nothing in
  `training/`, `data/`, `evaluation/` or `notebook/` calls it.
- **Runtime:** `models/react.py` intercepts `Action: tool("arg")` with one regex
  and appends a real `Observation:`; `models/sandbox.py` executes confinement;
  `models/foundation_memory.py` renders memory as `[MEM] topic = value` bytes.
- **Data** is real and streamed: `data/phase22_sources.py` (The Stack, GigaVerbo /
  fineweb-2 PT-BR, fineweb-edu / DCLM, open-web-math; no_robots, Tucano-SFT,
  distilabel-SFT, CodeFeedback, Minecraft VLA / MET-Bench).
- **Training** is three phases and one class: `training/phase22_train.py`
  (`--phase 1|2|3`) producing `checkpoint_phase1.pt`, `checkpoint_phase2.pt`,
  `final_model.pt` with `schema=3` state (weights, AdamW, GradScaler, RNG,
  `BytePacker` carry, dataset manifest).

## Phase 22: what is actually wired

The Phase 22 run is `configs/t4_scaled.yaml` + `training/phase22_train.py` +
`data/phase22_sources.py`. Three facts matter when answering questions about it:

1. The stack is RMSNorm + SwiGLU + RoPE + GQA (2 KV heads) + weight-tied
   Iterative Refinement (2 passes) at `d_model=1024`, `n_layers=12` — the same
   class and the same code path as the CPU test config, only the integers differ.
2. `precision: auto` exists because bfloat16 CUDA autocast is unavailable on the
   T4 target (sm_75); the recorded intent stays `requested_precision: bfloat16`
   and the resolved dtype is float16 plus `GradScaler` on Turing.
3. The three outputs are `checkpoint_phase1.pt`, `checkpoint_phase2.pt` and
   `final_model.pt`, all `schema=3`. Nothing in this repository is a trained
   Phase 22 checkpoint, so no quality, loss or speed claim about the run is
   available here.

## Phase 21: what is history

Phase 21 diagnosed the Phase 19 collapse (≈81 % of the byte probability mass on
byte 32, no NaN/Inf, healthy gradients) and is retained only as evidence under
`runs/phase21/`. Its scripts, the clone-based notebook and `configs/phone_100m.yaml`
were deleted in Phase 23; do not resurrect them, and do not treat the Phase 19
numbers as describing the current model.

## What you must not do

1. Do not reintroduce a tokenizer, a BPE vocabulary, or a vocabulary size other
   than 261.
2. Do not reintroduce the deleted stacks: diffusion core, pointer-generator copy
   head, `GeneratorConditioned`, control-byte frames, the conversation/vector
   stack, `ConfidenceHead`, adaptive halting, or an intent classifier. They are
   gone; the audit explains what replaced them (usually: nothing, or plain
   bytes).
3. Do not describe memory as "differentiable" without naming the one place it is:
   `MemoryModel.retrieve_differentiable` and its straight-through estimator.
   Writes are gradient-free by design.
4. Do not claim GPU numbers. VRAM, throughput and bfloat16/float16 behaviour are
   unmeasured in this environment; `precision: auto` exists because T4 (sm_75)
   cannot do bfloat16 autocast.
5. Do not claim the Phase 3 search tool performs a web search. It appends a
   constant string.
6. Do not hand-edit the three notebooks; regenerate with
   `python -m notebook.build_phase22_notebooks`.

## House style

- Report `[THINK]`, `[MAP]`, `[SOLVE]`, `[⚠️ WARN]`, `[EXECUTE]`, `[UPDATE]`,
  `[BENCHMARK]` for architecture work.
- State tensor shapes with every operation: `(B, L, D)` is `(4, 1024, 1024)` in
  the production preset.
- Separate measured from analytic from unmeasured. Quote the command that
  produced a number.
- Keep `pytest -q` green (92 tests, 1 skip). If a change alters an invariant in
  `ARCHITECTURE_AUDIT.md`, update the document and the corresponding test in
  `tests/test_foundation.py` / `test_memory_model.py` / `test_react_runtime.py`.

## Fast orientation commands

```bash
python -m pytest -q                                   # full regression suite
python -m evaluation.phase22_benchmark                # parameter/VRAM/inventory
python -m training.phase22_train --self-test          # CPU plumbing smoke test
git ls-files | sed 's|/[^/]*$||' | sort -u            # the whole repo, one screen
```
