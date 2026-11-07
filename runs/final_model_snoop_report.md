# ByteCore-DualMemory — Phase 26 Deep Model Snoop

- Checkpoint: `gabrielbosse9/airesearch2` / `Hub lookup`
- SHA-256: `not available`
- Training step: unknown
- Device: `auto`
- Parameters: not measured

## Executive Summary

Run status: **blocked before inference.**
Checkpoint/load error, if any: `ConnectError: TLS/SSL connection has been closed (EOF) (_ssl.c:992)`
This report distinguishes checkpoint facts, observed generations, and claims that still require human or task-level validation. A four-hour pre-training budget is not evidence of finished instruction-following capability. Text and code examples below are raw byte-model continuations, not curated outputs. If the run status is blocked, no model capability or activation claims are being made.

## Architecture Verification

- Production parameter-count match (366,316,833): **not run**
- Eight-block match: **not applicable**
- Sliding-window blocks: **not run**
- Skill adapters in every block: **not run**
- Config: `{}`

## Activation Forensics

Each probe records final-position sliding-window attention weights, per-layer SwiGLU/GLU output norms, latent thought-buffer norm and gates, and memory cross-attention softmax summaries when memory is present. These are instrumentation signals—not proof of useful reasoning or factual recall.

## Capability Matrix

- **PT-BR Fluency:** Not run /10
- **Code Logic:** Not run /10
- **Tool Use:** Not run /10
- **Memory Retrieval:** Not run /10

## The Toddler Limits

Do not infer a context-length capability from the configured maximum: this run tests only the prompt and generation lengths shown above. A syntactically valid Python snippet is not proof that it runs or is correct. The model has no built-in edit handler; `[EDIT]` is ordinary learned vocabulary. A tool request is only useful if the dedicated control ID is emitted and the surrounding host runtime executes it. Synthetic memory injection does not establish reliable real-world memory retrieval.

## Next Steps

1. Repeat deterministic prompts at several seeds/temperatures and compare against a held-out PT-BR/code set.
2. Evaluate Python outputs in an isolated sandbox with unit tests, not merely `compile()`.
3. Measure tool-call precision/recall and exact answer accuracy against a controlled search harness.
4. Compare injected-memory answer accuracy and target-slot attention against no-memory and shuffled-key controls.
5. Publish the checkpoint step, training-token count, tokenizer-free byte budget, and independent human review before assigning 0–10 capability scores.

## Forensic Limitations

- Checkpoint download or load failed; no generation or activation evidence was collected.
