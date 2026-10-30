# Phase 9: explicit cognitive edit arguments and native masked diffusion

## Requested one-byte diagnostic

Source: `Hello Worl` = `[72, 101, 108, 108, 111, 32, 87, 111, 114, 108]`. The explicit request is to replace **zero-based position 9**, byte 108, with byte 100; expected output is `Hello Word` = `[72, 101, 108, 108, 111, 32, 87, 111, 114, 100]`.

| Path | Outcome | What was actually supplied |
|---|---|---|
| Native mask-conditioned diffusion | **PASS**: `Hello Word`; masked model prediction `100`; 50 DDPM steps | Original bytes, mask at 9, and requested byte 100. Unmasked embeddings and bytes are preserved; the masked output is decoded from the model, not overwritten with 100. |
| Autonomous cognitive loop, natural instruction | **PASS**: `Hello Word`; EDIT arguments `[9, 2, 100]` | No memory, host controller or target canvas; **this exact example was included and oversampled in training**. |
| Guarded memory, isolated copy | Existing snapshot: **no match**. After explicitly storing the exact supplied mapping and reloading: **PASS**. | Explicitly stored answer; exact lookup/replay, not neural generation. The original memory file is not modified. |

The cognitive trace is `LOAD(0,10) -> THINK -> EDIT_OP [9,2,100] -> SAVE -> DONE`; SAVE is necessary because edits are staged. An optional *separate* structured request `[9,2,100]` also succeeds without a controller, but already specifies the edit and is an input, not evidence that the model parsed English. The earlier baseline had no native mask-conditioned feature, its host-side fallback predicted byte 0, and its autonomous cognitive result was `Hell? Worl`.

For a counterfactual desired byte 101, the same mask/seed predicted `101`. With no requested byte, mask-only inpainting predicted `127`; a mask cannot determine what replacement is intended. The same requested-byte diagnostic succeeded for 32/32 additional batched samples and preserved all other bytes.

## Independent synthetic evaluation

- **Natural-language cognitive generalization: 0/68 exact**, despite the trained-on Hello pass. The synthetic test examples have different source/target bytes but share a template; this is not a real-world language benchmark. The original eight real held-outs have no actionable edit instruction and remain **0/8** exact. The natural-language goal beyond the acceptance example is **unmet**.
- **Structured-request cognition: 254/264 exact** and 257/264 produced matching edit arguments. Position >255: 7/8 exact. These requests explicitly contain position, opcode and byte; this evaluates learned conditional copying plus control, **not** inference of the request from English. Three failures repeatedly issued EDIT without committing, and other failures chose incorrect spans; DONE alone is not success. Only single-byte replacement conditioning was trained.
- **Masked diffusion:** 78/80 exact complete rows, 1155/1159 correct masked bytes with all unmasked bytes preserved. A separate set supplies every requested byte value 0–255 once in a new ten-byte source: **235/256** exact, all unmasked bytes preserved. This is not a perfect or general text editor, and the desired values were explicitly supplied.
- These are deterministic, synthetic, source-content-disjoint splits, not semantic-independence proofs. All training examples were preflighted against registered real, processed, and experiment-specific held-outs; guards cannot audit unknown pretrained data or near duplicates. Test splits did not select the reported checkpoints.

## Training and selection

- Natural cognition: 10 epochs from the Phase 8 selected generator, 128 short + 4 long synthetic inputs, 10 curriculum anchors, and the user example repeated 32 times/epoch. Selection **prioritized the in-training Hello success**, then validation; epoch 4 was retained. Supervised trace targets train existing action, pointer, kind and byte heads. The 352,728-parameter count in `cognitive_metrics.json` reflects the code before the subsequently added optional structured adapter.
- Structured cognition: 544 separate training examples, 6 epochs; frozen natural policy plus trainable argument compatibility projections. Validation selected epoch 2, with 72/72 exact. The requested fields were explicitly supplied to both training and evaluation; no full target or controller enters inference.
- Diffusion: 640 synthetic raw-byte examples, 30% random masks (at least one selected byte per nonempty row), 600 updates, frozen denoising backbone. Validation selected step 300, with 610/610 correct masked bytes. A learned mask feature and requested-value projection condition the core; MSE supervises selected positions only.
- Ten-byte warm CPU microbenchmark, one thread, 30 runs, model load excluded: median 68.74 ms for 50-step masked diffusion, 16.31 ms for the five-action cognitive example. Times are machine-specific, not throughput or accuracy guarantees. See `runs/phase9/latency.json`.

## Argument wire format and compatibility

An autonomous EDIT event exposes `event.argument_bytes` and `event.control_and_arguments` (e.g. `(262,9,2,100)`); those literal 0–255 argument bytes are serialized from the existing jointly predicted heads, then parsed and validated **before** application. This is a structured control/argument stream, **not an autoregressive prediction of each argument byte**. Typed `CognitiveAction`, host-controller operations, staging, action replay and legacy RL scoring remain available.

- The first three argument bytes are `[position_low, opcode_and_flags, payload_byte]`. Short single-byte edits use insert=0, delete=1, replace=2, move=3; delete/move require a zero placeholder. Thus position 9 replace with 100 is `[9,2,100]`.
- Bit `0x80` marks position high bits encoded in canonical unsigned LEB128 after the three-byte header; position 300 replace with 100 is `[44,130,100,1]`. The default decoded canvas-position limit is 4096, configurable by the caller.
- Bit `0x40` encodes a non-unit span length as unsigned LEB128; bit `0x20` encodes a non-unit payload length followed by the remaining payload bytes. Extensions occur in position, span, payload order. MOVE then appends its original-workspace destination as unsigned LEB128; e.g. moving `[1,3)` to destination 8 is `[1,67,0,2,8]`. Trailing, noncanonical, malformed, oversized or impossible fields are rejected. Coordinates are relative to the loaded workspace.
- Multi-byte payloads and arbitrary spans are codec/host capabilities; autonomous insert/replace currently predict **one** payload byte per edit. The optional structured model rejects untrained operation kinds and non-unit replace spans by default. Neither Fourier positional features nor a valid wire format guarantee accuracy at long positions.

The width-64 generator now has 369,368 parameters, including 16,640 new structured-request parameters; 12,544 were trained in this experiment. Diffusion now has 200,065 parameters, including 4,160 new mask-conditioning parameters. No pretrained base checkpoint was overwritten. Entirely absent old heads migrate to zero and are marked untrained; partial current heads/metadata fail strict loading. Older software cannot strictly load new states, and optimizers should be rebuilt for changed parameter groups. Request/mask trained flags are caller-maintained metadata, not accuracy or provenance guarantees. Confidence calibration does not carry over to masked edits; masked adaptive halting is refused.

## Native diffusion API

```python
import torch
from training.train_masked_diffusion import load_model

model = load_model("configs/default.yaml", "runs/phase9/diffusion").eval()
ids = torch.tensor([[72,101,108,108,111,32,87,111,114,108]])
mask = torch.tensor([[0,0,0,0,0,0,0,0,0,1]], dtype=torch.bool)
request = torch.zeros_like(ids); request[0, 9] = 100
torch.manual_seed(42)
out = model.masked_edit(ids, mask, request)  # masked value is the model's prediction
assert out.tolist() == [[72,101,108,108,111,32,87,111,114,100]]
```

`p_sample_loop_with_mask(shape, edit_mask, original_emb=..., requested_byte_ids=...)` exposes the embedding-level path; `p_sample_loop` also accepts these keywords. `compute_loss` accepts `edit_mask=` or `random_edit_mask_probability=`, optional `source_emb` and `requested_byte_ids`. A mask without a requested value performs uncertain inpainting. The original diffusion core lacks a padding-attention mask, so padding width changes the context. `EditRouter` accepts `EditRequest(..., edit_mask=raw_binary_bytes, replacement_bytes=raw_bytes)` for structured fixed-length edits and rejects unparsed natural-language instructions.

## Reproduce

```bash
.venv/bin/python data/generate.py  # rebuilds the ignored processed holdout registry
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_positional_edits
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_structured_requests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_masked_diffusion
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase9_diagnostic
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase9_benchmark
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m evaluation.phase9_report
```

Original detailed evidence: `runs/phase9/cognitive_metrics.json`, `structured_metrics.json`, `diffusion_metrics.json`, `single_byte_edit_diagnostic.json` and `latency.json`. Verification: 101/101 passed, zero failures/errors/skips.
