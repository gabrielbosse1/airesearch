"""Build a transparent Phase 9 report from saved metrics and diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/phase9")
    parser.add_argument("--output", default="runs/phase9_report.md")
    args = parser.parse_args()
    root = Path(args.run_dir)
    diag = load(root / "single_byte_edit_diagnostic.json")
    cognitive = load(root / "cognitive_metrics.json")
    structured = load(root / "structured_metrics.json")
    diffusion = load(root / "diffusion_metrics.json")
    latency = load(root / "latency.json")
    verified_path = root / "verification.json"
    verification = load(verified_path) if verified_path.exists() else {}
    natural_test = cognitive["heldout_positional_test"]
    structured_test = structured["heldout_structured_test"]
    long_test = [row for row in structured_test["examples"] if row["position"] > 255]
    d_test = diffusion["heldout_multimask_test"]
    coverage = diffusion["heldout_all_byte_values_test"]
    d = diag["diffusion_native"]
    c = diag["cognitive_natural_only"]
    memory = diag["memory"]
    diff_ms = latency["measurements"]["diffusion_masked_50_steps"]["median_ms"]
    cog_ms = latency["measurements"]["cognitive_natural_5_actions"]["median_ms"]
    report = f'''# Phase 9: explicit cognitive edit arguments and native masked diffusion

## Requested one-byte diagnostic

Source: `Hello Worl` = `{diag['source_bytes']}`. The explicit request is to replace **zero-based position 9**, byte 108, with byte 100; expected output is `Hello Word` = `{diag['target_bytes_for_scoring']}`.

| Path | Outcome | What was actually supplied |
|---|---|---|
| Native mask-conditioned diffusion | **{'PASS' if d['exact'] else 'FAIL'}**: `{d['output_text']}`; masked model prediction `{d['predicted_masked_byte']}`; {d['sampling_steps']} DDPM steps | Original bytes, mask at 9, and requested byte 100. Unmasked embeddings and bytes are preserved; the masked output is decoded from the model, not overwritten with 100. |
| Autonomous cognitive loop, natural instruction | **{'PASS' if c['exact'] else 'FAIL'}**: `{c['output_text']}`; EDIT arguments `{next((a['argument_bytes'] for a in c['actions'] if a['token'] == 262), None)}` | No memory, host controller or target canvas; **this exact example was included and oversampled in training**. |
| Guarded memory, isolated copy | Existing snapshot: **{'match' if memory['existing_memory']['found'] else 'no match'}**. After explicitly storing the exact supplied mapping and reloading: **{'PASS' if memory['after_explicit_write_and_reload']['exact'] else 'FAIL'}**. | Explicitly stored answer; exact lookup/replay, not neural generation. The original memory file is not modified. |

The cognitive trace is `LOAD(0,10) -> THINK -> EDIT_OP [9,2,100] -> SAVE -> DONE`; SAVE is necessary because edits are staged. An optional *separate* structured request `[9,2,100]` also succeeds without a controller, but already specifies the edit and is an input, not evidence that the model parsed English. The earlier baseline had no native mask-conditioned feature, its host-side fallback predicted byte 0, and its autonomous cognitive result was `Hell? Worl`.

For a counterfactual desired byte 101, the same mask/seed predicted `{d['counterfactual_request_101']['predicted_byte']}`. With no requested byte, mask-only inpainting predicted `{d['mask_only_no_requested_byte']['predicted_byte']}`; a mask cannot determine what replacement is intended. The same requested-byte diagnostic succeeded for {d['repeated_samples']['exact_requested_byte']}/{d['repeated_samples']['samples']} additional batched samples and preserved all other bytes.

## Independent synthetic evaluation

- **Natural-language cognitive generalization: {natural_test['exact']}/{natural_test['count']} exact**, despite the trained-on Hello pass. The synthetic test examples have different source/target bytes but share a template; this is not a real-world language benchmark. The original eight real held-outs have no actionable edit instruction and remain **{cognitive['real_test_no_actionable_instruction']['exact']}/{cognitive['real_test_no_actionable_instruction']['count']}** exact. The natural-language goal beyond the acceptance example is **unmet**.
- **Structured-request cognition: {structured_test['exact']}/{structured_test['count']} exact** and {structured_test['correct_edit_arguments']}/{structured_test['count']} produced matching edit arguments. Position >255: {sum(row['exact'] for row in long_test)}/{len(long_test)} exact. These requests explicitly contain position, opcode and byte; this evaluates learned conditional copying plus control, **not** inference of the request from English. Three failures repeatedly issued EDIT without committing, and other failures chose incorrect spans; DONE alone is not success. Only single-byte replacement conditioning was trained.
- **Masked diffusion:** {d_test['exact']}/{d_test['count']} exact complete rows, {d_test['masked_byte_correct']}/{d_test['masked_byte_count']} correct masked bytes with all unmasked bytes preserved. A separate set supplies every requested byte value 0–255 once in a new ten-byte source: **{coverage['exact']}/{coverage['count']}** exact, all unmasked bytes preserved. This is not a perfect or general text editor, and the desired values were explicitly supplied.
- These are deterministic, synthetic, source-content-disjoint splits, not semantic-independence proofs. All training examples were preflighted against registered real, processed, and experiment-specific held-outs; guards cannot audit unknown pretrained data or near duplicates. Test splits did not select the reported checkpoints.

## Training and selection

- Natural cognition: {cognitive['configuration']['epochs']} epochs from the Phase 8 selected generator, {cognitive['training_example_counts']['synthetic_short']} short + {cognitive['training_example_counts']['synthetic_long']} long synthetic inputs, {cognitive['training_example_counts']['curriculum_anchors']} curriculum anchors, and the user example repeated {cognitive['training_example_counts']['diagnostic_repeats_per_epoch']} times/epoch. Selection **prioritized the in-training Hello success**, then validation; epoch {cognitive['selected_epoch']} was retained. Supervised trace targets train existing action, pointer, kind and byte heads. The 352,728-parameter count in `cognitive_metrics.json` reflects the code before the subsequently added optional structured adapter.
- Structured cognition: {structured['training_examples']} separate training examples, {structured['configuration']['epochs']} epochs; frozen natural policy plus trainable argument compatibility projections. Validation selected epoch {structured['selected_epoch']}, with {structured['selected_structured_validation']['exact']}/{structured['selected_structured_validation']['count']} exact. The requested fields were explicitly supplied to both training and evaluation; no full target or controller enters inference.
- Diffusion: {diffusion['training_examples']} synthetic raw-byte examples, 30% random masks (at least one selected byte per nonempty row), {diffusion['training_updates']} updates, frozen denoising backbone. Validation selected step {diffusion['selected_step']}, with {diffusion['selected_validation']['masked_byte_correct']}/{diffusion['selected_validation']['masked_byte_count']} correct masked bytes. A learned mask feature and requested-value projection condition the core; MSE supervises selected positions only.
- Ten-byte warm CPU microbenchmark, one thread, 30 runs, model load excluded: median {diff_ms:.2f} ms for 50-step masked diffusion, {cog_ms:.2f} ms for the five-action cognitive example. Times are machine-specific, not throughput or accuracy guarantees. See `runs/phase9/latency.json`.

## Argument wire format and compatibility

An autonomous EDIT event exposes `event.argument_bytes` and `event.control_and_arguments` (e.g. `(262,9,2,100)`); those literal 0–255 argument bytes are serialized from the existing jointly predicted heads, then parsed and validated **before** application. This is a structured control/argument stream, **not an autoregressive prediction of each argument byte**. Typed `CognitiveAction`, host-controller operations, staging, action replay and legacy RL scoring remain available.

- The first three argument bytes are `[position_low, opcode_and_flags, payload_byte]`. Short single-byte edits use insert=0, delete=1, replace=2, move=3; delete/move require a zero placeholder. Thus position 9 replace with 100 is `[9,2,100]`.
- Bit `0x80` marks position high bits encoded in canonical unsigned LEB128 after the three-byte header; position 300 replace with 100 is `[44,130,100,1]`. The default decoded canvas-position limit is 4096, configurable by the caller.
- Bit `0x40` encodes a non-unit span length as unsigned LEB128; bit `0x20` encodes a non-unit payload length followed by the remaining payload bytes. Extensions occur in position, span, payload order. MOVE then appends its original-workspace destination as unsigned LEB128; e.g. moving `[1,3)` to destination 8 is `[1,67,0,2,8]`. Trailing, noncanonical, malformed, oversized or impossible fields are rejected. Coordinates are relative to the loaded workspace.
- Multi-byte payloads and arbitrary spans are codec/host capabilities; autonomous insert/replace currently predict **one** payload byte per edit. The optional structured model rejects untrained operation kinds and non-unit replace spans by default. Neither Fourier positional features nor a valid wire format guarantee accuracy at long positions.

The width-64 generator now has {structured['parameters']['total']:,} parameters, including {structured['parameters']['new_structured_adapter_parameters']:,} new structured-request parameters; {structured['parameters']['trainable_replacement_parameters']:,} were trained in this experiment. Diffusion now has {diffusion['parameters']['total']:,} parameters, including {diffusion['parameters']['new_trainable_parameters']:,} new mask-conditioning parameters. No pretrained base checkpoint was overwritten. Entirely absent old heads migrate to zero and are marked untrained; partial current heads/metadata fail strict loading. Older software cannot strictly load new states, and optimizers should be rebuilt for changed parameter groups. Request/mask trained flags are caller-maintained metadata, not accuracy or provenance guarantees. Confidence calibration does not carry over to masked edits; masked adaptive halting is refused.

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

Original detailed evidence: `runs/phase9/cognitive_metrics.json`, `structured_metrics.json`, `diffusion_metrics.json`, `single_byte_edit_diagnostic.json` and `latency.json`. Verification: {verification.get('unit_tests', 'see test output and runs/phase9/verification.json')}.
'''
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
