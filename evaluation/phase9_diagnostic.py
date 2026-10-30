"""Rerun the one-byte diagnostic with the actual Phase 9 APIs.

Targets are used for scoring only, except for the separately labelled explicit
memory-write demonstration. The requested byte is an instruction and is passed
only at the masked position; it is never copied into the masked output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from evaluation.diagnostic_single_byte_edit import INSTRUCTION, POSITION, SOURCE, TARGET, memory_test
from models.cognitive import action_to_record
from training.train_confidence import load_generator
from training.train_masked_diffusion import load_model


def result_data(result) -> dict[str, Any]:
    return {"output_bytes": list(result.canvas), "output_text": result.canvas.decode("utf-8", errors="backslashreplace"),
            "done": result.done, "halt_reason": result.halt_reason, "steps": len(result.events),
            "exact": bool(result.done and result.canvas == TARGET),
            "actions": [dict(action_to_record(event.action),
                             argument_bytes=None if event.argument_bytes is None else list(event.argument_bytes),
                             control_and_arguments=list(event.control_and_arguments))
                        for event in result.events]}


@torch.no_grad()
def diffusion_diagnostic(model, *, seed: int, checkpoint: str) -> dict[str, Any]:
    model.eval()
    ids = torch.tensor([list(SOURCE)], dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool); mask[0, POSITION] = True
    original = model.bytes_to_emb(ids)
    requested = torch.zeros_like(ids); requested[0, POSITION] = 100
    # Sampling is performed by the model API; no expected output is passed.
    torch.manual_seed(seed)
    latent, stats = model.p_sample_loop_with_mask(tuple(original.shape), mask, original_emb=original,
                                                  requested_byte_ids=requested, return_stats=True)
    predicted = model.core.embed_to_bytes(latent)
    torch.manual_seed(seed)
    wrapped = model.masked_edit(ids, mask, requested)
    if not torch.equal(predicted[mask], wrapped[mask]):
        raise AssertionError("wrapper changed the model's masked prediction")
    if not torch.equal(latent[~mask], original[~mask]):
        raise AssertionError("unmasked embeddings were not preserved")
    output = bytes(wrapped[0].tolist())
    counterfactual = torch.zeros_like(ids); counterfactual[0, POSITION] = 101
    torch.manual_seed(seed)
    other = model.masked_edit(ids, mask, counterfactual)
    torch.manual_seed(seed)
    no_request = model.masked_edit(ids, mask)
    torch.manual_seed(seed + 17)
    batched_ids = ids.expand(32, -1).clone()
    batched_mask = mask.expand(32, -1).clone()
    batched_request = requested.expand(32, -1).clone()
    samples = model.masked_edit(batched_ids, batched_mask, batched_request)
    successes = int((samples[:, POSITION] == 100).sum().item())
    return {"checkpoint": checkpoint, "seed": seed,
            "mask": mask[0].int().tolist(), "requested_bytes_at_mask": [100],
            "predicted_masked_byte": int(predicted[0, POSITION]),
            "output_bytes": list(output), "output_text": output.decode("utf-8", errors="backslashreplace"),
            "exact": output == TARGET, "sampling_steps": int(stats.steps[0]),
            "unmasked_embedding_preserved": bool(torch.equal(latent[~mask], original[~mask])),
            "unmasked_bytes_preserved": all(output[i] == SOURCE[i] for i in range(len(SOURCE)) if i != POSITION),
            "model_prediction_not_request_overwrite": bool(torch.equal(predicted[mask], wrapped[mask])),
            "counterfactual_request_101": {"requested_byte": 101, "predicted_byte": int(other[0, POSITION]),
                                          "output_bytes": other[0].tolist()},
            "mask_only_no_requested_byte": {"predicted_byte": int(no_request[0, POSITION]),
                                            "output_bytes": no_request[0].tolist(),
                                            "note": "A mask alone does not specify which replacement is desired."},
            "repeated_samples": {"seed": seed + 17, "samples": 32, "exact_requested_byte": successes,
                                 "unmasked_preserved": bool(torch.equal(samples[~batched_mask], batched_ids[~batched_mask]))}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--diffusion-dir", default="runs/phase9/diffusion")
    parser.add_argument("--generator", default="runs/phase9/generator.pt")
    parser.add_argument("--structured-generator", default="runs/phase9/structured_generator.pt")
    parser.add_argument("--memory", default="runs/real_traces/memory.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--output", default="runs/phase9/single_byte_edit_diagnostic.json")
    args = parser.parse_args()
    if not 1 <= args.budget <= 4096:
        raise ValueError("budget must be in [1, 4096]")
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    diffusion = load_model(args.config, args.diffusion_dir)
    cognitive = load_generator(args.generator, "cpu", 4).eval()
    structured = load_generator(args.structured_generator, "cpu", 4).eval()
    # No controller, memory, target canvas, action, oracle, or target-derived budget.
    natural = cognitive.cognitive_generate(SOURCE, instruction_bytes=INSTRUCTION, max_steps=args.budget)
    # This second variant is explicitly given the user-requested edit fields.
    structured_result = structured.cognitive_generate(SOURCE, instruction_bytes=INSTRUCTION,
                                                       structured_request=bytes([9, 2, 100]), max_steps=args.budget)
    baseline_path = Path("runs/single_byte_edit_diagnostic.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    results = {"source_bytes": list(SOURCE), "source_text": SOURCE.decode("utf-8"),
               "target_bytes_for_scoring": list(TARGET), "target_text": TARGET.decode("utf-8"),
               "instruction": INSTRUCTION.decode("ascii"), "requested_position_zero_based": POSITION,
               "requested_old_byte": SOURCE[POSITION], "requested_new_byte": 100,
               "diffusion_native": diffusion_diagnostic(diffusion, seed=args.seed,
                                                        checkpoint=str(Path(args.diffusion_dir) / "model.pt")),
               "cognitive_natural_only": {"checkpoint": args.generator, "in_training_data": True,
                                           "memory": False, "controller": False, **result_data(natural)},
               "cognitive_explicit_request": {"checkpoint": args.structured_generator, "inherited_training_example": True,
                                              "request_argument_bytes": [9, 2, 100], "memory": False,
                                              "controller": False, **result_data(structured_result)},
               "memory": memory_test(args.memory),
               "pre_phase9_reference": None if baseline is None else {
                   "source": str(baseline_path),
                   "native_mask_supported_then": baseline["diffusion"]["phase7_mask_conditioned_feature_available"],
                   "host_inpainting_byte_then": baseline["diffusion"]["host_mask_fallback"]["predicted_byte_at_position_9"],
                   "cognitive_output_then": baseline["cognitive_selected"]["output_text"]}}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
