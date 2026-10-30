"""Run the requested ten-byte edit through the existing models and memory.

The held target is used only for scoring, except for the explicitly labelled
memory-write demonstration. A host-side masked diffusion experiment is a
fallback, not a mask-conditioned or instruction-trained model. Run with:

    OMP_NUM_THREADS=1 .venv/bin/python -m evaluation.diagnostic_single_byte_edit
"""
from __future__ import annotations

import argparse
import inspect
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml

from models.bytecore_diffusion import ByteCoreDiffusion
from models.cognitive import action_to_record
from models.edit_router import EditRequest, EditRouter
from models.memory_model import MemoryModel
from training.train_confidence import load_generator


SOURCE = bytes([72, 101, 108, 108, 111, 32, 87, 111, 114, 108])
TARGET = bytes([72, 101, 108, 108, 111, 32, 87, 111, 114, 100])
INSTRUCTION = b"Replace the byte at zero-based position 9 from 108 ('l') to 100 ('d')."
POSITION = 9


def load_diffusion(config_path: str, runs_dir: str) -> tuple[ByteCoreDiffusion, int]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    settings = config["model"]
    model = ByteCoreDiffusion(
        vocab_size=settings["vocab_size"], d_model=settings["d_model"], n_layers=settings["n_layers"],
        n_heads=settings["n_heads"], d_ff=settings["d_ff"], max_len=settings["max_len"],
        diffusion_timesteps=settings["diffusion_timesteps"], concept_capacity=settings["concept_capacity"],
        concept_dim=settings["concept_dim"], dropout=settings["dropout"], device="cpu",
    ).load(runs_dir).eval()
    return model, int(config["data"]["max_seq_len"])


@torch.no_grad()
def externally_masked_diffusion(model: ByteCoreDiffusion, source: bytes, instruction: bytes, *,
                                position: int, width: int, seed: int) -> bytes:
    """Try host-side DDPM inpainting without passing the expected answer.

    The existing core has no edit-mask input and no trained edit-instruction
    encoder. The host clamps the *other* positions to noised source embeddings
    at each step and averages raw instruction-byte embeddings as a concept.
    This is an experimental approximation, not the missing Phase 7 feature.
    """
    if not 0 <= position < len(source) <= width or width + 1 > model.core.max_len:
        raise ValueError("source, mask position or padded width is invalid")
    torch.manual_seed(seed)
    ids = torch.zeros((1, width), dtype=torch.long, device=model.betas.device)
    ids[0, :len(source)] = torch.tensor(list(source), dtype=torch.long, device=model.betas.device)
    clean = model.bytes_to_emb(ids)
    mask = torch.zeros((1, width, 1), dtype=torch.bool, device=ids.device)
    mask[0, position, 0] = True
    known_noise = torch.randn_like(clean)
    xt = torch.randn_like(clean)
    concept_ids = torch.tensor(list(instruction), dtype=torch.long, device=ids.device)
    concept = model.encode_concept(concept_ids)
    for t in range(model.diffusion_timesteps - 1, -1, -1):
        ab = model.alpha_bars[t]
        known_at_t = ab.sqrt() * clean + (1 - ab).sqrt() * known_noise
        xt = torch.where(mask, xt, known_at_t)
        time = torch.tensor([t], dtype=torch.long, device=ids.device)
        mean, variance, _ = model.p_mean_variance(xt, time, concept, guidance_scale=1.0)
        xt = mean + variance.sqrt() * torch.randn_like(mean) if t else mean
    predicted = int(model.core.embed_to_bytes(xt)[0, position].item())
    # Host masking preserves other bytes; this is not a model prediction of them.
    return source[:position] + bytes([predicted]) + source[position + 1:]


def diffusion_test(model: ByteCoreDiffusion, generator: Any, width: int, seed: int) -> dict[str, Any]:
    names = set(inspect.signature(model.generate).parameters)
    mask_supported = bool(names & {"mask", "edit_mask", "inpaint_mask"})
    router = EditRouter(model, generator, diffusion_pad_to=width)
    try:
        official = router.execute(EditRequest(SOURCE, "replace", length_preserving=True, instruction=INSTRUCTION))
    except NotImplementedError as exc:
        official_result: dict[str, Any] = {"status": "unsupported", "error": str(exc), "output_bytes": None}
    else:
        official_result = {"status": "completed", "error": None, "output_bytes": list(official.output),
                           "exact": official.completed and official.output == TARGET}
    fallback = externally_masked_diffusion(model, SOURCE, INSTRUCTION, position=POSITION, width=width, seed=seed)
    return {
        "native_generate_accepts_edit_mask": mask_supported,
        "official_instruction_directed_replace": official_result,
        "phase7_mask_conditioned_feature_available": mask_supported,
        "host_mask_fallback": {"seed": seed, "output_bytes": list(fallback), "output_text": fallback.decode("utf-8", errors="backslashreplace"),
                               "predicted_byte_at_position_9": fallback[POSITION], "exact": fallback == TARGET,
                               "limitation": "Host clamps all other positions; the mask is not a model input. "
                                             "The instruction is averaged into an untrained concept vector."},
    }


def cognitive_test(checkpoint: str, budget: int) -> dict[str, Any]:
    model = load_generator(checkpoint, "cpu", 4).eval()
    # The target is deliberately not passed to the model; the raw instruction is.
    result = model.cognitive_generate(SOURCE, instruction_bytes=INSTRUCTION, max_steps=budget, temperature=0.0)
    return {"checkpoint": checkpoint, "output_bytes": list(result.canvas),
            "output_text": result.canvas.decode("utf-8", errors="backslashreplace"),
            "done": result.done, "halt_reason": result.halt_reason, "steps": len(result.events),
            "exact": result.done and result.canvas == TARGET,
            "actions": [action_to_record(event.action) for event in result.events]}


def memory_test(path: str) -> dict[str, Any]:
    # Load a copy, never mutating the original checkpoint. Its guard is retained.
    memory = MemoryModel(d_model=64, capacity=128).load(path)
    with memory.read_only():
        try:
            old = memory.execute_procedure(SOURCE)
        except LookupError:
            previous: dict[str, Any] = {"found": False, "exact": False, "output_bytes": None}
        else:
            previous = {"found": True, "exact": old.output == TARGET, "output_bytes": list(old.output)}
    # The user supplied the answer. This deliberately stores the exact mapping
    # in an isolated copy and checks persistence + exact-match lookup, not inference.
    memory.write_procedure(SOURCE, TARGET, torch.zeros(1, memory.d_model), INSTRUCTION,
                           metadata={"type": "diagnostic", "split": "training"})
    with tempfile.TemporaryDirectory(prefix="bytecore-memory-diagnostic-") as directory:
        path_copy = str(Path(directory) / "memory.pt")
        memory.save(path_copy)
        restored = MemoryModel(d_model=64, capacity=128).load(path_copy)
        with restored.read_only():
            execution = restored.execute_procedure(SOURCE)
        output = execution.output
        audit = restored.audit_holdouts()
    return {"checkpoint": path, "existing_memory": previous,
            "after_explicit_write_and_reload": {"found": True, "output_bytes": list(output),
                                                 "output_text": output.decode("utf-8", errors="backslashreplace"),
                                                 "exact": output == TARGET, "guard_failed_slots": audit["failed_slots"],
                                                 "limitation": "Stored the exact answer first; this is lookup, not model-generated editing."}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--selected-generator", default="runs/phase8/generator.pt")
    parser.add_argument("--final-generator", default="runs/phase8/last_epoch.pt")
    parser.add_argument("--memory", default="runs/real_traces/memory.pt")
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/single_byte_edit_diagnostic.json")
    args = parser.parse_args()
    if not 1 <= args.budget <= 4096:
        raise ValueError("budget must be in [1, 4096]")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    diffusion, width = load_diffusion(args.config, args.runs_dir)
    selected = load_generator(args.selected_generator, "cpu", 4).eval()
    result = {
        "source_bytes": list(SOURCE), "source_text": SOURCE.decode("utf-8"),
        "requested_position_zero_based": POSITION, "requested_old_byte": SOURCE[POSITION],
        "requested_new_byte": TARGET[POSITION], "target_bytes": list(TARGET), "target_text": TARGET.decode("utf-8"),
        "instruction": INSTRUCTION.decode("utf-8"),
        "diffusion": diffusion_test(diffusion, selected, width, args.seed),
        "cognitive_selected": cognitive_test(args.selected_generator, args.budget),
        "cognitive_final_epoch": cognitive_test(args.final_generator, args.budget),
        "memory": memory_test(args.memory),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
