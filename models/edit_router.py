"""Explicit dispatch between fixed-length denoising and cognitive editing.

The caller supplies the operation, not a target. Dispatch is not a classifier,
not evidence of a successful edit, and not an instruction-following guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch

from .bytecore_diffusion import ByteCoreDiffusion
from .cognitive import CognitiveResult
from .generator_conditioned import GeneratorConditioned
from .procedural import raw_bytes

EditOperation = Literal["denoise", "substitute", "replace", "insert", "delete", "move"]


@dataclass(frozen=True)
class EditRequest:
    source: bytes
    operation: EditOperation
    # Required for replace: a replacement can change length.
    length_preserving: bool | None = None
    instruction: bytes = b""
    # Optional fixed-length, explicit native diffusion request. Non-masked
    # replacement bytes are ignored; the mask consists of literal 0/1 bytes.
    edit_mask: bytes | None = None
    replacement_bytes: bytes | None = None


@dataclass(frozen=True)
class EditOutcome:
    engine: Literal["diffusion", "cognitive"]
    output: bytes
    completed: bool
    computation_steps: int
    reason: str
    cognitive_result: CognitiveResult | None = None


def route_edit(request: EditRequest) -> Literal["diffusion", "cognitive"]:
    """Choose an engine from an explicit, caller-provided operation.

    A same-length substitution without an explicit mask means denoising.
    Native targeted diffusion requires a trained mask editor and explicit
    replacement bytes, not an unparsed natural-language instruction. Ambiguous
    replacement requests fail instead of assuming a length.
    """
    raw_bytes(request.source, "source")
    raw_bytes(request.instruction, "instruction")
    if request.operation in ("denoise", "substitute"):
        if request.length_preserving is False:
            raise ValueError("substitution/denoising must be length-preserving")
        return "diffusion"
    if request.operation == "replace":
        if not isinstance(request.length_preserving, bool):
            raise ValueError("replace requires an explicit length_preserving boolean")
        return "diffusion" if request.length_preserving else "cognitive"
    if request.operation in ("insert", "delete", "move"):
        return "cognitive"
    raise ValueError(f"unsupported edit operation: {request.operation!r}")


class EditRouter:
    """Route inference without taking, storing, or consulting the answer."""

    def __init__(self, diffusion: ByteCoreDiffusion, cognitive: GeneratorConditioned, *,
                 diffusion_pad_to: int | None):
        # Padding influences attention because the legacy diffusion core has no
        # padding mask. Callers must explicitly supply the width used in training;
        # None is appropriate only for a model trained on variable-width inputs.
        if diffusion_pad_to is not None and (not isinstance(diffusion_pad_to, int)
                or isinstance(diffusion_pad_to, bool) or not 1 <= diffusion_pad_to < diffusion.core.max_len):
            raise ValueError("diffusion_pad_to must fit the diffusion positional limit")
        self.diffusion = diffusion
        self.cognitive = cognitive
        self.diffusion_pad_to = diffusion_pad_to

    @torch.no_grad()
    def execute(self, request: EditRequest, *, memory_model: Any | None = None,
                max_steps: int = 64, denoise_timestep: int = 2,
                confidence_threshold: float | None = None,
                autonomous_think_steps: int = 1,
                adaptive_config: Mapping[str, Any] | None = None) -> EditOutcome:
        engine = route_edit(request)
        source = raw_bytes(request.source, "source")
        instruction = raw_bytes(request.instruction, "instruction")
        if engine != "diffusion" and (request.edit_mask is not None or request.replacement_bytes is not None):
            raise ValueError("mask-conditioned diffusion is fixed-length; structural edits use cognitive arguments")
        if engine == "diffusion":
            # Free-form instructions are not parsed by the diffusion network.
            # Do not silently ignore them even if structured values were given.
            if instruction:
                raise NotImplementedError("diffusion cannot parse free-form instruction-directed substitution; pass an explicit mask and replacement bytes")
            if request.replacement_bytes is not None and request.edit_mask is None:
                raise ValueError("replacement_bytes requires an edit_mask")
            if request.edit_mask is not None:
                raw_mask = raw_bytes(request.edit_mask, "edit_mask")
                if len(raw_mask) != len(source) or any(value not in (0, 1) for value in raw_mask):
                    raise ValueError("edit_mask must contain one binary byte per source byte")
                replacement = None if request.replacement_bytes is None else raw_bytes(request.replacement_bytes, "replacement_bytes")
                if replacement is not None and len(replacement) != len(source):
                    raise ValueError("replacement_bytes must have the same length as source")
                if confidence_threshold is not None or (adaptive_config is not None and adaptive_config.get("enabled", False)):
                    raise ValueError("adaptive confidence thresholds are not calibrated for masked edits")
                if not source:
                    return EditOutcome(engine, b"", True, 0, "empty_input")
                model = self.diffusion
                width = self.diffusion_pad_to if self.diffusion_pad_to is not None else len(source)
                if len(source) > width or width + 1 > model.core.max_len:
                    raise ValueError("source exceeds the configured diffusion input width")
                model.eval()
                ids = torch.zeros((1, width), dtype=torch.long, device=model.betas.device)
                ids[0, :len(source)] = torch.tensor(list(source), dtype=torch.long, device=model.betas.device)
                mask = torch.zeros_like(ids, dtype=torch.bool)
                mask[0, :len(source)] = torch.tensor(list(raw_mask), dtype=torch.bool, device=ids.device)
                requested = None
                if replacement is not None:
                    requested = torch.zeros_like(ids)
                    requested[0, :len(source)] = torch.tensor(list(replacement), dtype=torch.long, device=ids.device)
                output_ids, stats = model.masked_edit(ids, mask, requested, return_stats=True)
                output = bytes(output_ids[0, :len(source)].cpu().tolist())
                return EditOutcome(engine, output, True, int(stats.steps[0]), "mask_conditioned_prediction")
            if confidence_threshold is not None or (adaptive_config is not None and adaptive_config.get("enabled", False)):
                raise NotImplementedError("single-pass denoising has no adaptive sampling loop")
            if not source:
                return EditOutcome(engine, b"", True, 0, "empty_input")
            model = self.diffusion
            width = self.diffusion_pad_to if self.diffusion_pad_to is not None else len(source)
            if len(source) > width or width + 1 > model.core.max_len:
                raise ValueError("source exceeds the configured diffusion input width")
            if not isinstance(denoise_timestep, int) or isinstance(denoise_timestep, bool) or not 0 <= denoise_timestep < model.diffusion_timesteps:
                raise ValueError("denoise_timestep must be within the diffusion schedule")
            model.eval()
            ids = torch.zeros((1, width), dtype=torch.long, device=model.betas.device)
            ids[0, :len(source)] = torch.tensor(list(source), dtype=torch.long, device=model.betas.device)
            emb = model.bytes_to_emb(ids)
            t = torch.tensor([denoise_timestep], dtype=torch.long, device=ids.device)
            noisy = model.q_sample(emb, t)
            predicted = model.core(noisy, t.float() / model.diffusion_timesteps, None)
            output = bytes(model.core.embed_to_bytes(predicted)[0, :len(source)].cpu().tolist())
            return EditOutcome(engine, output, True, 1, "fixed_length_denoising")
        result = self.cognitive.cognitive_generate(
            source, memory_model=memory_model, max_steps=max_steps,
            confidence_threshold=confidence_threshold,
            autonomous_think_steps=autonomous_think_steps, adaptive_config=adaptive_config,
            instruction_bytes=instruction if instruction else None,
        )
        # DONE only means the control loop terminated, not that its edit is correct.
        return EditOutcome(engine, result.canvas, result.done, len(result.events),
                           result.halt_reason, result)
