"""Phase 26 model snoop: load a real ByteCore checkpoint and collect inference forensics.

The evaluator is deliberately explicit about what it can and cannot prove:
activation norms show execution, not learned capability; an attention mass is a
routing measurement, not proof of semantic retrieval. Hub access is only used by
``download_checkpoint``; tests and ``--checkpoint`` mode stay fully offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.encoders import TextSkillProjector
from models.foundation import (
    CONTROL_NAMES, EOS, CausalByteTransformer, FoundationConfig,
    apply_rotary, decode_ids, encode_bytes, repeat_kv, rotary_tables,
)
from models.memory_model import SkillBankView

DEFAULT_REPO = "gabrielbosse9/airesearch2"
EXPECTED_PRODUCTION_PARAMS = 366_316_833


def _hub_api() -> Any:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt (huggingface_hub) to fetch from the Hub") from exc
    return HfApi(), hf_hub_download


def select_checkpoint_file(files: list[str]) -> str:
    """Choose latest, then final, else the highest explicit checkpoint step."""
    if "checkpoints/checkpoint_latest.pt" in files:
        return "checkpoints/checkpoint_latest.pt"
    if "checkpoints/checkpoint_final.pt" in files:
        return "checkpoints/checkpoint_final.pt"
    candidates = [name for name in files if name.endswith(".pt") and "checkpoint" in name.lower()]
    if not candidates:
        raise FileNotFoundError("No checkpoint_latest.pt, checkpoint_final.pt, or stepped checkpoint found")

    def step(name: str) -> int:
        matches = re.findall(r"(?:step|checkpoint)[_-]?(\d+)", Path(name).stem.lower())
        return int(matches[-1]) if matches else -1

    return max(candidates, key=lambda name: (step(name), name))


def download_checkpoint(repo_id: str = DEFAULT_REPO, revision: str = "main",
                        token: str | None = None, cache_dir: str | None = None) -> tuple[Path, str]:
    """Resolve the latest published checkpoint file and download it via HF Hub."""
    api, hf_hub_download = _hub_api()
    files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision, token=token)
    filename = select_checkpoint_file(list(files))
    local = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model",
                            revision=revision, token=token, cache_dir=cache_dir)
    return Path(local), filename


def load_payload(path: str | Path) -> dict[str, Any]:
    """Load a standard tensor/dict training checkpoint with mmap and safe unpickling."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # older supported PyTorch lacks mmap keyword
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint root must be a mapping")
    if payload.get("schema_version") != 7:
        raise ValueError(f"Expected Phase 25.1 trainer checkpoint schema 7; got {payload.get('schema_version')!r}")
    for required in ("config", "model", "step"):
        if required not in payload:
            raise ValueError(f"Incomplete trainer checkpoint: missing {required!r}")
    return payload


def build_model(payload: Mapping[str, Any], device: torch.device) -> CausalByteTransformer:
    """Instantiate on meta and assign checkpoint tensors to avoid a second CPU copy."""
    config = FoundationConfig.from_mapping(payload["config"])
    with torch.device("meta"):
        model = CausalByteTransformer(config)
    try:
        model.load_state_dict(payload["model"], strict=True, assign=True)
    except TypeError as exc:
        raise RuntimeError("This evaluator needs PyTorch with load_state_dict(assign=True)") from exc
    model.to(device)
    model.eval()
    return model


def architecture_report(model: CausalByteTransformer) -> dict[str, Any]:
    cfg = model.config
    expected = {
        "d_model": 1024, "n_layers": 8, "n_heads": 8, "n_kv_heads": 2,
        "d_ff": 4096, "max_seq_len": 16384,
        "attention_type": "sliding_window", "window_size": 1024,
        "memory_dim": 8192, "memory_cross_attention": True,
        "refinement_steps": 4, "max_thought_steps": 4,
    }
    checks = {name: {"expected": value, "actual": getattr(cfg, name),
                     "match": getattr(cfg, name) == value}
              for name, value in expected.items()}
    layers = []
    for i, block in enumerate(model.blocks):
        layers.append({
            "layer": i,
            "attention": block.attention.attention_type,
            "window_size": block.attention.window_size,
            "query_weight": list(block.attention.query.weight.shape),
            "key_weight": list(block.attention.key.weight.shape),
            "ffn_value_weight": list(block.feed_forward.w_value.weight.shape)
            if hasattr(block.feed_forward, "w_value") else None,
            "skill_key_width": block.skill_adapter.d_model if block.has_skills else 0,
            "skill_value_width": block.skill_adapter.memory_dim if block.has_skills else 0,
            "skill_glu_weight": list(block.skill_adapter.glu.weight.shape) if block.has_skills else None,
            "cross_q_weight": list(block.skill_adapter.cross_q.weight.shape)
            if block.has_skills and block.skill_adapter.cross_attention_enabled else None,
            "cross_k_weight": list(block.skill_adapter.cross_k.weight.shape)
            if block.has_skills and block.skill_adapter.cross_attention_enabled else None,
            "cross_v_weight": list(block.skill_adapter.cross_v.weight.shape)
            if block.has_skills and block.skill_adapter.cross_attention_enabled else None,
            "cross_o_weight": list(block.skill_adapter.cross_o.weight.shape)
            if block.has_skills and block.skill_adapter.cross_attention_enabled else None,
            "memory_cross_attention": bool(block.has_skills and block.skill_adapter.cross_attention_enabled),
        })
    return {
        "parameters": model.parameters_count,
        "config": cfg.to_dict(),
        "global_shapes": {
            "token_embedding": list(model.token_embedding.weight.shape),
            "language_model_head": list(model.head.weight.shape),
            "refinement_steps": int(model.refinement.steps) if model.refinement is not None else 0,
            "thought_slots": (list(model.refinement.thoughts.slot_embeddings.shape)
                              if model.refinement is not None and model.refinement.thoughts is not None else None),
            "thought_buffer_capacity": (int(model.refinement.thoughts.capacity)
                                        if model.refinement is not None and model.refinement.thoughts is not None else 0),
            "confidence_head": (list(model.refinement.confidence_head.weight.shape)
                                if model.refinement is not None else None),
            "memory_key_value_widths": [cfg.d_model, cfg.memory_dim],
        },
        "production_parameter_count_match": model.parameters_count == EXPECTED_PRODUCTION_PARAMS,
        "expected_config_checks": checks,
        "production_config_match": all(item["match"] for item in checks.values()),
        "layer_count_match": len(model.blocks) == 8 if cfg.d_model == 1024 else None,
        "all_attention_sliding_window": all(b.attention.attention_type == "sliding_window" for b in model.blocks),
        "all_skill_adapters_present": all(b.has_skills for b in model.blocks),
        "layers": layers,
    }


class ProbeRecorder:
    """Collect only scalar/device-side summaries; no full activation retention."""

    def __init__(self, model: CausalByteTransformer) -> None:
        self.model = model
        self.handles: list[Any] = []
        self.scalars: dict[str, list[torch.Tensor]] = {}
        self.attention_inputs: dict[int, torch.Tensor] = {}
        self.cross_inputs: dict[int, torch.Tensor] = {}

    def _scalar(self, name: str, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return
        value = tensor.detach().float().norm(dim=-1).mean()
        self.scalars.setdefault(name, []).append(value)

    def __enter__(self) -> "ProbeRecorder":
        for index, block in enumerate(self.model.blocks):
            self.handles.append(block.attention.register_forward_pre_hook(
                lambda module, args, i=index: self.attention_inputs.__setitem__(i, args[0].detach())))
            self.handles.append(block.feed_forward.register_forward_hook(
                lambda module, args, output, i=index: self._scalar(f"ffn_layer_{i}_output_l2", output)))
            if hasattr(block.feed_forward, "w_out"):
                self.handles.append(block.feed_forward.w_out.register_forward_pre_hook(
                    lambda module, args, i=index: self._scalar(f"swiglu_layer_{i}_activation_l2", args[0])))
            if block.has_skills:
                self.handles.append(block.skill_adapter.cross_q.register_forward_pre_hook(
                    lambda module, args, i=index: self.cross_inputs.__setitem__(i, args[0].detach())))
                self.handles.append(block.skill_adapter.glu.register_forward_hook(
                    lambda module, args, output, i=index: self._scalar(f"glu_layer_{i}_projected_l2", output)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self, starts: Mapping[str, int] | None = None) -> dict[str, Any]:
        starts = starts or {}
        result = {}
        for name, values in self.scalars.items():
            selected = values[int(starts.get(name, 0)):]
            if selected:
                stacked = torch.stack(selected)
                result[name] = {
                    "calls": len(selected),
                    "mean_l2": float(stacked.mean().cpu()),
                    "max_l2": float(stacked.max().cpu()),
                }
        return result

    @torch.no_grad()
    def self_attention_last(self) -> dict[str, Any]:
        """Recompute the final-row sliding-window weights for each layer."""
        result: dict[str, Any] = {}
        for index, hidden in self.attention_inputs.items():
            attn = self.model.blocks[index].attention
            batch, length, _ = hidden.shape
            q = attn.query(hidden).view(batch, length, attn.n_heads, attn.head_dim).transpose(1, 2)
            k = attn.key(hidden).view(batch, length, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
            if attn.position_type == "rope":
                cos, sin = rotary_tables(length, attn.head_dim, attn.rope_theta, hidden.device, q.dtype)
                q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
            k = repeat_kv(k, attn.repeats)
            start = max(0, length - attn.window_size - 1)
            keys = k[:, :, start:length]
            scores = torch.matmul(q[:, :, -1:, :].float(), keys.transpose(-2, -1).float())
            scores = scores / math.sqrt(attn.head_dim)
            weights = torch.softmax(scores, dim=-1).mean(dim=1)[0, 0]
            values, indices = torch.topk(weights, min(8, weights.numel()))
            result[str(index)] = {
                "key_positions_in_window": indices.cpu().tolist(),
                "weights": values.cpu().tolist(),
                "attended_key_count": int(weights.numel()),
                "window_size": attn.window_size,
            }
        return result

    @torch.no_grad()
    def cross_attention_last(self, memory: SkillBankView | None,
                             target_slot: int | None = None) -> dict[str, Any]:
        """Exact softmax QK diagnostic for the final token, averaged over heads."""
        if memory is None or memory.keys.shape[-2] == 0:
            return {str(i): {"available": False, "reason": "no memory slots"}
                    for i in self.cross_inputs}
        result: dict[str, Any] = {}
        for index, hidden in self.cross_inputs.items():
            adapter = self.model.blocks[index].skill_adapter
            q = adapter.cross_q_norm(adapter.cross_q(hidden[:, -1:, :]))
            keys = adapter.cross_k_norm(adapter.cross_k(memory.keys))
            if keys.ndim == 2:
                keys = keys.unsqueeze(0).expand(q.shape[0], -1, -1)
            q = q.view(q.shape[0], 1, adapter.n_heads, adapter.cross_head_dim).transpose(1, 2)
            keys = keys.view(keys.shape[0], keys.shape[1], adapter.n_heads,
                             adapter.cross_head_dim).transpose(1, 2)
            scores = torch.matmul(q.float(), keys.float().transpose(-2, -1))
            scores = scores / math.sqrt(adapter.cross_head_dim)
            mask = memory.mask
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(q.shape[0], -1)
            scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1).mean(dim=1)[0, 0]
            entropy = -(weights.clamp_min(1e-30) * weights.clamp_min(1e-30).log()).sum()
            record = {
                "available": True,
                "slots": int(weights.numel()),
                "entropy_nats": float(entropy.cpu()),
                "max_weight": float(weights.max().cpu()),
                "top_slot": int(weights.argmax().cpu()),
            }
            if target_slot is not None:
                record["injected_fact_weight"] = float(weights[target_slot].cpu())
                record["injected_fact_rank"] = int((weights > weights[target_slot]).sum().cpu()) + 1
            result[str(index)] = record
        return result


def _decode_text(ids: list[int] | torch.Tensor) -> str:
    return decode_ids(ids, keep_controls=True).decode("utf-8", "replace")


@torch.no_grad()
def generate_trace(model: CausalByteTransformer, prompt: str, recorder: ProbeRecorder,
                   *, memory: SkillBankView | None, max_new_bytes: int,
                   temperature: float = 0.0, target_slot: int | None = None) -> dict[str, Any]:
    """Greedy byte decoding with forward-pass thought/attention/activation telemetry."""
    if max_new_bytes < 1:
        raise ValueError("max_new_bytes must be positive")
    prefix = encode_bytes(prompt.encode("utf-8"), bos=True).to(next(model.parameters()).device)
    ids = prefix.clone()
    generated: list[int] = []
    thought_trace: list[dict[str, Any]] = []
    refinement = model.refinement
    thoughts = refinement.thoughts if refinement is not None else None
    buffer_gate_value = (float(thoughts.buffer_gate.detach().float().cpu())
                         if thoughts is not None else None)
    context_gate_value = (float(thoughts.context_gate.detach().float().cpu())
                          if thoughts is not None else None)
    scalar_starts = {name: len(values) for name, values in recorder.scalars.items()}
    final_attention: dict[str, Any] = {}
    final_memory: dict[str, Any] = {}
    for _ in range(max_new_bytes):
        window = ids[-model.config.max_seq_len:]
        logits, info = model(window.unsqueeze(0), memory=memory, return_info=True)
        next_logits = logits[0, -1].float()
        if temperature <= 0:
            next_id = int(next_logits.argmax().item())
        else:
            next_id = int(torch.multinomial(torch.softmax(next_logits / temperature, dim=-1), 1).item())
        thought_trace.append({
            "steps": int(info.get("thought_steps", info.get("steps_run", 0)) or 0),
            "thought_buffer_l2": float(info.get("thought_norm") or 0.0),
            "buffer_gate": buffer_gate_value,
            "context_gate": context_gate_value,
            "confidence": info.get("confidence"),
        })
        if next_id == EOS:
            break
        generated.append(next_id)
        ids = torch.cat((ids, ids.new_tensor([next_id])))
    # Reconstruct attention maps only once per prompt, avoiding per-byte CPU/GPU sync.
    final_attention = recorder.self_attention_last()
    final_memory = recorder.cross_attention_last(memory, target_slot=target_slot)
    return {
        "prompt": prompt,
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "input_context_bytes": int(min(prefix.numel(), model.config.max_seq_len)),
        "continuation": _decode_text(generated),
        "generated_bytes": len(generated),
        "generated_control_ids": [value for value in generated if value > 255],
        "generated_control_tokens": [CONTROL_NAMES.get(value, f"<control:{value}>")
                                     for value in generated if value > 255],
        "adaptive": thought_trace[-1] if thought_trace else {},
        "thought_trace": thought_trace,
        "thought_buffer_l2_mean": (sum(x["thought_buffer_l2"] for x in thought_trace) /
                                    len(thought_trace)) if thought_trace else 0.0,
        "attention_last_generated_position": final_attention,
        "memory_attention_last_generated_position": final_memory,
        "activation_summary": recorder.summary(scalar_starts),
    }


def _memory_view(payload: Mapping[str, Any], device: torch.device,
                 max_slots: int) -> SkillBankView | None:
    memory = payload.get("memory")
    bank = memory.get("bank") if isinstance(memory, dict) else None
    if not isinstance(bank, dict) or not isinstance(bank.get("buffers"), dict):
        return None
    buffers = bank["buffers"]
    used = buffers.get("used")
    if not isinstance(used, torch.Tensor):
        return None
    indices = used.nonzero(as_tuple=False).flatten()[:max_slots]
    if indices.numel() == 0:
        return None
    keys = buffers["keys"].index_select(0, indices).to(device)
    values = buffers["values"].index_select(0, indices).to(device)
    mask = torch.ones(indices.numel(), dtype=torch.bool, device=device)
    return SkillBankView(keys, values, mask)


def _encode_fact(model: CausalByteTransformer, projector: TextSkillProjector,
                 text: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = encode_bytes(text.encode("utf-8"), bos=True).to(device).unsqueeze(0)
    pooled = model.embed(ids).mean(dim=1)
    keys, values = projector(pooled)
    return keys[0].detach(), values[0].detach()


def _install_text_projector(payload: Mapping[str, Any], model: CausalByteTransformer,
                            device: torch.device) -> TextSkillProjector | None:
    state = payload.get("text_projector")
    if not isinstance(state, dict):
        return None
    projector = TextSkillProjector(model.config.d_model, model.config.memory_dim,
                                   model.config.d_model)
    projector.load_state_dict(state, strict=True)
    return projector.to(device).eval()


def _syntax_check_python(text: str) -> dict[str, Any]:
    try:
        compile(text, "<generated>", "exec")
        valid = True
        error = None
    except (SyntaxError, ValueError, TypeError) as exc:
        valid = False
        error = f"{type(exc).__name__}: {exc}"
    return {"compiles": valid, "error": error}


def run_snoop(checkpoint: str | Path, *, repo_id: str, source_file: str,
              device: str = "auto", max_new_bytes: int = 128,
              memory_slots: int = 16) -> dict[str, Any]:
    path = Path(checkpoint)
    payload = load_payload(path)
    resolved_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else "cpu" if device == "auto" else device)
    config = FoundationConfig.from_mapping(payload["config"])
    if config.d_model != 1024 or config.n_layers != 8 or config.memory_dim != 8192:
        raise ValueError("Checkpoint geometry does not match the claimed Phase 25 production architecture")
    model = build_model(payload, resolved_device)
    report: dict[str, Any] = {
        "repo_id": repo_id,
        "checkpoint_file": source_file,
        "checkpoint_sha256": _sha256_file(path),
        "checkpoint_step": int(payload["step"]),
        "device": str(resolved_device),
        "architecture": architecture_report(model),
        "probes": {},
        "evidence_limits": [
            "Activation norms establish component execution, not semantic correctness or learning quality.",
            "Attention weights establish routing mass, not that retrieved information caused a correct answer.",
            "[EDIT] is a learned ordinary control token; this repository has no edit runtime handler.",
            "[TOOL_SEARCH] only invokes a tool when emitted as its dedicated token ID and handled by the host runtime.",
            "Byte-level Python syntax checks do not prove algorithmic correctness or safety.",
        ],
    }
    if memory_slots < 1:
        raise ValueError("memory_slots must be positive")
    bank = _memory_view(payload, resolved_device, memory_slots)
    memory_payload = payload.get("memory")
    bank_payload = memory_payload.get("bank") if isinstance(memory_payload, dict) else None
    used_slots = None
    capacity = None
    if isinstance(bank_payload, dict):
        capacity = bank_payload.get("capacity")
        buffers = bank_payload.get("buffers", {})
        used = buffers.get("used") if isinstance(buffers, dict) else None
        if isinstance(used, torch.Tensor):
            used_slots = int(used.sum().item())
    report["memory_bank"] = {
        "capacity": capacity,
        "active_slots_in_checkpoint": used_slots,
        "slots_selected_for_regular_probes": int(bank.keys.shape[0]) if bank is not None else 0,
        "selection": "first active slot indices; use --memory-slots to change this compute bound",
    }
    projector = _install_text_projector(payload, model, resolved_device)
    long_context = ("Context stress. The sky is blue because sunlight scatters in the atmosphere. " * 32
                    + "\nQuestion: Why does the sky look blue? Answer briefly.")
    cases = [
        ("ptbr_fluency", "O céu está azul hoje porque a luz do sol", max_new_bytes),
        ("quicksort", "def quicksort(arr):", max_new_bytes),
        ("latent_logic", "Resolva com cuidado: Ana é mais velha que Bia. Bia é mais velha que Caio. Quem é o mais jovem? Explique.", max_new_bytes),
        ("obscure_fact_tool_use", "What was the exact population of Tokyo in 1923? If uncertain, use [TOOL_SEARCH].", max_new_bytes),
        ("typo_edit", "Corrija esta frase e use [EDIT] se apropriado: O ceu esta azul hojé.", max_new_bytes),
        ("long_context_2k", long_context, max_new_bytes),
        ("minecraft_bot", "Write a Python Minecraft bot script that connects, gathers wood, crafts planks, and builds a small shelter.", max(512, max_new_bytes)),
    ]
    with ProbeRecorder(model) as recorder:
        for name, prompt, byte_limit in cases:
            report["probes"][name] = generate_trace(
                model, prompt, recorder, memory=bank, max_new_bytes=byte_limit)
            if name == "quicksort":
                full_text = prompt + report["probes"][name]["continuation"]
                report["probes"][name]["python_syntax"] = _syntax_check_python(full_text)
        question = "Qual é a cor favorita do Gabriel?"
        fact = "Gabriel's favorite color is neon green."
        if projector is not None:
            fact_key, fact_value = _encode_fact(model, projector, fact, resolved_device)
            distractor_count = min(8, bank.keys.shape[0]) if bank is not None else 0
            keys = bank.keys[:distractor_count] if bank is not None else fact_key.new_zeros((0, fact_key.numel()))
            values = bank.values[:distractor_count] if bank is not None else fact_value.new_zeros((0, fact_value.numel()))
            injected_slot = int(keys.shape[0])
            keys = torch.cat([keys, fact_key.unsqueeze(0)], dim=0)
            values = torch.cat([values, fact_value.unsqueeze(0)], dim=0)
            injected = SkillBankView(keys, values, torch.ones(keys.shape[0], dtype=torch.bool, device=resolved_device))
            baseline = generate_trace(model, question, recorder, memory=bank,
                                      max_new_bytes=max_new_bytes)
            injected_result = generate_trace(model, question, recorder, memory=injected,
                                              max_new_bytes=max_new_bytes, target_slot=injected_slot)
            report["probes"]["memory_injection"] = {
                "question": question,
                "injected_fact": fact,
                "baseline_memory_answer": baseline["continuation"],
                "injected_memory_answer": injected_result["continuation"],
                "baseline": baseline,
                "injected": injected_result,
                "injected_slot": injected_slot,
                "injected_fact_key_l2": float(fact_key.float().norm().cpu()),
                "injected_fact_value_l2": float(fact_value.float().norm().cpu()),
                "warning": "The key/value are generated by the checkpoint's text projector; a one-slot result would be trivial, so up to eight checkpoint bank slots are included as distractors. The learned query may still fail to rank the synthetic fact.",
            }
        else:
            report["probes"]["memory_injection"] = {
                "status": "not run: checkpoint has no saved text_projector state",
            }
    return report


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def capability_scores(report: Mapping[str, Any]) -> dict[str, str]:
    """Avoid unsupported automated scores; leave substantive grading to a reviewer."""
    if not report.get("probes"):
        return {name: "Not run /10" for name in ("PT-BR Fluency", "Code Logic", "Tool Use", "Memory Retrieval")}
    return {
        "PT-BR Fluency": "Human review required /10 (raw generation recorded)",
        "Code Logic": "Syntax checked only; execution/review required /10",
        "Tool Use": "Token-ID emissions recorded; task success requires host/tool validation /10",
        "Memory Retrieval": "Routing and generated answer recorded; semantic success requires review /10",
    }


def render_report(report: Mapping[str, Any]) -> str:
    """Honest, portable Markdown model card assembled from the measured run."""
    arch = report.get("architecture", {})
    probes = report.get("probes", {})
    lines = [
        "# ByteCore-DualMemory — Phase 26 Deep Model Snoop",
        "",
        f"- Checkpoint: `{report.get('repo_id', 'unknown')}` / `{report.get('checkpoint_file', 'unknown')}`",
        f"- SHA-256: `{report.get('checkpoint_sha256', 'not available')}`",
        f"- Training step: {report.get('checkpoint_step', 'unknown')}",
        f"- Device: `{report.get('device', 'not run')}`",
        f"- Parameters: {arch.get('parameters', 'not measured'):,}" if isinstance(arch.get('parameters'), int) else "- Parameters: not measured",
        "",
        "## Executive Summary",
        "",
        f"Run status: **{report.get('status', 'completed inference')}.**",
        f"Checkpoint/load error, if any: `{report.get('error', 'none')}`",
        "This report distinguishes checkpoint facts, observed generations, and claims that still require human or task-level validation. A four-hour pre-training budget is not evidence of finished instruction-following capability. Text and code examples below are raw byte-model continuations, not curated outputs. If the run status is blocked, no model capability or activation claims are being made.",
        "",
        "## Architecture Verification",
        "",
        f"- Production parameter-count match (366,316,833): **{arch.get('production_parameter_count_match', 'not run')}**",
        f"- Eight-block match: **{arch.get('layer_count_match', 'not applicable')}**",
        f"- Sliding-window blocks: **{arch.get('all_attention_sliding_window', 'not run')}**",
        f"- Skill adapters in every block: **{arch.get('all_skill_adapters_present', 'not run')}**",
        f"- Config: `{json.dumps(arch.get('config', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- Memory bank: `{json.dumps(report.get('memory_bank', {}), sort_keys=True)}`",
        "",
        "## Activation Forensics",
        "",
        "Each probe records final-position sliding-window attention weights, per-layer SwiGLU/GLU output norms, latent thought-buffer norm and gates, and memory cross-attention softmax summaries when memory is present. These are instrumentation signals—not proof of useful reasoning or factual recall.",
        "",
    ]
    for field, check in arch.get("expected_config_checks", {}).items():
        lines.append(f"- Geometry `{field}`: actual `{check['actual']}`, expected `{check['expected']}` — **{'MATCH' if check['match'] else 'MISMATCH'}**")
    lines.append("")
    for name, probe in probes.items():
        if not isinstance(probe, dict):
            continue
        lines.extend([f"### {name.replace('_', ' ').title()}", ""])
        if "continuation" in probe:
            lines.extend(["Prompt:", "", f"> {probe.get('prompt', '')}", "", "Continuation:", "", "```text", str(probe["continuation"]), "```", ""])
            lines.append(f"- Input context: {probe.get('input_context_bytes')} bytes (UTF-8 prompt: {probe.get('prompt_utf8_bytes')} bytes) | Generated: {probe.get('generated_bytes')} bytes | Thought norm (mean): {probe.get('thought_buffer_l2_mean')} | Final thought telemetry: `{json.dumps(probe.get('adaptive', {}), ensure_ascii=False, sort_keys=True)}`")
            lines.append(f"- Generated control IDs/tokens: `{json.dumps(list(zip(probe.get('generated_control_ids', []), probe.get('generated_control_tokens', []))), ensure_ascii=False)}`")
            if "python_syntax" in probe:
                lines.append(f"- Python syntax: `{json.dumps(probe['python_syntax'], sort_keys=True)}` (compile check only; not algorithmic verification).")
            lines.append(f"- Activation summary: `{json.dumps(probe.get('activation_summary', {}), sort_keys=True)}`")
            lines.append(f"- Sliding-window final-token attention: `{json.dumps(probe.get('attention_last_generated_position', {}), sort_keys=True)}`")
            if probe.get("memory_attention_last_generated_position"):
                lines.append(f"- Memory cross-attention: `{json.dumps(probe['memory_attention_last_generated_position'], sort_keys=True)}`")
        else:
            lines.append(f"Result: {json.dumps(probe, ensure_ascii=False, sort_keys=True)}")
        lines.append("")
    injection = probes.get("memory_injection")
    if isinstance(injection, dict) and "injected" in injection:
        lines.extend([
            "### Memory injection comparison", "",
            f"- Fake fact: `{injection['injected_fact']}`",
            f"- Baseline answer: `{injection.get('baseline_memory_answer', '')}`",
            f"- Injected-memory answer: `{injection.get('injected_memory_answer', '')}`",
            f"- Injected slot index: `{injection.get('injected_slot')}`",
            f"- Diagnostic caveat: {injection.get('warning', '')}", "",
        ])
    lines.extend(["## Capability Matrix", ""])
    for name, value in capability_scores(report).items():
        lines.append(f"- **{name}:** {value}")
    lines.extend(["", "## The Toddler Limits", "",
                  "Do not infer a context-length capability from the configured maximum: this run tests only the prompt and generation lengths shown above. A syntactically valid Python snippet is not proof that it runs or is correct. The model has no built-in edit handler; `[EDIT]` is ordinary learned vocabulary. A tool request is only useful if the dedicated control ID is emitted and the surrounding host runtime executes it. Synthetic memory injection does not establish reliable real-world memory retrieval."])
    long_probe = probes.get("long_context_2k", {})
    if isinstance(long_probe, dict) and "input_context_bytes" in long_probe:
        lines.append(f"- Long-context probe: {long_probe['input_context_bytes']} input bytes, {long_probe.get('generated_bytes')} output bytes; output text is shown above. This is a single qualitative sample, not evidence of a robust 2k context window.")
    minecraft = probes.get("minecraft_bot", {})
    if isinstance(minecraft, dict) and "generated_bytes" in minecraft:
        lines.append(f"- Minecraft bot request: {minecraft.get('generated_bytes')} bytes generated (cap 512); completeness and correctness require the emitted code to be reviewed/tested separately.")
    lines.extend(["", "## Next Steps", "",
                  "1. Repeat deterministic prompts at several seeds/temperatures and compare against a held-out PT-BR/code set.",
                  "2. Evaluate Python outputs in an isolated sandbox with unit tests, not merely `compile()`.",
                  "3. Measure tool-call precision/recall and exact answer accuracy against a controlled search harness.",
                  "4. Compare injected-memory answer accuracy and target-slot attention against no-memory and shuffled-key controls.",
                  "5. Publish the checkpoint step, training-token count, tokenizer-free byte budget, and independent human review before assigning 0–10 capability scores.",
                  "", "## Forensic Limitations", ""])
    for item in report.get("evidence_limits", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--checkpoint", type=Path, help="Use an already downloaded checkpoint (offline)")
    parser.add_argument("--cache-dir", type=str)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:N]")
    parser.add_argument("--max-new-bytes", type=int, default=128)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("runs/final_model_snoop_report.md"))
    parser.add_argument("--json-output", type=Path, default=Path("runs/final_model_snoop_report.json"))
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.checkpoint:
            checkpoint, source = args.checkpoint, str(args.checkpoint)
        else:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            checkpoint, source = download_checkpoint(args.repo_id, args.revision, token, args.cache_dir)
        result = run_snoop(checkpoint, repo_id=args.repo_id, source_file=source,
                           device=args.device, max_new_bytes=args.max_new_bytes,
                           memory_slots=args.memory_slots)
        result["status"] = "completed inference"
    except Exception as exc:
        result = {
            "status": "blocked before inference",
            "repo_id": args.repo_id,
            "checkpoint_file": str(args.checkpoint) if args.checkpoint else "Hub lookup",
            "error": f"{type(exc).__name__}: {exc}",
            "device": args.device,
            "probes": {},
            "evidence_limits": ["Checkpoint download or load failed; no generation or activation evidence was collected."],
        }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.write_text(render_report(result), encoding="utf-8")
    print(f"Snoop report: {args.output} ({result['status']})")
    if result.get("checkpoint_sha256"):
        print(f"checkpoint sha256: {result['checkpoint_sha256']}; step: {result['checkpoint_step']}; device: {result['device']}")
    else:
        print(result.get("error", "No checkpoint was evaluated"), file=sys.stderr)
    return 0 if result["status"] == "completed inference" else 2


if __name__ == "__main__":
    raise SystemExit(main())
