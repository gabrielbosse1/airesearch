"""Phase 22 shared runtime: precision, VRAM control, lossless bytes, resume.

This module is deliberately importable on CPU with no CUDA and no Hub access.
Every GPU/Hub-dependent operation is lazy and reports ``measured=False`` when
it cannot be executed here, so no VRAM number is ever invented.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
import yaml

from models.foundation import (BOS, EOS, PAD, FoundationConfig,
                               CausalByteTransformer)

CHECKPOINT_SCHEMA = 3


def load_config(path: str | Path = "configs/t4_scaled.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError(f"{path}: expected a mapping with a 'model' section")
    # Fail fast on typo'd model keys.
    FoundationConfig.from_mapping(config["model"])
    return config


def model_config(config: Mapping[str, Any], **overrides: Any) -> FoundationConfig:
    return FoundationConfig.from_mapping(config["model"]).scaled(**overrides)


# ---------------------------------------------------------------------------
# Precision: the T4 bf16 trap
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PrecisionPlan:
    requested: str
    resolved: str                 # bfloat16 | float16 | float32
    device: str
    capability: tuple[int, int] | None
    bf16_supported: bool
    autocast: bool
    scaler_required: bool
    fallback: bool
    reason: str

    @property
    def dtype(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[self.resolved]

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested, "resolved": self.resolved,
            "device": self.device, "capability": list(self.capability) if self.capability else None,
            "bf16_supported": self.bf16_supported, "autocast": self.autocast,
            "scaler_required": self.scaler_required, "fallback": self.fallback,
            "reason": self.reason,
        }

    def autocast_context(self) -> Any:
        if not self.autocast:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype)


def resolve_precision(device: torch.device | str,
                      requested: str = "bfloat16",
                      mode: str = "auto") -> PrecisionPlan:
    """Resolve a requested precision against the ACTUAL device.

    A Tesla T4 is Turing/sm_75: CUDA bf16 autocast is unsupported and raises
    ``RuntimeError: Current CUDA Device does not support bfloat16``. On sm75
    we therefore fall back to float16 + ``torch.amp.GradScaler``; both are
    2-byte formats, so the memory budget is unchanged, but fp16 needs loss
    scaling. Ampere+ (sm_80 or newer) gets real bfloat16 as requested.
    """
    device = torch.device(device)
    capability = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
    bf16_ok = bool(torch.cuda.is_bf16_supported()) if device.type == "cuda" else False
    requested = requested.lower()
    if requested not in ("bfloat16", "float16", "float32"):
        raise ValueError(f"unknown precision {requested!r}")

    if device.type != "cuda":
        return PrecisionPlan(requested, "float32", device.type, capability, False,
                             False, False, True,
                             "CPU: float32 master weights; no autocast (Colab resolves precision on GPU)")

    resolution = requested if mode != "auto" else requested
    fallback = False
    reason = f"explicit {resolution} on sm_{capability[0]}{capability[1]}"
    if resolution == "bfloat16" and not bf16_ok:
        resolution = "float16"
        fallback = True
        reason = ("T4/Turing sm_75 has no CUDA bf16 support; falling back to "
                  "float16 autocast + GradScaler (same 2-byte memory footprint)")
    elif resolution == "bfloat16":
        reason = f"native bf16 supported on sm_{capability[0]}{capability[1]}"
    if resolution != requested and mode != "auto":
        raise RuntimeError(f"requested {requested} is unsupported on this device: {reason}")
    return PrecisionPlan(
        requested=requested, resolved=resolution, device=str(device),
        capability=capability, bf16_supported=bf16_ok,
        autocast=resolution != "float32",
        scaler_required=resolution == "float16",
        fallback=fallback, reason=reason,
    )


def make_grad_scaler(plan: PrecisionPlan) -> Any:
    """Return ``torch.amp.GradScaler('cuda')`` only when fp16 requires it."""
    if not plan.scaler_required:
        return None
    return torch.amp.GradScaler("cuda", enabled=True)


# ---------------------------------------------------------------------------
# VRAM: analytic budget + real runtime probe
# ---------------------------------------------------------------------------
GB = 1024.0 ** 3


def estimate_vram(config: Mapping[str, Any], *, batch_size: int, seq_len: int,
                  parameters: int | None = None,
                  checkpointing: bool | None = None) -> dict[str, Any]:
    """Analytic worst-case budget for one micro-batch on a T4.

    The formula is explicit so it can be audited; it is an ESTIMATE, never a
    measurement. ``probe_vram`` below is the measured counterpart.
    """
    model = config["model"]
    if parameters is None:
        with torch.device("meta"):
            parameters = CausalByteTransformer(model_config(config)).parameters_count
    d = int(model["d_model"])
    layers = int(model["n_layers"])
    heads = int(model["n_heads"])
    kv_heads = int(model.get("n_kv_heads") or heads)
    d_ff = int(model["d_ff"])
    vocab = int(model["vocab_size"])
    refinement = int(model.get("refinement_steps", 0))
    b = int(batch_size)
    length = int(seq_len)

    # fp32 master weights + fp32 gradients + two AdamW fp32 moments.
    master = parameters * 4
    gradients = parameters * 4
    optimiser = parameters * 8
    # Autocast keeps a 2-byte copy of weights used by the kernels.
    autocast_weights = parameters * 2

    use_checkpointing = (bool(model.get("gradient_checkpointing", False))
                         if checkpointing is None else bool(checkpointing))
    # Gradient checkpointing stores only the input of each block plus the
    # refinement boundaries: (B, L, d) at 2 bytes per element.
    boundary = b * length * d * 2
    boundaries = (layers + 1 + refinement) * boundary
    # Recomputed peak inside one block (worst case on sm75, where SDPA may
    # fall back to a kernel that materialises B*H*L*L scores).
    qkv = b * length * (d + 2 * kv_heads * (d // heads)) * 2
    attention_scores = b * heads * length * length * 2
    swiglu_branches = 2 * b * length * d_ff * 2
    block_activations = qkv + attention_scores + swiglu_branches + 4 * boundary
    recompute = block_activations if use_checkpointing else layers * block_activations
    if not use_checkpointing:
        boundaries = (1 + refinement) * boundary
    logits = b * length * vocab * 2
    cross_entropy = b * length * vocab * 4  # fp32 CE intermediates
    activations = boundaries + recompute + logits + cross_entropy
    cuda_context = 0.6 * GB  # cuBLAS/cuDNN workspaces + framework reserve
    total = master + gradients + optimiser + autocast_weights + activations + cuda_context

    vram = config.get("vram", {})
    return {
        "estimate_only": True,
        "formula": ("16 B/param (fp32 master+grad+Adam) + 2 B/param (autocast copy) "
                    "+ (layers+1+refinement)*B*L*d*2 + one recomputed block "
                    "+ B*L*L*H*2 scores + 2*B*L*d_ff*2 SwiGLU branches "
                    "+ B*L*vocab*(2+4) + 0.6 GB CUDA context"),
        "parameters": parameters,
        "batch_size": b, "seq_len": length, "checkpointing": use_checkpointing,
        "master_gb": master / GB, "gradients_gb": gradients / GB,
        "optimiser_gb": optimiser / GB, "autocast_weights_gb": autocast_weights / GB,
        "boundaries_gb": boundaries / GB, "recompute_gb": recompute / GB,
        "logits_gb": (logits + cross_entropy) / GB,
        "cuda_context_gb": cuda_context / GB,
        "estimated_total_gb": round(total / GB, 3),
        "target_min_gb": float(vram.get("target_min_gb", 10.0)),
        "target_max_gb": float(vram.get("target_max_gb", 12.0)),
        "device_total_gb": float(vram.get("device_total_gb", 14.0)),
        "within_target_band": (float(vram.get("target_min_gb", 10.0)) <= total / GB
                               <= float(vram.get("target_max_gb", 12.0))),
    }


@dataclass
class VRAMMonitor:
    """Aggressive runtime VRAM telemetry. CPU returns ``available=False``."""

    path: str | Path = "runs/phase22/vram.jsonl"
    target_min_gb: float = 10.0
    target_max_gb: float = 12.0
    device_total_gb: float = 14.0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return torch.cuda.is_available()

    def snapshot(self, tag: str = "step", *, step: int = 0) -> dict[str, Any]:
        record: dict[str, Any] = {"tag": tag, "step": step, "timestamp": time.time(),
                                  "cuda_available": self.available, "measured": self.available}
        if self.available:
            free, total = torch.cuda.mem_get_info()
            record.update({
                "allocated_gb": round(torch.cuda.memory_allocated() / GB, 3),
                "reserved_gb": round(torch.cuda.memory_reserved() / GB, 3),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
                "free_gb": round(free / GB, 3),
                "total_gb": round(total / GB, 3),
                "utilisation_gb": round((total - free) / GB, 3),
                "within_target_band": self.target_min_gb <= (total - free) / GB <= self.target_max_gb,
            })
        else:
            record.update({"allocated_gb": None, "peak_allocated_gb": None,
                           "utilisation_gb": None, "within_target_band": None,
                           "reason": "no CUDA device in this environment"})
        self.history.append(record)
        self.write(record)
        return record

    def write(self, record: Mapping[str, Any]) -> None:
        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf8") as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")

    def assert_budget(self, *, estimated_gb: float | None = None) -> dict[str, Any]:
        """Fail loudly if the analytical estimate exceeds device memory."""
        verdict = {"estimated_gb": estimated_gb, "ok": True, "warnings": []}
        if estimated_gb is not None and estimated_gb > self.device_total_gb - 0.5:
            verdict["ok"] = False
            verdict["warnings"].append(
                f"estimate {estimated_gb:.2f} GB exceeds device {self.device_total_gb:.2f} GB")
        if not self.available:
            verdict["warnings"].append("no GPU: VRAM utilisation is UNMEASURED")
        return verdict


def vram_fill_ladder(config: Mapping[str, Any], *, device: torch.device,
                     model: CausalByteTransformer | None = None,
                     plan: PrecisionPlan | None = None) -> dict[str, Any]:
    """Aggressively target 10-12 GB utilisation with REAL probes on Colab.

    Finding: at 183.8M params, B=4, L=1024 and full gradient checkpointing the
    analytic budget is only ~3.9 GB. The mandated batch_size=4 / grad_accum=8
    / gradient_checkpointing=True stay as the SAFE DEFAULTS; this ladder grows
    the token throughput (seq_len -> 2048, then micro-batch up to 32, then
    relaxes checkpointing) until MEASURED peak utilisation enters the band.
    On CPU every candidate is reported with ``measured=False``.
    """
    vram = config.get("vram", {})
    low = float(vram.get("target_min_gb", 10.0))
    high = float(vram.get("target_max_gb", 12.0))
    max_batch = int(vram.get("max_micro_batch", 32))
    max_len = min(int(vram.get("max_seq_len", 2048)), int(config["model"]["max_seq_len"]))
    base_batch = int(config["training"].get("micro_batch_size", 4))
    base_len = int(config["training"].get("seq_len", 1024))
    candidates: list[tuple[int, int, bool]] = [
        (base_batch, base_len, True),
        (base_batch, max_len, True),
    ]
    for batch in (8, 12, 16, 24, max_batch):
        if batch <= base_batch or batch > max_batch:
            continue
        candidates.append((batch, max_len, True))
    if vram.get("allow_disable_checkpointing", True):
        for batch in (base_batch, 6, 8, max_batch):
            if batch <= max_batch:
                candidates.append((batch, max_len, False))
    seen: set[tuple[int, int, bool]] = set()
    trials: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for batch, length, checkpointing in candidates:
        key = (batch, length, checkpointing)
        if key in seen or length > int(config["model"]["max_seq_len"]):
            continue
        seen.add(key)
        estimate = estimate_vram(config, batch_size=batch, seq_len=length,
                                 checkpointing=checkpointing)
        over_device = estimate["estimated_total_gb"] > float(vram.get("device_total_gb", 14.0))
        trial: dict[str, Any] = {
            "micro_batch_size": batch, "seq_len": length,
            "gradient_checkpointing": checkpointing,
            "estimated_gb": estimate["estimated_total_gb"],
            "estimated_over_device": over_device,
            "measured": False, "peak_allocated_gb": None,
        }
        if over_device:
            trial["skipped"] = "analytic estimate exceeds device memory; would OOM"
            trials.append(trial)
            continue
        if device.type == "cuda" and model is not None and plan is not None:
            model.set_gradient_checkpointing(checkpointing)
            probe = probe_vram(model, plan, batch_size=batch, seq_len=length, device=device)
            trial["measured"] = bool(probe.get("measured"))
            trial["peak_allocated_gb"] = probe.get("peak_allocated_gb")
            trial["within_target_band"] = (low <= (probe.get("peak_allocated_gb") or 0) <= high)
        trials.append(trial)
        if trial["measured"] and trial.get("within_target_band") and selected is None:
            selected = {"micro_batch_size": batch, "seq_len": length,
                        "gradient_checkpointing": checkpointing,
                        "measured_peak_gb": trial["peak_allocated_gb"]}
    return {
        "target_band_gb": [low, high],
        "defaults": {"micro_batch_size": base_batch, "grad_accum": int(config["training"].get("grad_accum", 8)),
                     "seq_len": base_len, "gradient_checkpointing": True},
        "default_estimate_gb": estimate_vram(config, batch_size=base_batch,
                                             seq_len=base_len)["estimated_total_gb"],
        "trials": trials,
        "selected": selected,
        "measured": device.type == "cuda",
        "note": ("Probe each candidate on the real T4. If none reaches the band, "
                 "report target_met=False instead of padding memory artificially."),
        "target_met": selected is not None,
    }


def probe_vram(model: CausalByteTransformer, plan: PrecisionPlan, *,
               batch_size: int, seq_len: int, device: torch.device) -> dict[str, Any]:
    """Real forward+backward probe. Returns ``measured=False`` without CUDA."""
    if device.type != "cuda":
        return {"measured": False, "reason": "CUDA unavailable in this environment",
                "batch_size": batch_size, "seq_len": seq_len}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    try:
        ids = torch.randint(0, 256, (batch_size, seq_len), device=device)
        with plan.autocast_context():
            loss = model.loss(ids)
        loss.backward()
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated() / GB
        model.zero_grad(set_to_none=True)
        del ids, loss
        torch.cuda.empty_cache()
        return {"measured": True, "peak_allocated_gb": round(peak, 3),
                "batch_size": batch_size, "seq_len": seq_len,
                "oom": False, "precision": plan.resolved}
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {"measured": True, "peak_allocated_gb": None, "oom": True,
                "batch_size": batch_size, "seq_len": seq_len,
                "reason": str(exc)[:300], "precision": plan.resolved}


def autotune_batch(model: CausalByteTransformer, plan: PrecisionPlan,
                   config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Try (4,1024) -> (4,768) -> (2,1024) -> (2,512) -> (1,512) until it fits."""
    vram = config.get("vram", {})
    limit = float(vram.get("target_max_gb", 12.0))
    training = config.get("training", {})
    candidates = [
        (int(training.get("micro_batch_size", 4)), int(training.get("seq_len", 1024))),
        (4, 768), (2, 1024), (2, 768), (2, 512), (1, 512),
    ]
    trials = []
    for batch, length in candidates:
        if length > int(config["model"]["max_seq_len"]):
            continue
        result = probe_vram(model, plan, batch_size=batch, seq_len=length, device=device)
        trials.append(result)
        if result["measured"] and result["peak_allocated_gb"] <= limit:
            return {"selected": {"micro_batch_size": batch, "seq_len": length},
                    "trials": trials, "measured": True,
                    "target_max_gb": limit}
        if not result["measured"]:
            return {"selected": {"micro_batch_size": batch, "seq_len": length},
                    "trials": trials, "measured": False,
                    "note": "no GPU here; Colab will repeat this probe"}
    return {"selected": {"micro_batch_size": 1, "seq_len": 512},
            "trials": trials, "measured": True,
            "warning": f"no candidate stayed below {limit} GB; smallest chosen"}


# ---------------------------------------------------------------------------
# Lossless byte packing (no replacement chars, no mid-codepoint cuts)
# ---------------------------------------------------------------------------
def _utf8_boundary(buffer: list[int], cut: int) -> int:
    """Back off ``cut`` until it is not inside a UTF-8 continuation byte."""
    while cut > 0 and cut < len(buffer) and (buffer[cut] & 0xC0) == 0x80:
        cut -= 1
    return cut


class BytePacker:
    """Pack real UTF-8 documents into fixed-length next-byte LM sequences.

    Framing: ``BOS <document bytes> EOS``. A document is never truncated; when
    a sequence boundary would split a multi-byte codepoint the cut is backed
    off and the remaining bytes carry into the next sequence. The ``carry`` is
    part of the resume state, so NOTHING is discarded between checkpoints.
    """

    def __init__(self, seq_len: int, carry: Iterable[int] | None = None,
                 carry_labels: Iterable[float] | None = None) -> None:
        if seq_len < 8:
            raise ValueError("seq_len must be >= 8")
        self.seq_len = int(seq_len)
        self.buffer: list[int] = list(carry or [])
        self.labels: list[float] = (list(carry_labels) if carry_labels is not None
                                    else [1.0] * len(self.buffer))
        if len(self.labels) != len(self.buffer):
            raise ValueError("carry_labels must align with carry")
        self.documents = 0
        self.sequences = 0
        self.bytes_packed = 0

    def feed(self, document: bytes, *, trainable_prefix: int | None = None
             ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Append one framed document and emit any complete ``(ids, weights)``.

        ``trainable_prefix`` masks the first N raw bytes of the document (0.0
        weight) and keeps the remainder trainable (1.0). Phase 1 passes the
        whole document length; Phase 2 masks prompt/header bytes so only the
        assistant / ReAct continuation contributes to the loss.
        """
        if not isinstance(document, (bytes, bytearray)):
            raise TypeError("document must be raw bytes")
        masked = 0 if trainable_prefix is None else max(0, min(int(trainable_prefix), len(document)))
        self.buffer.append(BOS)
        self.labels.append(0.0)
        for index, value in enumerate(document):
            self.buffer.append(value)
            self.labels.append(0.0 if index < masked else 1.0)
        self.buffer.append(EOS)
        self.labels.append(1.0)
        self.documents += 1
        emitted: list[tuple[torch.Tensor, torch.Tensor]] = []
        while len(self.buffer) >= self.seq_len:
            cut = _utf8_boundary(self.buffer, self.seq_len)
            if cut < self.seq_len // 2:      # avoid pathological tiny fragments
                break
            chunk = self.buffer[:cut]
            weights = self.labels[:cut]
            emitted.append((torch.tensor(chunk, dtype=torch.long),
                            torch.tensor(weights, dtype=torch.float32)))
            del self.buffer[:cut]
            del self.labels[:cut]
            self.sequences += 1
            self.bytes_packed += sum(1 for value in chunk if value < 256)
        return emitted

    @property
    def carry(self) -> list[int]:
        return list(self.buffer)

    @property
    def carry_labels(self) -> list[float]:
        return list(self.labels)

    def state_dict(self) -> dict[str, Any]:
        return {"seq_len": self.seq_len, "carry": self.carry,
                "carry_labels": self.carry_labels,
                "documents": self.documents, "sequences": self.sequences,
                "bytes_packed": self.bytes_packed}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.buffer = [int(value) for value in state.get("carry", [])]
        self.labels = [float(value) for value in state.get("carry_labels", [1.0] * len(self.buffer))]
        self.documents = int(state.get("documents", 0))
        self.sequences = int(state.get("sequences", 0))
        self.bytes_packed = int(state.get("bytes_packed", 0))

    def decode_all(self, sequences: Iterable[Any]) -> bytes:
        """Reassemble raw bytes from emitted items + carry (audit helper)."""
        raw = bytearray()
        for item in sequences:
            tensor = item[0] if isinstance(item, (tuple, list)) else item
            for value in tensor.tolist():
                if 0 <= value < 256:
                    raw.append(value)
        for value in self.carry:
            if 0 <= value < 256:
                raw.append(value)
        return bytes(raw)


def pack_stream(documents: Iterable[tuple[bytes, int]], seq_len: int,
                carry: Iterable[int] | None = None,
                carry_labels: Iterable[float] | None = None
                ) -> tuple[Iterator[tuple[torch.Tensor, torch.Tensor]], BytePacker]:
    """``documents`` yields ``(raw_bytes, trainable_prefix_length)`` pairs."""
    packer = BytePacker(seq_len, carry, carry_labels)

    def generate() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for document, prefix in documents:
            for pair in packer.feed(document, trainable_prefix=prefix):
                yield pair

    return generate(), packer


# ---------------------------------------------------------------------------
# Checkpoints: atomic, exact, resumable, Hub-backed
# ---------------------------------------------------------------------------
def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def save_checkpoint(path: str | Path, model: CausalByteTransformer, *,
                    phase: int, step: int, optimizer: Any = None,
                    scaler: Any = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Atomic fp32 save: master weights + optimiser + scaler + RNG + packer.

    Weights stay fp32 so a resumed step is bit-identical to an uninterrupted
    run; ``save_deploy_copy`` writes the 2-byte inference copy separately.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "phase": int(phase), "step": int(step),
        "config": model.config.to_dict(),
        "parameters": model.parameters_count,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": {"python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        "extra": dict(extra or {}),
        "saved_at": time.time(),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, target)
    return {"path": str(target), "bytes": target.stat().st_size,
            "phase": int(phase), "step": int(step), "parameters": model.parameters_count}


def load_checkpoint(path: str | Path, model: CausalByteTransformer | None = None, *,
                    optimizer: Any = None, scaler: Any = None,
                    device: torch.device | str = "cpu",
                    expected_phase: int | None = None) -> dict[str, Any]:
    """Strictly restore a checkpoint, or raise with an actionable message."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    state = torch.load(source, map_location=device, weights_only=False)
    if int(state.get("schema", 0)) != CHECKPOINT_SCHEMA:
        raise ValueError(f"{source}: schema {state.get('schema')} != {CHECKPOINT_SCHEMA}")
    if expected_phase is not None and int(state["phase"]) != expected_phase:
        raise ValueError(f"{source}: phase {state['phase']} != {expected_phase}")
    config = FoundationConfig(**state["config"])
    if model is None:
        model = CausalByteTransformer(config)
    elif model.config.to_dict() != config.to_dict():
        raise ValueError(f"{source}: model config mismatch")
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    restored: dict[str, Any] = {"model": model, "config": config,
                                "phase": int(state["phase"]), "step": int(state["step"]),
                                "extra": dict(state.get("extra") or {}),
                                "optimizer_restored": False, "scaler_restored": False,
                                "rng_restored": False}
    if optimizer is not None and state.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(state["optimizer"])
            restored["optimizer_restored"] = True
        except (ValueError, KeyError) as exc:
            restored["optimizer_error"] = str(exc)
    if scaler is not None and state.get("scaler") is not None:
        try:
            scaler.load_state_dict(state["scaler"])
            restored["scaler_restored"] = True
        except (ValueError, KeyError) as exc:
            restored["scaler_error"] = str(exc)
    rng = state.get("rng") or {}
    if "python" in rng:
        random.setstate(rng["python"])
    if "torch" in rng:
        torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and rng.get("cuda"):
        torch.cuda.set_rng_state_all(rng["cuda"])
    restored["rng_restored"] = bool(rng)
    return restored


def save_deploy_copy(path: str | Path, checkpoint: str | Path) -> dict[str, Any]:
    """Write a 2-byte inference-only copy (half the disk / upload size)."""
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    slim = {"schema": CHECKPOINT_SCHEMA, "phase": state["phase"], "step": state["step"],
            "config": state["config"],
            "model": {key: (value.half() if value.is_floating_point() else value)
                      for key, value in state["model"].items()}}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, target)
    return {"path": str(target), "bytes": target.stat().st_size,
            "dtype": "float16", "note": "inference only; no optimiser/RNG state"}


def hf_upload(local_path: str | Path, repo_id: str | None,
              path_in_repo: str) -> dict[str, Any]:
    """Upload a checkpoint to the Hub. No-op with a reason when unconfigured."""
    if not repo_id:
        return {"uploaded": False, "reason": "hf_repo is not configured"}
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        return {"uploaded": False, "reason": f"huggingface_hub missing: {exc}"}
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo,
                    repo_id=repo_id, repo_type="model")
    return {"uploaded": True, "repo_id": repo_id, "path_in_repo": path_in_repo}


def hf_download(repo_id: str | None, filename_in_repo: str,
                local_path: str | Path) -> dict[str, Any]:
    """Fetch a previous checkpoint for resume; returns a reason on failure."""
    target = Path(local_path)
    if target.is_file():
        return {"downloaded": False, "reason": "local checkpoint already present",
                "path": str(target)}
    if not repo_id:
        return {"downloaded": False, "reason": "hf_repo is not configured"}
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        return {"downloaded": False, "reason": f"huggingface_hub missing: {exc}"}
    try:
        fetched = hf_hub_download(repo_id=repo_id, filename=filename_in_repo,
                                  repo_type="model")
    except Exception as exc:
        return {"downloaded": False, "reason": f"{type(exc).__name__}: {exc}"[:400],
                "note": "starting from scratch is safe; the run is resumable from step 0"}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path(fetched).read_bytes())
    return {"downloaded": True, "path": str(target), "repo_id": repo_id}


def resume_checkpoint(config: Mapping[str, Any], phase: int,
                      model: CausalByteTransformer, *, optimizer: Any = None,
                      scaler: Any = None, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Find local or Hub checkpoint for ``phase`` and restore it."""
    ck = config.get("checkpoints", {})
    name = {1: "phase1_name", 2: "phase2_name", 3: "phase3_name"}[phase]
    local = Path(ck.get("local_dir", "runs/phase22/checkpoints")) / ck[name]
    remote_name = f"{ck.get('hf_path_in_repo', 'phase22')}/{ck[name]}"
    fetched = (hf_download(ck.get("hf_repo"), remote_name, local)
               if ck.get("resume_from_hf", True) else {"downloaded": False, "reason": "disabled"})
    if not local.is_file():
        return {"resumed": False, "step": 0, "path": str(local), "hf": fetched,
                "reason": "no checkpoint found; starting this phase from scratch"}
    restored = load_checkpoint(local, model, optimizer=optimizer, scaler=scaler,
                               device=device, expected_phase=phase)
    return {"resumed": True, "step": restored["step"], "path": str(local), "hf": fetched,
            "optimizer_restored": restored["optimizer_restored"],
            "scaler_restored": restored["scaler_restored"],
            "rng_restored": restored["rng_restored"], "extra": restored["extra"]}


def initialise_from_phase(config: Mapping[str, Any], source_phase: int,
                          model: CausalByteTransformer, *,
                          device: torch.device | str = "cpu") -> dict[str, Any]:
    """Phase 2 starts from Phase 1 weights; Phase 3 starts from Phase 2.

    Missing weight files are a HARD ERROR when the notebook requests strict
    phase order -- silently training Phase 2 from random weights would
    reproduce the Phase 19/21 "everything is from scratch" failure mode.
    """
    ck = config.get("checkpoints", {})
    name = {1: "phase1_name", 2: "phase2_name", 3: "phase3_name"}[source_phase]
    local = Path(ck.get("local_dir", "runs/phase22/checkpoints")) / ck[name]
    fetched = (hf_download(ck.get("hf_repo"),
                           f"{ck.get('hf_path_in_repo', 'phase22')}/{ck[name]}", local)
               if ck.get("resume_from_hf", True) else {"downloaded": False, "reason": "disabled"})
    if not local.is_file():
        raise FileNotFoundError(
            f"{local}: Phase {source_phase} weights are required before this phase. "
            "Run the previous notebook, or set checkpoints.hf_repo to resume from the Hub.")
    restored = load_checkpoint(local, model, device=device,
                               expected_phase=source_phase)
    return {"loaded": True, "source_phase": source_phase, "source_step": restored["step"],
            "path": str(local), "hf": fetched,
            "parameters": model.parameters_count}


def write_report(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf8")
    return target
