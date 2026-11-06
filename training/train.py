"""Unified training script (Phase 25). One script, one pipeline.

    python -m training.train --config configs/t4_scaled.yaml \
        --hf-repo gabrielbosse9/airesearch2 --max-minutes 220

Everything the 4-hour Colab run needs lives here:

* precision resolution -- T4 (sm_75) CANNOT run bf16 autocast; ``auto``
  resolves to fp16 + GradScaler there, bf16 on Ampere+, fp32 on CPU;
* the streaming PT-BR-first 50/25/15/10 data mix (``data/sources.py``),
  quota-exact per 100-draw window (50% PT-BR docs), packed to bytes;
* STRICT streaming (Phase 24.2): every source is a verified
  ``IterableDataset`` opened with ``streaming=True`` and shuffled with a
  RAM-only ``buffer_size`` reservoir. The loop is STEP-based
  (``while step < max_steps``) -- never epochs, never ``len(dataset)``.
  There is deliberately no multi-process DataLoader: ``MixtureSampler`` +
  ``BatchBuilder`` are the custom single-process iterable wrapper (the
  ``num_workers=0`` configuration HF recommends for streaming), so worker
  deadlocks and disk caching are structurally impossible;
* bidirectional multimodal batches: caption bytes conditioned on a media
  skill matrix AND media codec bytes conditioned on a text skill matrix;
* causal-LM cross-entropy + confidence BCE + dynamic-memory policy loss;
* gradient checkpointing, grad accumulation, cosine LR with warmup;
* representative measured VRAM probe + batch autotune; the separate
  ``--simulate-t4`` accounting mode is explicitly labelled as an estimate;
* atomic checkpoints every 10 minutes (and every N steps), resumable locally
  or from the Hugging Face Hub;
* the TRUE tool runtime: ``[TOOL_SEARCH]`` token ids trigger a sandboxed
  subprocess search (deterministic mock for training, real ``ddgs`` at
  inference). ``[EDIT]`` has no host handler -- it is ordinary vocabulary.

``--smoke`` runs the entire pipeline on CPU with a tiny model and synthetic
fixtures in seconds. Smoke fixtures validate plumbing; they are NOT training
data and are never used in a real run.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.nn.functional as F
import yaml
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.foundation import (BOS, EOS, PAD, TOOL_SEARCH, TOOL_SEARCH_END, VOCAB_SIZE,
                               OBS_START, OBS_END, EDIT, EDIT_END,
                               CausalByteTransformer, FoundationConfig,
                               decode_ids, encode_bytes, pad_batch)
from models.memory_model import (KEY_DIM, VALUE_DIM, DualMemory, EvictionEvent,
                                 SkillBankView, stack_views)
from models.encoders import (MultimodalIngestor, TextSkillProjector,
                             build_encoder, matrix_to_bytes,
                             pooled_text_embedding, preprocess_image)
from data.sources import BytePacker, MixtureSampler, build_mixture, mixture_snapshot

CHECKPOINT_SCHEMA = 7  # v7: sliding-window attention + memory long-range hand-off


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str | Path = "configs/t4_scaled.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError(f"{path}: expected a mapping with a 'model' section")
    FoundationConfig.from_mapping(config["model"])  # fail fast on typos
    return config


# ---------------------------------------------------------------------------
# Simulated T4 accounting (ESTIMATE, never a CUDA measurement)
# ---------------------------------------------------------------------------
GB = 1_000_000_000
# Deliberately rounded UP, not presented as exact model-card counts. Each
# configured modality owns a separate frozen backbone in the ingestor registry.
FROZEN_ENCODER_PARAM_ESTIMATE = {"vit_b_16": 90_000_000, "wav2vec2": 95_000_000}


def _trainable_parameters(config: Mapping[str, Any], model_cfg: FoundationConfig) -> tuple[int, int]:
    """Exact trainable count on meta + conservative frozen-backbone count."""
    mem = dict(config.get("memory", {}))
    enc = dict(config.get("encoders", {}))
    key_dim = int(mem.get("key_dim", model_cfg.d_model))
    value_dim = int(mem.get("value_dim", model_cfg.memory_dim or VALUE_DIM))
    capacity = int(mem.get("capacity", int(mem.get("procedural_capacity", 1024))
                           + int(mem.get("episodic_capacity", 1024))))
    with torch.device("meta"):
        model = CausalByteTransformer(model_cfg)
        text = TextSkillProjector(model_cfg.d_model, value_dim, key_dim)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable += sum(p.numel() for p in text.parameters() if p.requires_grad)
        frozen = 0
        for name in enc.values():
            if name is None:
                continue
            ingestor = MultimodalIngestor(None, memory_dim=value_dim, key_dim=key_dim,
                                          encoder_dim=768)
            trainable += sum(p.numel() for p in ingestor.parameters() if p.requires_grad)
            frozen += FROZEN_ENCODER_PARAM_ESTIMATE.get(str(name), 100_000_000)
        trainable += 2 * capacity  # importance_raw + decay_raw
    return int(trainable), int(frozen)


def simulate_t4_vram(config: Mapping[str, Any], *, micro_batch: int | None = None,
                     seq_len: int | None = None, window_size: int | None = None,
                     chunk_size: int | None = None) -> dict[str, Any]:
    """Conservative simulated T4 peak for the sliding-window training path.

    This is an ESTIMATE, not a CUDA measurement. It assumes fp32 trainable
    weights/grads/AdamW moments, fp16 autocast, fp32 residual boundaries,
    all-block PLUS sub-block checkpointing, chunk-local sliding-window SDPA
    (one small mask/workspace live at a time), inline confidence auxiliary
    loss, and the configured frozen-encoder storage dtype. T4/sm_75 uses the
    memory-efficient SDPA backend; PyTorch Flash/cuDNN SDPA require sm_80+.
    """
    model_cfg = FoundationConfig.from_mapping(config["model"])
    train = dict(config.get("training", {}))
    mem = dict(config.get("memory", {}))
    vram = dict(config.get("vram", {}))
    batch = int(micro_batch if micro_batch is not None else train.get("micro_batch_size", 1))
    length = int(seq_len if seq_len is not None else train.get("seq_len", model_cfg.max_seq_len))
    window = int(window_size if window_size is not None else model_cfg.window_size)
    chunk = int(chunk_size if chunk_size is not None else model_cfg.sliding_chunk_size)
    if batch < 1 or length < 1 or length > model_cfg.max_seq_len or window < 1 or chunk < 1:
        raise ValueError("invalid simulated batch/length/window/chunk")
    if model_cfg.attention_type == "sliding_window" and not model_cfg.gradient_checkpointing:
        raise ValueError("sliding-window T4 simulation requires gradient_checkpointing=True")
    d = model_cfg.d_model
    h = model_cfg.n_heads
    kv = model_cfg.n_kv_heads or h
    ff = model_cfg.d_ff
    head_dim = d // h
    memory_dim = model_cfg.memory_dim or int(mem.get("value_dim", VALUE_DIM))
    key_dim = int(mem.get("key_dim", d))
    capacity = int(mem.get("capacity", int(mem.get("procedural_capacity", 1024))
                           + int(mem.get("episodic_capacity", 1024))))
    slots = capacity + sum(1 for name in config.get("encoders", {}).values() if name) + 1
    steps = max(1, model_cfg.refinement_steps)
    trainable, frozen = _trainable_parameters(config, model_cfg)
    act = batch * length * d
    frozen_dtype = str(vram.get("frozen_encoder_dtype", "float32")).lower()
    if frozen_dtype not in ("float32", "float16", "bfloat16", "auto"):
        raise ValueError("frozen_encoder_dtype must be float32, float16, bfloat16, or auto")
    frozen_bytes = 2 if frozen_dtype in ("float16", "bfloat16", "auto") else 4

    # One chunk-local SWA call: small bool mask + possible backend bias,
    # repeated GQA K/V for the local span, Q/output, and logsumexp workspace.
    local_chunk = min(chunk, length)
    key_span = min(length, window + local_chunk)
    sliding_workspace = (5 * local_chunk * key_span
                         + 2 * batch * h * key_span * head_dim * 2
                         + 2 * batch * h * local_chunk * head_dim * 2
                         + 4 * batch * h * local_chunk)
    norm_temp = 4 * act
    residual = 4 * act
    block_groups = {
        "attention": 2 * act * (4 + 2 * kv / h) + sliding_workspace + 2 * norm_temp + residual,
        "ffn": 2 * 4 * batch * length * ff + 2 * norm_temp + residual,
        "adapter_glu": (2 * batch * length * memory_dim + 2 * 3 * act
                        + 8 * batch * length * slots + norm_temp + residual),
        "adapter_cross": (2 * (6 * act + 2 * slots * d + batch * length * slots)
                          + norm_temp + residual),
    }
    components: dict[str, int] = {
        "fp32_params_grads_adam": 16 * trainable,
        "frozen_encoder_weights": frozen_bytes * frozen,
        "unified_memory_buffers": 4 * capacity * (key_dim + memory_dim + 8),
        "block_checkpoint_boundaries": 4 * (model_cfg.n_layers + 1) * act,
        # Sub-block checkpoints retain a few within-layer boundaries, but only
        # the largest recomputed branch is live during backward.
        "live_block_sub_boundaries": 3 * 4 * act,
        "live_block_max_subcheckpoint": max(block_groups.values()),
        # Inline confidence BCE avoids a second retained list of full states.
        "refinement_step_boundaries": 4 * steps * act,
        "refinement_live_thought_step": 2 * 4 * act + 6 * 2 * act,
        "refinement_live_cell": 2 * 6 * batch * length * model_cfg.refinement_rank + 4 * 4 * act,
        "confidence_and_ce_logits": 4 * 4 * batch * length * VOCAB_SIZE,
        "ids_targets_embeddings": 8 * 2 * batch * length + 4 * act,
    }
    if model_cfg.attention_type != "sliding_window":
        components["dense_sdpa_workspace"] = 2 * batch * h * length * length
    safety_gb = float(vram.get("simulator_safety_gb", 0.5))
    if safety_gb < 0:
        raise ValueError("simulator_safety_gb must be non-negative")
    components["allocator_backend_safety_reserve"] = int(safety_gb * GB)
    peak_bytes = sum(components.values())
    target = float(vram.get("target_max_gb", 13.5))
    return {
        "measured": False,
        "simulated": True,
        "device": "Tesla T4 16GB (simulated; no CUDA measurement)",
        "peak_gb": round(peak_bytes / GB, 3),
        "target_max_gb": target,
        "design_target_gb": 10.0,
        "fits": bool(peak_bytes / GB <= target),
        "meets_design_target": bool(peak_bytes / GB < 10.0),
        "micro_batch": batch,
        "seq_len": length,
        "attention_type": model_cfg.attention_type,
        "window_size": window,
        "sliding_chunk_size": chunk,
        "sdpa_backend_required": "memory_efficient (T4/sm_75); flash/cudnn require sm_80+",
        "trainable_parameters": trainable,
        "frozen_encoder_parameters_estimate": frozen,
        "frozen_encoder_bytes_per_parameter": frozen_bytes,
        "components_gb": {name: round(value / GB, 4) for name, value in components.items()},
        "block_group_estimates_gb": {name: round(value / GB, 4) for name, value in block_groups.items()},
        "assumptions": [
            "decimal GB; fp32 trainable weights/grads/AdamW moments (16 B/parameter)",
            "fp16 autocast inside blocks; fp32 residual/checkpoint boundaries",
            "all transformer blocks and attention/FFN/adapter branches checkpointed",
            "exact chunk-local sliding window; no (L,L) mask; one chunk workspace live",
            "T4 uses memory-efficient SDPA, not Flash/cuDNN (sm_80+ only)",
            "inline confidence BCE avoids retaining a second full refinement-state list",
            "full unified bank plus batch-local projector slots in cross-attention",
            "frozen encoder counts rounded up; storage dtype from vram.frozen_encoder_dtype",
            "safety reserve is an estimate, not observed allocator fragmentation",
        ],
    }



# ---------------------------------------------------------------------------
# Precision: the T4 bf16 trap
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PrecisionPlan:
    requested: str
    resolved: str
    device: str
    bf16_supported: bool
    autocast: bool
    scaler_required: bool
    reason: str

    @property
    def dtype(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[self.resolved]

    def autocast_context(self) -> Any:
        if not self.autocast:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    def as_dict(self) -> dict[str, Any]:
        return {"requested": self.requested, "resolved": self.resolved,
                "device": self.device, "bf16_supported": self.bf16_supported,
                "autocast": self.autocast, "scaler_required": self.scaler_required,
                "reason": self.reason}


def resolve_precision(device: torch.device | str, requested: str = "bfloat16",
                      mode: str = "auto") -> PrecisionPlan:
    """Resolve requested precision against the ACTUAL device.

    Tesla T4 = Turing/sm_75: bf16 autocast raises at runtime. There we fall
    back to float16 + GradScaler (same 2-byte budget, needs loss scaling).
    """
    device = torch.device(device)
    requested = requested.lower()
    if requested not in ("bfloat16", "float16", "float32"):
        raise ValueError(f"unknown precision {requested!r}")
    if device.type != "cuda":
        return PrecisionPlan(requested, "float32", device.type, False, False, False,
                             "CPU: float32, no autocast")
    bf16_ok = bool(torch.cuda.is_bf16_supported())
    if mode == "auto":
        if bf16_ok:
            return PrecisionPlan(requested, "bfloat16", "cuda", True, True, False,
                                 "Ampere+: native bfloat16")
        return PrecisionPlan(requested, "float16", "cuda", False, True, True,
                             "T4/sm_75: bf16 unsupported -> float16 + GradScaler")
    if mode == "bfloat16" and not bf16_ok:
        raise ValueError("bfloat16 requested but this CUDA device does not support it")
    resolved = mode if mode != "float32" else "float32"
    return PrecisionPlan(requested, resolved, "cuda", bf16_ok,
                         resolved != "float32", resolved == "float16",
                         f"explicit mode {mode}")


def make_grad_scaler(plan: PrecisionPlan) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=plan.scaler_required)


# ---------------------------------------------------------------------------
# True tooling: sandboxed search execution
# ---------------------------------------------------------------------------
_MOCK_SEARCH_SCRIPT = r"""
import hashlib, sys
query = sys.stdin.buffer.read().decode("utf8", "replace").strip()
digest = hashlib.sha256(query.encode("utf8")).hexdigest()
lines = []
for i in range(4):
    token = digest[i * 8:(i + 1) * 8]
    lines.append(f"[{i + 1}] {query} -- result {token} (mock ddgs)")
sys.stdout.write("\n".join(lines))
"""

_DDGS_SEARCH_SCRIPT = r"""
import sys
query = sys.stdin.buffer.read().decode("utf8", "replace").strip()
from ddgs import DDGS
results = DDGS().text(query, max_results=5)
lines = [f"[{i + 1}] {r.get('title', '')} -- {r.get('body', '')} ({r.get('href', '')})"
         for i, r in enumerate(results)]
sys.stdout.write("\n".join(lines) if lines else "(no results)")
"""


def _sandbox_limits() -> None:  # pragma: no cover - executed in the child
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def run_search(query: bytes, *, backend: str = "mock", timeout: float = 15.0,
               max_output_bytes: int = 4096) -> bytes:
    """Execute a search in an isolated subprocess (the 'true tool' path).

    ``mock``: deterministic stdlib-only script (``-I -S``: no site-packages,
    no env, no user site) -- used for training trajectories and tests.
    ``ddgs``: real DuckDuckGo search via the ``ddgs`` package (``-I``), for
    inference on a networked machine (Colab). CPU/RAM/file-size rlimits and a
    wall-clock timeout apply to both. No host-side regex anywhere: the query
    arrives as decoded token bytes from ``generate_with_tools``.
    """
    if backend not in ("mock", "ddgs"):
        raise ValueError("backend must be 'mock' or 'ddgs'")
    script = _MOCK_SEARCH_SCRIPT if backend == "mock" else _DDGS_SEARCH_SCRIPT
    argv = [sys.executable, "-I"]
    if backend == "mock":
        argv.append("-S")
    argv += ["-c", script]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    preexec = _sandbox_limits if os.name == "posix" else None
    try:
        with tempfile.TemporaryDirectory() as scratch:
            completed = subprocess.run(
                argv, input=query, capture_output=True, cwd=scratch, env=env,
                timeout=timeout, preexec_fn=preexec, check=False)
        output = completed.stdout[:max_output_bytes]
        if completed.returncode != 0 and not output:
            return f"(search failed: exit {completed.returncode})".encode("utf8")
        return output if output else b"(no results)"
    except subprocess.TimeoutExpired:
        return b"(search timed out)"
    except Exception as exc:
        return f"(search error: {type(exc).__name__})".encode("utf8")


def make_search_fn(backend: str = "mock", **kwargs: Any):
    def search(query: bytes) -> bytes:
        return run_search(query, backend=backend, **kwargs)
    return search


def inject_tool_trajectories(samples: Iterator[tuple[str, dict[str, Any]]], *,
                             rate: float = 0.02, seed: int = 24,
                             search_fn: Any = None) -> Iterator[tuple[str, dict[str, Any]]]:
    """Sprinkle REAL token-id tool/edit trajectories into the stream.

    After a text sample, with probability ``rate``, build a trajectory whose
    query comes from the corpus bytes and whose observation is produced by
    the actual sandboxed MOCK search (deterministic, offline). Emitted as
    ``kind="ids"`` so the control ids are trained by ordinary cross-entropy.
    This is how ``[TOOL_SEARCH]``/``[EDIT]`` enter the 4-hour pre-training mix
    without a legacy synthetic-data generator.
    """
    rng = random.Random(seed)
    if search_fn is None:
        # The built-in mock is deterministic, so memoise it: a sandbox spawn
        # costs ~17 ms and corpus-derived queries repeat heavily. Never cache
        # a caller-supplied search_fn (it may be live/stateful).
        base_search = make_search_fn("mock")

        @lru_cache(maxsize=4096)
        def search(query: bytes) -> bytes:
            return base_search(query)
    else:
        search = search_fn
    for group, sample in samples:
        yield group, sample
        if sample.get("kind") != "text" or rng.random() >= rate:
            continue
        words = bytes(sample["data"]).split()[:6]
        if len(words) < 2:
            continue
        query = b" ".join(words)[:64]
        if rng.random() < 0.5:
            result = search(query)[:256]
            ids = ([BOS] + list(b"search: ") + [TOOL_SEARCH] + list(query)
                   + [TOOL_SEARCH_END, OBS_START] + list(result) + [OBS_END]
                   + list(b"\nanswer: ") + list(result[:64]) + [EOS])
        else:
            old, new = words[0][:24], words[-1][:24]
            ids = ([BOS] + list(b"edit: ") + [EDIT] + list(old) + list(b" -> ")
                   + list(new) + [EDIT_END, EOS])
        yield "tools", {"kind": "ids", "ids": ids}


# ---------------------------------------------------------------------------
# Batch construction (text packing + bidirectional multimodal pairs)
# ---------------------------------------------------------------------------
@dataclass
class _Row:
    ids: torch.Tensor
    targets: torch.Tensor
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None


class BatchBuilder:
    """Turns mixture samples into padded training batches with memory views.

    Text -> packed next-byte pairs. Multimodal -> TWO rows per sample:
      A) media->text: caption bytes, conditioned on the media skill matrix;
      B) text->media: media codec bytes, conditioned on the TEXT skill matrix.
    Rows without a matrix get mask=False slots, so the adapter retrieves
    exactly zero for them (no cross-sample leakage).
    """

    def __init__(self, model: CausalByteTransformer, ingestors: Mapping[str, MultimodalIngestor],
                 text_projector: TextSkillProjector, seq_len: int, *,
                 memory: DualMemory | None = None, ingest_to_memory: bool = False) -> None:
        self.model = model
        self.ingestors = dict(ingestors)
        self.text_projector = text_projector
        self.seq_len = int(seq_len)
        self.device = next(model.parameters()).device
        self.packer = BytePacker(seq_len)
        self.memory = memory
        self.ingest_to_memory = ingest_to_memory
        self._rows: list[_Row] = []
        self._pending_writes: list[tuple[torch.Tensor, torch.Tensor, str, dict[str, Any]]] = []
        self.stats = {"text_rows": 0, "media_to_text": 0, "text_to_media": 0,
                      "skipped": 0, "ingested": 0}

    def _features(self, sample: Mapping[str, Any]) -> torch.Tensor | None:
        modality = sample["modality"]
        if "features" in sample:  # pre-computed / smoke fixtures: matrix-agnostic
            features = torch.as_tensor(sample["features"], dtype=torch.float32,
                                       device=self.device)
            return features.unsqueeze(0) if features.ndim == 1 else features
        ingestor = self.ingestors.get(modality)
        if ingestor is None or ingestor.encoder is None:
            return None
        if modality in ("image", "video") and sample.get("image") is not None:
            raw = preprocess_image(sample["image"]).to(self.device)
            return ingestor.encoder(raw)
        if modality == "audio" and sample.get("waveform") is not None:
            waveform = sample["waveform"].to(self.device)
            rate = sample.get("sample_rate") or 16000
            if rate != 16000:
                import torchaudio
                waveform = torchaudio.functional.resample(waveform, rate, 16000)
            return ingestor.encoder(waveform.unsqueeze(0))
        return None

    def _add_multimodal(self, sample: Mapping[str, Any]) -> None:
        features = self._features(sample)
        if features is None:
            self.stats["skipped"] += 1
            return
        modality = sample["modality"]
        ingestor = self.ingestors.get(modality)
        if ingestor is None:
            self.stats["skipped"] += 1
            return
        key, value = ingestor.project_features(features)  # gradients ON
        caption = bytes(sample["caption"])[: self.seq_len // 4]

        if self.ingest_to_memory and self.memory is not None:
            # Delay commitment until AFTER the current batch is assembled.
            # Otherwise a sample fed to fill this batch could be retrieved by
            # that same batch (future/self leakage through the teleportation hub).
            self._pending_writes.append((key[0].detach(), value[0].detach(), modality,
                                         {"caption": caption[:64].decode("utf8", "replace")}))

        # Direction A: media matrix -> caption bytes.
        prompt = b"caption: "
        full = encode_bytes(prompt + caption, bos=True, eos=True)
        full = full[: self.seq_len + 1]
        ids_a, targets_a = full[:-1].clone(), full[1:].clone()
        targets_a[: len(prompt)] = PAD  # learn the caption, not the prompt
        self._rows.append(_Row(ids_a, targets_a, key[0], value[0]))
        self.stats["media_to_text"] += 1

        # Direction B: text matrix -> media codec bytes.
        caption_ids = encode_bytes(caption, bos=True).unsqueeze(0).to(self.device)
        pooled = pooled_text_embedding(self.model, caption_ids, PAD)
        text_key, text_value = self.text_projector(pooled)
        codec = matrix_to_bytes(features[0])
        prefix = b"matrix: " + caption + b"\n"
        budget = self.seq_len - len(prefix) - 2  # BOS + EOS framing
        if budget >= 64:
            payload = codec[:budget]
            full_b = encode_bytes(prefix + payload, bos=True, eos=True)
            ids_b, targets_b = full_b[:-1].clone(), full_b[1:].clone()
            targets_b[: len(prefix)] = PAD  # learn only the codec bytes + EOS
            self._rows.append(_Row(ids_b, targets_b, text_key[0], text_value[0]))
            self.stats["text_to_media"] += 1

    def feed(self, sample: Mapping[str, Any]) -> None:
        if sample["kind"] == "text":
            for ids, targets in self.packer.feed(sample["data"]):
                self._rows.append(_Row(ids, targets))
                self.stats["text_rows"] += 1
        elif sample["kind"] == "ids":
            # Pre-built id sequences (tool/edit trajectories): control ids are
            # trained by the SAME cross-entropy as every byte.
            ids = torch.as_tensor(sample["ids"], dtype=torch.long)[: self.seq_len + 1]
            self._rows.append(_Row(ids[:-1].clone(), ids[1:].clone()))
            self.stats["text_rows"] += 1
        elif sample["kind"] == "multimodal":
            self._add_multimodal(sample)
        else:
            raise ValueError(f"unknown sample kind {sample['kind']!r}")

    def ready(self, batch_size: int) -> bool:
        return len(self._rows) >= batch_size

    def pop_batch(self, batch_size: int) -> dict[str, Any] | None:
        if len(self._rows) < batch_size:
            return None
        rows = self._rows[:batch_size]
        self._rows = self._rows[batch_size:]
        ids, targets = pad_batch([row.ids for row in rows], self.seq_len)
        # pad_batch re-derives targets from ids; use OUR masked targets instead.
        targets = torch.full_like(ids, PAD)
        for index, row in enumerate(rows):
            length = min(row.targets.shape[0], self.seq_len)
            targets[index, :length] = row.targets[:length]
        memory = None
        if any(row.key is not None for row in rows):
            key_dim = self.text_projector.key.key_dim
            value_dim = self.text_projector.value.memory_dim
            keys = torch.zeros(batch_size, 1, key_dim, device=self.device)
            values = torch.zeros(batch_size, 1, value_dim, device=self.device)
            mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device)
            for index, row in enumerate(rows):
                if row.key is not None:
                    keys[index, 0] = row.key
                    values[index, 0] = row.value
                    mask[index, 0] = True
            memory = SkillBankView(keys, values, mask)
        # Merge the persistent bank so retrieval keys face real distractors.
        # (With a single slot, softmax == 1 is constant and the key projector
        # would receive zero gradient; bank slots make matching learnable.)
        if self.memory is not None and len(self.memory) > 0:
            memory = stack_views(memory, self.memory.view(), batch_size=batch_size)
        # The returned batch sees only the PRIOR committed timeline. Pending
        # writes become visible to subsequent micro-batches, never this one.
        self.commit_pending_writes()
        return {"ids": ids, "targets": targets, "memory": memory}

    def commit_pending_writes(self) -> int:
        if self.memory is None or not self._pending_writes:
            return 0
        count = 0
        for key, value, modality, metadata in self._pending_writes:
            self.memory.write_skill(key, value, modality=modality, metadata=metadata)
            count += 1
        self._pending_writes.clear()
        self.stats["ingested"] += count
        return count


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(path: str | Path, *, model: CausalByteTransformer,
                    ingestors: Mapping[str, MultimodalIngestor],
                    text_projector: TextSkillProjector, memory: DualMemory | None,
                    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
                    step: int, extra: Mapping[str, Any] | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "config": model.config.to_dict(),
        "model": model.state_dict(),
        # Only the TRAINABLE projectors are persisted; frozen encoders are
        # rebuildable from the registry and never carry learned state.
        "ingestors": {name: {"key_projector": ing.key_projector.projection.state_dict(),
                             "value_projector": ing.value_projector.projection.state_dict()}
                      for name, ing in ingestors.items()},
        "text_projector": text_projector.state_dict(),
        "memory": ({"bank": memory.procedural.state_payload()} if memory is not None else None),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": int(step),
        "rng": {"python": random.getstate(), "torch": torch.get_rng_state()},
    }
    if extra:
        payload.update(dict(extra))
    tmp = target.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)  # atomic: a crash never corrupts the latest state
    return target


def load_checkpoint(path: str | Path, *, model: CausalByteTransformer,
                    ingestors: Mapping[str, MultimodalIngestor],
                    text_projector: TextSkillProjector, memory: DualMemory | None,
                    optimizer: torch.optim.Optimizer | None = None,
                    scaler: torch.amp.GradScaler | None = None,
                    device: str | torch.device = "cpu",
                    return_payload: bool = False) -> int | tuple[int, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("not a Phase 25.1 unified checkpoint (schema 7)")
    required = {"config", "model", "ingestors", "text_projector", "memory", "optimizer", "scaler", "step"}
    if not required.issubset(payload):
        raise ValueError(f"incomplete unified checkpoint: missing {sorted(required - set(payload))}")
    if payload["config"] != model.config.to_dict():
        raise ValueError("checkpoint architecture/config differs from the requested model; "
                         "resume with the saved config instead of silently changing geometry")
    model.load_state_dict(payload["model"])
    device = next(model.parameters()).device
    for name, state in payload["ingestors"].items():
        if name not in ingestors:
            # Recreate the projector-only ingestor so optimizer param groups
            # line up exactly with the saved state (order preserved).
            key_w = state["key_projector"]["weight"]
            value_w = state["value_projector"]["weight"]
            recreated = MultimodalIngestor(None, memory_dim=value_w.shape[0],
                                           key_dim=key_w.shape[0],
                                           encoder_dim=key_w.shape[1]).to(device)
            ingestors[name] = recreated
            if optimizer is not None:
                optimizer.add_param_group(
                    {"params": [p for p in recreated.parameters() if p.requires_grad]})
        ingestors[name].key_projector.projection.load_state_dict(state["key_projector"])
        ingestors[name].value_projector.projection.load_state_dict(state["value_projector"])
    text_projector.load_state_dict(payload["text_projector"])
    if memory is not None and payload.get("memory") is not None:
        if "bank" not in payload["memory"]:
            raise ValueError("unified checkpoint requires a single bank payload")
        memory.procedural.load_payload(payload["memory"]["bank"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if "rng" in payload:
        random.setstate(payload["rng"]["python"])
        torch.set_rng_state(payload["rng"]["torch"].cpu())
    step = int(payload["step"])
    return (step, payload) if return_payload else step


DEFAULT_HF_REPO = "gabrielbosse9/airesearch2"
ARTIFACT_ALLOW = ["*.json", "*.jsonl", "*.log", "*.yaml", "*.yml", "*.txt", "*.zip"]
ARTIFACT_IGNORE = ["checkpoints/**", "*.pt", "*.tmp", "*.npz", "**/*.pt", "**/*.tmp"]


def utc_stamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def resolve_hf_token() -> str | None:
    """Read auth from the environment; never accept it in CLI argv or logs."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def redact_secrets(text: str, *extra: str | None) -> str:
    """Best-effort redaction before anything is printed or persisted."""
    values = {value for value in extra if value and len(value) >= 4}
    for key, value in os.environ.items():
        if any(word in key.upper() for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")):
            if value and len(value) >= 4:
                values.add(value)
    for value in sorted(values, key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    # Small literal scanner, not a model-output parser: keep the runtime free of
    # regex while still masking common Authorization/token query leak shapes.
    for marker in ("bearer ", "token="):
        lower = text.lower()
        pos = 0
        while True:
            index = lower.find(marker, pos)
            if index < 0:
                break
            start = index + len(marker)
            end = start
            while end < len(text) and text[end] not in " \t\r\n&\"'<>":
                end += 1
            if end > start:
                text = text[:start] + "[REDACTED]" + text[end:]
                lower = text.lower()
            pos = start + len("[REDACTED]")
    return text


def error_record(exc: BaseException, *, token: str | None = None) -> dict[str, Any]:
    return {"type": type(exc).__name__,
            "message": redact_secrets(str(exc), token)[:2000],
            "traceback": redact_secrets(traceback.format_exc(), token)[-16000:]}


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(redact_secrets(json.dumps(payload, indent=2, ensure_ascii=False, default=str)),
                   encoding="utf8")
    os.replace(tmp, target)
    return target


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hf_upload(local_path: str | Path, repo_id: str | None, path_in_repo: str, *,
              token: str | None = None, retries: int = 3, strict: bool = True,
              commit_message: str | None = None) -> dict[str, Any]:
    """Upload one artifact, retrying transient failures.

    A configured remote is a durability contract: by default exhausted retries
    raise instead of silently continuing a GPU run with no recoverable state.
    New repositories are private; an existing repository's visibility is kept.
    """
    if not repo_id:
        return {"uploaded": False, "reason": "no hf_repo configured"}
    token = token or resolve_hf_token()
    attempts = max(1, int(retries))
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True, token=token)
            result = api.upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo,
                                     repo_id=repo_id, repo_type="model", token=token,
                                     commit_message=commit_message or f"Save {path_in_repo}")
            return {"uploaded": True, "repo_id": repo_id, "path_in_repo": path_in_repo,
                    "attempts": attempt + 1, "commit": getattr(result, "oid", None),
                    "commit_url": getattr(result, "commit_url", None)}
        except Exception as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 8))
    record = {"uploaded": False, "repo_id": repo_id, "path_in_repo": path_in_repo,
              "attempts": attempts, "error": error_record(last, token=token) if last else None}
    if strict:
        raise RuntimeError(f"HF upload failed for {repo_id}/{path_in_repo}: {record['error']}")
    return record


def sync_run_folder(run_dir: str | Path, repo_id: str | None, path_in_repo: str, *,
                    token: str | None = None, retries: int = 3, strict: bool = True,
                    commit_message: str | None = None) -> dict[str, Any]:
    """Commit diagnostics together; checkpoints are deliberately excluded."""
    if not repo_id:
        return {"uploaded": False, "reason": "no hf_repo configured"}
    token = token or resolve_hf_token()
    attempts = max(1, int(retries))
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True, token=token)
            result = api.upload_folder(folder_path=str(run_dir), repo_id=repo_id,
                                       repo_type="model", path_in_repo=path_in_repo,
                                       allow_patterns=ARTIFACT_ALLOW, ignore_patterns=ARTIFACT_IGNORE,
                                       token=token, commit_message=commit_message or "Sync run diagnostics")
            return {"uploaded": True, "repo_id": repo_id, "path_in_repo": path_in_repo,
                    "attempts": attempt + 1, "commit": getattr(result, "oid", None),
                    "commit_url": getattr(result, "commit_url", None)}
        except Exception as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 8))
    record = {"uploaded": False, "repo_id": repo_id, "path_in_repo": path_in_repo,
              "attempts": attempts, "error": error_record(last, token=token) if last else None}
    if strict:
        raise RuntimeError(f"HF diagnostics sync failed for {repo_id}/{path_in_repo}: {record['error']}")
    return record


def hf_download(repo_id: str | None, path_in_repo: str, local_path: str | Path, *,
                token: str | None = None, strict: bool = True) -> dict[str, Any]:
    """Download atomically. Missing is normal; auth/network failures are not."""
    if not repo_id:
        return {"downloaded": False, "reason": "no hf_repo configured", "not_found": False}
    token = token or resolve_hf_token()
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
        downloaded = hf_hub_download(repo_id=repo_id, filename=path_in_repo,
                                     repo_type="model", token=token)
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".download")
        shutil.copyfile(downloaded, tmp)
        os.replace(tmp, target)
        return {"downloaded": True, "path": str(target), "not_found": False}
    except Exception as exc:
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
        missing = isinstance(exc, (EntryNotFoundError, RepositoryNotFoundError))
        record = {"downloaded": False, "not_found": missing,
                  "error": error_record(exc, token=token)}
        if strict and not missing:
            raise RuntimeError(f"HF download failed for {repo_id}/{path_in_repo}: {record['error']}") from exc
        return record


def verify_hf_access(repo_id: str = DEFAULT_HF_REPO, *, token: str | None = None,
                     run_dir: str | Path = "runs/colab_latest") -> dict[str, Any]:
    """Prove identity and write access with a tiny preflight before training."""
    token = token or resolve_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Add a write-scoped Colab secret and enable notebook access.")
    from huggingface_hub import HfApi
    who = HfApi(token=token).whoami(token=token)
    path = Path(run_dir) / "hf_preflight.json"
    payload = {"repo_id": repo_id, "user": who.get("name"), "checked_at": utc_stamp(),
               "purpose": "write-access preflight", "secret_included": False}
    write_json(path, payload)
    uploaded = hf_upload(path, repo_id, f"runs/preflight/{path.stem}-{utc_stamp()}.json",
                         token=token, strict=True, commit_message="Verify checkpoint write access")
    return {"repo_id": repo_id, "user": who.get("name"), "write_verified": True,
            "upload": uploaded}


def environment_manifest() -> dict[str, Any]:
    import importlib.metadata as metadata
    import platform
    packages = {}
    for name in ("torch", "torchvision", "torchaudio", "datasets", "huggingface_hub",
                 "numpy", "pillow", "pyyaml", "ddgs"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {"name": props.name, "capability": list(torch.cuda.get_device_capability(0)),
               "total_memory_bytes": props.total_memory,
               "driver": getattr(torch, "cuda_driver_version", None)}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "cuda_available": torch.cuda.is_available(),
            "gpu": gpu, "cwd": str(Path.cwd())}


def git_provenance() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  timeout=10, check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "status": git("status", "--porcelain"), "diff": git("diff", "--binary")}


def source_snapshot(target: str | Path, *, max_file_bytes: int = 4_000_000) -> Path:
    """Small code-only snapshot for later sandbox debugging; never data/cache."""
    root = Path.cwd()
    output = Path(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    excluded = {".git", ".venv", "runs", "data_cache", "user_memory", "__pycache__", "node_modules"}
    suffixes = {".py", ".yaml", ".yml", ".md", ".txt", ".ipynb"}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if not path.is_file() or excluded.intersection(rel.parts):
                continue
            if path.suffix not in suffixes or path.stat().st_size > max_file_bytes:
                continue
            if any(word in path.name.lower() for word in ("secret", "credential", ".env")):
                continue
            archive.write(path, rel.as_posix())
    return output


def make_debug_bundle(run_dir: str | Path) -> Path:
    """Zip logs/config/provenance for download; exclude multi-GB checkpoints."""
    root = Path(run_dir)
    target = root / "debug_bundle.zip"
    tmp = root / "debug_bundle.zip.tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path in (target, tmp):
                continue
            rel = path.relative_to(root)
            if "checkpoints" in rel.parts or path.suffix in (".pt", ".tmp", ".npz", ".download"):
                continue
            archive.write(path, rel.as_posix())
    os.replace(tmp, target)
    return target


class _ConsoleTee:
    def __init__(self, original: Any, handle: Any, token: str | None) -> None:
        self.original, self.handle, self.token = original, handle, token

    def write(self, text: str) -> int:
        clean = redact_secrets(text, self.token)
        self.original.write(clean)
        self.handle.write(clean)
        self.handle.flush()
        return len(text)

    def flush(self) -> None:
        self.original.flush()
        self.handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.original, "isatty", lambda: False)())



# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf8") as handle:
        handle.write(redact_secrets(json.dumps(record, ensure_ascii=False, default=str)) + "\n")
        handle.flush()


class TrainingInterrupted(RuntimeError):
    """Cooperative SIGINT/SIGTERM stop; hard VM kills still need periodic saves."""


class Trainer:
    """The single training loop: CE + confidence BCE, time-boxed, resumable."""

    def __init__(self, config: Mapping[str, Any], *, device: str = "auto",
                 samples: Iterator[tuple[str, dict[str, Any]]] | None = None,
                 hf_repo: str | None = None, resume: bool = True) -> None:
        self.config = copy.deepcopy(dict(config))
        train_cfg = dict(self.config.get("training", {}))
        ckpt_cfg = dict(self.config.get("checkpoints", {}))
        output_cfg = dict(self.config.get("output", {}))
        mem_cfg = dict(self.config.get("memory", {}))
        enc_cfg = dict(self.config.get("encoders", {}))
        seed = int(train_cfg.get("seed", 24))
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.plan = resolve_precision(self.device,
                                      train_cfg.get("requested_precision", "bfloat16"),
                                      train_cfg.get("precision", "auto"))
        self.model_config = FoundationConfig.from_mapping(config["model"])
        self.model = CausalByteTransformer(self.model_config).to(self.device)
        self.seq_len = int(train_cfg.get("seq_len", 1024))
        self.micro_batch = int(train_cfg.get("micro_batch_size", 4))
        self.grad_accum = int(train_cfg.get("grad_accum", 8))
        self.confidence_weight = float(train_cfg.get("confidence_loss_weight", 0.1))
        self.entropy_threshold = float(train_cfg.get("entropy_threshold", 0.35))
        self.decay_reg_weight = float(train_cfg.get("decay_reg_weight", 0.01))
        self.decay_margin = float(mem_cfg.get("decay_margin", 0.25))
        if self.decay_reg_weight < 0 or self.decay_margin <= 0:
            raise ValueError("decay_reg_weight must be >= 0 and decay_margin must be > 0")
        vram_cfg = dict(config.get("vram", {}))
        self._frozen_encoder_dtype = str(vram_cfg.get("frozen_encoder_dtype", "float32")).lower()
        if self._frozen_encoder_dtype not in ("float32", "float16", "bfloat16", "auto"):
            raise ValueError("frozen_encoder_dtype must be float32, float16, bfloat16, or auto")
        if self.device.type != "cuda":
            self._frozen_encoder_dtype = "float32"
        elif self._frozen_encoder_dtype == "auto":
            self._frozen_encoder_dtype = self.plan.resolved
        require_checkpoint = bool(vram_cfg.get("require_gradient_checkpointing",
                                               self.model_config.attention_type == "sliding_window"))
        if require_checkpoint and not self.model_config.gradient_checkpointing:
            raise ValueError("this preset requires gradient_checkpointing=True for all blocks")
        self.model.set_gradient_checkpointing(self.model_config.gradient_checkpointing)
        self.vram_report: dict[str, Any] | None = None
        self.log_every = max(1, int(train_cfg.get("log_every", 10)))
        self.checkpoint_every = int(train_cfg.get("checkpoint_every", 100))
        self.checkpoint_minutes = float(train_cfg.get("checkpoint_minutes", 10))
        self.upload_minutes = float(ckpt_cfg.get("upload_minutes", 5))
        self.upload_retries = max(1, int(ckpt_cfg.get("upload_retries", 3)))
        self.strict_uploads = bool(ckpt_cfg.get("fail_on_upload_error", True))
        self.keep_local_previous = bool(ckpt_cfg.get("keep_local_previous", True))
        if self.checkpoint_every < 0 or self.checkpoint_minutes < 0 or self.upload_minutes < 0:
            raise ValueError("checkpoint/upload cadences must be non-negative (0 disables)")
        self.hf_repo = hf_repo or ckpt_cfg.get("hf_repo") or os.environ.get("HF_REPO")
        self.hf_path = ckpt_cfg.get("hf_path_in_repo", "checkpoints/checkpoint_latest.pt")
        self._hf_token = resolve_hf_token()
        self.local_dir = Path(ckpt_cfg.get("local_dir", "runs/checkpoints"))
        self.run_dir = Path(output_cfg.get("run_dir", self.local_dir.parent))
        self.metrics_path = Path(output_cfg.get("metrics_jsonl", self.run_dir / "metrics.jsonl"))
        self.events_path = self.run_dir / "events.jsonl"
        self.console_path = self.run_dir / "console.log"
        self.run_id = str(output_cfg.get("run_id") or os.environ.get("BYTECORE_RUN_ID")
                          or f"run-{utc_stamp()}-{uuid.uuid4().hex[:8]}")
        self.session_id = f"session-{utc_stamp()}-{uuid.uuid4().hex[:8]}"
        self.remote_run_path = f"runs/{self.run_id}/{self.session_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.config.setdefault("checkpoints", {})["hf_repo"] = self.hf_repo
        self.config.setdefault("output", {})["run_dir"] = str(self.run_dir)
        self.config["output"]["run_id"] = self.run_id
        self._remote_ready = False
        self._last_upload_time = time.monotonic()
        self._console_handle = None
        self._total_tokens = 0
        self._completed_micros = 0
        self._training_seconds = 0.0
        self._last_metrics: dict[str, Any] = {}
        self._resume_info: dict[str, Any] = {"resumed": False}
        self._artifact_errors: list[dict[str, Any]] = []
        self._signal_received: str | None = None
        (self.run_dir / "config.resolved.yaml").write_text(
            yaml.safe_dump(self.config, sort_keys=False, allow_unicode=True), encoding="utf8")
        write_json(self.run_dir / "environment.json", environment_manifest())
        write_json(self.run_dir / "provenance.json", git_provenance())
        source_snapshot(self.run_dir / "source_snapshot.zip")

        # Multimodal ingestion paths (frozen encoders are built lazily: the
        # CPU/smoke path uses pre-computed features and never downloads).
        self.ingestors: dict[str, MultimodalIngestor] = {}
        self.text_projector = TextSkillProjector(
            self.model_config.d_model,
            mem_cfg.get("value_dim", self.model_config.memory_dim or 8192),
            mem_cfg.get("key_dim", self.model_config.d_model)).to(self.device)
        self._enc_cfg = enc_cfg
        self._mem_cfg = mem_cfg
        capacity = mem_cfg.get("capacity")
        self.memory = DualMemory(
            procedural_capacity=int(mem_cfg.get("procedural_capacity", 1024)),
            episodic_capacity=int(mem_cfg.get("episodic_capacity", 1024)),
            key_dim=int(mem_cfg.get("key_dim", self.model_config.d_model)),
            value_dim=int(mem_cfg.get("value_dim", self.model_config.memory_dim or 8192)),
            capacity=int(capacity) if capacity is not None else None,
        ).to(self.device)
        self.memory.procedural.eviction_events = deque(
            self.memory.procedural.eviction_events,
            maxlen=int(mem_cfg.get("max_eviction_events", 256)))

        self.samples = samples
        self.mixture: MixtureSampler | None = (
            samples if isinstance(samples, MixtureSampler) else None)
        self.builder: BatchBuilder | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scaler = make_grad_scaler(self.plan)
        self.step = 0
        self._micro = 0
        self._last_ckpt_time = time.monotonic()
        self._start_time = time.monotonic()
        self._resume = resume
        self.history: list[dict[str, Any]] = []

    # -- lazy pieces ------------------------------------------------------
    def _prepare_frozen_encoder(self, encoder: Any) -> Any:
        encoder = encoder.to(self.device)
        if self.device.type == "cuda" and self._frozen_encoder_dtype != "float32":
            encoder = encoder.to(dtype=getattr(torch, self._frozen_encoder_dtype))
        return encoder

    def ensure_ingestor(self, modality: str, *, feature_dim: int | None = None
                        ) -> MultimodalIngestor | None:
        if modality in self.ingestors:
            existing = self.ingestors[modality]
            name = self._enc_cfg.get(modality)
            if existing.encoder is None and name:
                # Upgrade a resumed projector-only ingestor with its frozen
                # encoder so raw media can keep flowing.
                existing.encoder = self._prepare_frozen_encoder(build_encoder(name, pretrained=True))
            return existing
        name = self._enc_cfg.get(modality)
        memory_dim = int(self._mem_cfg.get("value_dim", self.model_config.memory_dim or 8192))
        key_dim = int(self._mem_cfg.get("key_dim", self.model_config.d_model))
        if name:
            encoder = self._prepare_frozen_encoder(build_encoder(name, pretrained=True))
            ingestor = MultimodalIngestor(encoder, memory_dim=memory_dim, key_dim=key_dim)
        elif feature_dim is not None:
            # Matrix-agnostic path: pre-computed features, projector only.
            ingestor = MultimodalIngestor(None, memory_dim=memory_dim, key_dim=key_dim,
                                          encoder_dim=feature_dim)
        else:
            return None
        ingestor = ingestor.to(self.device)
        self.ingestors[modality] = ingestor
        if self.optimizer is not None:
            self.optimizer.add_param_group(
                {"params": [p for p in ingestor.parameters() if p.requires_grad]})
        return ingestor

    def _build_optimizer(self) -> torch.optim.Optimizer:
        train_cfg = self.config.get("training", {})
        params = [p for module in [self.model, self.text_projector, self.memory, *self.ingestors.values()]
                  for p in module.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=float(train_cfg.get("learning_rate", 3e-4)),
                                 weight_decay=float(train_cfg.get("weight_decay", 0.1)),
                                 betas=tuple(train_cfg.get("adam_betas", (0.9, 0.95))))

    def _lr_at(self, step: int, max_steps: int) -> float:
        train_cfg = self.config.get("training", {})
        base = float(train_cfg.get("learning_rate", 3e-4))
        warmup = int(train_cfg.get("warmup_steps", 500))
        if step < warmup:
            return base * (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(max_steps - warmup, 1)
        return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    # -- batches ----------------------------------------------------------
    def _next_batch(self) -> dict[str, Any] | None:
        """Pull the next micro-batch from the stream (single process).

        This is the custom iterable wrapper that replaces a PyTorch
        DataLoader: ``num_workers=0`` semantics by construction, so HF
        streaming never deadlocks in worker subprocesses and nothing is
        prefetched to disk. ``StopIteration`` ends the run cleanly.
        """
        assert self.builder is not None and self.samples is not None
        while not self.builder.ready(self.micro_batch):
            try:
                _, sample = next(self.samples)
            except StopIteration:
                return None
            if sample["kind"] == "multimodal":
                feature_dim = None
                if "features" in sample:
                    feature_dim = int(torch.as_tensor(sample["features"]).shape[-1])
                self.ensure_ingestor(sample["modality"], feature_dim=feature_dim)
                # BatchBuilder holds the ingestor dict by reference at build
                # time; refresh so newly created projectors are visible.
                self.builder.ingestors = self.ingestors
            self.builder.feed(sample)
        batch = self.builder.pop_batch(self.micro_batch)
        if batch is not None:
            batch["ids"] = batch["ids"].to(self.device)
            batch["targets"] = batch["targets"].to(self.device)
            if batch["memory"] is not None:
                batch["memory"] = batch["memory"].to(self.device)
        return batch

    # -- VRAM ---------------------------------------------------------------
    def _probe_memory_view(self, batch: int) -> SkillBankView:
        """Full detached bank + differentiable local slots from every projector."""
        key_dim = int(self._mem_cfg.get("key_dim", self.model_config.d_model))
        value_dim = int(self._mem_cfg.get("value_dim", self.model_config.memory_dim or 8192))
        capacity = self.memory.procedural.capacity
        keys = torch.randn(capacity, key_dim, device=self.device)
        values = torch.randn(capacity, value_dim, device=self.device)
        mask = torch.ones(capacity, dtype=torch.bool, device=self.device)
        global_view = SkillBankView(keys, values, mask)
        local_keys, local_values = [], []
        pooled = torch.randn(batch, self.model_config.d_model, device=self.device)
        key, value = self.text_projector(pooled)
        local_keys.append(key), local_values.append(value)
        for ingestor in self.ingestors.values():
            width = ingestor.key_projector.projection.in_features
            features = torch.randn(batch, width, device=self.device)
            key, value = ingestor.project_features(features)
            local_keys.append(key), local_values.append(value)
        local = SkillBankView(torch.stack(local_keys, dim=1), torch.stack(local_values, dim=1),
                              torch.ones(batch, len(local_keys), dtype=torch.bool, device=self.device))
        return stack_views(global_view, local, batch_size=batch)

    def probe_vram(self) -> dict[str, Any]:
        """MEASURED representative forward+backward+Adam-state peak on CUDA.

        Includes the full unified bank, batch-local projector slots, confidence
        path, policy loss, all-block checkpointing, and worst-case AdamW moment
        allocation. Existing optimizer moments are temporarily detached from
        the optimizer object so a resumed probe does not double-count them or
        corrupt them (lr=0 step). CPU reports measured=False, never a fake peak.
        """
        if self.device.type != "cuda":
            return {"measured": False, "simulated": False, "reason": "CPU device; run --simulate-t4 for an estimate"}
        if self.optimizer is None:
            self.optimizer = self._build_optimizer()
        # Load configured frozen backbones/projectors so their persistent weights
        # are present. Encoder transient activations occur before the transformer
        # backward and are covered separately by the simulator safety reserve.
        for modality, name in self._enc_cfg.items():
            if name:
                self.ensure_ingestor(modality)
        assert self.optimizer is not None
        old_state = self.optimizer.state
        old_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.optimizer.state = defaultdict(dict)
        for group in self.optimizer.param_groups:
            group["lr"] = 0.0
        probe_scaler = make_grad_scaler(self.plan)
        self.model.zero_grad(set_to_none=True)
        self.text_projector.zero_grad(set_to_none=True)
        self.memory.zero_grad(set_to_none=True)
        for ingestor in self.ingestors.values():
            ingestor.zero_grad(set_to_none=True)
        bank = self.memory.procedural
        saved_events = list(bank.eviction_events)
        rng_python = random.getstate()
        rng_torch = torch.get_rng_state()
        rng_cuda = [state.cpu() for state in torch.cuda.get_rng_state_all()]
        bank.eviction_events.clear()
        retained = torch.arange(1, min(bank.capacity, 9), device=self.device)
        ages = torch.linspace(0.0, 1.0, bank.capacity, device=self.device)
        bank.eviction_events.append(EvictionEvent(0, retained, ages, 1.0, False))
        # Verify the actual T4-safe backend before the expensive probe. A
        # silent math fallback would invalidate the memory estimate.
        backend_width = self.model_config.d_model // self.model_config.n_heads
        probe_q = torch.randn(1, 1, 128, backend_width, device=self.device, dtype=self.plan.dtype)
        probe_k = torch.randn(1, 1, 192, backend_width, device=self.device, dtype=self.plan.dtype)
        chunk_mask = torch.ones(128, 192, device=self.device, dtype=torch.bool)
        cross_mask = torch.ones(1, 1, 1, 17, device=self.device, dtype=torch.bool)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            F.scaled_dot_product_attention(probe_q, probe_q, probe_q)
            F.scaled_dot_product_attention(probe_q, probe_k, probe_k,
                                           attn_mask=chunk_mask[None, None], is_causal=False)
            F.scaled_dot_product_attention(probe_q, probe_q[:, :, :17], probe_q[:, :, :17],
                                           attn_mask=cross_mask, is_causal=False)
        del probe_q, probe_k, chunk_mask, cross_mask
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        ids = memory_view = loss = total = policy = None
        try:
            ids = torch.randint(0, 256, (self.micro_batch, self.seq_len), device=self.device)
            memory_view = self._probe_memory_view(self.micro_batch)
            with self.plan.autocast_context():
                loss, metrics = self.model.loss(
                    ids, memory=memory_view, confidence_weight=self.confidence_weight,
                    entropy_threshold=self.entropy_threshold, return_metrics=True)
                policy = self.memory.policy_loss(margin=self.decay_margin)
                total = loss + self.decay_reg_weight * policy.to(loss.dtype)
            probe_scaler.scale(total / max(1, self.grad_accum)).backward()
            # Allocate moments even for parameters whose synthetic probe path did
            # not touch them; this is the worst-case optimizer residency.
            for group in self.optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.requires_grad and parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
            probe_scaler.unscale_(self.optimizer)
            probe_scaler.step(self.optimizer)  # lr=0: weights unchanged, moments allocated
            probe_scaler.update()
            peak_gb = torch.cuda.max_memory_allocated(self.device) / 1e9
            return {"measured": True, "simulated": False, "peak_gb": round(peak_gb, 3),
                    "micro_batch": self.micro_batch, "seq_len": self.seq_len,
                    "attention_type": self.model_config.attention_type,
                    "window_size": self.model_config.window_size,
                    "sliding_chunk_size": self.model_config.sliding_chunk_size,
                    "sdpa_backend_verified": "memory_efficient (self + chunk-local mask + cross mask)",
                    "optimizer_state_allocated": True,
                    "paths": ["ce", "confidence", "memory_cross_attention", "policy", "adamw"]}
        finally:
            bank.eviction_events.clear()
            bank.eviction_events.extend(saved_events)
            self.optimizer.state = old_state
            for group, lr in zip(self.optimizer.param_groups, old_lrs):
                group["lr"] = lr
            self.model.zero_grad(set_to_none=True)
            self.text_projector.zero_grad(set_to_none=True)
            self.memory.zero_grad(set_to_none=True)
            for ingestor in self.ingestors.values():
                ingestor.zero_grad(set_to_none=True)
            del ids, memory_view, loss, total, policy
            random.setstate(rng_python)
            torch.set_rng_state(rng_torch)
            for index, rng_state in enumerate(rng_cuda):
                torch.cuda.set_rng_state(rng_state, device=index)
            torch.cuda.empty_cache()

    def autotune_batch(self, target_max_gb: float = 13.5) -> dict[str, Any]:
        """Probe and halve micro-batch; fail closed above the hard ceiling.

        The sliding window is part of the architecture contract (W=1024), so
        autotune does not silently shrink long-range connectivity. Chunk-local
        masks already avoid quadratic allocation; if batch 1 still exceeds the
        ceiling, the run must stop rather than claim a fit.
        """
        if target_max_gb <= 0:
            raise ValueError("target_max_gb must be positive")
        probe = self.probe_vram()
        while probe.get("measured") and probe["peak_gb"] > target_max_gb and self.micro_batch > 1:
            self.micro_batch = max(1, self.micro_batch // 2)
            probe = self.probe_vram()
        if probe.get("measured") and probe["peak_gb"] > target_max_gb:
            raise MemoryError(
                f"measured T4 peak {probe['peak_gb']:.3f} GB exceeds {target_max_gb:.3f} GB "
                "at micro_batch=1; refusing to start. Use a smaller model/context, "
                "optimizer offload, or a lower-memory attention kernel.")
        if not probe.get("measured"):
            probe["simulated"] = simulate_t4_vram(
                self.config, micro_batch=self.micro_batch, seq_len=self.seq_len,
                window_size=self.model_config.window_size,
                chunk_size=self.model_config.sliding_chunk_size)
        self.vram_report = probe
        return probe

    # -- checkpoint / resume / diagnostics ---------------------------------
    def _cuda_memory(self) -> dict[str, float] | None:
        if self.device.type != "cuda":
            return None
        return {"allocated_gb": round(torch.cuda.memory_allocated(self.device) / 1e9, 4),
                "reserved_gb": round(torch.cuda.memory_reserved(self.device) / 1e9, 4),
                "max_allocated_gb": round(torch.cuda.max_memory_allocated(self.device) / 1e9, 4)}

    def _runtime_state(self) -> dict[str, Any]:
        sampler_state = None
        if self.mixture is not None:
            sampler_state = {"counts": dict(self.mixture.counts),
                             "weights": self.mixture.weights,
                             "failures": dict(self.mixture.failures),
                             "queue": list(self.mixture._queue),
                             "rng_python": self.mixture._rng.getstate()}
        return {
            "run_id": self.run_id, "session_id": self.session_id,
            "total_tokens": self._total_tokens,
            "completed_micro_batches": self._completed_micros,
            "incomplete_micro_batches": self._micro % self.grad_accum,
            "training_seconds": round(self._training_seconds, 3),
            "last_metrics": self._last_metrics,
            "grad_accum": self.grad_accum, "micro_batch": self.micro_batch,
            "seq_len": self.seq_len, "sampler": sampler_state,
            "cuda_rng": ([state.cpu() for state in torch.cuda.get_rng_state_all()]
                         if self.device.type == "cuda" else None),
            "data_resume": {
                "exact_stream_cursor": False,
                "policy": "Resume model/optimizer/scaler/bank/RNG and seeded mixture state; "
                          "live HF iterators/shuffle buffers are reopened, not rewound to an "
                          "exact document cursor. Incomplete gradient accumulation is discarded.",
            },
        }

    def _restore_runtime_state(self, payload: Mapping[str, Any]) -> None:
        state = dict(payload.get("training_state") or {})
        self._total_tokens = int(state.get("total_tokens", 0))
        self._completed_micros = int(state.get("completed_micro_batches", self.step * self.grad_accum))
        self._training_seconds = float(state.get("training_seconds", 0.0))
        self._last_metrics = dict(state.get("last_metrics") or {})
        self._micro = 0
        self._resume_info = {"resumed": True, "step": self.step,
                             "previous_run_id": state.get("run_id"),
                             "previous_session_id": state.get("session_id"),
                             "discarded_incomplete_micro_batches": int(state.get("incomplete_micro_batches", 0)),
                             "exact_stream_cursor": False}
        if self.mixture is not None and state.get("sampler") is not None:
            saved = state["sampler"]
            # Counts/failures are diagnostic; RNG/queue restore the seeded quota
            # sequence when the same live sources remain available.
            self.mixture.counts.update(saved.get("counts") or {})
            self.mixture.failures.update(saved.get("failures") or {})
            self.mixture._queue = deque(saved.get("queue") or [])
            if saved.get("rng_python") is not None:
                self.mixture._rng.setstate(saved["rng_python"])
        if self.device.type == "cuda" and state.get("cuda_rng"):
            for index, rng_state in enumerate(state["cuda_rng"]):
                if index < torch.cuda.device_count():
                    torch.cuda.set_rng_state(rng_state.cpu(), device=index)
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def _artifact_manifest(self, reason: str) -> dict[str, Any]:
        return {"run_id": self.run_id, "session_id": self.session_id, "reason": reason,
                "updated_at": utc_stamp(), "step": self.step, "hf_repo": self.hf_repo,
                "remote_run_path": self.remote_run_path, "checkpoint_path": self.hf_path,
                "total_tokens": self._total_tokens, "training_seconds": self._training_seconds,
                "resume": self._resume_info, "artifact_errors": self._artifact_errors[-20:],
                "secret_included": False}

    def _write_diagnostics(self, reason: str) -> None:
        write_json(self.run_dir / "manifest.json", self._artifact_manifest(reason))
        write_json(self.run_dir / "diagnostics.json", {
            "reason": reason, "step": self.step, "cuda_memory": self._cuda_memory(),
            "vram_probe": self.vram_report, "precision": self.plan.as_dict(),
            "batch_stats": dict(self.builder.stats) if self.builder else {},
            "memory_slots": len(self.memory), "resume": self._resume_info,
            "data_mix": mixture_snapshot(self.mixture) if self.mixture is not None else None,
            "runtime_state_json_safe": {key: value for key, value in self._runtime_state().items()
                                        if key not in ("cuda_rng",)},
        })

    def _prepare_remote(self) -> None:
        if not self.hf_repo or self._remote_ready:
            return
        try:
            access = verify_hf_access(self.hf_repo, token=self._hf_token, run_dir=self.run_dir)
            write_json(self.run_dir / "hf_access.json", access)
            self._sync_artifacts("startup", strict=self.strict_uploads)
            self._remote_ready = True
        except Exception as exc:
            record = error_record(exc, token=self._hf_token)
            self._artifact_errors.append({"stage": "startup", "error": record})
            append_jsonl(self.run_dir / "artifact_errors.jsonl", record)
            if self.strict_uploads:
                raise
            print("WARNING: remote persistence unavailable; local-only run explicitly allowed", flush=True)

    def _sync_artifacts(self, reason: str, *, strict: bool = False) -> dict[str, Any]:
        if self._console_handle is not None:
            self._console_handle.flush()
        self._write_diagnostics(reason)
        if reason in ("final", "crash", "interrupted"):
            make_debug_bundle(self.run_dir)
        if not self.hf_repo:
            return {"uploaded": False, "reason": "local run; no hf_repo"}
        result = sync_run_folder(self.run_dir, self.hf_repo, self.remote_run_path,
                                 token=self._hf_token, retries=self.upload_retries,
                                 strict=strict, commit_message=f"{reason}: step {self.step}")
        pointer = write_json(self.run_dir / "remote_pointer.json", self._artifact_manifest(reason))
        pointer_result = hf_upload(pointer, self.hf_repo, "runs/latest.json",
                                   token=self._hf_token, retries=self.upload_retries,
                                   strict=strict, commit_message=f"Latest run pointer: {self.run_id}")
        result["pointer"] = pointer_result
        if not result.get("uploaded") or not pointer_result.get("uploaded"):
            record = {"stage": reason, "result": result}
            self._artifact_errors.append(record)
            append_jsonl(self.run_dir / "artifact_errors.jsonl", record)
            print(f"WARNING: diagnostics sync incomplete ({reason}); local files retained", flush=True)
        self._last_upload_time = time.monotonic()
        return result

    def _save(self, *, upload: bool = True, reason: str = "periodic",
              final: bool = False, emergency: bool = False) -> dict[str, Any]:
        assert self.optimizer is not None
        path = self.local_dir / "checkpoint_latest.pt"
        previous = self.local_dir / "checkpoint_previous.pt"
        if path.exists() and self.keep_local_previous:
            shutil.copy2(path, previous)
        # Checkpoints represent committed optimizer boundaries. Any incomplete
        # accumulation is explicitly discarded on resume, never half-applied.
        self.optimizer.zero_grad(set_to_none=True)
        save_checkpoint(path, model=self.model, ingestors=self.ingestors,
                        text_projector=self.text_projector, memory=self.memory,
                        optimizer=self.optimizer, scaler=self.scaler, step=self.step,
                        extra={"precision": self.plan.as_dict(),
                               "training_state": self._runtime_state()})
        record: dict[str, Any] = {"step": self.step, "checkpoint": str(path),
                                  "reason": reason, "bytes": path.stat().st_size,
                                  "sha256": file_sha256(path), "created_at": utc_stamp()}
        write_json(self.run_dir / "checkpoint_manifest.json", record)
        if upload and self.hf_repo:
            strict = self.strict_uploads and not emergency
            record["hf"] = hf_upload(path, self.hf_repo, self.hf_path,
                                     token=self._hf_token, retries=self.upload_retries,
                                     strict=strict, commit_message=f"Checkpoint step {self.step} ({reason})")
            if final:
                record["hf_final"] = hf_upload(path, self.hf_repo, "checkpoints/checkpoint_final.pt",
                                               token=self._hf_token, retries=self.upload_retries,
                                               strict=strict, commit_message=f"Final checkpoint step {self.step}")
            write_json(self.run_dir / "checkpoint_manifest.json", record)
            record["diagnostics"] = self._sync_artifacts(reason, strict=strict)
        self._last_ckpt_time = time.monotonic()
        append_jsonl(self.events_path, {"event": "checkpoint", **record})
        return record

    def _maybe_resume(self) -> None:
        if not self._resume or self.optimizer is None:
            return
        path = self.local_dir / "checkpoint_latest.pt"
        previous = self.local_dir / "checkpoint_previous.pt"
        candidates = [path, previous]
        if not path.exists() and self.hf_repo:
            result = hf_download(self.hf_repo, self.hf_path, path,
                                 token=self._hf_token, strict=True)
            self._resume_info["remote_download"] = result
            if not result.get("downloaded") and not result.get("not_found"):
                raise RuntimeError(f"cannot establish resume state: {result}")
        errors = []
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                self.step, payload = load_checkpoint(
                    candidate, model=self.model, ingestors=self.ingestors,
                    text_projector=self.text_projector, memory=self.memory,
                    optimizer=self.optimizer, scaler=self.scaler,
                    device=self.device, return_payload=True)
                self._restore_runtime_state(payload)
                self._resume_info["source"] = str(candidate)
                append_jsonl(self.events_path, {"event": "resume", **self._resume_info})
                return
            except Exception as exc:
                errors.append(error_record(exc, token=self._hf_token))
        if errors:
            raise RuntimeError(f"local checkpoints are unreadable/incompatible: {errors}")

    def _evaluate_final(self) -> dict[str, Any]:
        prompts = {"ptbr": "Olá, eu sou um modelo de linguagem e",
                   "code": "def fibonacci(n):"}
        result: dict[str, Any] = {"step": self.step, "prompts": {}}
        self.model.eval()
        try:
            view = self.memory.view()
            for name, prompt in prompts.items():
                prefix = encode_bytes(prompt.encode("utf8"), bos=True, eos=False).to(self.device)
                with torch.no_grad():
                    ids = self.model.generate(prefix, max_new_bytes=100, temperature=0.0, memory=view)
                    _, info = self.model(prefix.unsqueeze(0), memory=view, return_info=True)
                continuation = decode_ids(ids).decode("utf8", "replace")
                result["prompts"][name] = {"prompt": prompt, "continuation": continuation,
                                           "full_text": prompt + continuation,
                                           "generated_bytes": int(ids.numel()),
                                           "adaptive": {key: info.get(key) for key in
                                                        ("steps_run", "confidence", "thought_steps", "thought_norm")}}
            try:
                ids, stats = self.model.generate_with_tools(
                    encode_bytes(b"search: ").to(self.device), make_search_fn("mock"),
                    max_new_bytes=96)
                result["tool_demo"] = {"generated_ids": int(ids.numel()), **stats,
                                       "transcript": decode_ids(ids, keep_controls=True)[:300].decode("utf8", "replace")}
            except Exception as exc:
                result["tool_error"] = error_record(exc, token=self._hf_token)
        except Exception as exc:
            result["error"] = error_record(exc, token=self._hf_token)
        finally:
            self.model.train()
        write_json(self.run_dir / "evaluation.json", result)
        return result

    def _handle_crash(self, exc: BaseException) -> None:
        record = error_record(exc, token=self._hf_token)
        traceback.clear_frames(exc.__traceback__)
        write_json(self.run_dir / "failure.json", {"step": self.step, "run_id": self.run_id,
                                                   "session_id": self.session_id, "error": record})
        append_jsonl(self.events_path, {"event": "failure", "step": self.step, "error": record})
        print(redact_secrets(traceback.format_exc(), self._hf_token), flush=True)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            if self.optimizer is not None:
                crash = self._save(upload=bool(self.hf_repo), reason="crash", emergency=True)
                write_json(self.run_dir / "crash_checkpoint.json", crash)
        except Exception as save_exc:
            append_jsonl(self.run_dir / "artifact_errors.jsonl",
                         {"stage": "crash_checkpoint", "error": error_record(save_exc, token=self._hf_token)})
        try:
            self._sync_artifacts("crash", strict=False)
        except Exception as sync_exc:
            append_jsonl(self.run_dir / "artifact_errors.jsonl",
                         {"stage": "crash_sync", "error": error_record(sync_exc, token=self._hf_token)})

    @contextlib.contextmanager
    def _captured_run(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        previous_handlers = {}
        def interrupt(signum, _frame):
            raise TrainingInterrupted(f"received signal {signum}")
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                previous_handlers[signum] = signal.signal(signum, interrupt)
            except ValueError:
                pass  # non-main thread: KeyboardInterrupt still works interactively
        with open(self.console_path, "a", encoding="utf8") as handle:
            self._console_handle = handle
            stdout = _ConsoleTee(sys.stdout, handle, self._hf_token)
            stderr = _ConsoleTee(sys.stderr, handle, self._hf_token)
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    yield
            finally:
                self._console_handle = None
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)

    def run(self, *, max_steps: int = 1000, max_minutes: float | None = None,
            checkpoint_minutes: float | None = None) -> dict[str, Any]:
        with self._captured_run():
            try:
                return self._run_impl(max_steps=max_steps, max_minutes=max_minutes,
                                      checkpoint_minutes=checkpoint_minutes)
            except (Exception, KeyboardInterrupt) as exc:
                self._handle_crash(exc)
                raise

    # -- the loop -----------------------------------------------------------
    def _run_impl(self, *, max_steps: int = 1000, max_minutes: float | None = None,
                  checkpoint_minutes: float | None = None) -> dict[str, Any]:
        if self.samples is None:
            data_cfg = self.config.get("data", {})
            self.mixture = build_mixture(data_cfg, seed=int(
                self.config.get("training", {}).get("seed", 24)))
            self.samples = iter(self.mixture)
        # Phase 24.2 streaming contract: the sample source must be a len-less
        # iterator. A sized object (list, map-style Dataset, ...) means
        # epoch-style training over materialised data -- reject it loudly
        # instead of silently downloading a corpus to the Colab disk.
        if hasattr(self.samples, "__len__"):
            raise TypeError(
                "Trainer samples must be a streaming iterator without __len__ "
                f"(got {type(self.samples).__name__}); epoch-based training over "
                "sized datasets is forbidden (Phase 24.2 strict streaming)")
        tool_rate = float(self.config.get("training", {}).get("tool_trajectory_rate", 0.02))
        if tool_rate > 0:
            self.samples = inject_tool_trajectories(self.samples, rate=tool_rate,
                                                    seed=int(self.config.get(
                                                        "training", {}).get("seed", 24)))
        if checkpoint_minutes is not None:
            self.checkpoint_minutes = float(checkpoint_minutes)
        # Smoke/CPU path: pre-computed features need no frozen backbone.
        self.builder = BatchBuilder(self.model, self.ingestors, self.text_projector,
                                    self.seq_len, memory=self.memory,
                                    ingest_to_memory=bool(self.config.get("training", {}).get(
                                        "ingest_to_memory", False)))
        if self.optimizer is None:
            self.optimizer = self._build_optimizer()
        self._maybe_resume()
        self._prepare_remote()
        vram = self.autotune_batch(float(self.config.get("vram", {}).get("target_max_gb", 13.5)))
        write_json(self.run_dir / "vram_probe.json", vram)
        self.model.train()
        self._start_time = time.monotonic()
        self._last_ckpt_time = time.monotonic()
        self._last_upload_time = time.monotonic()
        deadline = None if max_minutes is None else self._start_time + float(max_minutes) * 60.0
        last_metrics: dict[str, Any] = dict(self._last_metrics)
        stop_reason = "max_steps"
        pending_tokens = 0
        accum_sums = defaultdict(float)
        accum_weight = 0
        accum_micros = 0
        # STEP-BASED loop: count optimizer boundaries, never epochs or len(data).
        while self.step < max_steps:
            if deadline is not None and time.monotonic() >= deadline:
                stop_reason = "time_budget"
                break
            batch = self._next_batch()
            if batch is None:
                stop_reason = "data_exhausted"
                break
            with self.plan.autocast_context():
                loss, metrics = self.model.loss(
                    batch["ids"], batch["targets"], batch["memory"],
                    confidence_weight=self.confidence_weight,
                    entropy_threshold=self.entropy_threshold, return_metrics=True)
                policy = self.memory.policy_loss(margin=self.decay_margin)
                total = loss + self.decay_reg_weight * policy.to(loss.dtype)
            if not bool(torch.isfinite(total).item()):
                raise FloatingPointError(f"non-finite loss at step {self.step}: {metrics}")
            metrics["policy_loss"] = float(policy.detach().item())
            metrics["total"] = float(total.detach().item())
            valid = int((batch["targets"] != PAD).sum().item())
            weight = valid
            accum_weight += weight
            accum_micros += 1
            for key in ("ce", "confidence_bce", "total", "policy_loss"):
                accum_sums[key] += float(metrics.get(key) or 0.0) * weight
            if metrics.get("mean_entropy") is not None:
                accum_sums["mean_entropy"] += float(metrics["mean_entropy"]) * weight
            self.scaler.scale(total / self.grad_accum).backward()
            self.memory.procedural.clear_policy_events()
            self._micro += 1
            self._completed_micros += 1
            pending_tokens += valid
            if self._micro % self.grad_accum == 0:
                lr = self._lr_at(self.step, max_steps)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group["params"]],
                    float(self.config.get("training", {}).get("max_grad_norm", 1.0)))
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                self._total_tokens += pending_tokens
                pending_tokens = 0
                denominator = max(1, accum_weight)
                averaged = {key: value / denominator for key, value in accum_sums.items()}
                if "mean_entropy" not in averaged:
                    averaged["mean_entropy"] = None
                last_metrics = {**metrics, **averaged,
                                "steps_run": int(metrics.get("steps_run") or 0),
                                "thought_norm": float(metrics.get("thought_norm") or 0.0)}
                self._last_metrics = last_metrics
                event = {"event": "optimizer_step", "step": self.step,
                         "tokens": self._total_tokens, "micro_batches": self._completed_micros,
                         "elapsed_s": round(time.monotonic() - self._start_time, 3),
                         "lr": lr, "grad_norm": float(grad_norm.detach().item()),
                         "scaler_scale": self.scaler.get_scale(),
                         "optimizer_update_skipped": bool(self.scaler.is_enabled() and self.scaler.get_scale() < scale_before),
                         "accumulated_micro_batches": accum_micros,
                         "accumulated_target_bytes": accum_weight, **last_metrics}
                if self.device.type == "cuda":
                    event["vram_gb"] = round(torch.cuda.max_memory_allocated(self.device) / 1e9, 3)
                append_jsonl(self.events_path, event)
                accum_sums.clear()
                accum_weight = 0
                accum_micros = 0
                if self.step % self.log_every == 0 or self.step == 1:
                    record = {key: value for key, value in event.items() if key != "event"}
                    if self.mixture is not None:
                        record["data_mix"] = mixture_snapshot(self.mixture)["observed_share"]
                    self.history.append(record)
                    append_jsonl(self.metrics_path, record)
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                due_time = self.checkpoint_minutes > 0 and time.monotonic() - self._last_ckpt_time >= self.checkpoint_minutes * 60.0
                due_steps = self.checkpoint_every > 0 and self.step % self.checkpoint_every == 0
                if due_time or due_steps:
                    record = self._save(reason="periodic")
                    self.history.append({"checkpoint_event": record})
                elif self.upload_minutes > 0 and time.monotonic() - self._last_upload_time >= self.upload_minutes * 60.0:
                    self._sync_artifacts("telemetry", strict=False)
        self._training_seconds += time.monotonic() - self._start_time
        final = self._save(reason="final", final=True)
        evaluation = None
        if bool(self.config.get("output", {}).get("evaluate_final", bool(self.hf_repo))):
            evaluation = self._evaluate_final()
            self._sync_artifacts("evaluation", strict=self.strict_uploads)
        data_mix = mixture_snapshot(self.mixture) if self.mixture is not None else None
        summary = {"steps": self.step, "tokens": self._total_tokens, "stop_reason": stop_reason,
                   "data_mix": data_mix, "run_id": self.run_id, "session_id": self.session_id,
                   "run_dir": str(self.run_dir), "remote_run_path": self.remote_run_path,
                   "precision": self.plan.as_dict(), "vram_probe": vram,
                   "final_checkpoint": final, "last_metrics": last_metrics,
                   "resume": self._resume_info, "evaluation": evaluation,
                   "batch_stats": dict(self.builder.stats) if self.builder else {},
                   "parameters": self.model.parameters_count,
                   "debug_bundle": str(self.run_dir / "debug_bundle.zip")}
        write_json(self.run_dir / "summary.json", summary)
        append_jsonl(self.metrics_path, {"summary": summary})
        self._sync_artifacts("final", strict=self.strict_uploads)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str), flush=True)
        return summary



# ---------------------------------------------------------------------------
# Smoke fixtures (plumbing validation ONLY -- never real training data)
# ---------------------------------------------------------------------------
def smoke_config() -> dict[str, Any]:
    return {
        "model": {"d_model": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
                  "d_ff": 128, "max_seq_len": 256, "dropout": 0.0,
                  "attention_type": "sliding_window", "window_size": 64,
                  "sliding_chunk_size": 16,
                  "refinement_steps": 4, "refinement_rank": 32,
                  "memory_dim": 128, "gradient_checkpointing": True},
        "memory": {"key_dim": 64, "value_dim": 128,
                   "capacity": 32},
        "training": {"micro_batch_size": 2, "grad_accum": 2, "seq_len": 128,
                     "learning_rate": 3e-4, "warmup_steps": 2, "log_every": 1,
                     "checkpoint_every": 0, "checkpoint_minutes": 30,
                     "confidence_loss_weight": 0.1, "entropy_threshold": 0.35,
                     "decay_reg_weight": 0.01,
                     "seed": 24, "precision": "auto", "ingest_to_memory": True},
        "checkpoints": {"local_dir": "runs/smoke/checkpoints"},
        "output": {"metrics_jsonl": "runs/smoke/metrics.jsonl"},
    }


def smoke_samples(count: int = 64) -> Iterator[tuple[str, dict[str, Any]]]:
    """Deterministic fixtures: text, tool trajectories, EDIT sequences and
    multimodal samples with pre-computed features (matrix-agnostic path)."""
    rng = random.Random(24)
    texts = [
        b"def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n",
        b"O Brasil e um pais da America do Sul. A lingua oficial e o portugues.\n",
        b"The quick brown fox jumps over the lazy dog. 0123456789.\n",
        b"theorem: for all n, sum_{k=1}^n k = n(n+1)/2. proof by induction.\n",
    ]
    tool_ids = ([BOS] + list(b"query: ") + [TOOL_SEARCH] + list(b"python bytes")
                + [TOOL_SEARCH_END, OBS_START] + list(b"mock result") + [OBS_END]
                + list(b" answer.") + [EOS])
    edit_ids = ([BOS] + list(b"before ") + [EDIT] + list(b"old->new") + [EDIT_END]
                + list(b" after") + [EOS])
    for index in range(count):
        draw = index % 8
        if draw == 6:
            yield "tools", {"kind": "ids", "ids": tool_ids}
        elif draw == 7:
            yield "tools", {"kind": "ids", "ids": edit_ids}
        elif draw in (4, 5):
            modality = "image" if draw == 4 else "audio"
            features = torch.randn(768, generator=torch.Generator().manual_seed(index))
            yield "multimodal", {"kind": "multimodal", "modality": modality,
                                 "caption": f"a synthetic {modality} number {index}".encode(),
                                 "features": features, "image": None,
                                 "waveform": None, "sample_rate": None}
        else:
            yield "text", {"kind": "text", "data": texts[draw % len(texts)] * rng.randint(1, 3)}


def run_smoke(*, steps: int = 4, workdir: str | Path = "runs/smoke") -> dict[str, Any]:
    """Full-pipeline CPU validation: train, checkpoint, resume, tool loop."""
    config = smoke_config()
    config["checkpoints"]["local_dir"] = str(Path(workdir) / "checkpoints")
    config["output"]["metrics_jsonl"] = str(Path(workdir) / "metrics.jsonl")
    trainer = Trainer(config, device="cpu", samples=smoke_samples(256), resume=False)
    summary = trainer.run(max_steps=steps)
    # Resume exactly where we stopped.
    resumed = Trainer(config, device="cpu", samples=smoke_samples(256), resume=True)
    resumed.optimizer = resumed._build_optimizer()
    resumed._maybe_resume()
    summary["resume_step"] = resumed.step
    # True tooling: token-triggered sandboxed mock search.
    search = make_search_fn("mock")
    prompt = encode_bytes(b"question: what are python bytes?\n")
    ids, stats = trainer.model.generate_with_tools(prompt, search, max_new_bytes=48)
    summary["tool_demo"] = {"generated_ids": int(ids.numel()), **stats}
    # Coherence eyeball: a greedy PT-BR continuation from the trained model.
    with torch.no_grad():
        continuation = trainer.model.generate(encode_bytes(b"O Brasil e um pais"),
                                              max_new_bytes=48)
    summary["sample_continuation"] = decode_ids(continuation).decode("utf8", "replace")
    # Phase 24.4: latent scratchpad telemetry (activation norm proves usage).
    refinement = trainer.model.refinement
    if refinement is not None and refinement.thoughts is not None:
        trainer.model.eval()
        with torch.no_grad():
            trainer.model(encode_bytes(b"O Brasil e um pais da America do Sul.").unsqueeze(0))
        thoughts = refinement.thoughts
        summary["thought"] = {
            "buffer_shape": list(thoughts.last_buffer.shape),
            "thought_steps": int(thoughts.last_steps),
            "activation_norm": float(thoughts.last_norm),
            "buffer_gate": float(thoughts.buffer_gate.detach()),
            "context_gate": float(thoughts.context_gate.detach()),
            "thought_confidence_weight_norm": float(
                refinement.thought_confidence.weight.detach().norm()),
        }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ByteCore-DualMemory unified trainer")
    parser.add_argument("--config", default="configs/t4_scaled.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--checkpoint-minutes", type=float, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--upload-minutes", type=float, default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--hf-repo", default=None,
                        help=f"HF model repo for checkpoints/diagnostics (preset default: {DEFAULT_HF_REPO})")
    parser.add_argument("--local-only", action="store_true",
                        help="explicitly disable Hub persistence for a local experiment")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="CPU plumbing validation with synthetic fixtures")
    parser.add_argument("--report", default=None)
    parser.add_argument("--simulate-t4", action="store_true",
                        help="print conservative analytic T4 VRAM estimate and exit (no GPU needed)")
    args = parser.parse_args(argv)

    if args.simulate_t4:
        report = simulate_t4_vram(load_config(args.config))
        text = json.dumps(report, indent=2)
        print(text)
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(text, encoding="utf8")
        raise SystemExit(0 if report["fits"] else 1)

    if args.smoke:
        summary = run_smoke(steps=args.max_steps or 4)
    else:
        config = load_config(args.config)
        train_cfg = config.setdefault("training", {})
        ckpt_cfg = config.setdefault("checkpoints", {})
        output_cfg = config.setdefault("output", {})
        if args.checkpoint_every is not None:
            train_cfg["checkpoint_every"] = args.checkpoint_every
        if args.upload_minutes is not None:
            ckpt_cfg["upload_minutes"] = args.upload_minutes
        if args.run_dir:
            output_cfg["run_dir"] = args.run_dir
            ckpt_cfg["local_dir"] = str(Path(args.run_dir) / "checkpoints")
            output_cfg["metrics_jsonl"] = str(Path(args.run_dir) / "metrics.jsonl")
        if args.run_id:
            output_cfg["run_id"] = args.run_id
        if args.local_only:
            ckpt_cfg["hf_repo"] = None
        repo = None if args.local_only else (args.hf_repo or ckpt_cfg.get("hf_repo"))
        if args.no_resume and repo:
            print("WARNING: --no-resume starts fresh and may replace the remote latest checkpoint", flush=True)
        trainer = Trainer(config, device=args.device, hf_repo=repo,
                          resume=not args.no_resume)
        summary = trainer.run(
            max_steps=args.max_steps if args.max_steps is not None else int(train_cfg.get("max_steps", 200000)),
            max_minutes=args.max_minutes if args.max_minutes is not None else train_cfg.get("max_minutes"),
            checkpoint_minutes=args.checkpoint_minutes)
    text = json.dumps(summary, indent=2, default=str)
    print(text)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text, encoding="utf8")


if __name__ == "__main__":
    main()
