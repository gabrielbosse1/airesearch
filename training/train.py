"""Unified training script (Phase 25). One script, one pipeline.

    python -m training.train --config configs/t4_scaled.yaml \
        --hf-repo gabrielbosse9/airesearch --max-minutes 240

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
* representative measured VRAM probe + batch/sparse autotune; the separate
  ``--simulate-t4`` accounting mode is explicitly labelled as an estimate;
* atomic checkpoints every 30 minutes (and every N steps), resumable locally
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
import contextlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import yaml

from models.foundation import (BOS, EOS, PAD, TOOL_SEARCH, TOOL_SEARCH_END, VOCAB_SIZE,
                               OBS_START, OBS_END, EDIT, EDIT_END,
                               BigBirdSparseMask, CausalByteTransformer, FoundationConfig,
                               decode_ids, encode_bytes, pad_batch)
from models.memory_model import (KEY_DIM, VALUE_DIM, DualMemory, EvictionEvent,
                                 SkillBankView, stack_views)
from models.encoders import (MultimodalIngestor, TextSkillProjector,
                             build_encoder, matrix_to_bytes,
                             pooled_text_embedding, preprocess_image)
from data.sources import BytePacker, MixtureSampler, build_mixture, mixture_snapshot

CHECKPOINT_SCHEMA = 6  # v6: unified dynamic memory + BigBird/cross-attention


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
                     seq_len: int | None = None, random_tokens: int | None = None,
                     global_tokens: int | None = None) -> dict[str, Any]:
    """Return a conservative simulated peak and its transparent components.

    The model assumes fp32 master weights/grads/Adam moments (16 bytes per
    trainable parameter), fp16 autocast inside blocks, fp32 residual/checkpoint
    boundaries, all-block checkpointing, whole-loop refinement checkpointing,
    and ONE BigBird overlay live at a time. It does NOT claim block-sparse
    FLOPs or measure backend workspace; the safety reserve covers allocator
    fragmentation and unmodelled temporaries.
    """
    model_cfg = FoundationConfig.from_mapping(config["model"])
    train = dict(config.get("training", {}))
    mem = dict(config.get("memory", {}))
    vram = dict(config.get("vram", {}))
    batch = int(micro_batch if micro_batch is not None else train.get("micro_batch_size", 1))
    length = int(seq_len if seq_len is not None else train.get("seq_len", model_cfg.max_seq_len))
    r = int(random_tokens if random_tokens is not None else model_cfg.bigbird_random_tokens)
    g = int(global_tokens if global_tokens is not None else model_cfg.bigbird_global_tokens)
    if batch < 1 or length < 1 or length > model_cfg.max_seq_len or r < 0 or g < 1:
        raise ValueError("invalid simulated batch/length/sparse counts")
    if model_cfg.attention_type == "bigbird" and not model_cfg.gradient_checkpointing:
        raise ValueError("BigBird T4 simulation requires gradient_checkpointing=True")
    d, h, kv, ff = model_cfg.d_model, model_cfg.n_heads, model_cfg.n_kv_heads or model_cfg.n_heads, model_cfg.d_ff
    memory_dim = model_cfg.memory_dim or int(mem.get("value_dim", VALUE_DIM))
    key_dim = int(mem.get("key_dim", d))
    capacity = int(mem.get("capacity", int(mem.get("procedural_capacity", 1024))
                           + int(mem.get("episodic_capacity", 1024))))
    slots = capacity + len(config.get("encoders", {})) + 1  # persistent + local projector slots
    steps = max(1, model_cfg.refinement_steps)
    trainable, frozen = _trainable_parameters(config, model_cfg)
    act = batch * length * d
    components: dict[str, int] = {
        # fp32 param + grad + exp_avg + exp_avg_sq. GradScaler adds negligible state.
        "fp32_params_grads_adam": 16 * trainable,
        "frozen_encoder_weights": 4 * frozen,
        "unified_memory_buffers": 4 * capacity * (key_dim + memory_dim + 8),
        # Autocast leaves residual/checkpoint boundaries in fp32.
        "block_checkpoint_boundaries": 4 * (model_cfg.n_layers + 1) * act,
        # One live recomputed block: Q/K/V/output/RoPE temporaries, SwiGLU,
        # fp32 norm temporaries, wide GLU retrieval, cross-attention, SDPA work.
        "live_block_attention": 2 * act * (4 + 2 * kv / h),
        "live_block_ffn": 2 * 4 * batch * length * ff,
        "live_block_norms": 4 * 4 * act,
        "live_adapter_glu": 2 * batch * length * memory_dim + 2 * 3 * act,
        # Manual GLU retrieval materialises fp16 scores, fp32 softmax, fp16 weights.
        "adapter_retrieval_score_workspace": 8 * batch * length * slots,
        "live_memory_cross_attention": 2 * (6 * act + 2 * slots * d + batch * length * slots),
        "live_sdpa_workspace": 2 * 4 * act,
        # Confidence needs every refined state; the loop checkpoint retains
        # those outputs, while only one cell/thought step is live in backward.
        "refinement_state_outputs": 4 * steps * act,
        # Pure thought steps pass slots/state explicitly across checkpoints:
        # hidden boundaries are retained, but only ONE step's context K/V is live.
        "refinement_step_boundaries": 4 * steps * act,
        "refinement_live_thought_step": 2 * 4 * act + 6 * 2 * act,
        "refinement_live_cell": 2 * 6 * batch * length * model_cfg.refinement_rank + 4 * 4 * act,
        # Streamed entropy avoids stacking S logits/log-probs; final CE still
        # materialises logits and softmax temporaries at byte vocabulary width.
        "confidence_and_ce_logits": 4 * 4 * batch * length * VOCAB_SIZE,
        "ids_targets_embeddings": 8 * 2 * batch * length + 4 * act,
    }
    if model_cfg.attention_type == "bigbird":
        # Shared base + ONE lazy head overlay + a possible fp16 additive bias
        # conversion by the efficient backend. Per-head coordinates are tiny.
        components["bigbird_base_and_one_overlay"] = 2 * length * length
        components["bigbird_backend_bias_workspace"] = 2 * length * length
        components["bigbird_random_coordinates"] = 8 * h * length * r
    else:
        components["dense_causal_mask"] = length * length
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
        "fits": bool(peak_bytes / GB <= target),
        "micro_batch": batch,
        "seq_len": length,
        "bigbird_random_tokens": r,
        "bigbird_global_tokens": g,
        "trainable_parameters": trainable,
        "frozen_encoder_parameters_estimate": frozen,
        "components_gb": {name: round(value / GB, 4) for name, value in components.items()},
        "assumptions": [
            "decimal GB; fp32 weights/grads/AdamW moments (16 B/trainable parameter)",
            "fp16 autocast inside blocks; fp32 residual/checkpoint boundaries",
            "all transformer blocks checkpointed; whole refinement loop checkpointed",
            "one BigBird head overlay materialised at a time; no block-sparse FLOP claim",
            "full unified bank plus batch-local projector slots in cross-attention",
            "frozen encoder counts rounded up; encoder transient activations covered by reserve",
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
                    device: str | torch.device = "cpu") -> int:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("not a Phase 25 unified checkpoint (schema 6)")
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
    return int(payload["step"])


def hf_upload(local_path: str | Path, repo_id: str | None, path_in_repo: str) -> dict[str, Any]:
    """Upload a checkpoint to the Hub. Returns measured=False when skipped."""
    if not repo_id:
        return {"uploaded": False, "reason": "no hf_repo configured"}
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo,
                        repo_id=repo_id, repo_type="model")
        return {"uploaded": True, "repo_id": repo_id, "path_in_repo": path_in_repo}
    except Exception as exc:
        return {"uploaded": False, "reason": f"{type(exc).__name__}: {exc}"}


def hf_download(repo_id: str | None, path_in_repo: str, local_path: str | Path) -> dict[str, Any]:
    if not repo_id:
        return {"downloaded": False, "reason": "no hf_repo configured"}
    try:
        import shutil
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=repo_id, filename=path_in_repo,
                                     repo_type="model")
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, target)
        return {"downloaded": True, "path": str(target)}
    except Exception as exc:
        return {"downloaded": False, "reason": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class Trainer:
    """The single training loop: CE + confidence BCE, time-boxed, resumable."""

    def __init__(self, config: Mapping[str, Any], *, device: str = "auto",
                 samples: Iterator[tuple[str, dict[str, Any]]] | None = None,
                 hf_repo: str | None = None, resume: bool = True) -> None:
        self.config = dict(config)
        train_cfg = dict(config.get("training", {}))
        ckpt_cfg = dict(config.get("checkpoints", {}))
        mem_cfg = dict(config.get("memory", {}))
        enc_cfg = dict(config.get("encoders", {}))
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
        require_checkpoint = bool(vram_cfg.get("require_gradient_checkpointing",
                                               self.model_config.attention_type == "bigbird"))
        if require_checkpoint and not self.model_config.gradient_checkpointing:
            raise ValueError("this preset requires gradient_checkpointing=True for all blocks")
        self.model.set_gradient_checkpointing(self.model_config.gradient_checkpointing)
        self.vram_report: dict[str, Any] | None = None
        self.log_every = int(train_cfg.get("log_every", 10))
        self.checkpoint_every = int(train_cfg.get("checkpoint_every", 500))
        self.checkpoint_minutes = float(train_cfg.get("checkpoint_minutes", 30))
        self.local_dir = Path(ckpt_cfg.get("local_dir", "runs/checkpoints"))
        self.hf_repo = hf_repo or ckpt_cfg.get("hf_repo")
        self.hf_path = ckpt_cfg.get("hf_path_in_repo", "checkpoints/checkpoint_latest.pt")
        self.metrics_path = Path(config.get("output", {}).get("metrics_jsonl", "runs/metrics.jsonl"))

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
    def ensure_ingestor(self, modality: str, *, feature_dim: int | None = None
                        ) -> MultimodalIngestor | None:
        if modality in self.ingestors:
            existing = self.ingestors[modality]
            name = self._enc_cfg.get(modality)
            if existing.encoder is None and name:
                # Upgrade a resumed projector-only ingestor with its frozen
                # encoder so raw media can keep flowing.
                existing.encoder = build_encoder(name, pretrained=True).to(self.device)
            return existing
        name = self._enc_cfg.get(modality)
        memory_dim = int(self._mem_cfg.get("value_dim", self.model_config.memory_dim or 8192))
        key_dim = int(self._mem_cfg.get("key_dim", self.model_config.d_model))
        if name:
            encoder = build_encoder(name, pretrained=True).to(self.device)
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
        bank.eviction_events.clear()
        retained = torch.arange(1, min(bank.capacity, 9), device=self.device)
        ages = torch.linspace(0.0, 1.0, bank.capacity, device=self.device)
        bank.eviction_events.append(EvictionEvent(0, retained, ages, 1.0, False))
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
                    "bigbird_random_tokens": self.model_config.bigbird_random_tokens,
                    "bigbird_global_tokens": self.model_config.bigbird_global_tokens,
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
            torch.cuda.empty_cache()

    def _rebuild_sparse_mask(self, random_tokens: int, global_tokens: int) -> None:
        self.model_config = replace(self.model_config, bigbird_random_tokens=random_tokens,
                                    bigbird_global_tokens=global_tokens)
        self.model.config = self.model_config
        self.model.attention_mask = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.model.attention_mask = BigBirdSparseMask(
            self.model_config.max_seq_len, self.model_config.n_heads,
            window=self.model_config.bigbird_window, random_tokens=random_tokens,
            global_tokens=global_tokens, seed=self.model_config.bigbird_seed,
            device=self.device)

    def autotune_batch(self, target_max_gb: float = 13.5) -> dict[str, Any]:
        """Probe, halve batch, then reduce R/G; fail closed above the ceiling."""
        if target_max_gb <= 0:
            raise ValueError("target_max_gb must be positive")
        probe = self.probe_vram()
        reductions: list[dict[str, int]] = []
        while probe.get("measured") and probe["peak_gb"] > target_max_gb and self.micro_batch > 1:
            self.micro_batch = max(1, self.micro_batch // 2)
            probe = self.probe_vram()
        vram_cfg = dict(self.config.get("vram", {}))
        if (probe.get("measured") and probe["peak_gb"] > target_max_gb
                and bool(vram_cfg.get("auto_reduce_sparse", True))
                and self.model_config.attention_type == "bigbird"):
            min_r = int(vram_cfg.get("min_random_tokens", 0))
            min_g = int(vram_cfg.get("min_global_tokens", 1))
            r, g = self.model_config.bigbird_random_tokens, self.model_config.bigbird_global_tokens
            while probe["peak_gb"] > target_max_gb and (r > min_r or g > min_g):
                if r > min_r:
                    r = max(min_r, r // 2)
                elif g > min_g:
                    g = max(min_g, g // 2)
                self._rebuild_sparse_mask(r, g)
                reductions.append({"random_tokens": r, "global_tokens": g})
                probe = self.probe_vram()
        probe["sparse_reductions"] = reductions
        if probe.get("measured") and probe["peak_gb"] > target_max_gb:
            raise MemoryError(
                f"measured T4 peak {probe['peak_gb']:.3f} GB exceeds {target_max_gb:.3f} GB "
                "after batch/sparse autotune; refusing to start. Dense boolean-mask SDPA "
                "allocation does not necessarily shrink with R/G; use a block-sparse kernel "
                "or a smaller model/context.")
        if not probe.get("measured"):
            probe["simulated"] = simulate_t4_vram(
                self.config, micro_batch=self.micro_batch, seq_len=self.seq_len,
                random_tokens=self.model_config.bigbird_random_tokens,
                global_tokens=self.model_config.bigbird_global_tokens)
        self.vram_report = probe
        return probe

    # -- checkpoint / resume ------------------------------------------------
    def _save(self, *, upload: bool = True) -> dict[str, Any]:
        assert self.optimizer is not None
        path = self.local_dir / "checkpoint_latest.pt"
        save_checkpoint(path, model=self.model, ingestors=self.ingestors,
                        text_projector=self.text_projector, memory=self.memory,
                        optimizer=self.optimizer, scaler=self.scaler, step=self.step,
                        extra={"precision": self.plan.as_dict()})
        record: dict[str, Any] = {"step": self.step, "checkpoint": str(path)}
        if upload and self.hf_repo:
            record["hf"] = hf_upload(path, self.hf_repo, self.hf_path)
        self._last_ckpt_time = time.monotonic()
        return record

    def _maybe_resume(self) -> None:
        if not self._resume or self.optimizer is None:
            return
        path = self.local_dir / "checkpoint_latest.pt"
        if not path.exists() and self.hf_repo:
            result = hf_download(self.hf_repo, self.hf_path, path)
            if not result.get("downloaded"):
                return
        if path.exists():
            self.step = load_checkpoint(path, model=self.model, ingestors=self.ingestors,
                                        text_projector=self.text_projector, memory=self.memory,
                                        optimizer=self.optimizer, scaler=self.scaler,
                                        device=self.device)

    # -- the loop -----------------------------------------------------------
    def run(self, *, max_steps: int = 1000, max_minutes: float | None = None,
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
        vram = self.autotune_batch(float(self.config.get("vram", {}).get("target_max_gb", 13.5)))
        self.model.train()
        self._start_time = time.monotonic()
        self._last_ckpt_time = time.monotonic()
        deadline = None if max_minutes is None else self._start_time + max_minutes * 60.0
        tokens = 0
        last_metrics: dict[str, Any] = {}
        stop_reason = "max_steps"
        # STEP-BASED loop (Phase 24.2): equivalent to
        #   for step, batch in enumerate(dataloader):
        #       if step >= max_steps: break
        # but counting OPTIMIZER steps (grad_accum micro-batches each).
        # Streams have no len() and no epochs: we pull one micro-batch at a
        # time from the single-process iterable pipeline and stop on
        # max_steps, the wall-clock budget, or stream exhaustion.
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
            metrics["policy_loss"] = float(policy.detach().item())
            metrics["total"] = float(total.detach().item())
            self.scaler.scale(total / self.grad_accum).backward()
            self.memory.procedural.clear_policy_events()
            self._micro += 1
            tokens += int((batch["targets"] != PAD).sum().item())
            if self._micro % self.grad_accum == 0:
                for group in self.optimizer.param_groups:
                    group["lr"] = self._lr_at(self.step, max_steps)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group["params"]],
                    float(self.config.get("training", {}).get("max_grad_norm", 1.0)))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                last_metrics = metrics
                if self.step % self.log_every == 0 or self.step == 1:
                    record = {"step": self.step, "tokens": tokens,
                              "elapsed_s": round(time.monotonic() - self._start_time, 2),
                              **metrics}
                    if self.device.type == "cuda":
                        record["vram_gb"] = round(torch.cuda.max_memory_allocated(self.device) / 1e9, 3)
                    if self.mixture is not None:
                        # Effective (post-failure) weights + realised doc share.
                        record["data_mix"] = mixture_snapshot(self.mixture)["observed_share"]
                    self.history.append(record)
                    append_jsonl(self.metrics_path, record)
                    print(json.dumps(record), flush=True)
                due_time = time.monotonic() - self._last_ckpt_time >= self.checkpoint_minutes * 60.0
                due_steps = self.checkpoint_every > 0 and self.step % self.checkpoint_every == 0
                if due_time or due_steps:
                    record = self._save()
                    self.history.append({"checkpoint_event": record})
        final = self._save()
        data_mix = mixture_snapshot(self.mixture) if self.mixture is not None else None
        summary = {"steps": self.step, "tokens": tokens, "stop_reason": stop_reason,
                   "data_mix": data_mix,
                   "precision": self.plan.as_dict(), "vram_probe": vram,
                   "final_checkpoint": final, "last_metrics": last_metrics,
                   "batch_stats": dict(self.builder.stats) if self.builder else {},
                   "parameters": self.model.parameters_count}
        append_jsonl(self.metrics_path, {"summary": summary})
        return summary


# ---------------------------------------------------------------------------
# Smoke fixtures (plumbing validation ONLY -- never real training data)
# ---------------------------------------------------------------------------
def smoke_config() -> dict[str, Any]:
    return {
        "model": {"d_model": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
                  "d_ff": 128, "max_seq_len": 256, "dropout": 0.0,
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
    parser.add_argument("--hf-repo", default=None)
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
        train_cfg = config.get("training", {})
        trainer = Trainer(config, device=args.device, hf_repo=args.hf_repo,
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
