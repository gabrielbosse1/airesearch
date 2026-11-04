"""Phase 22 offline benchmark: measurable local facts only.

Writes ``runs/phase22/benchmark.json``. GPU VRAM, HF throughput and model
quality are deliberately reported as ``unmeasured``: the Arena sandbox has no
CUDA and cannot reach huggingface.co.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch

from data.phase22_sources import (HF_SEARCH_QUERIES, LICENSE_NOTES,
                                _self_test as sources_self_test)
from evaluation.final_agent_eval import evaluate
from models.foundation import CausalByteTransformer, FoundationConfig
from training.phase22_common import (estimate_vram, load_config, model_config,
                                     resolve_precision, vram_fill_ladder,
                                     write_report)
from training.phase22_reward import dominance_report
from training.phase22_train import smoke_test

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "phase22"


def timed_forward(model: CausalByteTransformer, batch: int, length: int,
                  *, repeats: int = 3) -> dict[str, float | int]:
    ids = torch.randint(0, 256, (batch, length))
    model.eval()
    with torch.no_grad():
        model(ids)  # warm-up (lazy kernel init)
        start = time.perf_counter()
        for _ in range(repeats):
            logits = model(ids)
        elapsed = time.perf_counter() - start
    return {"batch": batch, "seq_len": length, "repeats": repeats,
            "mean_forward_ms": round(1000 * elapsed / repeats, 3),
            "logits_shape": list(logits.shape)}


def main() -> dict[str, object]:
    config = load_config(ROOT / "configs" / "t4_scaled.yaml")
    t0 = time.perf_counter()
    with torch.device("meta"):
        meta_model = CausalByteTransformer(model_config(config))
    meta_params = meta_model.parameters_count
    meta_seconds = round(time.perf_counter() - t0, 4)

    legacy = CausalByteTransformer(FoundationConfig(n_layers=2, d_model=32,
                                                    n_heads=2, d_ff=64))
    legacy_params = legacy.parameters_count

    # REAL (CPU) forward measurement of the full 183.8M parameter model.
    t0 = time.perf_counter()
    full = CausalByteTransformer(model_config(config)).eval()
    build_seconds = round(time.perf_counter() - t0, 3)
    full_forward_64 = timed_forward(full, 1, 64, repeats=2)
    full_forward_256 = timed_forward(full, 1, 256, repeats=2)
    del full

    tiny = CausalByteTransformer(FoundationConfig(
        n_layers=1, d_model=32, n_heads=4, n_kv_heads=2, d_ff=64,
        max_seq_len=64, dropout=0.0, norm_type="rmsnorm", ffn_type="swiglu",
        position_type="rope", refinement_steps=2, refinement_rank=8,
        gradient_checkpointing=True)).train()
    tiny_forward = timed_forward(tiny, 2, 16, repeats=5)
    ids = torch.randint(0, 256, (2, 16))
    start = time.perf_counter()
    loss = tiny.loss(ids)
    loss.backward()
    tiny_backward_ms = round((time.perf_counter() - start) * 1000, 3)

    estimates = {
        f"B{b}-L{l}-ckpt{c}": estimate_vram(config, batch_size=b, seq_len=l,
                                            checkpointing=c)["estimated_total_gb"]
        for b, l, c in ((4, 1024, True), (4, 2048, True), (32, 2048, True),
                        (4, 2048, False), (6, 2048, False))
    }
    ladder = vram_fill_ladder(config, device=torch.device("cpu"))
    precision = resolve_precision(torch.device("cpu"), "bfloat16", "auto").as_dict()
    dominance = dominance_report()
    final_eval = evaluate(root=ROOT)
    sources = sources_self_test()
    smoke = smoke_test()

    d, layers, heads, kv, d_ff, vocab = 1024, 12, 8, 2, 4096, 261
    rank = 256
    attn_per_layer = (d * d + d) + 2 * (d * (kv * (d // heads)) + kv * (d // heads)) + (d * d + d)
    swiglu_per_layer = 3 * d * d_ff
    norms_per_layer = 2 * d
    layers_total = layers * (attn_per_layer + swiglu_per_layer + norms_per_layer)
    refinement_total = 3 * d * rank + d + 2
    embedding_total = vocab * d
    head_total = d * vocab + vocab
    final_norm_total = d
    breakdown_sum = (layers_total + refinement_total + embedding_total +
                     head_total + final_norm_total)
    architecture = {
        "components": {
            "pre_norm_rmsnorm": True, "swiglu_ffn": True, "rope": True,
            "gqa": {"n_heads": heads, "n_kv_heads": kv, "cache_fraction": kv / heads},
            "iterative_refinement": {"steps": 2, "rank": rank, "weight_tied": True,
                                     "alpha_init": 0.0},
            "gradient_checkpointing": True, "learned_absolute_positions": False,
        },
        "parameter_breakdown": {
            "attention_per_layer": attn_per_layer,
            "swiglu_per_layer": swiglu_per_layer,
            "norms_per_layer": norms_per_layer,
            "layers_total": layers_total,
            "refinement_total": refinement_total,
            "embedding_total": embedding_total,
            "head_total": head_total,
            "final_norm_total": final_norm_total,
            "sum": breakdown_sum,
            "meta_count": meta_params,
            "exact_match": breakdown_sum == meta_params,
        },
        "max_seq_len": config["model"]["max_seq_len"],
        "training_seq_len": config["training"]["seq_len"],
    }

    dataset_groups: dict[str, list[dict[str, str]]] = {}
    for phase in ("phase1", "phase2"):
        dataset_groups[phase] = [
            {"id": spec["id"], "config": spec.get("name", ""),
             "data_dir": spec.get("data_dir", ""),
             "license": LICENSE_NOTES.get(spec["id"], "unverified"),
             "url": f"https://huggingface.co/datasets/{spec['id']}",
             "downloaded_here": "false"}
            for group in config["data"][phase] for spec in group["sources"]
        ]
    inventory = {
        "phase1": dataset_groups["phase1"], "phase2": dataset_groups["phase2"],
        "search_queries": list(HF_SEARCH_QUERIES),
        "rejected": [
            {"id": "bigcode/the-stack-v2", "reason": "gated (terms + contact info)"},
            {"id": "oscar-corpus/OSCAR-2301", "reason": "manually gated / access suspended"},
            {"id": "HuggingFaceTB/stack-edu", "reason": "blob_id only; needs separate S3 fetch"},
        ],
        "no_downloads_in_sandbox": True,
        "reason": "huggingface.co is not on the sandbox network allowlist; Colab runs load_dataset",
    }

    report: dict[str, object] = {
        "schema": 1,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "cuda_available": torch.cuda.is_available(),
            "threads": torch.get_num_threads(),
        },
        "scope": ("OFFLINE CPU benchmark. No HF dataset downloaded, no GPU/VRAM "
                  "measurement, no model-quality claim."),
        "architecture": architecture,
        "dataset_inventory": inventory,
        "parameters": {
            "scaled": meta_params, "scaled_in_target_150_200m": 150_000_000 <= meta_params <= 200_000_000,
            "legacy_phase16_tiny": legacy_params,
            "ratio": round(meta_params / legacy_params, 1),
            "meta_construction_seconds": meta_seconds,
            "full_cpu_build_seconds": build_seconds,
        },
        "cpu_forward": {
            "scaled_L64": full_forward_64, "scaled_L256": full_forward_256,
            "tiny_gated_stack": tiny_forward,
            "tiny_forward_backward_ms": tiny_backward_ms,
            "tiny_loss": round(float(loss.detach()), 6),
            "note": "measured on CPU; T4 numbers require Colab",
        },
        "vram": {
            "analytic_estimates_gb": estimates,
            "default_B4_L1024_gb": estimates["B4-L1024-ckptTrue"],
            "target_band_gb": ladder["target_band_gb"],
            "fill_ladder_measured": ladder["measured"],
            "fill_ladder_target_met": ladder["target_met"],
            "candidates": ladder["trials"],
            "measured_vram_gb": None,
            "unmeasured_reason": "no CUDA device in the Arena sandbox",
        },
        "precision": precision,
        "reward": {"all_orderings_satisfied": dominance["all_satisfied"],
                   "ordering": dominance["ordering"],
                   "example_totals": {k: v["total"] for k, v in dominance["rewards"].items()}},
        "final_eval": {"oracle_mode": final_eval["oracle_mode"],
                       "cases_passed": final_eval["cases_passed"],
                       "cases_total": final_eval["cases_total"],
                       "decontamination": final_eval["decontamination"]["status"],
                       "note": "scripted oracle proves the scorer, NOT the model"},
        "sources": sources,
        "pipeline_smoke": smoke,
        "checkpoint_names": ["checkpoint_phase1.pt", "checkpoint_phase2.pt", "final_model.pt"],
        "notebooks": ["01_setup_and_pretrain.ipynb", "02_distill_and_align.ipynb",
                      "03_rl_and_agent.ipynb"],
    }
    write_report(OUT / "benchmark.json", report)
    write_report(OUT / "vram_estimate.json", {
        "estimates_gb": estimates, "ladder": ladder,
        "measured": False, "reason": "no CUDA device"},
    )
    write_report(OUT / "reward_dominance.json", dominance)
    write_report(OUT / "architecture_report.json", architecture)
    write_report(OUT / "dataset_inventory.json", inventory)
    (OUT / "README.md").write_text(
        "# Phase 22 offline reports\n\n"
        "`benchmark.json` -- CPU-measured parameter counts, forward/backward timings, "
        "analytic VRAM estimates, precision fallback, reward ordering and the "
        "pipeline smoke test.\n\n"
        "`vram_estimate.json` -- the full fill ladder (all `measured=false` here).\n\n"
        "`reward_dominance.json` -- Search>Guess / Efficient Edit>Thrashing proof.\n\n"
        "`final_eval.json` -- scripted-oracle scorer proof (`oracle_mode=true`).\n\n"
        "`architecture_report.json` -- component inventory and exact parameter breakdown.\n\n"
        "`dataset_inventory.json` -- verified HF IDs, configs, licences and gating rejects.\n\n"
        "**No GPU VRAM utilisation, HF download, checkpoint training or model "
        "quality is claimed here.** Run the three notebooks on a Colab T4.\n",
        encoding="utf8")
    return report


if __name__ == "__main__":
    payload = main()
    print(json.dumps({
        "parameters": payload["parameters"],
        "cpu_forward": payload["cpu_forward"],
        "vram": {"default_B4_L1024_gb": payload["vram"]["default_B4_L1024_gb"],
                 "measured_vram_gb": payload["vram"]["measured_vram_gb"],
                 "target_band_gb": payload["vram"]["target_band_gb"]},
        "reward": payload["reward"],
        "final_eval": payload["final_eval"],
        "pipeline_smoke": payload["pipeline_smoke"],
    }, indent=2))
