"""Phase 4 evaluation: reconstruction accuracy, byte-noise robustness,
inference timing, and concept write/read (learn without retraining)."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from data.dataset import ByteMultimodalDataset, collate
from models.bytecore_diffusion import ByteCoreDiffusion
from models.holdout import HoldoutGuard, HoldoutViolation
from training.train import count_params


def corrupt_sequence(seq, f, mode, rng):
    b = list(seq)
    n = len(b)
    if n == 0 or f <= 0:
        return b
    k = max(1, int(round(f * n)))
    if mode == "substitute":
        idx = rng.sample(range(n), min(k, n))
        for i in idx:
            b[i] = rng.randint(0, 255)
    elif mode == "insert":
        # Distribute insertion sites throughout the sequence. The previous
        # integer-division loop placed most insertions at the end and cropping
        # removed them, overstating recovery.
        sites = set(rng.sample(range(n + 1), min(k, n + 1)))
        out = []
        for i in range(n + 1):
            if i in sites:
                out.append(rng.randint(0, 255))
            if i < n:
                out.append(b[i])
        b = out[:n]
    elif mode == "delete":
        kill = set(rng.sample(range(n), min(k, n)))
        b = [byte for i, byte in enumerate(b) if i not in kill]
        b = b + [0] * (n - len(b))
    return b


@torch.no_grad()
def denoise_predictions(model, x0, t_level=2):
    """Predict bytes after one light Gaussian denoising step."""
    b = x0.shape[0]
    x0_emb = model.bytes_to_emb(x0)
    t_int = torch.full((b,), t_level, device=x0.device, dtype=torch.long)
    xt = model.q_sample(x0_emb, t_int)
    pred = model.core(xt, t_int.float() / model.diffusion_timesteps, None)
    return model.core.embed_to_bytes(pred)


@torch.no_grad()
def denoise_accuracy(model, x0, t_level=2, guidance_scale=1.0):
    """Exact match against the supplied input, including padding (legacy metric)."""
    return (denoise_predictions(model, x0, t_level) == x0).float().mean().item()


@torch.no_grad()
def timing(model, loader, device, batches=20):
    """Wall-clock time per sample for one denoising forward pass."""
    model.eval()
    elapsed = 0.0
    total_samples = 0
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        x0 = batch["input_ids"].to(device)
        emb = model.bytes_to_emb(x0)
        t_int = torch.full((x0.shape[0],), 2, device=device, dtype=torch.long)
        xt = model.q_sample(emb, t_int)
        tf = t_int.float() / model.diffusion_timesteps
        if i == 0:
            _ = model.core(xt, tf)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model.core(xt, tf)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        total_samples += x0.shape[0]
    return elapsed / max(total_samples, 1) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--report", default="runs/report.md")
    args = ap.parse_args()
    torch.manual_seed(42)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = args.device
    mcfg = cfg["model"]
    d_cfg = cfg["data"]
    ev = cfg["evaluation"]

    os.makedirs(args.runs_dir, exist_ok=True)

    # ---- Load model -------------------------------------------------
    model = ByteCoreDiffusion(
        vocab_size=mcfg["vocab_size"], d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"], n_heads=mcfg["n_heads"],
        d_ff=mcfg["d_ff"], max_len=mcfg["max_len"],
        diffusion_timesteps=mcfg["diffusion_timesteps"],
        concept_capacity=mcfg["concept_capacity"],
        concept_dim=mcfg["concept_dim"],
        dropout=mcfg["dropout"],
        device=device,
    )
    model.load(args.runs_dir, map_location=device)
    model.eval()
    params = count_params(model)

    # ---- Load test set ----------------------------------------------
    test_ds = ByteMultimodalDataset(
        os.path.join(d_cfg["processed_dir"], "test.jsonl"),
        max_len=d_cfg["max_seq_len"])
    loader_test = DataLoader(test_ds, batch_size=64, shuffle=False,
                             collate_fn=collate)

    # Synthetic data may repeat across splits. Report this rather than implying
    # all reported reconstruction inputs are novel to the training corpus.
    def read_samples(split):
        with open(os.path.join(d_cfg["processed_dir"], f"{split}.jsonl")) as f:
            return [json.loads(line) for line in f if line.strip()]
    train_samples = read_samples("train")
    validation_samples = read_samples("val")
    test_samples = read_samples("test")
    training_bytes = {bytes(row["bytes"]) for row in train_samples}
    test_overlap = sum(bytes(row["bytes"]) in training_bytes for row in test_samples)
    protected = [{"input": bytes(row["bytes"])} for row in validation_samples + test_samples]
    # Also protect the real-edit evaluation fixture when it is present.
    fixture = Path("data/real_edit_pairs.jsonl")
    if fixture.exists():
        from data.ingest_real_traces import load_pairs
        protected += load_pairs(fixture, split="validation") + load_pairs(fixture, split="test")
    model.bank.set_holdout_guard(HoldoutGuard.from_records(protected))
    results = {"model": "ByteCore-Diffusion", "params": params,
               "params_mb": params * 4 / 1e6,
               "data_split_overlap": {"exact_test_rows_in_train": test_overlap, "test_rows": len(test_samples)}}

    # Reconstruction accuracy (clean, light denoising t=2)
    rec = denoise_accuracy(model, next(iter(loader_test))["input_ids"].to(device))
    results["reconstruction_accuracy"] = rec

    # Per-modality reconstruction
    per_mod = {}
    for batch in loader_test:
        x0 = batch["input_ids"].to(device)
        mods = batch["modalities"]
        # group by modality
        from collections import defaultdict
        by_mod = defaultdict(list)
        for i, m in enumerate(mods):
            by_mod[m].append(x0[i:i+1])
        for m, chunks in by_mod.items():
            xm = torch.cat(chunks, dim=0)
            per_mod.setdefault(m, []).append(denoise_accuracy(model, xm))
    results["per_modality"] = {
        m: sum(v) / len(v) for m, v in per_mod.items()
    }
    # Also report true byte-weighted accuracy without counting padded zeros.
    real_counts = {}
    novel_correct = novel_total = novel_rows = offset = 0
    for batch in loader_test:
        original = batch["input_ids"].to(device)
        prediction = denoise_predictions(model, original)
        for i, (mod, length) in enumerate(zip(batch["modalities"], batch["real_lens"])):
            correct, total = real_counts.get(mod, (0, 0))
            hit = int((prediction[i, :length] == original[i, :length]).sum())
            real_counts[mod] = (correct + hit, total + length)
            if bytes(test_samples[offset + i]["bytes"]) not in training_bytes:
                novel_correct += hit
                novel_total += length
                novel_rows += 1
        offset += len(batch["real_lens"])
    results["per_modality_real_bytes"] = {m: c / max(n, 1) for m, (c, n) in real_counts.items()}
    results["real_byte_reconstruction"] = sum(c for c, _ in real_counts.values()) / max(sum(n for _, n in real_counts.values()), 1)
    results["novel_exact_byte_rows"] = {"count": novel_rows, "bytes": novel_total,
                                        "reconstruction": novel_correct / novel_total if novel_total else None}

    # ---- Noise robustness -------------------------------------------
    noise_results = {}
    recovery_results = {}
    rng = random.Random(7)
    for mode in ev["noise_modes"]:
        noise_results[mode] = {}
        recovery_results[mode] = {}
        for level in ev["noise_levels"]:
            if level == 0.0:
                continue
            # corrupt test bytes
            corrupted = []
            with open(os.path.join(d_cfg["processed_dir"], "test.jsonl")) as f:
                samples = [json.loads(l) for l in f if l.strip()]
            for s in samples:
                cb = corrupt_sequence(s["bytes"], level, mode, rng)
                corrupted.append({"modality": s["modality"], "concept": s["concept"],
                                  "bytes": cb})
            path = os.path.join(args.runs_dir, f"_eval_noise_{mode}_{int(level*1000)}.jsonl")
            with open(path, "w") as fh:
                for s in corrupted:
                    fh.write(json.dumps(s) + "\n")
            cds = ByteMultimodalDataset(path, max_len=d_cfg["max_seq_len"])
            cl = DataLoader(cds, batch_size=64, shuffle=False, collate_fn=collate)
            input_correct = input_total = recovery_correct = recovery_total = 0
            for batch, clean in zip(cl, loader_test):
                corrupted_ids = batch["input_ids"].to(device)
                clean_ids = clean["input_ids"].to(device)
                pred = denoise_predictions(model, corrupted_ids)
                input_correct += int((pred == corrupted_ids).sum())
                input_total += pred.numel()
                for i, length in enumerate(clean["real_lens"]):
                    recovery_correct += int((pred[i, :length] == clean_ids[i, :length]).sum())
                    recovery_total += length
            noise_results[mode][level] = input_correct / max(input_total, 1)
            recovery_results[mode][level] = recovery_correct / max(recovery_total, 1)
    results["noise"] = noise_results
    results["recovery_vs_original_real_bytes"] = recovery_results

    # Robustness score = mean acc at max noise / clean acc
    max_level = max(ev["noise_levels"])
    scores = []
    for mode in ev["noise_modes"]:
        acc_n = noise_results[mode].get(max_level, 0.0)
        scores.append(acc_n / max(rec, 1e-9))
    results["robustness_score"] = sum(scores) / max(len(scores), 1)
    results["original_recovery_ratio"] = sum(
        recovery_results[mode].get(max_level, 0.0) / max(results["real_byte_reconstruction"], 1e-9)
        for mode in ev["noise_modes"]
    ) / max(len(ev["noise_modes"]), 1)

    # ---- Timing -----------------------------------------------------
    results["timing_ms_per_sample"] = timing(model, loader_test, device,
                                             ev["inference_timing_batches"])

    # ---- Concept write/read (learn without retraining) --------------
    # Write new concepts from example bytes, then generate conditioned.
    concept_demo = {}
    # Use training examples only. Exact matches and protected substrings are
    # rejected, including train/test duplicates in the synthetic corpus.
    for modality in ["text", "image", "audio"]:
        name = f"new:{modality}"
        ex = None
        for candidate in train_samples:
            if candidate["modality"] != modality:
                continue
            try:
                model.write_concept(name, candidate["bytes"])
            except HoldoutViolation:
                continue
            ex = candidate
            break
        if ex is None:
            concept_demo[modality] = {"concept_written": None, "gen_vs_example_byte_match": None,
                                      "gen_vs_example_real_byte_match": None}
            continue
        cv = model.read_concept(name)
        gen = model.generate(1, d_cfg["max_seq_len"], concept=cv,
                             guidance_scale=2.0)[0]
        # similarity = fraction of bytes matching the example (first real_len bytes)
        ex_bytes = torch.tensor(ex["bytes"][:d_cfg["max_seq_len"]],
                                dtype=torch.long, device=device)
        # pad example to max_seq_len for comparison
        if ex_bytes.shape[0] < d_cfg["max_seq_len"]:
            ex_bytes = torch.cat([ex_bytes,
                                  torch.zeros(d_cfg["max_seq_len"] - ex_bytes.shape[0],
                                              dtype=torch.long, device=device)])
        match = (gen == ex_bytes).float().mean().item()
        real_len = min(len(ex["bytes"]), d_cfg["max_seq_len"])
        real_match = (gen[:real_len] == ex_bytes[:real_len]).float().mean().item() if real_len else 0.0
        concept_demo[modality] = {
            "concept_written": name,
            "gen_vs_example_byte_match": match,
            "gen_vs_example_real_byte_match": real_match,
        }
    results["concept_demo"] = concept_demo

    # ---- Report -----------------------------------------------------
    lines = [
        "# ByteCore-Diffusion Evaluation Report",
        "",
        "## Overview",
        f"- **Model**: ByteCore-Diffusion (byte-level diffusion, raw bytes 0-255)",
        f"- **Parameters**: {params:,}  ({params * 4 / 1e6:.2f} MB, < 50 MB)",
        f"- **Reconstruction (first batch, padded)**: {rec:.4f}",
        f"- **Reconstruction (all real bytes)**: {results['real_byte_reconstruction']:.4f}",
        f"- **Corrupted-input reconstruction ratio** (not recovery): {results['robustness_score']:.4f}",
        f"- **Original-byte recovery ratio at maximum corruption**: {results['original_recovery_ratio']:.4f}",
        f"- **Inference time**: {results['timing_ms_per_sample']:.3f} ms/sample",
        f"- **Exact test rows also present in train**: {test_overlap}/{len(test_samples)} (synthetic data; not a fully novel test set)",
        f"- **Exact-byte-novel test subset**: {novel_rows} rows, reconstruction {results['novel_exact_byte_rows']['reconstruction']:.4f} (near duplicates not excluded)",
        "",
        "## Per-modality reconstruction (real bytes, no padding)",
        "",
        "| Modality | Accuracy |",
        "|----------|----------|",
    ]
    for m, a in results["per_modality_real_bytes"].items():
        lines.append(f"| {m} | {a:.4f} |")

    lines += [
        "",
        "## Corrupted-input reconstruction (includes padding)",
        "This is a self-reconstruction metric, not recovery of the original.",
        "",
    ]
    lines.append("### Per noise mode")
    lines.append("| Mode | clean | 0.02 | 0.05 | 0.10 |")
    lines.append("|------|-------|------|------|------|")
    for mode in ev["noise_modes"]:
        row = [f"{rec:.4f}"]
        for l in (0.02, 0.05, 0.10):
            v = noise_results[mode].get(l, float("nan"))
            row.append(f"{v:.4f}" if v == v else "n/a")
        lines.append(f"| {mode} | " + " | ".join(row) + " |")

    lines += ["", "## Recovery against the original (real bytes, position-wise)", "",
              "| Mode | 0.02 | 0.05 | 0.10 |", "|------|------|------|------|"]
    for mode in ev["noise_modes"]:
        row = [f"{recovery_results[mode].get(level, float('nan')):.4f}" for level in (0.02, 0.05, 0.10)]
        lines.append(f"| {mode} | " + " | ".join(row) + " |")

    lines += [
        "",
        "## Concept storage — learn new concepts without retraining",
        "",
        "New concepts are written directly to the concept bank (a memory write, "
        "no gradient step). The model can then generate conditioned on them.",
        "",
        "| Modality | Concept written | Match on real bytes | Match including padding |",
        "|----------|-----------------|---------------------|-------------------------|",
    ]
    for modality, d in concept_demo.items():
        if d["concept_written"] is None:
            lines.append(f"| {modality} | no eligible training example | n/a | n/a |")
        else:
            lines.append(f"| {modality} | {d['concept_written']} | "
                         f"{d['gen_vs_example_real_byte_match']:.4f} | {d['gen_vs_example_byte_match']:.4f} |")

    lines += [
        "",
        "## Notes",
        "- Raw bytes (0-255) throughout; no external tokenizers.",
        "- Reconstruction = exact byte match after one denoising step (light noise, t=2/50).",
        "- Corrupted-input reconstruction compares to the corrupted input, not the clean target.",
        "- Original-byte recovery excludes padding and compares to clean bytes at the same positions; "
        "insertions/deletions are not realigned.",
        "- Ratios divide maximum-corruption accuracy by the corresponding clean accuracy.",
        "- Concept examples come from the training split and pass an exact-byte holdout guard; "
        "no validation/test examples are written to memory.",
        "- Exact split-overlap counts do not detect near duplicates or prove checkpoint training provenance.",
        "- Concept bank: new concepts stored as memory entries (write, not train); "
        "instantly available at inference — no retraining required.",
        "- Inference time measured on CPU, one denoising step.",
    ]

    report = "\n".join(lines)
    Path(args.report).write_text(report)
    with open(os.path.join(args.runs_dir, "evaluation.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(report)
    print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
