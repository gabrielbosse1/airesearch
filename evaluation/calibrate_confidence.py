"""Phase 5: held-out confidence calibration, error records and threshold analysis.

Temperature is fitted on even validation examples; thresholds are selected on
odd validation examples; test data are never used for fitting or selection.
Thresholds concern q-sampled denoising and teacher-forced next-byte prediction,
NOT scratchpad thinking or ancestral sampling. Nothing enables halting by default.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.plotting import canvas, line_panel, save
from models.bytecore_diffusion import ByteCoreDiffusion
from models.generator_conditioned import GeneratorConditioned, GeneratorState
from training.train_confidence import get_loader, load_generator


@torch.no_grad()
def collect_predictions(
    diffusion: ByteCoreDiffusion, generator: GeneratorConditioned, loader: Any,
    *, seed: int, diffusion_steps: tuple[int, ...], generator_noise: tuple[float, ...],
) -> dict[str, dict[str, np.ndarray]]:
    """Record one row per *real byte*, retaining example IDs and corruption levels."""
    if not diffusion_steps or not generator_noise:
        raise ValueError("at least one noise level is required for each model")
    parts: dict[str, dict[str, list[np.ndarray]]] = {
        model: {key: [] for key in ("logit", "error", "correct", "sample", "position", "level")}
        for model in ("diffusion", "generator")
    }
    device = diffusion.betas.device
    if generator.byte_emb.weight.device != device:
        raise ValueError("models must use the same device")
    torch.manual_seed(seed)
    offset = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        b, length = ids.shape
        lengths = torch.tensor(batch["real_lens"], device=device)
        sample = torch.arange(offset, offset + b, device=device).unsqueeze(1)
        positions = torch.arange(length, device=device).unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)
        clean = diffusion.bytes_to_emb(ids)
        for level_idx, step in enumerate(diffusion_steps):
            if not 0 <= step < diffusion.diffusion_timesteps:
                raise ValueError("diffusion step is outside the configured schedule")
            t = torch.full((b,), step, device=device, dtype=torch.long)
            noisy = diffusion.q_sample(clean, t)
            pred, logits = diffusion.core(noisy, t.float() / diffusion.diffusion_timesteps, return_confidence=True)
            correct = diffusion.core.embed_to_bytes(pred) == ids
            mse = (pred - clean).square().mean(dim=-1)
            values = {"logit": logits, "error": mse, "correct": correct,
                      "sample": sample.expand(b, length), "position": positions.expand(b, length),
                      "level": torch.full_like(ids, level_idx)}
            for key, value in values.items():
                parts["diffusion"][key].append(value[valid].detach().cpu().numpy())
        if length < 2:
            raise ValueError("generator calibration requires at least two sequence positions")
        target = ids[:, 1:]
        valid_next = torch.arange(length - 1, device=device).unsqueeze(0) < (lengths - 1).clamp_min(0).unsqueeze(1)
        for level_idx, fraction in enumerate(generator_noise):
            if not 0 <= fraction <= 1:
                raise ValueError("generator noise fractions must be in [0, 1]")
            inputs = ids[:, :-1].clone()
            for i, real_len in enumerate(lengths.tolist()):
                n = max(0, real_len - 1)
                count = min(n, round(n * fraction))
                rng = random.Random(seed * 1000003 + (offset + i) * 101 + level_idx)
                for index in rng.sample(range(n), count):
                    old = int(inputs[i, index].item())
                    inputs[i, index] = (old + rng.randint(1, 255)) % 256
            state = generator(inputs, return_state=True)
            assert isinstance(state, GeneratorState)
            logits = state.output.clone()
            logits[..., generator.LOAD_SPAN:] = -torch.inf
            ce = F.cross_entropy(logits.reshape(-1, generator.VOCAB_SIZE), target.reshape(-1), reduction="none").reshape_as(target)
            correct = logits.argmax(dim=-1) == target
            pos = torch.arange(length - 1, device=device).unsqueeze(0)
            values = {"logit": state.confidence_logits, "error": ce, "correct": correct,
                      "sample": sample.expand(b, length - 1), "position": pos.expand(b, length - 1),
                      "level": torch.full_like(target, level_idx)}
            for key, value in values.items():
                parts["generator"][key].append(value[valid_next].detach().cpu().numpy())
        offset += b
    return {name: {key: np.concatenate(chunks) for key, chunks in values.items()} for name, values in parts.items()}


def reliability(confidence: np.ndarray, correct: np.ndarray, error: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """ECE/MCE compare confidence with correctness; errors are reported separately."""
    if len(confidence) == 0 or len(confidence) != len(correct) or len(correct) != len(error):
        raise ValueError("calibration inputs must be non-empty and equally sized")
    if not np.isfinite(confidence).all() or not np.isfinite(error).all():
        raise ValueError("calibration inputs must be finite")
    records = []
    ece = mce = 0.0
    bin_index = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    for index in range(bins):
        mask = bin_index == index
        count = int(mask.sum())
        if not count:
            continue
        c = float(confidence[mask].mean()); acc = float(correct[mask].mean()); err = float(error[mask].mean())
        gap = abs(c - acc)
        ece += count / len(confidence) * gap
        mce = max(mce, gap)
        records.append({"bin": index, "count": count, "mean_confidence": c,
                        "accuracy": acc, "mean_error": err, "gap": gap})
    return {"count": len(confidence), "accuracy": float(correct.mean()), "mean_error": float(error.mean()),
            "ece": ece, "mce": mce, "bins": records}


def wilson_lower(successes: int, n: int, z: float = 1.645) -> float:
    if n <= 0:
        return 0.0
    p = successes / n
    denominator = 1.0 + z * z / n
    return (p + z*z/(2*n) - z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denominator


def select_thresholds(confidence: np.ndarray, correct: np.ndarray, targets: tuple[float, ...] = (0.90, 0.95, 0.99)) -> dict[str, Any]:
    """Maximum observed coverage subject to an exploratory Wilson lower bound.

    Observations must be one per independent example. Selection over many
    candidate thresholds is not multiplicity-adjusted and is not a guarantee.
    """
    if confidence.size != correct.size or confidence.size == 0:
        raise ValueError("threshold selection needs equally sized observations")
    order = np.argsort(-confidence, kind="stable")
    p, labels = confidence[order], correct[order].astype(np.int64)
    cumulative = np.cumsum(labels)
    boundaries = np.flatnonzero(np.r_[p[:-1] != p[1:], True])
    result: dict[str, Any] = {}
    for target in targets:
        candidates = [(int(i) + 1, float(p[i]), int(cumulative[i])) for i in boundaries
                      if wilson_lower(int(cumulative[i]), int(i) + 1) >= target]
        key = f"{target:.2f}"
        if not candidates:
            result[key] = {"threshold": None, "coverage": 0.0, "count": 0,
                           "reason": "no candidate met the selection bound"}
        else:
            n, cutoff, hits = max(candidates, key=lambda item: item[0])
            result[key] = {"threshold": cutoff, "coverage": n / len(p), "count": n,
                           "accuracy": hits / n, "wilson_lower_90pct": wilson_lower(hits, n)}
    return result


def independent_observations(data: dict[str, np.ndarray], confidence: np.ndarray, *, kind: str, levels: int,
                             level_stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """One condition/outcome per example; balance levels for a strided sample split."""
    if levels < 1 or level_stride < 1:
        raise ValueError("levels and level_stride must be positive")
    out_conf, out_correct = [], []
    for sample in np.unique(data["sample"]):
        mask = (data["sample"] == sample) & (data["level"] == (int(sample) // level_stride) % levels)
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        if kind == "diffusion":
            out_conf.append(float(confidence[indices].min()))
            out_correct.append(bool(data["correct"][indices].all()))
        else:
            # One deterministic position; this does not validate latent THINK.
            index = indices[-1]
            out_conf.append(float(confidence[index])); out_correct.append(bool(data["correct"][index]))
    return np.asarray(out_conf, dtype=np.float64), np.asarray(out_correct, dtype=bool)


def evaluate_selected(thresholds: dict[str, Any], confidence: np.ndarray, correct: np.ndarray) -> dict[str, Any]:
    result = {}
    for target, item in thresholds.items():
        threshold = item["threshold"]
        if threshold is None:
            result[target] = {"count": 0, "coverage": 0.0, "accuracy": None}
            continue
        mask = confidence >= threshold
        count = int(mask.sum()); hits = int(correct[mask].sum())
        result[target] = {"count": count, "coverage": count / max(len(confidence), 1),
                          "accuracy": hits / count if count else None,
                          "wilson_lower_90pct": wilson_lower(hits, count) if count else None}
    return result


def write_config(path: Path, thresholds: dict[str, Any]) -> None:
    """Replace only our marked block; preserve unrelated later YAML sections."""
    marker = "\n# Phase 5 confidence thresholds (diagnostic-only; adaptive halting stays off).\n"
    end_marker = "# End Phase 5 confidence thresholds.\n"
    original = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(original) or {}
    suffix = ""
    if marker in original:
        prefix, tail = original.split(marker, 1)
        if end_marker in tail:
            _, suffix = tail.split(end_marker, 1)
        else:
            # Migrate the initial unclosed format without deleting later
            # top-level sections or comments following the final child key.
            rows = tail.splitlines(keepends=True)
            if not rows or not rows[0].startswith("phase5_calibration:"):
                raise ValueError("unrecognized Phase 5 calibration block")
            last_child = 0
            for i, row in enumerate(rows[1:], 1):
                if row.strip() and not row[0].isspace() and not row.lstrip().startswith("#"):
                    break
                if row.strip() and row[0].isspace() and not row.lstrip().startswith("#"):
                    last_child = i
            suffix = "".join(rows[last_child + 1:])
    else:
        if "phase5_calibration" in parsed:
            raise ValueError("refusing to replace an unmarked phase5_calibration section")
        prefix = original
    block: dict[str, Any] = {"phase5_calibration": {
        "adaptive_halting_enabled": False,
        "source": "validation_selection; thresholds require independent deployment validation",
        "diffusion_denoising_thresholds": {k: v["threshold"] for k, v in thresholds["diffusion"].items()},
        "generator_next_byte_thresholds": {k: v["threshold"] for k, v in thresholds["generator"].items()},
    }}
    content = prefix.rstrip() + "\n" + marker + yaml.safe_dump(block, sort_keys=False) + end_marker + suffix
    yaml.safe_load(content)  # reject malformed output before writing
    path.write_text(content, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--generator", default="runs/dual_memory/generator.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--generator-heads", type=int, default=4)
    ap.add_argument("--write-config", action="store_true")
    args = ap.parse_args()
    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mc, dc = cfg["model"], cfg["data"]
    diffusion = ByteCoreDiffusion(vocab_size=mc["vocab_size"], d_model=mc["d_model"], n_layers=mc["n_layers"],
        n_heads=mc["n_heads"], d_ff=mc["d_ff"], max_len=mc["max_len"], diffusion_timesteps=mc["diffusion_timesteps"],
        concept_capacity=mc["concept_capacity"], concept_dim=mc["concept_dim"], dropout=mc["dropout"], device=args.device,
    ).load(args.runs_dir, map_location=args.device).eval()
    generator = load_generator(args.generator, args.device, args.generator_heads)
    steps = tuple(sorted(set(min(diffusion.diffusion_timesteps - 1, x) for x in (2, 10, 25, 45))))
    fractions = (0.0, 0.02, 0.05, 0.10)
    data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for split, seed in (("val", 171), ("test", 172)):
        path = Path(dc["processed_dir"]) / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; first run python data/generate.py")
        loader = get_loader(str(path), dc["max_seq_len"], args.limit, args.batch_size)
        data[split] = collect_predictions(diffusion, generator, loader, seed=seed,
                                          diffusion_steps=steps, generator_noise=fractions)
    output = Path(args.runs_dir); output.mkdir(parents=True, exist_ok=True)
    fit_heads = {"diffusion": diffusion.core.confidence_head, "generator": generator.confidence_head}
    scopes = {"diffusion": "diffusion_denoising", "generator": "next_byte"}
    report: dict[str, Any] = {"protocol": {
        "temperature_fit": "even validation sample IDs", "threshold_selection": "odd validation sample IDs; one independent observation each with balanced noise conditions",
        "test": "held-out test sample IDs; not used for selection", "diffusion_timesteps": list(steps),
        "generator_substitution_fractions": list(fractions),
        "warning": "Thresholds are exploratory for these proxy distributions only, not validated for thought or ancestral sampling."}, "models": {}}
    arrays: dict[str, np.ndarray] = {}
    all_thresholds = {}
    for name in ("diffusion", "generator"):
        val = data["val"][name]; test = data["test"][name]
        fit_mask = val["sample"] % 2 == 0
        select_mask = ~fit_mask
        temperature = fit_heads[name].calibrate(torch.from_numpy(val["logit"][fit_mask]),
                                                torch.from_numpy(val["correct"][fit_mask].astype(np.float32)),
                                                scope=scopes[name])
        probabilities = {}
        for split, row in (("val", val), ("test", test)):
            logits = torch.from_numpy(row["logit"]).float()
            probabilities[split] = torch.sigmoid(logits / temperature).numpy()
            for key, value in row.items():
                arrays[f"{name}_{split}_{key}"] = value
            arrays[f"{name}_{split}_confidence"] = probabilities[split]
        subset = {key: value[select_mask] for key, value in val.items()}
        select_prob = probabilities["val"][select_mask]
        levels = len(steps) if name == "diffusion" else len(fractions)
        # Odd validation IDs have stride two; divide before modulo so the
        # selection subset covers every noise condition rather than only odds.
        sel_c, sel_y = independent_observations(subset, select_prob, kind=name, levels=levels, level_stride=2)
        test_c, test_y = independent_observations(test, probabilities["test"], kind=name, levels=levels)
        thresholds = select_thresholds(sel_c, sel_y)
        all_thresholds[name] = thresholds
        report["models"][name] = {
            "scope": scopes[name], "temperature": temperature,
            "validation_selection": reliability(select_prob, subset["correct"], subset["error"]),
            "test": reliability(probabilities["test"], test["correct"], test["error"]),
            "thresholds": thresholds, "test_threshold_outcomes": evaluate_selected(thresholds, test_c, test_y),
            "selection_examples": len(sel_c), "test_examples": len(test_c),
            "error_metric": "per-byte embedding MSE" if name == "diffusion" else "per-byte cross-entropy nats",
        }
    np.savez_compressed(output / "calibration_predictions.npz", **arrays)
    calibrated = output / "phase5_calibrated"; calibrated.mkdir(exist_ok=True)
    torch.save(diffusion.state_dict(), calibrated / "diffusion.pt")
    torch.save(generator.state_dict(), calibrated / "generator.pt")
    (output / "calibration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    img, draw = canvas("Confidence calibration and observed error", "Held-out test data; confidence is correctness probability, not a direct error estimate")
    for column, name in enumerate(("diffusion", "generator")):
        stat = report["models"][name]["test"]
        rows = stat["bins"]
        x0 = 30 + column * 790
        line_panel(draw, (x0, 110, x0 + 760, 505), f"{name.title()}: reliability (ECE {stat['ece']:.3f}, MCE {stat['mce']:.3f})",
            [("observed", [(r["mean_confidence"], r["accuracy"]) for r in rows], "#087e8b")],
            x_label="Predicted confidence", y_label="Accuracy", diagonal=True)
        ymax = max([r["mean_error"] for r in rows] + [0.001]) * 1.15
        line_panel(draw, (x0, 520, x0 + 760, 930), f"{name.title()}: mean observed error",
            [("observed", [(r["mean_confidence"], r["mean_error"]) for r in rows], "#d36a3b")],
            x_label="Predicted confidence", y_label="MSE" if name == "diffusion" else "Cross-entropy", y_range=(0, ymax))
    save(img, output / "calibration.png")
    if args.write_config:
        write_config(config_path, all_thresholds)
    print(json.dumps({name: {"test_ece": report["models"][name]["test"]["ece"],
                              "test_mce": report["models"][name]["test"]["mce"],
                              "thresholds": all_thresholds[name]} for name in all_thresholds}, indent=2))
    print(f"Wrote {output / 'calibration.png'}, calibration.json, per-byte calibration_predictions.npz, and separate calibrated checkpoints.")


if __name__ == "__main__":
    main()
