"""Scoped, paired adaptive-computation experiments with untouched test data.

Diffusion: reverse a q-sampled held-out byte sequence. Confidence temperature
is fitted on one validation subset and thresholds selected on another. Test
sequences are exact-byte-disjoint from train and validation. This does NOT
calibrate unconditional generation or establish out-of-distribution safety.

THINK: no held-out data contain both success and failure in the synthetic task,
so the head cannot be temperature-calibrated there. We report a clearly marked
exploratory policy against both four-step and existing one-step baselines.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.bytecore_diffusion import ByteCoreDiffusion
from training.train_cognitive import task
from training.train_confidence import load_generator


def load_rows(root: str, per_modality: int) -> tuple[list[dict], list[dict], list[dict]]:
    def read(split: str) -> list[dict]:
        return [json.loads(line) for line in Path(root, split + ".jsonl").read_text().splitlines() if line.strip()]
    train, val, test = read("train"), read("val"), read("test")
    train_bytes = {bytes(r["bytes"]) for r in train}
    val_bytes = {bytes(r["bytes"]) for r in val}
    subsets: list[list[dict]] = [[], [], []]
    for modality in ("text", "image", "audio"):
        def unique(rows: list[dict], forbidden: set[bytes]) -> list[dict]:
            seen = set(forbidden)
            result = []
            for r in rows:
                b = bytes(r["bytes"])
                if r["modality"] == modality and b and b not in seen:
                    seen.add(b); result.append(r)
            random.Random(810 + len(modality)).shuffle(result)
            return result
        v = unique(val, train_bytes)
        t = unique(test, train_bytes | val_bytes)
        if len(v) < 2 * per_modality or len(t) < per_modality:
            raise ValueError(f"not enough exact-byte-distinct {modality} examples")
        subsets[0].extend(v[:per_modality]); subsets[1].extend(v[per_modality:2*per_modality])
        subsets[2].extend(t[:per_modality])
    return subsets[0], subsets[1], subsets[2]


@torch.no_grad()
def trajectory(model: ByteCoreDiffusion, record: dict, seed: int, input_width: int) -> dict[str, Any]:
    """Full paired trajectory, using the same input padding as model training."""
    real_length = len(record["bytes"])
    if real_length > input_width:
        raise ValueError("test sequence exceeds training input width")
    ids = torch.zeros((1, input_width), dtype=torch.long, device=model.betas.device)
    ids[0, :real_length] = torch.tensor(record["bytes"], dtype=torch.long, device=ids.device)
    t_max = torch.tensor([model.diffusion_timesteps - 1], dtype=torch.long, device=ids.device)
    torch.manual_seed(seed)
    initial = model.q_sample(model.bytes_to_emb(ids), t_max)
    random_state = torch.random.get_rng_state().clone()
    xt = initial.clone()
    predictions, logits = [], []
    final = None
    for t in range(model.diffusion_timesteps - 1, -1, -1):
        ti = torch.tensor([t], dtype=torch.long, device=ids.device)
        pred, conf_logits = model.core(xt, ti.float() / model.diffusion_timesteps, None, return_confidence=True)
        predictions.append(model.core.embed_to_bytes(pred)[0, :real_length].cpu())
        logits.append(conf_logits[0].cpu())
        ab_t = model.alpha_bars[ti].view(1, 1, 1)
        ab_prev = model.alpha_bars[ti - 1].clone()
        if t == 0:
            ab_prev.fill_(1.0)
        ab_prev = ab_prev.view(1, 1, 1)
        beta, alpha = model.betas[ti].view(1, 1, 1), model.alphas[ti].view(1, 1, 1)
        mean = ab_prev.sqrt() * beta / (1.0 - ab_t) * pred + alpha.sqrt() * (1.0 - ab_prev) / (1.0 - ab_t) * xt
        if t > 0:
            xt = mean + beta.sqrt() * torch.randn_like(mean)
        else:
            final = model.core.embed_to_bytes(mean)[0, :real_length].cpu()
    return {"target": ids[0, :real_length].cpu(), "predictions": torch.stack(predictions), "logits": torch.stack(logits),
            "full": final, "initial": initial.cpu(), "random_state": random_state, "modality": record["modality"]}


def metrics(rows: list[dict], temperature: float, threshold: float | None, min_steps: int = 1,
            reduction: str = "min") -> dict[str, Any]:
    correct = total = exact = early = 0
    steps: list[int] = []
    changes: list[int] = []
    for row in rows:
        probs = torch.sigmoid(row["logits"] / temperature)
        confidence = probs.amin(dim=1) if reduction == "min" else probs.mean(dim=1)
        length = len(confidence)
        stop = length
        if threshold is not None:
            for i in range(length - 1):
                if i + 1 >= min_steps and float(confidence[i]) >= threshold:
                    stop = i + 1
                    break
        output = row["full"] if stop == length else row["predictions"][stop - 1]
        target = row["target"]
        got = int((output == target).sum()); full = int((row["full"] == target).sum())
        correct += got; total += len(target); exact += int(got == len(target)); steps.append(stop)
        early += int(stop < length); changes.append(got - full)
    return {"count": len(rows), "bytes": total, "correct_bytes": correct, "byte_accuracy": correct / total,
            "exact": exact, "mean_steps": sum(steps) / len(steps), "early_samples": early,
            "samples_with_fewer_correct_bytes_than_full": sum(x < 0 for x in changes),
            "samples_with_more_correct_bytes_than_full": sum(x > 0 for x in changes),
            "steps": steps}


def calibration_metrics(rows: list[dict], temperature: float) -> dict[str, float]:
    logits = torch.cat([row["logits"][::5, :len(row["target"])].reshape(-1) for row in rows])
    labels = torch.cat([(row["predictions"][::5] == row["target"].unsqueeze(0)).reshape(-1) for row in rows]).float()
    probabilities = torch.sigmoid(logits / temperature)
    bins = torch.clamp((probabilities * 10).long(), max=9)
    ece = 0.0
    for i in range(10):
        mask = bins == i
        if bool(mask.any()):
            ece += float(mask.float().mean() * (probabilities[mask].mean() - labels[mask].mean()).abs())
    return {"observations": len(labels), "correct_fraction": float(labels.mean()),
            "mean_confidence": float(probabilities.mean()), "ece_10_bins": ece}


def select_diffusion(rows: list[dict], temperature: float) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    full = metrics(rows, temperature, None)
    candidates = []
    for reduction in ("min", "mean"):
        for minimum in (1, 5, 10, 20, 35, 45):
            for threshold in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
                result = metrics(rows, temperature, threshold, minimum, reduction)
                if (result["early_samples"] > 0 and result["byte_accuracy"] >= full["byte_accuracy"]
                        and result["exact"] >= full["exact"]
                        and result["samples_with_fewer_correct_bytes_than_full"] == 0):
                    candidates.append((result["mean_steps"], -threshold, minimum, reduction, result))
    if not candidates:
        return None, full
    _, negative_threshold, minimum, reduction, selected = min(candidates, key=lambda x: x[:4])
    return {"threshold": -negative_threshold, "min_steps": minimum, "reduction": reduction,
            "validation": selected, "candidate_count": len(candidates)}, full


def diffusion_experiment(config: dict, checkpoint: str, per_modality: int) -> dict[str, Any]:
    mc, dc = config["model"], config["data"]
    model = ByteCoreDiffusion(vocab_size=mc["vocab_size"], d_model=mc["d_model"], n_layers=mc["n_layers"],
        n_heads=mc["n_heads"], d_ff=mc["d_ff"], max_len=mc["max_len"], diffusion_timesteps=mc["diffusion_timesteps"],
        concept_capacity=mc["concept_capacity"], concept_dim=mc["concept_dim"], dropout=mc["dropout"], device="cpu")
    model.load(checkpoint, map_location="cpu"); model.eval()
    calibration, selection, test = load_rows(dc["processed_dir"], per_modality)
    print("Collecting paired diffusion trajectories", flush=True)
    groups = [[trajectory(model, record, 10000 + group * 1000 + i, dc["max_seq_len"]) for i, record in enumerate(records)]
              for group, records in enumerate((calibration, selection, test))]
    # Subsample timesteps deterministically; all calibration rows remain grouped
    # separately from selection/test. Labels are byte predictions, not full-sequence
    # correctness and not an unconditional-sampling quality signal.
    logits = torch.cat([row["logits"][::5, :len(row["target"])].reshape(-1) for row in groups[0]])
    labels = torch.cat([(row["predictions"][::5] == row["target"].unsqueeze(0)).reshape(-1) for row in groups[0]]).float()
    temperature = model.core.confidence_head.calibrate(logits, labels, scope="diffusion_sampling")
    selected, validation_full = select_diffusion(groups[1], temperature)
    test_full = metrics(groups[2], temperature, None)
    test_selected = None
    if selected:
        test_selected = metrics(groups[2], temperature, selected["threshold"], selected["min_steps"], selected["reduction"])
    # Also verify production sampling if selection found no acceptable policy.
    check_policy = selected or {"threshold": 0.05, "min_steps": 1, "reduction": "mean"}
    paired_verified = True
    for row in groups[2][:min(5, len(groups[2]))]:
        torch.random.set_rng_state(row["random_state"])
        out, stat = model.p_sample_loop(tuple(row["initial"].shape), guidance_scale=1.0,
            initial_xt=row["initial"], confidence_threshold=check_policy["threshold"],
            min_steps=check_policy["min_steps"], confidence_reduction=check_policy["reduction"], return_stats=True)
        actual = model.core.embed_to_bytes(out)[0, :len(row["target"])].cpu()
        one = metrics([row], temperature, check_policy["threshold"], check_policy["min_steps"], check_policy["reduction"])
        expected_step = one["steps"][0]
        expected = row["full"] if expected_step == model.diffusion_timesteps else row["predictions"][expected_step - 1]
        paired_verified &= int(stat.steps[0]) == expected_step and torch.equal(actual, expected)
    if not paired_verified:
        raise AssertionError("paired simulator disagrees with p_sample_loop")
    aggressive = metrics(groups[2], temperature, 0.05, 1, "mean")
    return {"protocol": {"task": "q-sampled last-timestep reconstruction followed by reverse DDPM; not unconditional generation",
                         "disjoint": "calibration/threshold-selection/test distinct raw examples, exact train/validation exclusions; near duplicates remain",
                         "calibration": "byte-correctness temperature fit on calibration samples, every fifth timestep; correlated observations",
                         "selection": "grid on independent validation; require no per-sample byte loss and no aggregate byte/exact loss",
                         "baseline": "50 steps; training-width padding; one sample per run; identical initial state and per-step random draws",
                         "population_per_subset": len(calibration), "paired_production_check": paired_verified},
            "calibration": {"logits": len(logits), "correct_fraction": float(labels.mean()), "temperature": temperature,
                            "fit": calibration_metrics(groups[0], temperature),
                            "selection": calibration_metrics(groups[1], temperature),
                            "test": calibration_metrics(groups[2], temperature)},
            "selection": {"full": validation_full, "policy": selected},
            "test": {"full": test_full, "selected": test_selected, "aggressive_005_mean": aggressive},
            "default_enabled": False,
            "caveat": "Small scoped experiment; do not reuse this calibration for other noise, sequence, guidance or unconditional generation."}


@torch.no_grad()
def cognitive_runs(generator: Any, payloads: list[bytes], *, steps: int, threshold: float | None) -> dict[str, Any]:
    correct = 0; used: list[int] = []; outcomes: list[bool] = []; confidences: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for payload in payloads:
            for command in b"IDRM":
                source = bytes([command]) + b":" + payload
                expected = task(generator, source)[1]
                result = generator.cognitive_generate(source, max_steps=8, autonomous_think_steps=steps,
                                                       confidence_threshold=threshold)
                ok = bool(result.done and result.canvas == expected)
                correct += int(ok); outcomes.append(ok)
                used.append(sum(event.thought_steps for event in result.events))
                confidences.extend(event.confidence for event in result.events
                                   if event.action.token == generator.THINK and event.confidence is not None)
    return {"count": len(outcomes), "exact": correct, "exact_fraction": correct / len(outcomes),
            "mean_think_steps": sum(used) / len(used), "steps": used, "outcomes": outcomes,
            "observed_confidence_range": [min(confidences), max(confidences)] if confidences else None}


def cognitive_experiment(path: str, count: int) -> dict[str, Any]:
    generator = load_generator(path, "cpu", 4).eval()
    payloads = [bytes(x) for x in itertools.product(b"abcdefgh", repeat=3)]
    random.Random(23).shuffle(payloads)
    # The training checkpoint used payloads [:48]; Phase 5 used [48:112].
    val, test = payloads[112:112+count], payloads[112+count:112+2*count]
    if len(val) != count or len(test) != count:
        raise ValueError("not enough unseen synthetic payloads")
    val_full = cognitive_runs(generator, val, steps=4, threshold=None)
    candidates = []
    for threshold in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.8, 0.99):
        result = cognitive_runs(generator, val, steps=4, threshold=threshold)
        if (all(not full or early for full, early in zip(val_full["outcomes"], result["outcomes"]))
                and result["mean_think_steps"] < val_full["mean_think_steps"]):
            candidates.append((result["mean_think_steps"], -threshold, result))
    policy = None
    if candidates:
        _, neg, result = min(candidates, key=lambda item: (item[0], item[1]))
        policy = {"threshold": -neg, "validation": result,
                  "test": cognitive_runs(generator, test, steps=4, threshold=-neg)}
    test_full = cognitive_runs(generator, test, steps=4, threshold=None)
    test_default = cognitive_runs(generator, test, steps=1, threshold=None)
    return {"protocol": {"task": "synthetic I/D/R/M edits only; three-byte payloads",
                         "splits": "payload-disjoint from 48 training payloads and from Phase 5 test payloads",
                         "calibration": "not calibrated for thought; validation all-correct may not fit a binary temperature",
                         "selection": "experimental grid with no validation exact losses; no safety guarantee",
                         "comparison": "four-step policy is experimental; existing production default is one THINK step"},
            "validation_full": val_full, "policy": policy, "test_full": test_full,
            "test_existing_default_one_step": test_default, "default_enabled": False}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--generator", default="runs/dual_memory/generator.pt")
    p.add_argument("--output", default="runs/phase6/halting.json")
    p.add_argument("--per-modality", type=int, default=16)
    p.add_argument("--payloads", type=int, default=32)
    args = p.parse_args()
    if args.per_modality < 1 or args.payloads < 1:
        raise ValueError("sample counts must be positive")
    cfg = yaml.safe_load(Path(args.config).read_text())
    report = {"diffusion": diffusion_experiment(cfg, args.runs_dir, args.per_modality),
              "thought": cognitive_experiment(args.generator, args.payloads)}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"diffusion_selected": report["diffusion"]["selection"]["policy"],
                      "diffusion_test": report["diffusion"]["test"],
                      "thought_policy": report["thought"]["policy"],
                      "thought_test_full": report["thought"]["test_full"],
                      "thought_default": report["thought"]["test_existing_default_one_step"]}, indent=2))


if __name__ == "__main__":
    main()
