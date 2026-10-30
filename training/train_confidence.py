"""Train only the confidence heads and calibrate on held-out raw-byte data.

Example: python training/train_confidence.py --output-dir runs/calibrated
Base model weights are frozen. Validation fits temperature; the separate test
split reports NLL/Brier/ECE. Calibration is specific to the sampled noise and
next-byte distributions, not a guarantee for arbitrary deployment inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from data.dataset import ByteMultimodalDataset, collate
from models.bytecore_diffusion import ByteCoreDiffusion
from models.epistemic import ConfidenceHead
from models.generator_conditioned import GeneratorConditioned, GeneratorState


def load_generator(path: str, device: str, n_heads: int) -> GeneratorConditioned:
    state = torch.load(path, map_location=device, weights_only=True)
    d_model = int(state["byte_emb.weight"].shape[1])
    n_layers = len([k for k in state if k.endswith("self_attn.in_proj_weight") and k.startswith("blocks.")])
    d_ff = int(state["blocks.0.linear1.weight"].shape[0])
    max_len = int(state["pos_emb.weight"].shape[0])
    adapter_ids = sorted({int(k.split(".")[1]) for k in state if k.startswith("memory_adapters.")})
    inject_every = adapter_ids[1] - adapter_ids[0] if len(adapter_ids) > 1 else (n_layers if adapter_ids else 1)
    max_slots = int(state["memory_adapters.0.cross_attn.rank_emb.weight"].shape[0]) if adapter_ids else 64
    model = GeneratorConditioned(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                                 d_ff=d_ff, max_len=max_len, memory_inject_every=inject_every,
                                 max_memory_slots=max_slots).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def get_loader(path: str, max_len: int, limit: int, batch_size: int) -> DataLoader:
    ds = ByteMultimodalDataset(path, max_len=max_len)
    if not len(ds) or limit < 1:
        raise ValueError("confidence training requires a non-empty dataset and positive limit")
    return DataLoader(Subset(ds, range(min(len(ds), limit))), batch_size=batch_size,
                      shuffle=False, collate_fn=collate)


@torch.no_grad()
def collect_diffusion(model: ByteCoreDiffusion, loader: DataLoader, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    captured: list[torch.Tensor] = []
    handle = model.core.confidence_head.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach()))
    try:
        for batch in loader:
            ids = batch["input_ids"].to(model.betas.device)
            lengths = torch.tensor(batch["real_lens"], device=ids.device)
            mask = torch.arange(ids.shape[1], device=ids.device)[None, :] < lengths[:, None]
            clean = model.bytes_to_emb(ids)
            t = torch.randint(0, model.diffusion_timesteps, (ids.shape[0],), device=ids.device)
            noisy = model.q_sample(clean, t)
            prediction, _ = model.core(noisy, t.float() / model.diffusion_timesteps, return_confidence=True)
            correct = (model.core.embed_to_bytes(prediction) == ids).float()
            features.append(captured.pop()[mask].cpu())
            labels.append(correct[mask].cpu())
    finally:
        handle.remove()
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def collect_generator(model: GeneratorConditioned, loader: DataLoader, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        ids = batch["input_ids"].to(model.byte_emb.weight.device)
        inputs, targets = ids[:, :-1], ids[:, 1:]
        lengths = torch.tensor(batch["real_lens"], device=ids.device)
        mask = torch.arange(targets.shape[1], device=ids.device)[None, :] < (lengths - 1).clamp_min(0)[:, None]
        state = model(inputs, return_state=True)
        assert isinstance(state, GeneratorState)
        raw_logits = state.output.clone()
        # Normal generate() cannot select cognitive controls.
        raw_logits[..., model.LOAD_SPAN:] = -torch.inf
        correct = (raw_logits.argmax(dim=-1) == targets).float()
        features.append(torch.cat([state.hidden, state.thought], dim=-1)[mask].cpu())
        labels.append(correct[mask].cpu())
    return torch.cat(features), torch.cat(labels)


def metrics(logits: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0) -> dict[str, float]:
    if logits.numel() == 0:
        raise ValueError("no valid byte predictions were collected")
    probability = torch.sigmoid(logits / temperature)
    ece = 0.0
    for i in range(10):
        lower, upper = i / 10, (i + 1) / 10
        selected = (probability >= lower) & (probability < upper if i < 9 else probability <= upper)
        if bool(selected.any()):
            ece += float(selected.float().mean() * (probability[selected].mean() - targets[selected].mean()).abs())
    return {
        "count": int(targets.numel()),
        "correct_fraction": float(targets.mean()),
        "mean_confidence": float(probability.mean()),
        "nll": float(F.binary_cross_entropy_with_logits(logits / temperature, targets)),
        "brier": float((probability - targets).square().mean()),
        "ece_10_bins": ece,
    }


def fit_head(
    head: ConfidenceHead,
    train: tuple[torch.Tensor, torch.Tensor],
    validation: tuple[torch.Tensor, torch.Tensor],
    test: tuple[torch.Tensor, torch.Tensor],
    *,
    scope: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("training hyperparameters must be positive")
    device = head.temperature.device
    features, targets = train
    if features.shape[0] == 0 or not bool((targets == 0).any() and (targets == 1).any()):
        raise ValueError("training needs both correct and incorrect predictions")
    torch.manual_seed(seed)
    head.train()
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    history = []
    for _ in range(epochs):
        perm = torch.randperm(len(targets))
        total = 0.0
        for indices in perm.split(batch_size):
            x, y = features[indices].to(device), targets[indices].to(device)
            logits = head(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(indices)
        history.append(total / len(targets))
    head.eval()

    @torch.no_grad()
    def predict(data: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = data
        parts = [head(part.to(device)).cpu() for part in x.split(batch_size)]
        return torch.cat(parts), y

    val_logits, val_targets = predict(validation)
    test_logits, test_targets = predict(test)
    before = metrics(test_logits, test_targets)
    temperature = head.calibrate(val_logits, val_targets, scope=scope)
    after = metrics(test_logits, test_targets, temperature)
    return {"scope": scope, "epochs": epochs, "train_bce": history,
            "temperature": temperature, "test_before_temperature": before,
            "test_after_temperature": after}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-dir", default="runs")
    parser.add_argument("--generator", default="runs/dual_memory/generator.pt")
    parser.add_argument("--generator-heads", type=int, default=4)
    parser.add_argument("--output-dir", default="runs/calibrated")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-examples", type=int, default=1200)
    parser.add_argument("--eval-examples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mc, dc = cfg["model"], cfg["data"]
    torch.manual_seed(42)
    diffusion = ByteCoreDiffusion(
        vocab_size=mc["vocab_size"], d_model=mc["d_model"], n_layers=mc["n_layers"],
        n_heads=mc["n_heads"], d_ff=mc["d_ff"], max_len=mc["max_len"],
        diffusion_timesteps=mc["diffusion_timesteps"], concept_capacity=mc["concept_capacity"],
        concept_dim=mc["concept_dim"], dropout=mc["dropout"], device=args.device,
    ).load(args.model_dir, map_location=args.device).eval()
    generator = load_generator(args.generator, args.device, args.generator_heads)
    datasets = []
    for split, limit in (("train", args.train_examples), ("val", args.eval_examples), ("test", args.eval_examples)):
        datasets.append(get_loader(os.path.join(dc["processed_dir"], split + ".jsonl"), dc["max_seq_len"], limit, args.batch_size))
    report = {}
    for name, model, collector, scope in (
        ("diffusion", diffusion, collect_diffusion, "diffusion_denoising"),
        ("generator", generator, collect_generator, "next_byte"),
    ):
        print(f"Collecting {name} train/validation/test confidence labels...", flush=True)
        data = [collector(model, loader, 101 + i) for i, loader in enumerate(datasets)]
        head = diffusion.core.confidence_head if name == "diffusion" else generator.confidence_head
        report[name] = fit_head(head, *data, scope=scope, epochs=args.epochs,
                                batch_size=args.feature_batch_size, learning_rate=args.learning_rate, seed=42)
        print(f"{name}: {json.dumps(report[name])}", flush=True)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(diffusion.state_dict(), out / "diffusion.pt")
    torch.save(generator.state_dict(), out / "generator.pt")
    (out / "confidence_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved confidence-only checkpoints and held-out metrics to {out}")


if __name__ == "__main__":
    main()
