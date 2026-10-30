"""Matched retraining experiment for the soft-DTW objective.

Both arms start from the same frozen-byte-embedding checkpoint and receive the
same minibatches, insert/delete corruptions, dropout and Gaussian noise seeds.
Only the alignment-loss weight differs. Selection uses validation, never test.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import ByteMultimodalDataset, collate
from evaluation.evaluate import corrupt_sequence, denoise_predictions
from models.bytecore_diffusion import ByteCoreDiffusion
from training.train import train_epoch


def edit_distance(a: list[int], b: list[int]) -> int:
    """Independent O(min(n,m)) memory Levenshtein distance for reporting."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1, previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


@torch.no_grad()
def assess(model: ByteCoreDiffusion, loader: DataLoader, *, seed: int, aligned: bool = False,
           records: bool = False) -> dict[str, Any]:
    model.eval()
    result: dict[str, Any] = {}
    for mode in ("clean", "insert", "delete"):
        rng = random.Random(seed + {"clean": 1, "insert": 2, "delete": 3}[mode])
        torch.manual_seed(seed + 900 + {"clean": 1, "insert": 2, "delete": 3}[mode])
        matches = total = exact = count = edits = distance_total = 0
        examples = []
        for batch in loader:
            clean = batch["input_ids"].to(model.betas.device)
            corrupted = clean.clone()
            for i, length in enumerate(batch["real_lens"]):
                if mode != "clean":
                    source = clean[i, :length].tolist()
                    corrupted[i, :length] = torch.tensor(corrupt_sequence(source, 0.1, mode, rng), device=clean.device)
            pred = denoise_predictions(model, corrupted)
            for i, length in enumerate(batch["real_lens"]):
                target = clean[i, :length].tolist(); output = pred[i, :length].tolist()
                c = sum(int(x == y) for x, y in zip(output, target))
                matches += c; total += length; count += 1; exact += int(output == target)
                if aligned or records:
                    distance = edit_distance(output, target)
                    edits += distance; distance_total += max(len(output), len(target))
                if records:
                    examples.append({"modality": batch["modalities"][i], "target_hex": bytes(target).hex(),
                                     "input_hex": bytes(corrupted[i, :length].tolist()).hex(),
                                     "output_hex": bytes(output).hex(), "position_matches": c,
                                     "length": length, "edit_distance": distance, "exact": output == target})
        row = {"bytes": total, "count": count, "position_accuracy": matches / max(total, 1),
               "sequence_exact": exact / max(count, 1)}
        if aligned or records:
            row["edit_aligned_similarity"] = 1 - edits / max(distance_total, 1)
        if records:
            row["examples"] = examples
        result[mode] = row
    return result


def clean_score(stats: dict[str, Any]) -> float:
    return stats["clean"]["position_accuracy"]


def recovery_score(stats: dict[str, Any]) -> float:
    return (stats["insert"]["position_accuracy"] + stats["delete"]["position_accuracy"]) / 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-dir", default="runs/alignment")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-examples", type=int, default=1200)
    parser.add_argument("--val-examples", type=int, default=200)
    parser.add_argument("--test-examples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.00003)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--clean-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    if min(args.train_examples, args.val_examples, args.test_examples, args.epochs, args.batch_size) < 1 or args.lr <= 0 or args.alignment_weight <= 0 or not 0 <= args.clean_tolerance < 1:
        raise ValueError("experiment sizes, learning rate and alignment weight must be positive")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); mc, dc = cfg["model"], cfg["data"]
    def dataset(split: str, limit: int) -> Subset:
        ds = ByteMultimodalDataset(str(Path(dc["processed_dir"]) / f"{split}.jsonl"), max_len=dc["max_seq_len"])
        if not len(ds):
            raise ValueError("dataset split is empty")
        return Subset(ds, range(min(len(ds), limit)))
    train = dataset("train", args.train_examples); val = dataset("val", args.val_examples); test = dataset("test", args.test_examples)
    vloader = DataLoader(val, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    tloader = DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    base = ByteCoreDiffusion(vocab_size=mc["vocab_size"], d_model=mc["d_model"], n_layers=mc["n_layers"],
        n_heads=mc["n_heads"], d_ff=mc["d_ff"], max_len=mc["max_len"], diffusion_timesteps=mc["diffusion_timesteps"],
        concept_capacity=mc["concept_capacity"], concept_dim=mc["concept_dim"], dropout=mc["dropout"], device=args.device,
    ).load(args.runs_dir, map_location=args.device).eval()
    original_val = assess(base, vloader, seed=451)
    original_test = assess(base, tloader, seed=452, aligned=True, records=True)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    arms: dict[str, Any] = {}
    for name, weight in (("no_alignment_retrained", 0.0), ("soft_dtw_retrained", args.alignment_weight)):
        model = copy.deepcopy(base)
        opt = torch.optim.Adam(model.core.parameters(), lr=args.lr)
        best_state = None; best_score = -1.0; best_epoch = None
        history = []
        for epoch in range(1, args.epochs + 1):
            seed = 1000 + epoch
            torch.manual_seed(seed)
            shuffle = torch.Generator().manual_seed(seed)
            loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=shuffle, collate_fn=collate)
            train_loss = train_epoch(model, loader, opt, args.device, alignment_loss_weight=weight,
                                     alignment_gamma=float(cfg["training"].get("alignment_gamma", 0.02)),
                                     edit_noise_probability=0.5, edit_noise_fraction=0.1,
                                     rng=random.Random(seed), timestep_limit=min(8, mc["diffusion_timesteps"]),
                                     confidence_weight=0.0)
            stats = assess(model, vloader, seed=451)
            score = recovery_score(stats)
            eligible = clean_score(stats) >= clean_score(original_val) - args.clean_tolerance
            history.append({"epoch": epoch, "training_loss": train_loss, "validation": stats, "eligible": eligible})
            if eligible and score > best_score:
                best_score = score; best_state = copy.deepcopy(model.state_dict()); best_epoch = epoch
            print(f"{name} epoch {epoch}/{args.epochs}: loss={train_loss:.5f} val_clean={clean_score(stats):.4f} val_insdel={score:.4f} eligible={eligible}", flush=True)
        if best_state is None:
            # Keep a result even when no checkpoint satisfies the clean criterion.
            best_state = copy.deepcopy(model.state_dict()); best_epoch = args.epochs
            selection = "no epoch met clean tolerance; last epoch retained for diagnosis, not recommended"
        else:
            selection = "best validation mean insert/delete recovery among clean-tolerance-eligible epochs"
        model.load_state_dict(best_state)
        model.eval()
        test_stats = assess(model, tloader, seed=452, aligned=True, records=True)
        torch.save(model.state_dict(), output / f"{name}.pt")
        arms[name] = {"alignment_loss_weight": weight, "selected_epoch": best_epoch, "selection": selection,
                      "history": history, "test": test_stats,
                      "confidence_calibrated": bool(model.core.confidence_head.calibrated)}
    report = {"protocol": {"train_examples": len(train), "validation_examples": len(val), "test_examples": len(test),
              "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
              "seed_pairing": "same initialization, minibatch order, corruptions and torch random seeds per epoch",
              "corruption": "10% random byte insertions cropped or deletions padded to original length; light Gaussian noise",
              "selection": f"validation recovery with clean tolerance {args.clean_tolerance:.3f}; held-out test not used for selection",
              "metric": "position accuracy compares against original at original indices; edit similarity uses Levenshtein realignment",
              "warning": "Fixed-length denoising cannot infer missing random bytes; alignment loss itself is training-only. Experiment is synthetic."},
              "baseline_validation": original_val, "baseline_test": original_test, "arms": arms}
    path = output / "metrics.json"; path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    def brief(stats: dict[str, Any]) -> dict[str, Any]:
        return {k: {"position_accuracy": v["position_accuracy"], "edit_aligned_similarity": v.get("edit_aligned_similarity")}
                for k, v in stats.items()}
    print(json.dumps({"baseline_test": brief(original_test), **{k: brief(v["test"]) for k, v in arms.items()}}, indent=2))


if __name__ == "__main__":
    main()
