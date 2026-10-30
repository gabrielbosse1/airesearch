"""Train the ByteCore v2 foundation: plain next-byte prediction on ReAct bytes.

There is no special objective here. Every byte of every trajectory -- the
thought, the action, the observation the sandbox returned and the answer -- is
predicted by the same cross-entropy over a standard causal Transformer. Tool
use, copying and format adherence are therefore learned as language modelling
rather than bolted on.

The same code path runs the CPU test configuration and the production
configuration; only the YAML changes.

    python -m training.train_foundation --epochs 4
    python -m training.train_foundation --config configs/default.yaml --preset production --limit 2000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data.scalable_react_dataset import (Trajectory, build_all, dataset_stats,
                                         write_jsonl)
from models.foundation import (PAD, FoundationConfig, CausalByteTransformer,
                               encode_bytes, load_foundation, pad_batch, save_foundation)
from models.react import extract_answer

SCHEMA_VERSION = 2


def load_configuration(path: str, preset: str | None, overrides: dict[str, int | float]) -> tuple[FoundationConfig, dict]:
    """Read ``configs/default.yaml``; CLI values override the file."""
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    section = dict(document.get("foundation") or {})
    presets = dict(section.pop("presets", {}) or {})
    if preset:
        if preset not in presets:
            raise ValueError(f"unknown preset {preset!r}; available: {sorted(presets)}")
        section.update(presets[preset] or {})
    training = dict(document.get("foundation_training") or {})
    section.update({key: value for key, value in overrides.items() if value is not None})
    return FoundationConfig.from_mapping(section), training


def read_jsonl(path: Path) -> list[Trajectory]:
    rows: list[Trajectory] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            rows.append(Trajectory(
                request=payload["request"].encode("latin-1"),
                body=payload["body"].encode("latin-1"),
                memory=payload["memory"].encode("latin-1"),
                kind=payload["kind"],
                text=payload["text"].encode("latin-1"),
            ))
    return rows


def _encode_all(rows: list[Trajectory], max_len: int) -> list[torch.Tensor]:
    ids = [encode_bytes(row.text, bos=True, eos=True) for row in rows]
    return [sequence[:max_len] for sequence in ids]


def _batches(sequences: list[torch.Tensor], batch_size: int, shuffle: bool,
             generator: random.Random, token_budget: int = 8192):
    """Batches bounded by **tokens**, not just rows.

    Attention activations scale as ``rows * heads * length^2``, so a fixed row
    count can explode on long trajectories. Sorting by length inside a shuffled
    window and then greedily filling each batch to a token budget keeps peak
    memory flat and cuts padding at the same time.
    """
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    order = list(range(len(sequences)))
    if shuffle:
        generator.shuffle(order)
        order.sort(key=lambda index: int(sequences[index].shape[0]))
        batches: list[list[int]] = []
        current: list[int] = []
        longest = 0
        for index in order:
            length = int(sequences[index].shape[0])
            candidate = max(longest, length)
            if current and (len(current) + 1 > batch_size
                            or (len(current) + 1) * candidate > token_budget):
                batches.append(current)
                current, longest = [index], length
            else:
                current.append(index)
                longest = candidate
        if current:
            batches.append(current)
        generator.shuffle(batches)
        for indices in batches:
            chunk = [sequences[index] for index in indices]
            length = max(int(sequence.shape[0]) for sequence in chunk)
            yield pad_batch(chunk, length)
        return
    for start in range(0, len(order), batch_size):
        chunk = [sequences[index] for index in order[start:start + batch_size]]
        length = max(int(sequence.shape[0]) for sequence in chunk)
        yield pad_batch(chunk, length)


@torch.no_grad()
def evaluate(model: CausalByteTransformer, sequences: list[torch.Tensor], batch_size: int,
             token_budget: int = 8192) -> dict[str, float]:
    """Teacher-forced loss and next-byte accuracy (padding excluded)."""
    model.eval()
    total_loss, correct, counted = 0.0, 0, 0
    batches = 0
    for ids, targets in _batches(sequences, batch_size, shuffle=False, generator=random.Random(0),
                                 token_budget=token_budget):
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=PAD)
        mask = targets != PAD
        predictions = logits.argmax(dim=-1)
        correct += int(((predictions == targets) & mask).sum().item())
        counted += int(mask.sum().item())
        total_loss += float(loss.item())
        batches += 1
    return {"loss": total_loss / max(batches, 1), "next_byte_accuracy": correct / max(counted, 1),
            "tokens": counted}


@torch.no_grad()
def free_running_sample(model: CausalByteTransformer, prompt: bytes, max_new_bytes: int = 140) -> bytes:
    """Greedy continuation from a prompt -- what the agent actually does."""
    model.eval()
    ids = encode_bytes(prompt, bos=True)[-model.config.max_seq_len:]
    generated = model.generate(ids, max_new_bytes=max_new_bytes)
    return prompt + bytes(generated.tolist())


def _write_report(output: Path, config, model, statistics, longest, history, extra, *,
                  test: dict | None = None, sample: dict | None = None,
                  checkpoint: dict | None = None) -> dict:
    """Write (and return) foundation_metrics.json. Called after every epoch."""
    report = {
        "schema_version": SCHEMA_VERSION,
        "configuration": config.to_dict() | extra,
        "parameters": model.parameters_count,
        "dataset": statistics,
        "longest_trajectory_bytes": longest,
        "epochs": history,
        "final_validation": history[-1]["validation"] if history else None,
        "final_test": test,
        "checkpoint": checkpoint or {"path": str(output / "foundation.pt")},
        "sample_generation": sample,
        "selection": "fixed final epoch; no test-based checkpoint selection",
    }
    (output / "foundation_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--preset", default=None, help="optional scale preset from the YAML")
    parser.add_argument("--output-dir", default="runs/phase16")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap training trajectories")
    parser.add_argument("--rebuild", action="store_true", help="regenerate the dataset")
    parser.add_argument("--init-from", default=None,
                        help="continue from a checkpoint instead of starting fresh")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--token-budget", type=int, default=None,
                        help="max tokens per batch (memory is bounded by this)")
    args = parser.parse_args()

    config, training_defaults = load_configuration(
        args.config, args.preset,
        {"d_model": args.d_model, "n_layers": args.n_layers, "n_heads": args.n_heads,
         "d_ff": args.d_ff, "max_seq_len": args.max_seq_len, "dropout": args.dropout})
    epochs = int(args.epochs if args.epochs is not None else training_defaults.get("epochs", 4))
    batch_size = int(args.batch_size if args.batch_size is not None else training_defaults.get("batch_size", 32))
    lr = float(args.lr if args.lr is not None else training_defaults.get("lr", 0.003))
    seed = int(args.seed if args.seed is not None else training_defaults.get("seed", 16001))
    weight_decay = float(training_defaults.get("weight_decay", 0.01))
    token_budget = int(args.token_budget if args.token_budget is not None
                       else training_defaults.get("token_budget", 8192))
    warmup = int(training_defaults.get("warmup_epochs", 1))

    torch.manual_seed(seed)
    random.seed(seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    paths = {split: output / f"react_{split}.jsonl" for split in ("train", "validation", "test")}
    if args.rebuild or not all(path.exists() for path in paths.values()):
        print("building trajectories with the real sandbox (one run per snippet)...", flush=True)
        started = time.time()
        built = build_all()
        for split, rows in built.items():
            write_jsonl(rows, paths[split])
        print(f"dataset built in {time.time() - started:.1f}s", flush=True)
    raw = {split: read_jsonl(path) for split, path in paths.items()}
    if args.limit:
        raw["train"] = raw["train"][: args.limit]
    statistics = {split: dataset_stats(rows) for split, rows in raw.items()}
    print(json.dumps({"dataset": statistics}), flush=True)

    sequences = {split: _encode_all(rows, config.max_seq_len) for split, rows in raw.items()}
    longest = max(int(sequence.shape[0]) for sequence in sequences["train"])
    if longest > config.max_seq_len:
        raise ValueError(f"trajectory length {longest} exceeds max_seq_len {config.max_seq_len}")

    # --------------------------------------------------------------- model
    if args.init_from:
        # Continue training: weights and config both come from the checkpoint.
        model = load_foundation(args.init_from, device=args.device)
        print(json.dumps({"initialized_from": args.init_from}), flush=True)
    else:
        model = CausalByteTransformer(config).to(args.device)
    print(json.dumps({"config": config.to_dict(), "parameters": model.parameters_count}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = random.Random(seed)

    history: list[dict] = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        # Linear warmup then cosine decay: standard, and stable at small scale.
        # The decay is spread over (epochs - warmup + 1) steps so the final
        # epoch still learns: decaying to exactly zero on the last epoch would
        # waste it (progress = 1 gives rate = 0).
        if epoch <= warmup:
            rate = lr * epoch / max(warmup, 1)
        else:
            progress = (epoch - warmup) / max(epochs - warmup + 1, 1)
            rate = lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        for group in optimizer.param_groups:
            group["lr"] = rate

        running, correct, counted, batches = 0.0, 0, 0, 0
        for ids, targets in _batches(sequences["train"], batch_size, shuffle=True, generator=generator,
                                 token_budget=token_budget):
            ids, targets = ids.to(args.device), targets.to(args.device)
            logits = model(ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=PAD)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            mask = targets != PAD
            correct += int(((logits.argmax(dim=-1) == targets) & mask).sum().item())
            counted += int(mask.sum().item())
            running += float(loss.item())
            batches += 1

        validation = evaluate(model, sequences["validation"], batch_size, token_budget)
        record = {
            "epoch": epoch,
            "learning_rate": rate,
            "train_loss": running / max(batches, 1),
            "train_next_byte_accuracy": correct / max(counted, 1),
            "gradient_norm": float(norm.item()),
            "validation": validation,
            "elapsed_s": round(time.time() - started, 1),
        }
        history.append(record)
        # Checkpoint *and* metrics every epoch: a long CPU run can be stopped
        # at any point and the last completed epoch stays loadable and
        # documented, instead of leaving a bare checkpoint behind.
        save_foundation(model, output / "foundation.pt",
                        extra={"schema_version": SCHEMA_VERSION, "seed": seed, "epoch": epoch})
        _write_report(output, config, model, statistics, longest, history,
                      {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                       "weight_decay": weight_decay, "seed": seed, "warmup_epochs": warmup,
                       "device": str(args.device), "preset": args.preset, "complete": False})
        print(json.dumps({key: (round(value, 6) if isinstance(value, float) else value)
                          for key, value in record.items() if key != "validation"}
                         | {"val_loss": round(validation["loss"], 6),
                            "val_next_byte_accuracy": round(validation["next_byte_accuracy"], 6)}), flush=True)

    # ------------------------------------------------------------ artifacts
    test = evaluate(model, sequences["test"], batch_size, token_budget)
    sample_prompt = b"Request: Calculate 6 * 7.\nThought:"
    sample = free_running_sample(model, sample_prompt)
    checkpoint = save_foundation(model, output / "foundation.pt",
                                 extra={"schema_version": SCHEMA_VERSION, "seed": seed, "epochs": epochs})
    _write_report(output, config, model, statistics, longest, history,
                           {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                            "weight_decay": weight_decay, "seed": seed, "warmup_epochs": warmup,
                            "device": str(args.device), "preset": args.preset, "complete": True},
                           test=test,
                           sample={"prompt": sample_prompt.decode("latin-1"),
                                   "continuation": sample.decode("latin-1", "replace"),
                                   "answer": extract_answer(sample).decode("latin-1", "replace")},
                           checkpoint=checkpoint)
    print(json.dumps({"checkpoint": checkpoint, "test": test}, indent=2))


if __name__ == "__main__":
    main()
