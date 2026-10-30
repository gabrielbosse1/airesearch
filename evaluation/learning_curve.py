"""Train/validation learning curves for co-training on raw byte continuations."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import ByteMultimodalDataset
from evaluation.diagnostics_common import ece
from evaluation.plotting import canvas, line_panel, save
from models.generator_conditioned import GeneratorConditioned, GeneratorState
from models.memory_model import MemoryModel
from training.co_train import ExampleDataset, _byte_mean_embedding, co_train_step, collate, ingest_examples
from training.train_confidence import load_generator


def make_examples(path: str, limit: int, *, exclude: set[bytes] | None = None) -> list[dict[str, Any]]:
    dataset = ByteMultimodalDataset(path, max_len=128)
    seen = set(exclude or ()); rows = []
    for index, sample in enumerate(dataset.samples):
        raw = bytes(sample["bytes"])
        if len(raw) < 4:
            continue
        cut = min(8, max(1, len(raw)//2))
        source, target = raw[:cut], raw[cut:cut+16]
        if not target or source in seen:
            continue
        seen.add(source)
        rows.append({"input": list(source), "output": list(target), "context": {
            "concept": str(sample["concept"]), "modality": str(sample["modality"]), "id": f"sample-{index}"}})
        if len(rows) >= limit:
            break
    if not rows:
        raise ValueError("no unique continuation examples found")
    return rows


@torch.no_grad()
def score(memory: MemoryModel, g: GeneratorConditioned, examples: list[dict[str, Any]], device: str) -> dict[str, Any]:
    g.eval(); memory.eval()
    loader = DataLoader(ExampleDataset(examples), batch_size=16, shuffle=False, collate_fn=collate)
    loss_sum = n_tokens = n_correct = retrieval_correct = n_queries = 0
    all_conf, all_correct, retrieval_details = [], [], []
    for batch in loader:
        inp = batch["input_ids"].to(device); imask = batch["input_mask"].to(device)
        output = batch["output_ids"].to(device); omask = batch["output_mask"].to(device)
        query = _byte_mean_embedding(inp, g.byte_emb.weight, imask)
        values, _, mask = memory.retrieve_differentiable(query, top_k=g.memory_top_k)
        records, logits, _, _ = memory._gather_candidates(memory.query_encoder(query))
        top = logits.argmax(dim=1).tolist() if records else [-1]*len(inp)
        for i, pos in enumerate(top):
            actual = records[pos][3].get("concept") if pos >= 0 else None
            expected = batch["contexts"][i].get("concept")
            hit = actual == expected
            retrieval_correct += int(hit); n_queries += 1
            retrieval_details.append({"expected": expected, "retrieved": actual, "hit": hit})
        start = torch.full((len(inp), 1), g.FUNC_CALL, dtype=torch.long, device=device)
        state = g(torch.cat([start, output[:, :-1]], dim=1), values, memory_mask=mask, return_state=True)
        assert isinstance(state, GeneratorState)
        logits_out = state.output
        loss = F.cross_entropy(logits_out.reshape(-1, g.VOCAB_SIZE), output.reshape(-1), reduction="none").reshape_as(output)
        generation = logits_out.clone(); generation[..., g.LOAD_SPAN:] = -torch.inf
        correct = generation.argmax(-1) == output
        loss_sum += float(loss[omask].sum()); n_tokens += int(omask.sum()); n_correct += int(correct[omask].sum())
        all_conf.append(state.confidence[omask].cpu().numpy()); all_correct.append(correct[omask].cpu().numpy())
    confidence = np.concatenate(all_conf); correct_np = np.concatenate(all_correct)
    return {"loss": loss_sum/max(n_tokens,1), "accuracy": n_correct/max(n_tokens,1),
            "ece": ece(confidence, correct_np), "retrieval_accuracy": retrieval_correct/max(n_queries,1),
            "retrieval_queries": n_queries, "bytes": n_tokens, "retrieval_details": retrieval_details}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--generator", default="runs/dual_memory/generator.pt")
    p.add_argument("--output", default="runs/learning_curve.json")
    p.add_argument("--plot", default="runs/learning_curve.png")
    p.add_argument("--train-examples", type=int, default=96)
    p.add_argument("--val-examples", type=int, default=64)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if min(args.train_examples, args.val_examples, args.epochs, args.batch_size) < 1 or args.lr <= 0:
        raise ValueError("training sizes and learning rate must be positive")
    cfg = yaml.safe_load(Path(args.config).read_text())["data"]
    train = make_examples(str(Path(cfg["processed_dir"]) / "train.jsonl"), args.train_examples)
    validation = make_examples(str(Path(cfg["processed_dir"]) / "val.jsonl"), args.val_examples,
                               exclude={bytes(row["input"]) for row in train})
    torch.manual_seed(620); random.seed(620)
    g = load_generator(args.generator, args.device, 4)
    m = MemoryModel(d_model=g.d_model, capacity=max(512, 4 * len(train))).to(args.device)
    ingest_examples(m, g, train, args.device, epochs=0)
    opt_m = torch.optim.Adam(m.parameters(), lr=args.lr); opt_g = torch.optim.Adam(g.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs + 1):
        train_stats = score(m, g, train, args.device); val_stats = score(m, g, validation, args.device)
        history.append({"epoch": epoch, "train": train_stats, "validation": val_stats})
        print(f"epoch {epoch}: train_nll={train_stats['loss']:.4f} val_nll={val_stats['loss']:.4f} "
              f"val_acc={val_stats['accuracy']:.4f} val_retrieval={val_stats['retrieval_accuracy']:.4f} val_ece={val_stats['ece']:.4f}", flush=True)
        if epoch == args.epochs:
            break
        shuffle = torch.Generator().manual_seed(621 + epoch)
        loader = DataLoader(ExampleDataset(train), batch_size=args.batch_size, shuffle=True, generator=shuffle, collate_fn=collate)
        for batch in loader:
            co_train_step(m, g, batch, args.device, opt_m, opt_g)
    report = {"protocol": {"task": "teacher-forced raw-byte continuation conditioned on top-K procedural memories",
                           "split": "unique validation prefixes excluded from training prefix set; validation outputs never written to memory",
                           "retrieval_accuracy": "top-1 stored metadata concept equals validation example concept, not exact-output replay",
                           "confidence": "ECE measures the current uncalibrated head; it is not an adaptive-halting validation",
                           "train_examples": len(train), "validation_examples": len(validation), "epochs": args.epochs,
                           "batch_size": args.batch_size, "learning_rate": args.lr}, "history": history}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2) + "\n")
    image, draw = canvas("Co-training learning curves", "Unique-prefix continuation task; held-out validation, metadata-label retrieval and uncalibrated confidence")
    panels = [("loss", "Teacher-forced NLL", (30, 110, 785, 500)),
              ("accuracy", "Next-byte accuracy", (815, 110, 1570, 500)),
              ("retrieval_accuracy", "Top-1 concept retrieval accuracy", (30, 520, 785, 920)),
              ("ece", "Confidence ECE (10 bins)", (815, 520, 1570, 920))]
    for key, title, rect in panels:
        values = [row[split][key] for row in history for split in ("train", "validation")]
        # Use labeled, data-dependent scales so small but real changes remain visible.
        if key == "loss":
            yrange = (max(0.0, min(values) * 0.97), max(values) * 1.03 + 0.001)
        else:
            yrange = (0.0, min(1.0, max(0.05, max(values) * 1.20 + 0.001)))
        series = [("train", [(row["epoch"], row["train"][key]) for row in history], "#087e8b"),
                  ("validation", [(row["epoch"], row["validation"][key]) for row in history], "#d36a3b")]
        line_panel(draw, rect, title, series, x_label="Epoch", y_label=key.replace("_", " "),
                   x_range=(0, max(args.epochs, 1)), y_range=yrange)
    save(image, args.plot)


if __name__ == "__main__":
    main()
