"""Training loop for ByteCore-Diffusion (denoising score matching on raw bytes)."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from data.dataset import ByteMultimodalDataset, collate
from models.bytecore_diffusion import ByteCoreDiffusion


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x0 = batch["input_ids"].to(device)  # (B, L) byte ids
        x0_emb = model.bytes_to_emb(x0)  # (B, L, d)
        opt.zero_grad()
        loss = model.compute_loss(x0_emb)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        x0 = batch["input_ids"].to(device)
        x0_emb = model.bytes_to_emb(x0)
        loss = model.compute_loss(x0_emb)
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def reconstruction_accuracy(model, loader, device, n_batches=25):
    """Fraction of byte positions exactly reconstructed after a denoising
    pass at a low noise level (t near 0). Measures how well the model
    recovers clean bytes from a slightly noised input."""
    model.eval()
    correct = total = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        x0 = batch["input_ids"].to(device)  # (B, L)
        b, l = x0.shape
        x0_emb = model.bytes_to_emb(x0)
        # Lightly noise (t ~ 0.05)
        t_int = torch.full((b,), 2, device=device, dtype=torch.long)
        xt_emb = model.q_sample(x0_emb, t_int)
        pred_emb = model.core(xt_emb, t_int.float() / model.diffusion_timesteps)
        pred_bytes = model.core.embed_to_bytes(pred_emb)  # (B, L)
        correct += int((pred_bytes == x0).sum().item())
        total += b * l
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = args.device
    mcfg = cfg["model"]
    tr_cfg = cfg["training"]
    d_cfg = cfg["data"]

    os.makedirs(args.runs_dir, exist_ok=True)

    train_ds = ByteMultimodalDataset(
        os.path.join(d_cfg["processed_dir"], "train.jsonl"),
        max_len=d_cfg["max_seq_len"])
    val_ds = ByteMultimodalDataset(
        os.path.join(d_cfg["processed_dir"], "val.jsonl"),
        max_len=d_cfg["max_seq_len"])

    loader_train = DataLoader(train_ds, batch_size=tr_cfg["batch_size"],
                              shuffle=True, collate_fn=collate)
    loader_val = DataLoader(val_ds, batch_size=tr_cfg["batch_size"],
                            shuffle=False, collate_fn=collate)

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
    print(f"[ByteCore-Diffusion] params: {count_params(model):,}  "
          f"({count_params(model) * 4 / 1e6:.2f} MB)")

    opt = torch.optim.Adam(model.core.parameters(), lr=tr_cfg["lr"])
    # concept bank is NOT trained (memory writes only).

    best_path = os.path.join(args.runs_dir, "bytecore_diffusion_best.pt")
    best_val = float("inf")
    for epoch in range(1, tr_cfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, loader_train, opt, device)
        val_loss = evaluate(model, loader_val, device)
        print(f"[epoch {epoch:2d}] train_loss {train_loss:.4f} "
              f"val_loss {val_loss:.4f}  ({time.time() - t0:.1f}s)",
              flush=True)
        if val_loss <= best_val:
            best_val = val_loss
            model.save(args.runs_dir)  # model + concept bank (empty)
    print(f"Best val loss: {best_val:.4f} -> {best_path}")

    # Reconstruction accuracy of the best model.
    model.load(args.runs_dir, map_location=device)
    rec_acc = reconstruction_accuracy(model, loader_val, device)
    print(f"[reconstruction accuracy] {rec_acc:.4f}")

    with open(os.path.join(args.runs_dir, "results.json"), "w") as f:
        json.dump({
            "params": count_params(model),
            "params_mb": count_params(model) * 4 / 1e6,
            "best_val_loss": best_val,
            "reconstruction_accuracy": rec_acc,
        }, f, indent=2)


if __name__ == "__main__":
    main()
