"""Training loop for ByteCore-Diffusion (denoising score matching on raw bytes)."""

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


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def corrupt_edit_batch(ids, lengths, *, rng, probability=0.5, fraction=0.1):
    """Produce paired insertion/deletion inputs at the original tensor width.

    Insertions are cropped and deletions are padded with byte 0, exactly as in
    the position-wise robustness benchmark; the clean target mask distinguishes
    literal NUL bytes from padding. This is a fixed-length corruption task, not
    proof that the model can infer the true output length.
    """
    if not 0 <= probability <= 1 or not 0 <= fraction <= 1:
        raise ValueError("edit corruption probability and fraction must be in [0,1]")
    if ids.ndim != 2 or len(lengths) != ids.shape[0]:
        raise ValueError("ids and lengths must agree")
    result = ids.clone()
    for i, raw_length in enumerate(lengths):
        length = int(raw_length)
        if not 0 <= length <= ids.shape[1]:
            raise ValueError("sequence length exceeds its tensor")
        if not length or not fraction or rng.random() >= probability:
            continue
        source = ids[i, :length].tolist()
        k = min(length, max(1, round(length * fraction)))
        if rng.randrange(2) == 0:
            sites = set(rng.sample(range(length + 1), k))
            edited = []
            for j in range(length + 1):
                if j in sites:
                    edited.append(rng.randrange(256))
                if j < length:
                    edited.append(source[j])
            edited = edited[:length]
        else:
            removed = set(rng.sample(range(length), k))
            edited = [byte for j, byte in enumerate(source) if j not in removed]
            edited.extend([0] * (length - len(edited)))
        result[i, :length] = torch.tensor(edited, dtype=ids.dtype, device=ids.device)
    return result


def train_epoch(model, loader, opt, device, *, alignment_loss_weight=0.0, alignment_gamma=0.02,
                edit_noise_probability=0.0, edit_noise_fraction=0.1, rng=None, timestep_limit=None,
                confidence_weight=0.1):
    model.train()
    total_loss = 0.0
    rng = rng if rng is not None else random.Random()
    for batch in loader:
        x0 = batch["input_ids"].to(device)
        x0_emb = model.bytes_to_emb(x0)
        lengths = torch.tensor(batch["real_lens"], device=device)
        valid = torch.arange(x0.shape[1], device=device)[None, :] < lengths[:, None]
        corrupted = corrupt_edit_batch(x0, batch["real_lens"], rng=rng, probability=edit_noise_probability,
                                       fraction=edit_noise_fraction) if edit_noise_probability else None
        source_emb = model.bytes_to_emb(corrupted) if corrupted is not None else None
        opt.zero_grad()
        loss = model.compute_loss(x0_emb, valid_mask=valid, corrupted_emb=source_emb,
                                  alignment_loss_weight=alignment_loss_weight, alignment_gamma=alignment_gamma,
                                  timestep_limit=timestep_limit, confidence_weight=confidence_weight)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, *, alignment_loss_weight=0.0, alignment_gamma=0.02, timestep_limit=None):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        x0 = batch["input_ids"].to(device)
        x0_emb = model.bytes_to_emb(x0)
        lengths = torch.tensor(batch["real_lens"], device=device)
        valid = torch.arange(x0.shape[1], device=device)[None, :] < lengths[:, None]
        loss = model.compute_loss(x0_emb, valid_mask=valid, alignment_loss_weight=alignment_loss_weight,
                                  alignment_gamma=alignment_gamma, timestep_limit=timestep_limit)
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
        lengths = torch.tensor(batch["real_lens"], device=device)
        valid = torch.arange(l, device=device)[None, :] < lengths[:, None]
        correct += int(((pred_bytes == x0) & valid).sum().item())
        total += int(valid.sum().item())
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
    torch.manual_seed(int(tr_cfg.get("seed", 42)))
    edit_rng = random.Random(int(tr_cfg.get("seed", 42)) + 1)

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
        alignment_weight = float(tr_cfg.get("alignment_loss_weight", 0.0))
        alignment_gamma = float(tr_cfg.get("alignment_gamma", 0.02))
        timestep_limit = tr_cfg.get("timestep_limit")
        train_loss = train_epoch(model, loader_train, opt, device, alignment_loss_weight=alignment_weight,
                                 alignment_gamma=alignment_gamma, timestep_limit=timestep_limit,
                                 edit_noise_probability=float(tr_cfg.get("edit_noise_probability", 0.0)),
                                 edit_noise_fraction=float(tr_cfg.get("edit_noise_fraction", 0.1)), rng=edit_rng)
        val_loss = evaluate(model, loader_val, device, alignment_loss_weight=alignment_weight,
                            alignment_gamma=alignment_gamma, timestep_limit=timestep_limit)
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
