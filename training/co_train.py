"""Co-training loop: MemoryModel <-> GeneratorConditioned.

The two models improve *each other* rather than being trained in isolation:

  1. Generator -> Memory pressure: the generator must produce output that is
     consistent with what memory knows about the input. The consistency loss
     pushes the generator's expected output embedding toward the memory
     context retrieved for that input — so the generator learns to be
     "memorable" and consistent with long-term memory.

  2. Memory -> Generator pressure: the memory model's retrieved context must
     actually help the generator predict the right next bytes. The memory
     encoders receive gradients through the generator's loss so memory learns
     to be a better context provider.

This mirrors the user's request: "the model that generates later is trained so
it understands better what the model that remember does, and the opposite."

External examples adjust memory *and* knowledge: new (input, output, context)
triples are written to memory (fast write) and then both models co-train on
them, so a skill learned from several examples becomes stable knowledge
without full retraining.
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.memory_model import MemoryModel
from models.generator_conditioned import GeneratorConditioned


# --------------------------------------------------------------------------- #
# Small helpers (defined first so they can be used below)
# --------------------------------------------------------------------------- #
def math_sqrt(x: float) -> float:
    return float(x) ** 0.5


def _byte_mean_embedding(ids, byte_emb_weight, pad_id: int = 0):
    """Mean byte embedding of a sequence (differentiable)."""
    mask = (ids != pad_id).float().unsqueeze(-1)  # (B, L, 1)
    emb = byte_emb_weight[ids.clamp(max=255)]  # (B, L, d)
    emb = emb * mask
    return emb.sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def _expected_embedding(logits, byte_emb_weight, pad_id: int = 0):
    """Differentiable expected byte embedding from generator logits.

    logits: (B, L, V). Returns (B, d) mean expected byte embedding.
    """
    probs = F.softmax(logits, dim=-1)  # (B, L, V)
    byte_probs = probs[..., : GeneratorConditioned.VOCAB_BYTES]  # (B, L, 256)
    byte_probs = byte_probs / byte_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    emb = byte_probs @ byte_emb_weight  # (B, L, d)
    return emb.mean(dim=1)  # (B, d)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ExampleDataset(Dataset):
    """External examples for skill/memory acquisition.

    Each item: {"input": [byte ids], "output": [byte ids], "context": dict}
    where context holds {person, modality, session, type, ...}.
    """

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch):
    max_in = max(len(e["input"]) for e in batch)
    max_out = max(len(e["output"]) for e in batch)
    pad = 0
    input_ids = torch.full((len(batch), max_in), pad, dtype=torch.long)
    output_ids = torch.full((len(batch), max_out), pad, dtype=torch.long)
    contexts = []
    for i, e in enumerate(batch):
        input_ids[i, : len(e["input"])] = torch.tensor(e["input"], dtype=torch.long)
        output_ids[i, : len(e["output"])] = torch.tensor(e["output"], dtype=torch.long)
        contexts.append(e["context"])
    return {"input_ids": input_ids, "output_ids": output_ids, "contexts": contexts}


def encode_text(text: str) -> list:
    return list(text.encode("utf-8"))


# --------------------------------------------------------------------------- #
# Core co-training step
# --------------------------------------------------------------------------- #
def co_train_step(
    memory: MemoryModel,
    generator: GeneratorConditioned,
    batch,
    device,
    opt_mem,
    opt_gen,
    lambda_consistency: float = 1.0,
    lambda_retrieval: float = 1.0,
):
    """One co-training step on a batch of examples.

    Returns dict of losses for logging.
    """
    input_ids = batch["input_ids"].to(device)      # (B, Lin)
    output_ids = batch["output_ids"].to(device)    # (B, Lout)
    contexts = batch["contexts"]
    B = input_ids.shape[0]
    pad_id = 0

    byte_emb_weight = generator.byte_emb.weight  # (256, d)

    # ---- 1. Fast write inputs into persistent memory (no_grad, inference) ----
    with torch.no_grad():
        input_emb = generator.byte_emb(input_ids.clamp(max=255)).mean(dim=1)  # (B, d)
    for i in range(B):
        memory.write(
            input_emb[i],
            {**contexts[i], "type": "example_input"},
        )

    # ---- 2. Differentiable memory retrieval for each input ----
    keys, values = memory.encode(input_emb)  # (B, d), (B, d)
    queries = memory.query_encoder(input_emb)  # (B, d)

    logits = (queries @ keys.T) / math_sqrt(memory.d_model)  # (B, B)
    attn = F.softmax(logits, dim=-1)  # (B, B)
    context = attn @ values  # (B, d) retrieved context per input

    # ---- 3. Generator produces output conditioned on retrieved memory ----
    start = torch.full(
        (B, 1), GeneratorConditioned.VOCAB_BYTES, dtype=torch.long, device=device
    )
    gen_input = torch.cat([start, output_ids[:, :-1]], dim=1)  # (B, Lout)
    gen_logits = generator(gen_input, context)  # (B, Lout, V)
    V = GeneratorConditioned.VOCAB_SIZE

    # Generator next-byte loss on the real output (teacher forcing).
    gen_loss = F.cross_entropy(
        gen_logits.reshape(-1, V), output_ids.reshape(-1), ignore_index=pad_id
    )

    # ---- 4. Memory -> Generator pressure (retrieval quality) ----
    # Retrieved context should be similar to the real output embedding.
    real_out_emb = _byte_mean_embedding(output_ids, byte_emb_weight, pad_id)  # (B, d)
    context_norm = F.normalize(context, dim=-1)
    real_norm = F.normalize(real_out_emb, dim=-1)
    retrieval_sim = (context_norm * real_norm).sum(dim=-1).mean()
    retrieval_loss = -retrieval_sim  # memory maximizes relevance

    # ---- 5. Generator -> Memory pressure (consistency) ----
    # Generator's expected output embedding should match retrieved memory context.
    gen_out_emb = _expected_embedding(gen_logits, byte_emb_weight, pad_id)  # (B, d)
    g_norm = F.normalize(gen_out_emb, dim=-1)
    c_norm = F.normalize(context.detach(), dim=-1)  # context fixed for gen step
    consistency_sim = (g_norm * c_norm).sum(dim=-1).mean()
    consistency_loss = -consistency_sim  # generator matches memory context

    # ---- 6. Combined objectives ----
    gen_total = gen_loss + lambda_consistency * consistency_loss
    mem_total = retrieval_loss

    opt_gen.zero_grad()
    gen_total.backward(retain_graph=True)
    opt_gen.step()

    opt_mem.zero_grad()
    mem_total.backward()
    opt_mem.step()

    # Slow consolidation of hot memory slots (turn examples into stable knowledge).
    memory.consolidate(lr=0.005)

    return {
        "gen_loss": float(gen_loss.item()),
        "retrieval_sim": float(retrieval_sim.item()),
        "consistency_sim": float(consistency_sim.item()),
        "consistency_loss": float(consistency_loss.item()),
    }


# --------------------------------------------------------------------------- #
# External example ingestion
# --------------------------------------------------------------------------- #
def ingest_examples(
    memory: MemoryModel,
    generator: GeneratorConditioned,
    examples: list,
    device,
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 1e-3,
):
    """Ingest external examples: write to memory (fast) then co-train so both
    models adjust their knowledge. A skill learned from several examples
    becomes stable memory without full retraining."""
    opt_mem = torch.optim.Adam(memory.parameters(), lr=lr)
    opt_gen = torch.optim.Adam(generator.parameters(), lr=lr)

    ds = ExampleDataset(examples)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    logs = []
    for epoch in range(epochs):
        epoch_logs = []
        for batch in dl:
            step = co_train_step(
                memory, generator, batch, device, opt_mem, opt_gen
            )
            epoch_logs.append(step)
        avg = {
            k: sum(d[k] for d in epoch_logs) / len(epoch_logs) for k in epoch_logs[0]
        } if epoch_logs else {}
        logs.append(avg)
        print(
            f"[ingest epoch {epoch+1}/{epochs}] "
            f"gen_loss {avg.get('gen_loss',0):.4f} "
            f"retrieval_sim {avg.get('retrieval_sim',0):.4f} "
            f"consistency_sim {avg.get('consistency_sim',0):.4f}",
            flush=True,
        )
    return logs


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_system(memory, generator, dirpath: str):
    import os
    os.makedirs(dirpath, exist_ok=True)
    memory.save(os.path.join(dirpath, "memory.pt"))
    torch.save(generator.state_dict(), os.path.join(dirpath, "generator.pt"))


def load_system(memory, generator, dirpath: str, map_location="cpu"):
    memory.load(os.path.join(dirpath, "memory.pt"), map_location=map_location)
    generator.load_state_dict(
        torch.load(os.path.join(dirpath, "generator.pt"), map_location=map_location)
    )
    return memory, generator
