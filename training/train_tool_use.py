"""Train the Phase 15 tool-routing head and tool-argument decoder.

Only the two new heads are optimized. The Phase 12 generator is loaded, frozen
and verified bit-for-bit afterwards, so every Phase 10/12 decoder that is bound
to the generator checkpoint hash stays valid and the seven-way intent
classifier is untouched. Tool routing is an independent head by design: adding
an eighth intent class would resize the trained classifier and break strict
loading of the existing checkpoints.

Reproduction:
    OMP_NUM_THREADS=1 python -m training.train_tool_use \
        --generator runs/phase12/generator.pt --output-dir runs/phase15
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from data.tool_use_examples import ToolExample, tool_examples
from models.sandbox import ByteSandbox
from models.tool_use import ROUTE_NAMES, TOOL_NAMES, ToolArgumentDecoder, ToolCallPolicy, pack_tool_request
from training.train_confidence import load_generator


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def cache_rows(policy: ToolCallPolicy, rows: list[ToolExample], batch_size: int = 64):
    """Freeze the backbone view of every row once; the generator never moves."""
    width = max(len(frame) for frame, _ in
                (pack_tool_request(row.request, row.result, max_len=policy.generator.max_len) for row in rows))
    contexts, memories, masks, identifiers = [], [], [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        ids, _, hidden, mask = policy.encode([row.request for row in chunk],
                                             [row.result for row in chunk], width=width)
        contexts.append(policy.pool(hidden, mask))
        memories.append(hidden)
        masks.append(mask)
        identifiers.append(ids)
    routes = torch.tensor([row.route for row in rows], dtype=torch.long)
    return torch.cat(contexts), torch.cat(memories), torch.cat(masks), torch.cat(identifiers), routes


def argument_tensors(rows: list[ToolExample], device: torch.device):
    """Teacher-forced previous IDs and targets; padded rows use IGNORE."""
    width = max(len(row.argument) + 2 for row in rows)
    previous = torch.full((len(rows), width), ToolArgumentDecoder.START, dtype=torch.long, device=device)
    targets = torch.full((len(rows), width), ToolArgumentDecoder.IGNORE, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        if not row.tool:
            continue
        argument = list(row.argument)
        previous[index, 1:len(argument) + 1] = torch.tensor(argument, dtype=torch.long, device=device)
        targets[index, :len(argument)] = torch.tensor(argument, dtype=torch.long, device=device)
        targets[index, len(argument)] = ToolArgumentDecoder.EOS
    return previous, targets


def batch_loss(policy: ToolCallPolicy, batch: torch.Tensor, cached, tensors) -> tuple[torch.Tensor, dict]:
    contexts, memories, masks, identifiers, routes = cached
    previous, targets = tensors
    context, memory, mask = contexts[batch], memories[batch], masks[batch]
    memory_bytes = identifiers[batch]
    route_logits = policy.route_head(context)
    route_loss = F.cross_entropy(route_logits, routes[batch])
    active = routes[batch] != 0
    argument_loss = route_logits.sum() * 0.0
    correct_bytes = total_bytes = 0
    if bool(active.any()):
        rows = batch[active]
        # Log probabilities at position i are conditioned on previous[:i+1] and
        # predict targets[i]: START therefore predicts the first argument byte.
        log_probabilities = policy.argument_decoder(context[active], routes[rows], previous[rows][:, :-1],
                                                    memory[active], mask[active], memory_bytes[active])
        labels = targets[rows][:, :-1]
        loss = F.nll_loss(log_probabilities.reshape(-1, log_probabilities.shape[-1]), labels.reshape(-1),
                          ignore_index=ToolArgumentDecoder.IGNORE)
        argument_loss = loss
        predictions = log_probabilities.argmax(dim=-1)
        valid = labels != ToolArgumentDecoder.IGNORE
        correct_bytes = int(((predictions == labels) & valid).sum())
        total_bytes = int(valid.sum())
    total = route_loss + argument_loss
    metrics = {"route_loss": float(route_loss.detach()),
               "argument_loss": float(argument_loss.detach()),
               "route_correct": int((route_logits.argmax(dim=-1) == routes[batch]).sum()),
               "byte_correct": correct_bytes, "byte_total": total_bytes, "count": int(batch.numel())}
    return total, metrics


@torch.no_grad()
def route_metrics(policy: ToolCallPolicy, cached) -> dict:
    """Cheap cached-feature route loss/accuracy (no byte decoding)."""
    contexts, _, _, _, routes = cached
    logits = torch.cat([policy.route_head(contexts[start:start + 128])
                        for start in range(0, contexts.shape[0], 128)])
    return {"loss": float(F.cross_entropy(logits, routes)),
            "accuracy": float((logits.argmax(dim=-1) == routes).float().mean())}


@torch.no_grad()
def evaluate_rows(policy: ToolCallPolicy, rows: list[ToolExample], *, sample: int | None = None,
                  max_bytes: int = 64) -> dict:
    """Greedy decode: exact frame (route + argument) and route-only accuracy."""
    # Stratified across families: rows are generated family by family, so the
    # first N rows would only measure the first few families.
    if sample is None or sample >= len(rows):
        scored = rows
    else:
        step = max(len(rows) // sample, 1)
        scored = rows[::step][:sample]
    route_hits = frame_hits = 0
    calls = call_hits = declines = decline_hits = 0
    for row in scored:
        generation = policy.generate(row.request, row.result, threshold=0.0, max_bytes=max_bytes)
        if generation.route != row.route:
            continue
        route_hits += 1
        if row.tool:
            calls += 1
            if generation.call is not None and generation.call.argument == row.argument:
                frame_hits += 1
                call_hits += 1
        else:
            declines += 1
            frame_hits += 1
            if generation.call is None:
                decline_hits += 1
    return {"route_accuracy": route_hits / max(len(scored), 1),
            "frame_accuracy": frame_hits / max(len(scored), 1),
            "route_correct": route_hits, "frame_correct": frame_hits, "count": len(scored),
            "call_rows": calls, "call_frame_accuracy": call_hits / max(calls, 1),
            "decline_rows": declines, "decline_accuracy": decline_hits / max(declines, 1)}


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    generator = load_generator(args.generator, "cpu", args.heads).eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    original = {key: value.clone() for key, value in generator.state_dict().items()}
    policy = ToolCallPolicy(generator)
    parameters = policy.parameters()
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lambda epoch: args.lr_decay if epoch and epoch % args.lr_step == 0 else 1.0)

    with ByteSandbox(timeout=args.sandbox_timeout) as sandbox:
        train_rows = tool_examples("train", sandbox)
        validation_rows = tool_examples("validation", sandbox)
        sandbox_root = str(sandbox.root)

    train_cached = cache_rows(policy, train_rows)
    validation_cached = cache_rows(policy, validation_rows)
    train_tensors = argument_tensors(train_rows, generator.byte_emb.weight.device)
    device = generator.byte_emb.weight.device

    history = []
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        losses, norms, totals = [], [], {"route_correct": 0, "byte_correct": 0, "byte_total": 0, "count": 0}
        for start in range(0, len(order), args.batch_size):
            batch = torch.tensor(order[start:start + args.batch_size], dtype=torch.long, device=device)
            loss, metrics = batch_loss(policy, batch, train_cached, train_tensors)
            optimizer.zero_grad()
            loss.backward()
            norms.append(float(torch.nn.utils.clip_grad_norm_(parameters, args.clip)))
            optimizer.step()
            losses.append(float(loss.detach()))
            for key in ("route_correct", "byte_correct", "byte_total", "count"):
                totals[key] += metrics[key]
        validation = evaluate_rows(policy, validation_rows, sample=args.validation_sample)
        validation["cached_route"] = route_metrics(policy, validation_cached)
        scheduler.step()
        row = {"epoch": epoch, "learning_rate": float(optimizer.param_groups[0]["lr"]),
               "loss": sum(losses) / max(len(losses), 1),
               "train_route_accuracy": totals["route_correct"] / max(totals["count"], 1),
               "train_byte_accuracy": totals["byte_correct"] / max(totals["byte_total"], 1),
               "gradient_norm_max": max(norms) if norms else 0.0,
               "gradient_norm_mean": sum(norms) / max(len(norms), 1),
               "validation": validation}
        history.append(row)
        print(json.dumps(row), flush=True)

    policy.route_head.tool_trained.fill_(True)
    policy.argument_decoder.tool_trained.fill_(True)
    for key, value in generator.state_dict().items():
        if not torch.equal(value, original[key]):
            raise RuntimeError(f"frozen generator tensor changed: {key}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "tool_policy.pt"
    policy.save(checkpoint, generator_path=args.generator)
    final = evaluate_rows(policy, validation_rows)
    report = {
        "configuration": vars(args),
        "sandbox_root": sandbox_root,
        "generator_sha256": sha256(args.generator),
        "checkpoint_sha256": sha256(checkpoint),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "route_names": list(ROUTE_NAMES),
        "tool_names": [name.decode("ascii") for name in TOOL_NAMES],
        "selection": "fixed final epoch; no test-based checkpoint selection",
        "epochs": history,
        "final_validation_full": final,
        "limits": ("Template-bound synthetic tasks; prompts and code templates recur across instance-disjoint "
                   "splits. Measures tool routing and template copying, not code understanding. Results come "
                   "from real sandbox execution of the generated code, never from hand-written outputs."),
    }
    (output / "tool_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint), "validation": final}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="runs/phase12/generator.pt")
    parser.add_argument("--output-dir", default="runs/phase15")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--lr-step", type=int, default=25, help="multiply the learning rate every N epochs")
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=15001)
    parser.add_argument("--validation-sample", type=int, default=80)
    parser.add_argument("--sandbox-timeout", type=float, default=5.0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
