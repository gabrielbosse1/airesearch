"""Train cognitive controls on a small, explicit synthetic byte-edit task.

This is a capability smoke test, not general instruction tuning. Tasks start
with I:, D:, R:, or M: and operate on the following three raw ASCII bytes.
Only new cognitive components and control-vocabulary rows are trained; normal
byte/function-token logits and the pretrained backbone are left untouched.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.cognitive import CognitiveAction, apply_edit
from models.generator_conditioned import GeneratorConditioned
from training.cognitive_train import cognitive_supervision_loss
from training.train_confidence import load_generator


def task(g: GeneratorConditioned, canvas: bytes) -> tuple[CognitiveAction, bytes]:
    if len(canvas) != 5 or canvas[1:2] != b":":
        raise ValueError("synthetic task must have a one-byte command and three-byte payload")
    if canvas[:1] == b"I":
        action = CognitiveAction(g.EDIT_OP, start=5, kind="insert", data=b"!")
    elif canvas[:1] == b"D":
        action = CognitiveAction(g.EDIT_OP, start=2, end=3, kind="delete")
    elif canvas[:1] == b"R":
        action = CognitiveAction(g.EDIT_OP, start=2, end=3, kind="replace", data=b"?")
    elif canvas[:1] == b"M":
        action = CognitiveAction(g.EDIT_OP, start=4, end=5, destination=2, kind="move")
    else:
        raise ValueError("unknown synthetic command")
    return action, apply_edit(canvas, action, max_length=g.max_canvas_bytes)


def supervised_trace_loss(g: GeneratorConditioned, canvas: bytes) -> torch.Tensor:
    """Teacher force LOAD, THINK, EDIT, SAVE, DONE with real latent transitions."""
    device = g.byte_emb.weight.device
    scratch = g.scratchpad.clone()
    edit, updated = task(g, canvas)
    load = CognitiveAction(g.LOAD_SPAN, start=0, end=len(canvas))
    loss, _ = cognitive_supervision_loss(g, canvas, scratch, load)
    scratch = g.read_scratchpad(scratch, torch.tensor([list(canvas)], device=device))
    think = CognitiveAction(g.THINK)
    part, _ = cognitive_supervision_loss(g, canvas, scratch, think, workspace=canvas, last_token=g.LOAD_SPAN)
    loss = loss + part
    policy = g.cognitive_policy(canvas, scratch, pointer_bytes=canvas, loaded=True, last_token=g.LOAD_SPAN)
    scratch = g.think_scratchpad(scratch, policy.context.unsqueeze(0), steps=1).scratchpad
    part, _ = cognitive_supervision_loss(g, canvas, scratch, edit, workspace=canvas, last_token=g.THINK)
    loss = loss + part
    part, _ = cognitive_supervision_loss(g, updated, scratch, CognitiveAction(g.SAVE_SPAN),
                                         workspace=updated, dirty=True, last_token=g.EDIT_OP)
    loss = loss + part
    part, _ = cognitive_supervision_loss(g, updated, scratch, CognitiveAction(g.DONE),
                                         workspace=updated, last_token=g.SAVE_SPAN)
    return loss + part


def freeze_backbone(g: GeneratorConditioned) -> list[torch.nn.Parameter]:
    for parameter in g.parameters():
        parameter.requires_grad_(False)
    prefixes = ("scratchpad_reader.", "scratchpad_thinker.", "think_context.", "think_norm.",
                "cognitive_state_emb.", "pointer_boundary", "pointer_query.", "pointer_start.",
                "pointer_end.", "pointer_destination.", "edit_kind_head.", "edit_byte_head.",
                "reasoning_predictor.", "head.weight", "head.bias", "control_emb.weight")
    selected = []
    for name, parameter in g.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(True); selected.append(parameter)
    def mask_old_rows(grad: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(grad); mask[g.LOAD_SPAN:] = 1
        return grad * mask
    g.head.weight.register_hook(mask_old_rows)
    g.head.bias.register_hook(mask_old_rows)
    def mask_old_controls(grad: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(grad); mask[g.LOAD_SPAN - g.VOCAB_BYTES:] = 1
        return grad * mask
    g.control_emb.weight.register_hook(mask_old_controls)
    return selected


@torch.no_grad()
def evaluate(g: GeneratorConditioned, canvases: list[bytes]) -> dict[str, object]:
    total = 0; done = 0; correct = 0
    by_command = {chr(c): {"correct": 0, "count": 0} for c in b"IDRM"}
    for canvas in canvases:
        expected = task(g, canvas)[1]
        result = g.cognitive_generate(canvas, max_steps=8)
        ok = result.done and result.canvas == expected
        total += 1; done += int(result.done); correct += int(ok)
        record = by_command[chr(canvas[0])]
        record["count"] += 1; record["correct"] += int(ok)
    return {"count": total, "done_fraction": done / max(total, 1),
            "exact_task_fraction": correct / max(total, 1), "by_command": by_command}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", default="runs/dual_memory/generator.pt")
    parser.add_argument("--generator-heads", type=int, default=4)
    parser.add_argument("--output", default="runs/cognitive_generator.pt")
    parser.add_argument("--metrics", default="runs/cognitive_metrics.json")
    parser.add_argument("--train-payloads", type=int, default=48)
    parser.add_argument("--test-payloads", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.train_payloads, args.test_payloads, args.epochs) < 1 or args.lr <= 0:
        raise ValueError("training options must be positive")
    torch.manual_seed(23)
    g = load_generator(args.generator, args.device, args.generator_heads).eval()
    before_weight, before_bias = g.head.weight[:g.LOAD_SPAN].clone(), g.head.bias[:g.LOAD_SPAN].clone()
    before_controls = g.control_emb.weight[:g.LOAD_SPAN - g.VOCAB_BYTES].clone()
    params = freeze_backbone(g)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    payloads = [bytes(x) for x in itertools.product(b"abcdefgh", repeat=3)]
    random.Random(23).shuffle(payloads)
    if args.train_payloads + args.test_payloads > len(payloads):
        raise ValueError("requested more distinct payloads than available")
    train = [bytes([cmd]) + b":" + p for p in payloads[:args.train_payloads] for cmd in b"IDRM"]
    test = [bytes([cmd]) + b":" + p for p in payloads[args.train_payloads:args.train_payloads + args.test_payloads] for cmd in b"IDRM"]
    losses = []
    for epoch in range(args.epochs):
        random.Random(100 + epoch).shuffle(train)
        total = 0.0
        for canvas in train:
            optimizer.zero_grad()
            loss = supervised_trace_loss(g, canvas)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total += float(loss.detach())
        losses.append(total / len(train))
        print(f"epoch {epoch + 1}/{args.epochs}: mean trace loss={losses[-1]:.4f}", flush=True)
    if (not torch.equal(before_weight, g.head.weight[:g.LOAD_SPAN])
            or not torch.equal(before_bias, g.head.bias[:g.LOAD_SPAN])
            or not torch.equal(before_controls, g.control_emb.weight[:g.LOAD_SPAN - g.VOCAB_BYTES])):
        raise RuntimeError("training unexpectedly changed legacy byte/function rows")
    report = {"scope": "synthetic I/D/R/M byte edits, three-byte payload", "train_trace_loss": losses,
              "train": evaluate(g, train), "held_out": evaluate(g, test)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    torch.save(g.state_dict(), args.output)
    Path(args.metrics).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
