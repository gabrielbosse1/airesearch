"""Causal inference interventions plus a matched alignment *training* ablation."""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.diagnostics_common import contexts, load_memory, score_generator, test_batches
from models.generator_conditioned import AdaptiveThought
from training.train_cognitive import evaluate as evaluate_toy, task
from training.train_confidence import load_generator


@contextmanager
def without_adapter(generator, layer: str):
    """Remove both prefix and cross-memory paths at one layer, then restore."""
    adapter = generator.memory_adapters.pop(layer)
    try:
        yield
    finally:
        generator.memory_adapters[layer] = adapter


def delta(result, baseline):
    return {"nll_delta_vs_full": result["nll"] - baseline["nll"],
            "accuracy_delta_vs_full": result["accuracy"] - baseline["accuracy"],
            "accuracy": result["accuracy"], "nll": result["nll"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--generator", default="runs/dual_memory/generator.pt")
    p.add_argument("--memory", default="runs/dual_memory/memory.pt")
    p.add_argument("--alignment-report", default="runs/alignment/metrics.json")
    p.add_argument("--output", default="runs/ablation.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--examples", type=int, default=128)
    p.add_argument("--toy-payloads", type=int, default=32)
    args = p.parse_args()
    torch.manual_seed(44)
    g = load_generator(args.generator, args.device, 4)
    m = load_memory(args.memory, args.device)
    batches = test_batches(args.config, args.examples, 16, args.device)
    full_ctx = contexts(g, m, batches)
    baseline = score_generator(g, batches, full_ctx)
    no_memory = score_generator(g, batches)
    layer_results = {}
    for layer in list(g.memory_adapters.keys()):
        with without_adapter(g, layer):
            result = score_generator(g, batches, full_ctx)
        layer_results[layer] = delta(result, baseline)
    tiers = ("episodic", "semantic", "procedural")
    tier_results = {}
    for tier in tiers:
        allowed = [name for name in tiers if name != tier]
        ablated = score_generator(g, batches, contexts(g, m, batches, allowed))
        tier_results[tier] = {**delta(ablated, baseline),
                              "used_slots": int(getattr(m, tier).used.sum().item())}

    # The confidence head is observational: replacing its score cannot alter
    # next-byte logits unless a separate adaptive policy uses that score.
    calibration_path = Path("runs/calibration.json")
    prior = 0.5
    if calibration_path.exists():
        prior = json.loads(calibration_path.read_text())["models"]["generator"]["validation_selection"]["accuracy"]
    observed = baseline["accuracy"]
    prior_brier = observed*(1-prior)**2 + (1-observed)*prior**2
    with torch.no_grad():
        inputs = batches[0]["input_ids"][:, :-1]; tokens, mask = full_ctx[0]
        bare = g(inputs, tokens, memory_mask=mask, memory_chunk_size=g.retro_chunk_size)
        state = g(inputs, tokens, memory_mask=mask, memory_chunk_size=g.retro_chunk_size, return_state=True)
        confidence_output_delta = float((bare - state.output).abs().max())
    confidence = {"head_ece": baseline["ece"], "head_brier": baseline["brier"],
                  "constant_validation_prior": prior, "constant_ece": abs(prior-observed),
                  "constant_brier": prior_brier, "max_output_delta_when_requesting_confidence": confidence_output_delta,
                  "scope": "next-byte on memory-conditioned distribution; temperature was fitted without memory"}

    payloads = [bytes(x) for x in itertools.product(b"abcdefgh", repeat=3)]
    random.Random(23).shuffle(payloads)
    payloads = payloads[48:48 + args.toy_payloads]
    if not payloads:
        raise ValueError("toy-payloads must be positive")
    canvases = [bytes([cmd]) + b":" + payload for payload in payloads for cmd in b"IDRM"]
    full_cognitive = evaluate_toy(g, canvases)
    # Zero the latent thought output at each layer boundary. This keeps the
    # hidden-state output (and any current-layer gate coupling) intact, so it
    # isolates propagation of the thought state rather than deleting a block.
    thought_layers = {}
    for index, layer in enumerate(g.thought_layers):
        def zero_thought(_module, _inputs, output):
            hidden, thought = output
            return hidden, torch.zeros_like(thought)
        handle = layer.register_forward_hook(zero_thought)
        try:
            measured = score_generator(g, batches, full_ctx)
            toy = evaluate_toy(g, canvases)
        finally:
            handle.remove()
        thought_layers[str(index)] = {**delta(measured, baseline),
            "ece_delta": measured["ece"] - baseline["ece"],
            "brier_delta": measured["brier"] - baseline["brier"],
            "toy_cognitive_exact_fraction": toy["exact_task_fraction"]}
    no_edit_count = sum(int(canvas == task(g, canvas)[1]) for canvas in canvases)
    original_think = g.think_scratchpad
    def identity_think(self, scratchpad, context, **kwargs):
        b = scratchpad.shape[0]
        return AdaptiveThought(scratchpad, scratchpad.new_zeros(b), torch.zeros(b, device=scratchpad.device, dtype=torch.long))
    g.think_scratchpad = types.MethodType(identity_think, g)
    try:
        no_think = evaluate_toy(g, canvases)
    finally:
        g.think_scratchpad = original_think
    original_read = g.read_scratchpad
    def identity_read(self, scratchpad, byte_ids):
        return scratchpad
    g.read_scratchpad = types.MethodType(identity_read, g)
    try:
        no_read = evaluate_toy(g, canvases)
    finally:
        g.read_scratchpad = original_read
    cognitive = {"full": full_cognitive, "without_think_transition": no_think,
                 "without_scratchpad_read": no_read,
                 "without_cognitive_edit_loop_exact_fraction": no_edit_count / len(canvases),
                 "scope": "narrow synthetic I/D/R/M; identity interventions do not remove actions"}
    alignment = None
    alignment_path = Path(args.alignment_report)
    if alignment_path.exists():
        raw = json.loads(alignment_path.read_text())
        alignment = {name: {"selected_epoch": arm["selected_epoch"],
                            "test_position_accuracy": {mode: v["position_accuracy"] for mode, v in arm["test"].items()},
                            "alignment_loss_weight": arm["alignment_loss_weight"]}
                     for name, arm in raw["arms"].items()}
        alignment["interpretation"] = "Matched retraining comparison; alignment is a training loss, not an inference module."
    report = {"protocol": {"generator": args.generator, "memory": args.memory, "examples": args.examples,
                           "metric": "held-out teacher-forced byte NLL and accuracy; negative NLL delta means removal helped",
                           "retrieval": "previous-chunk top-K, frozen scores; slot access counters not mutated",
                           "warning": "Single interventions on one checkpoint are conditional effects, not universal component importance."},
              "full_memory": baseline, "all_memory_removed": delta(no_memory, baseline),
              "per_layer_memory_removed": layer_results, "memory_tiers_removed": tier_results,
              "per_layer_thought_output_zeroed": thought_layers,
              "confidence": confidence, "cognitive": cognitive, "alignment_retraining_ablation": alignment}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"full": {"accuracy": baseline["accuracy"], "nll": baseline["nll"]},
                      "no_memory": report["all_memory_removed"], "layers": layer_results,
                      "tiers": tier_results, "cognitive_exact": {"full": full_cognitive["exact_task_fraction"],
                      "no_think": no_think["exact_task_fraction"], "no_read": no_read["exact_task_fraction"]}}, indent=2))


if __name__ == "__main__":
    main()
