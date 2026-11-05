"""CPU smoke + coherence probe (Phase 24.3 / 24.4). Offline, seconds, no GPU.

    python smoke_test_coherence.py --steps 4

Runs the full training pipeline on synthetic fixtures (the same ``run_smoke``
path as ``python -m training.train --smoke``), then reports:

* pipeline health: steps, resume, loss, the token-triggered tool demo;
* **Latent Thought Buffer telemetry** -- the scratchpad activation norm after
  training (the ReZero gates start at 0, so a growing norm proves the model
  is actually using the workspace, not just carrying it);
* a **gates-open scratchpad probe**: per-slot activation norms on a PT-BR
  prompt, proving the ``(B, 4, d_model)`` buffer updates slot by slot;
* a greedy byte continuation (coherence eyeball -- at smoke scale expect
  byte-level structure, NOT fluent text; that needs the 4-hour T4 run).

Exit code 0 = every probe passed. This script never touches the network.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.foundation import CausalByteTransformer, FoundationConfig, encode_bytes
from training.train import run_smoke, smoke_config

PTBR_PROMPT = b"O Brasil e um pais da America do Sul. A lingua oficial e o "


def thought_probe() -> dict:
    """Fresh tiny model with the ReZero gates forced open: prove the
    scratchpad activates and the slot recurrence advances."""
    config = smoke_config()
    model = CausalByteTransformer(FoundationConfig.from_mapping(config["model"]))
    thoughts = model.refinement.thoughts
    assert thoughts is not None, "smoke config must enable the latent thought buffer"
    with torch.no_grad():
        thoughts.buffer_gate.fill_(1.0)
        thoughts.context_gate.fill_(1.0)
        thoughts.slot_embeddings.normal_(0.0, 0.02)
    model.eval()
    with torch.no_grad():
        model(encode_bytes(PTBR_PROMPT).unsqueeze(0))
    return {
        "buffer_shape": list(thoughts.last_buffer.shape),
        "thought_steps": int(thoughts.last_steps),
        "activation_norm": round(float(thoughts.last_norm), 6),
        "slot_norms": [round(float(thoughts.last_buffer[0, t].norm()), 6)
                       for t in range(thoughts.capacity)],
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="ByteCore-DualMemory smoke + coherence probe")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--workdir", default="runs/smoke_coherence")
    parser.add_argument("--report", default=None, help="optional JSON report path")
    args = parser.parse_args(argv)

    summary = run_smoke(steps=args.steps, workdir=args.workdir)
    print("== pipeline smoke ==")
    print(json.dumps({key: summary[key] for key in
                      ("steps", "resume_step", "stop_reason", "tool_demo")}, indent=2))
    print("last_metrics:", json.dumps(summary["last_metrics"]))

    thought = summary.get("thought")
    print("\n== latent thought buffer (after training) ==")
    print(json.dumps(thought, indent=2))

    probe = thought_probe()
    print("\n== scratchpad probe (gates forced open) ==")
    print(json.dumps(probe, indent=2))
    assert probe["buffer_shape"][1] == 4, "buffer capacity must match the loop count"
    assert probe["thought_steps"] == 4, "every slot must be written in a full loop"
    assert probe["activation_norm"] > 0.0, "scratchpad never activated"
    assert all(norm > 0.0 for norm in probe["slot_norms"]), "a thought slot stayed dead"

    print("\n== coherence eyeball (greedy continuation, smoke scale) ==")
    print(repr(summary.get("sample_continuation", "")))
    print("note: smoke-scale bytes are plumbing validation, not fluent text.")

    result = {"smoke": summary, "probe": probe}
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, default=str), encoding="utf8")
    print("\nOK: smoke + coherence + latent thought probes passed")
    return result


if __name__ == "__main__":
    main()
