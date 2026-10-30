"""Copy-fidelity probe for the Phase 15 tool policy.

The tool-routing benchmark scores a whole frame, so it cannot say *why* a frame
is wrong. This probe isolates the mechanism that produces the argument bytes:
verbatim copying of an unseen snippet out of the request.

For each family it reports how often the greedy argument is byte-identical to
the target, on held-out (validation) items only. Families:

``run``     valid code the model has to copy verbatim (``Run this Python: ...``)
``repair``  broken code pasted into a repair prompt (first attempt)
``wild``    broken code from the wide randomized space (first attempt)

The ``wild`` and ``run`` families are the honest ones: their content is drawn
from a space too large to memorise, so a hit can only come from copying.

Usage::

    python -m evaluation.copy_probe --output runs/phase15/copy_probe.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.tool_use_examples import (REPAIR_PROMPTS, repair_snippets, run_rows,
                                    wild_requests)
from models.sandbox import ByteSandbox
from models.tool_use import load_tool_policy
from training.train_confidence import load_generator


def _copy_rate(policy, items: list[tuple[bytes, bytes]]) -> dict:
    """Byte-exact and near-miss rates for greedy arguments on held-out items."""
    exact = 0
    near = 0
    for request, target in items:
        call = policy.generate(request, None, threshold=0.0, max_bytes=192).call
        argument = call.argument if call is not None else b""
        if argument == target:
            exact += 1
            continue
        overlap = sum(a == b for a, b in zip(argument, target))
        if target and overlap / len(target) >= 0.8:
            near += 1
    return {"exact": exact, "near": near, "total": len(items),
            "exact_rate": round(exact / max(len(items), 1), 4),
            "near_rate": round(near / max(len(items), 1), 4)}


def run_probe(generator_path: str, policy_path: str) -> dict:
    generator = load_generator(generator_path, torch.device("cpu"), 4)
    policy = load_tool_policy(policy_path, generator=generator, generator_path=generator_path)

    repair = repair_snippets("validation")
    repair_items = [(REPAIR_PROMPTS[index % len(REPAIR_PROMPTS)].format(code=broken.decode("utf-8")).encode("utf-8"),
                     broken) for index, (broken, _) in enumerate(repair)]
    wild_items = [(request, broken) for request, broken, _ in wild_requests("validation")]

    with ByteSandbox() as sandbox:  # keeps the import-time contract honest
        sandbox_root = str(sandbox.root)

    return {
        "note": ("Held-out (validation) families only. 'exact' is a byte-identical greedy "
                 "argument; 'near' is >=80% of the target bytes aligned from the start. "
                 "This measures copying, not understanding."),
        "checkpoint": policy_path,
        "sandbox_root": sandbox_root,
        "families": {
            "run": _copy_rate(policy, run_rows("validation")),
            "repair": _copy_rate(policy, repair_items),
            "wild": _copy_rate(policy, wild_items),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 15 copy-fidelity probe")
    parser.add_argument("--generator", default="runs/phase12/generator.pt")
    parser.add_argument("--policy", default="runs/phase15/tool_policy.pt")
    parser.add_argument("--output", default="runs/phase15/copy_probe.json")
    args = parser.parse_args()
    report = run_probe(args.generator, args.policy)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["families"], indent=2))


if __name__ == "__main__":
    main()
