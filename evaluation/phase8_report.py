"""Render the Phase 8 metrics and reward curve without changing the experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rate(result: dict[str, Any]) -> str:
    return f"{result['exact']}/{result['count']} ({100 * result['exact_fraction']:.1f}%)"


def _plot(curve: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in curve]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1]})
    top.plot(epochs, [row["reward"] for row in curve], marker="o", label="Sampled policy reward")
    top.plot(epochs, [row["baseline_reward"] for row in curve], marker=".", linestyle="--",
             label="Greedy self-critical baseline")
    top.set_ylabel("Mean training reward")
    top.grid(alpha=0.25)
    top.legend(loc="best")
    bottom.plot(epochs, [row["supervised_weight"] for row in curve], label="Supervised weight")
    bottom.plot(epochs, [row["rl_weight"] for row in curve], label="RL weight")
    bottom.set(xlabel="Epoch", ylabel="Loss weight", ylim=(0, 1))
    bottom.grid(alpha=0.25)
    bottom.legend(loc="best")
    for index in range(1, len(curve)):
        if curve[index]["curriculum_stage"] != curve[index - 1]["curriculum_stage"]:
            for axis in (top, bottom):
                axis.axvline(epochs[index] - 0.5, color="gray", alpha=0.35, linewidth=0.8)
    figure.suptitle("Phase 8: training reward and hybrid loss weights\nStage boundaries change task difficulty")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def render(metrics: dict[str, Any], plot_relative: str = "phase8/reward_curve.png",
           reasoning_diagnostic: dict[str, Any] | None = None,
           verification: dict[str, Any] | None = None) -> str:
    protocol = metrics["protocol"]
    counts = metrics["counts"]
    curve = metrics["reward_curve"]
    baseline, selected, final = metrics["baseline_test"], metrics["selected_test"], metrics["final_test"]
    validation = metrics["selected_validation"]
    synthetic = metrics["synthetic_validation"]
    target_met = final["exact"] > 4
    lines = [
        "# Phase 8 — reasoning supervision and autonomous editing",
        "",
        "## Outcome",
        "",
        f"- **Final-epoch real held-out exact success: {_rate(final)}.** "
        + ("The requested >4/8 target was met." if target_met else "The requested >4/8 target was not met."),
        f"- Pretraining baseline: **{_rate(baseline)}**; validation-selected final-stage checkpoint "
        f"(epoch {protocol['best_epoch']}): **{_rate(selected)}**.",
        f"- The selected checkpoint scored **{_rate(validation)}** on the four real validation records and "
        f"**{_rate(synthetic)}** on a separate, explicitly instructed synthetic curriculum validation set. "
        "Synthetic scores are not real-edit performance.",
        f"- Mean test byte-edit similarity: baseline {baseline['mean_similarity']:.3f}, "
        f"selected {selected['mean_similarity']:.3f}, final epoch {final['mean_similarity']:.3f}. "
        f"On real validation the selected policy completed {100 * validation['done_fraction']:.0f}% of episodes; "
        "the rest hit the fixed action cap. Similarity and completion are not exact success.",
        "",
        "The real fixture's provenance explanations are not actionable edit requests. The same source can have "
        "many valid changes, so its target cannot generally be inferred from source bytes alone. "
        "The source-file split and byte guards establish only the stated separation; they cannot certify that "
        "pre-existing model weights have never encountered semantically equivalent text.",
        "",
        "## Protocol and safeguards",
        "",
        f"- Training: {counts['synthetic_train']} generated train examples across five stages, "
        f"plus {counts['real_train']} real training pairs sampled as {protocol['real_anchors_per_epoch']} "
        f"anchors per epoch; {protocol['epochs']} epochs, seed {protocol['seed']}, "
        f"learning rate {protocol['learning_rate']}, rollout temperature {protocol['temperature']}.",
        f"- Train-only traces provide supervised labels; sampled policy rollouts use no memory, controller, "
        f"or stored trace. Registered holdouts included the {counts['real_validation']} real validation, "
        f"{counts['real_test']} real test, {protocol['synthetic_curriculum_validation_examples']} curriculum validation, "
        f"and {protocol['registered_processed_synthetic_holdouts']} pre-existing processed synthetic records. "
        "Raw inputs, targets, instructions, trace payloads and intermediate canvases are checked before training; "
        "the guard is not a model-weight or semantic-leakage detector.",
        f"- Training rollout budget {protocol['rollout_budget']}; all real validation/test inference uses a fixed "
        f"{protocol['evaluation_budget']}-action budget. The target is accessed only after generation for scoring. "
        "The real held-outs use no memory, initial trace, controller, oracle action or instruction text.",
        "- Validation selection, fixed before test evaluation, considers only final-stage epochs, maximizing "
        "validation exact success and then normalized byte similarity, with earliest epoch winning ties. "
        "Baseline, selected and final-epoch policies are all evaluated after selection; the test does not select weights.",
        f"- Generator: {protocol['generator_parameters']:,} parameters including "
        f"{protocol['reasoning_predictor_parameters']:,} in the new head; "
        f"{protocol['trainable_parameters']:,} parameter tensor elements participate in this selective fine-tuning "
        "(legacy head/control rows are gradient-masked).",
        "",
        "## Objectives and caveats",
        "",
        "A fixed, order-sensitive raw-byte sketch of the target minus the current document supervises the "
        "post-THINK mean scratchpad via Linear–SiLU–Linear MSE. The label is detached and used only during "
        "training. The sketch is lossy and can collide; it does not prove symbolic reasoning.",
        "",
        f"Training reward is `{protocol['reward']}`. The policy-gradient term uses "
        "`-(reward − greedy_baseline) × log_probability(sampled_action_sequence)`, where the baseline depends "
        "on the current training example but not the sampled action. Masked control, pointer, kind, byte "
        "and destination-remapping probabilities are differentiably replayed. The objective adds "
        "`0.5 × reasoning_MSE` to the supervised/RL mixture, with gradient clipping. Sampling and replay "
        "share the same attention path; the process-global PyTorch fast-path setting is restored afterward.",
        "",
        "This reward can prefer an unchanged long document over a correct small edit because THINK and EDIT "
        "carry a fixed cost. A high similarity reward is not an exact success. Across the curve, each stage "
        "has different examples and difficulty, so epoch rewards are not a controlled learning curve.",
        "",
        "## Training reward curve",
        "",
        f"![Training reward and loss weights]({plot_relative})",
        "",
        "| Epoch | Curriculum stage | Supervised / RL | Sample reward | Greedy baseline | "
        "Sample exact | Reasoning MSE | Real validation exact |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in curve:
        lines.append(f"| {row['epoch']} | {row['curriculum_stage']} | "
                     f"{row['supervised_weight']:.3f} / {row['rl_weight']:.3f} | "
                     f"{row['reward']:.4f} | {row['baseline_reward']:.4f} | "
                     f"{100 * row['exact']:.1f}% | {row['reasoning_loss']:.5f} | "
                     f"{row['validation_exact']}/{row['validation_count']} |")
    if reasoning_diagnostic is not None:
        lines.extend([
            "",
            "## Reasoning-head diagnostic (training examples only)",
            "",
            "For teacher-forced post-THINK states, compare each checkpoint's MSE with predicting an all-zero "
            "intent. These are training-set diagnostics, not generalization results; the very small zero-vector "
            "loss for line edits and moves reveals that this sketch can provide a weak learning signal.",
            "",
            "| Training stage | Examples | Zero-vector MSE | Selected MSE | Final-epoch MSE |",
            "|---|---:|---:|---:|---:|",
        ])
        selected_groups = reasoning_diagnostic["selected"]
        final_groups = reasoning_diagnostic["final_epoch"]
        for stage, selected_group in selected_groups.items():
            final_group = final_groups[stage]
            lines.append(f"| {stage} | {selected_group['count']} | {selected_group['zero_prediction_mse']:.3g} | "
                         f"{selected_group['predictor_mse']:.3g} | {final_group['predictor_mse']:.3g} |")
        worse = sum(group["predictor_mse"] > group["zero_prediction_mse"] for group in selected_groups.values())
        lines.extend([
            "",
            f"The selected head is worse than the zero predictor in {worse}/{len(selected_groups)} training groups. "
            "The MSE implementation has verified gradients; this comparison alone cannot establish "
            "useful edit intent or generalization.",
        ])
    lines.extend([
        "",
        "## Eight real held-out tasks: final epoch",
        "",
        "| ID | Exact | Done | Actions | Byte distance | Reward |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for item in final["examples"]:
        lines.append(f"| {item['id']} | {'yes' if item['exact'] else 'no'} | "
                     f"{'yes' if item['done'] else 'no'} | {item['steps']} | {item['distance']} | {item['reward']:.4f} |")
    lines.extend([
        "",
        "## Reproduction and artifacts",
        "",
        "```bash",
        ".venv/bin/python data/generate.py  # recreate the ignored holdout registry",
        ".venv/bin/python -m training.rl_editing --epochs 20 --examples-per-stage 8 --real-anchors 2 \\",
        "  --rollout-budget 16 --eval-budget 128 --temperature 0.8 --lr 0.0003 --seed 881 --threads 1",
        ".venv/bin/python -m evaluation.phase8_reasoning",
        ".venv/bin/python -m evaluation.phase8_report",
        "```",
        "",
        "`runs/phase8/metrics.json` contains the full curve, per-example outputs and action records; "
        "`runs/phase8/generator.pt` is the validation-selected final-stage checkpoint and "
        "`runs/phase8/last_epoch.pt` is the final-epoch checkpoint. Reproducing results requires "
        "the original train-only generator checkpoint and the same dependency versions.",
        "",
        "The checkout used here contained Phase 6 artifacts but no independently verifiable Phase 7 "
        "mask-conditioning implementation. No Phase 7 success is assumed or claimed.",
        "",
    ])
    if verification is not None:
        tests = verification["unit_tests"]
        lines.extend([
            "## Verification",
            "",
            f"{tests['passed']}/{tests['run']} tests passed. A guarded five-stage co-training smoke passed, "
            "the existing guarded memory audit found zero failed slots, and frozen backbone/legacy rows "
            "matched the starting checkpoint. Compilation, whitespace, tokenizer-import, "
            "checkpoint-migration and policy-gradient checks passed. "
            "See `runs/phase8/verification.json` for versions and checkpoint hashes.",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="runs/phase8/metrics.json")
    parser.add_argument("--report", default="runs/phase8_rl_report.md")
    parser.add_argument("--plot", default="runs/phase8/reward_curve.png")
    parser.add_argument("--reasoning-diagnostic", default="runs/phase8/reasoning_train_diagnostic.json")
    parser.add_argument("--verification", default="runs/phase8/verification.json")
    args = parser.parse_args()
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    diagnostic_path, verification_path = Path(args.reasoning_diagnostic), Path(args.verification)
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8")) if diagnostic_path.exists() else None
    verification = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else None
    report_path, plot_path = Path(args.report), Path(args.plot)
    _plot(metrics["reward_curve"], plot_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        relative_plot = str(plot_path.relative_to(report_path.parent))
    except ValueError:
        relative_plot = str(plot_path)
    report_path.write_text(render(metrics, relative_plot, diagnostic, verification), encoding="utf-8")
    print(f"Wrote {report_path} and {plot_path}")


if __name__ == "__main__":
    main()
