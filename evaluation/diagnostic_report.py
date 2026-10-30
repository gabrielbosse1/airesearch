"""Aggregate measured Phase 5 diagnostics into a reproducible Markdown report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.diagnostics_common import load_memory
from models.bytecore_diffusion import ByteCoreDiffusion
from training.train_confidence import load_generator


def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the corresponding Phase 5 diagnostic first")
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value, digits=2):
    return "n/a" if value is None else f"{100*value:.{digits}f}%"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output", default="runs/diagnostic_report.md")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    runs = Path(args.runs_dir)
    cal = load(runs / "calibration.json"); real = load(runs / "real_traces/metrics.json")
    align = load(runs / "alignment/metrics.json"); ab = load(runs / "ablation.json")
    heads = load(runs / "head_attribution.json"); importance = load(runs / "importance_validation.json")
    edits = load(runs / "edit_operation_analysis.json"); learning = load(runs / "learning_curve.json")
    failures = load(runs / "failure_detection.json"); evaluation = load(runs / "evaluation.json")
    cfg = yaml.safe_load(Path(args.config).read_text()); mc = cfg["model"]
    generator = load_generator(str(runs / "dual_memory/generator.pt"), args.device, 4)
    memory = load_memory(str(runs / "dual_memory/memory.pt"), args.device)
    diffusion = ByteCoreDiffusion(vocab_size=mc["vocab_size"], d_model=mc["d_model"], n_layers=mc["n_layers"],
        n_heads=mc["n_heads"], d_ff=mc["d_ff"], max_len=mc["max_len"], diffusion_timesteps=mc["diffusion_timesteps"],
        concept_capacity=mc["concept_capacity"], concept_dim=mc["concept_dim"], dropout=mc["dropout"], device=args.device,
    ).load(str(runs), map_location=args.device)
    counts = {name: sum(p.numel() for p in model.parameters()) for name, model in (("diffusion", diffusion), ("generator", generator), ("memory", memory))}
    supported = sum(cal['models'][name]['thresholds'][target]['threshold'] is not None
                    for name in ('diffusion', 'generator') for target in ('0.90', '0.95', '0.99'))
    aligned_test = align['arms']['soft_dtw_retrained']['test']
    reached = all(aligned_test[mode]['position_accuracy'] > .7 for mode in ('insert', 'delete'))
    clean_delta = aligned_test['clean']['position_accuracy'] - align['baseline_test']['clean']['position_accuracy']
    clean_description = 'regressed' if clean_delta < 0 else 'did not regress'
    lines = ["# Phase 5: Full-Spectrum Grounding & Diagnostic Attribution", "",
             "**Historical report.** Phase 6 replaced the contaminated verification memory, changed the training defaults, and reran evaluation with a fixed (not target-derived) budget. See [Phase 6 report](phase6_report.md).", "",
             "## Executive summary", "",
             f"- **Calibration is not capability.** On the held-out proxy tasks, {supported}/6 head/accuracy-target combinations supported a threshold under the independent-example selection criterion. Adaptive halting remains off.",
             f"- **Stored replay is not autonomous editing.** {real['replay']['matched']}/{real['replay']['count']} stored procedures round-tripped, while the trace-trained generator solved {real['trained_train']['exact']}/{real['trained_train']['count']} training excerpts and {real['trained_test']['exact']}/{real['trained_test']['count']} held-out excerpts without a controller.",
             f"- **The >70% insertion/deletion recovery target was {'reached' if reached else 'not reached'}.** In a matched synthetic retraining experiment, soft-DTW yielded {pct(aligned_test['insert']['position_accuracy'])} insert and {pct(aligned_test['delete']['position_accuracy'])} delete position accuracy. Clean accuracy changed from {pct(align['baseline_test']['clean']['position_accuracy'])} to {pct(aligned_test['clean']['position_accuracy'])}; it {clean_description}.",
             f"- **Memory utility is task-dependent.** Removing all memory changed teacher-forced NLL by {ab['all_memory_removed']['nll_delta_vs_full']:+.5f} nats in this checkpoint/test; a negative number means removing it improved NLL.", "",
             "## Scope and protocol", "",
             "Text, images and audio remain raw bytes, without external tokenizers. The confidence and diffusion datasets are synthetic; real edits are short, curated before/after excerpts from this repository, not an external benchmark. Model selection and reporting are exploratory and sample sizes are small. Exact output replay returns stored data; it never runs stored bytes as code. NUL remains a valid byte.", "",
             "## A. Confidence calibration", "",
             "Temperature fit uses even validation examples, threshold selection uses odd validation examples, and test is separate. Diffusion thresholds use one sequence-level all-bytes-correct observation per sample; generator thresholds use one next-byte observation per sample. Wilson bounds are exploratory and not multiplicity-adjusted. Thresholds do **not** establish THINK or ancestral-sampling reliability.", "",
             "| Head / scope | Test byte predictions | Correctness | ECE-10 | MCE-10 | 0.90 / 0.95 / 0.99 thresholds |", "|---|---:|---:|---:|---:|---|"]
    for name in ("diffusion", "generator"):
        model = cal["models"][name]; stat = model["test"]
        thresholds = " / ".join("unsupported" if model["thresholds"][key]["threshold"] is None else f"{model['thresholds'][key]['threshold']:.4f}" for key in ("0.90", "0.95", "0.99"))
        lines.append(f"| {name} / {model['scope']} | {stat['count']:,} | {pct(stat['accuracy'])} | {stat['ece']:.4f} | {stat['mce']:.4f} | {thresholds} |")
    lines += ["", "Per-prediction confidence, actual embedding MSE / cross-entropy, sample ID and noise level are in `calibration_predictions.npz`; calibrated copies are separate from defaults.", "",
              "![Calibration](calibration.png)", "", "## B. Real byte traces", "",
              f"The repository excerpt fixture has {real['counts']['train']} train, {real['counts']['validation']} validation and {real['counts']['test']} test examples, separated by source file. The experiment used {real['protocol']['train_epochs']} co-training epochs; held-out outputs were written only after generator evaluation.",
              f"- Exact procedure persistence/retrieval: **{real['replay']['matched']}/{real['replay']['count']}**. Ordered latent snapshots and explanation bytes were checked.",
              f"- Autonomous baseline / trained held-out exact: **{real['baseline_test']['exact']}/{real['baseline_test']['count']}** / **{real['trained_test']['exact']}/{real['trained_test']['count']}**; conditioned on training memory: **{real['trained_test_with_train_memory']['exact']}/{real['trained_test_with_train_memory']['count']}**.",
              "- Dynamic programming gives an optimal unit-cost byte Levenshtein trace. MOVE is available to the cognitive loop, but is not a primitive in that optimality claim. Byte insert/replace targets are single-byte actions.",
              f"- {real.get('phase6_checkpoint_update', 'The historical verification memory contained held-out targets; do not use it for a fresh held-out performance claim.')}",
              "- The historical autonomous budget depended on target-derived edit distance; the Phase 6 rerun uses a fixed budget.", "",
              "## C. Alignment and recovery", "",
              "The trained loss is MSE + λ·soft-DTW in embedding space (and optional confidence BCE). Masks exclude padding; soft-DTW is entropy-smoothed and can be negative. Fixed-length denoising cannot itself infer target length or reconstruct information removed by random deletion. The paired arms share initialization, data, noise and seeds; selection was on validation with a 1 percentage-point clean tolerance.", "",
              "| Checkpoint | Clean | 10% insert, position-wise | 10% delete, position-wise | Insert / delete edit similarity |", "|---|---:|---:|---:|---:|"]
    models = [("Original", align["baseline_test"]), ("Retrained without alignment", align["arms"]["no_alignment_retrained"]["test"]),
              ("Retrained with soft-DTW", align["arms"]["soft_dtw_retrained"]["test"])]
    for title, row in models:
        lines.append(f"| {title} | {pct(row['clean']['position_accuracy'])} | {pct(row['insert']['position_accuracy'])} | {pct(row['delete']['position_accuracy'])} | {pct(row['insert']['edit_aligned_similarity'])} / {pct(row['delete']['edit_aligned_similarity'])} |")
    lines += ["", "Levenshtein similarity realigns sequences and was already high in the baseline; it is **not** recovery of deleted bytes. No trained alignment checkpoint replaced the original.", "",
              "## D. Attribution and diagnostics", "", "### Inference ablations", "",
              f"Generator full-memory next-byte accuracy: {pct(ab['full_memory']['accuracy'])}; NLL: {ab['full_memory']['nll']:.4f}. Interventions remove one component from the same checkpoint; values are conditional on this task.", "",
              "| Intervention | Δ NLL (ablated − full) | Δ accuracy |", "|---|---:|---:|"]
    lines.append(f"| All memory | {ab['all_memory_removed']['nll_delta_vs_full']:+.5f} | {ab['all_memory_removed']['accuracy_delta_vs_full']:+.5f} |")
    for layer, row in ab["per_layer_memory_removed"].items():
        lines.append(f"| Memory layer {layer} | {row['nll_delta_vs_full']:+.5f} | {row['accuracy_delta_vs_full']:+.5f} |")
    for tier, row in ab["memory_tiers_removed"].items():
        lines.append(f"| {tier.title()} tier | {row['nll_delta_vs_full']:+.5f} | {row['accuracy_delta_vs_full']:+.5f} |")
    thought_layers = ab.get("per_layer_thought_output_zeroed", {})
    if thought_layers:
        lines += ["", "Zeroing each layer's thought-state output (while keeping its hidden output) produced these confidence/cognitive changes:", "",
                  "| Thought layer | Δ ECE | Δ Brier | Synthetic edit exact |", "|---|---:|---:|---:|"]
        for layer, row in thought_layers.items():
            lines.append(f"| {layer} | {row['ece_delta']:+.5f} | {row['brier_delta']:+.5f} | {pct(row['toy_cognitive_exact_fraction'])} |")
    cognitive = ab["cognitive"]
    lines += ["", f"On the narrow synthetic edit task, exact success was {pct(cognitive['full']['exact_task_fraction'])} normally, {pct(cognitive['without_think_transition']['exact_task_fraction'])} when THINK was an identity transition, and {pct(cognitive['without_scratchpad_read']['exact_task_fraction'])} when READ was an identity transition. These interventions are not general edit benchmarks.",
              f"The confidence head's observed ECE/Brier was {ab['confidence']['head_ece']:.4f}/{ab['confidence']['head_brier']:.4f}; a constant validation prior had {ab['confidence']['constant_ece']:.4f}/{ab['confidence']['constant_brier']:.4f}. Requesting confidence changed logits by at most {ab['confidence']['max_output_delta_when_requesting_confidence']:.2g}; the head by itself does not change output.",
              "Alignment is compared by **retraining**, not falsely treated as an inference-time layer.", "", "### Cross-attention heads", "",
              f"{len(heads['heads'])} RETRO cross-attention heads were individually zeroed at the output projection; self-attention and memory prefix tokens remained active. Text examples are short enough that preceding-chunk retrieval may not activate; zero effects are not evidence that a head is universally irrelevant.", "",
              "![Head attribution](head_attribution.png)", "", "### Memory importance", ""]
    pearson = importance["pearson"]; spearman = importance["spearman"]
    pf = "undefined" if pearson is None else f"{pearson:.3f}"
    sf = "undefined" if spearman is None else f"{spearman:.3f}"
    active = sum(row["retrieval_count"] > 0 for row in importance["slots"])
    lines += [f"For {importance['selected_slots']} selected slots ({active} actually retrieved), Pearson/Spearman correlations between importance and paired NLL utility were **{pf}/{sf}**. Removing a slot can substitute a different neighbor; many zero effects and the small selected sample prevent broad conclusions.", "",
              "![Importance](importance_validation.png)", "", "### Edit operations", "",
              "The real excerpt counts overlap when optimal traces require multiple edit kinds. Thought confidence was not calibrated for THINK.", "",
              "| Operation | Synthetic exact | Real test exact | Real mean THINK steps | Real mean thought confidence |", "|---|---:|---:|---:|---:|"]
    for kind in ("insert", "delete", "replace", "move"):
        syn = edits["synthetic"]["by_operation"][kind]; rr = edits["real_test"]["by_operation"][kind]
        ts = "n/a" if rr["mean_think_steps"] is None else f"{rr['mean_think_steps']:.2f}"
        cf = "n/a" if rr["mean_thought_confidence"] is None else f"{rr['mean_thought_confidence']:.3f}"
        lines.append(f"| {kind} | {pct(syn['exact_fraction'])} (n={syn['count']}) | {pct(rr['exact_fraction'])} (n={rr['count']}) | {ts} | {cf} |")
    first = learning["history"][0]["validation"]; last = learning["history"][-1]["validation"]
    lines += ["", "### Learning curves and failures", "",
              f"On a separate unique-prefix continuation experiment, validation NLL changed {first['loss']:.4f} → {last['loss']:.4f}; accuracy {pct(first['accuracy'])} → {pct(last['accuracy'])}; top-1 concept-label retrieval {pct(first['retrieval_accuracy'])} → {pct(last['retrieval_accuracy'])}; uncalibrated ECE {first['ece']:.4f} → {last['ece']:.4f}.",
              f"Real test edit failure classes: {failures['real_edits']['classes']}. See [failure_modes.md](failure_modes.md) for separate denominators, high-confidence errors and diffusion shift proxies.", "",
              "![Learning curves](learning_curve.png)", "", "## Recommendations (evidence-based)", "",
              "1. Keep adaptive halting disabled; gather independently held-out, deployment-matched thought and ancestral trajectories and calibrate each scope after training.",
              "2. Expand real edit data with explicit instructions, more repositories and deduplicated splits; evaluate autonomous execution separately from stored-answer replay and host-controlled traces.",
              "3. Investigate an explicitly length-aware decoder and stronger paired insert/delete supervision; compare against copy/unaligned baselines and require non-regressing clean accuracy before replacing the default.",
              "4. Train and evaluate retrieval against task outcomes, audit tier/router collapse and assess slot utility under realistic queries before tuning importance or eviction.",
              "5. Repeat intervention and calibration studies across seeds, modalities, longer sequences and independent data before claiming causal generality.", "",
              "## Compatibility, cost and final benchmark", "",
              f"- Included checkpoint parameter counts: diffusion **{counts['diffusion']:,}**, generator **{counts['generator']:,}**, memory (capacity {memory.capacity}) **{counts['memory']:,}**. Soft-DTW adds **0 learned parameters** and no checkpoint tensors; it adds quadratic sequence-time/memory cost during training.",
              "- The Phase 5 experiment used λ=0.1 and 50% synthetic edit corruption; Phase 6 sets both defaults to zero. This affects future training, not historical checkpoints. Retraining invalidates calibration; optimizers should be recreated when architecture dimensions change.",
              f"- Original diffusion final benchmark: clean real-byte reconstruction {pct(evaluation['real_byte_reconstruction'])}; 10% insertion/deletion position recovery {pct(evaluation['recovery_vs_original_real_bytes']['insert']['0.1'])}/{pct(evaluation['recovery_vs_original_real_bytes']['delete']['0.1'])}. See [report.md](report.md).", "",
              "Artifacts: [calibration metrics](calibration.json), [real traces](real_traces/metrics.json), [alignment experiment](alignment/metrics.json), [ablations](ablation.json), [head results](head_attribution.json), [importance results](importance_validation.json), [edit results](edit_operation_analysis.json), [learning metrics](learning_curve.json), [failure analysis](failure_modes.md)."]
    verification = runs / "phase5_verification.json"
    if verification.exists():
        check = load(verification)
        lines.append(f"- Verification: {check.get('summary', 'see phase5_verification.json')}")
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
