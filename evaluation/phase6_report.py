"""Render Phase 6's measured outputs without conflating lookup and generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase-dir", default="runs/phase6")
    p.add_argument("--output", default="runs/phase6_report.md")
    args = p.parse_args()
    root = Path(args.phase_dir)
    def load(name: str):
        return json.loads((root / name).read_text(encoding="utf-8"))
    real = load("real_traces/metrics.json"); legacy = load("legacy_audit.json")
    ev = load("evaluation/evaluation.json"); routing = load("routing.json"); halting = load("halting.json")
    verify_path = root / "verification.json"
    verification = json.loads(verify_path.read_text()) if verify_path.exists() else None
    h, th = halting["diffusion"], halting["thought"]
    test = h["test"]; thought_policy = th["policy"]
    pct = lambda x: f"{100*x:.2f}%"
    count = lambda item: f"{item['exact']}/{item['count']}"
    rec = ev["real_byte_reconstruction"]; novel = ev["novel_exact_byte_rows"]
    over = ev["data_split_overlap"]
    lines = [
        "# Phase 6 — Fix the Fundamentals", "",
        "## Outcome", "",
        f"- **Clean reconstruction:** {pct(rec)} across real (non-padding) bytes; {pct(novel['reconstruction'])} "
        f"on {novel['count']} test rows not exactly present in train. The original checkpoint was evaluated, not retrained.",
        f"- **Guarded, train-only replay:** {real['replay']['matched']}/{real['replay']['count']} stored training procedures; "
        f"{real['holdout_checks']['attempts_rejected']}/12 held-out write attempts rejected. Lookup is not model editing.",
        f"- **Autonomous real edits:** {count(real['trained_test'])} test excerpts without memory; "
        f"{count(real['trained_test_with_train_memory'])} with guarded train memory. The target is not passed to the model.",
        "- **Adaptive halting:** implemented and config-wired, but left disabled: no validated quality-preserving diffusion policy was selected; thought confidence remains uncalibrated. This requested target was not met.",
        "- **Architectural dispatch:** caller-declared structural operations use the cognitive loop; fixed-length denoising uses diffusion. This does not establish instruction following or success.", "",
        "## 1. Memory contamination audit", "",
        f"The previous `runs/real_traces/memory.pt` contained {legacy['historical_real_trace_memory']['exact_heldout_pairs_found']} exact validation/test pairs "
        f"among 22 procedures. It was replaced with the audited guarded train-only snapshot. Historical Phase 5 22/22 numbers describe the old verification run, not the replacement.",
        f"The separate legacy `runs/dual_memory/memory.pt` has {legacy['legacy_dual_memory']['used_slots']} used slots with missing, unverifiable provenance; "
        "that is not proof all slots are contaminated, but it cannot be certified clean and was not used as evaluation memory.",
        f"The new checkpoint has {real['holdout_checks']['audit_after']['checked_slots']} audited slots and "
        f"{real['holdout_checks']['audit_after']['failed_slots']} failures. Its registry covers the 12 real held-out pairs plus "
        f"{real['holdout_checks']['synthetic_heldout_records']} available synthetic validation/test records. Test-time retrieval runs read-only, "
        "including access statistics; the persisted snapshot is saved before test evaluation.", "",
        "The guard uses domain-separated SHA-256, length-framed pair identities, exact input/output matching, "
        "and registered substrings of at least 16 bytes. It checks nested metadata and decodes `*_hex` fields; "
        "explicit test/validation split labels are blocked even without a registry. Whole ingestion batches, intermediate symbolic canvases "
        "and consecutive trace payload runs are checked before mutation. "
        "Guarded checkpoints persist the policy, reject unproven legacy slots and reject direct public bank/state loading.", "",
        "**Limits:** it is opt-in for unlabeled embedding writes, depends on a complete caller-supplied registry and honest raw provenance, "
        "and cannot establish what is inside opaque vectors or pretrained weights. Near duplicates, short embedded substrings, "
        "alternate encodings and direct tensor mutation can evade it. It is not a security boundary or a privacy guarantee.", "",
        "## 2. Autonomous edits and routing", "",
        f"Repository fixture: {real['counts']['train']} train, {real['counts']['validation']} validation, {real['counts']['test']} test; "
        "split by source file, not an external benchmark. The current run uses a fixed 128-action budget, with no target, oracle action, "
        "target-derived budget or host controller during generation. The earlier Phase 5 budget was target-derived, an additional evaluation caveat.",
        f"The real-trained checkpoint produces {count(real['trained_train'])} train and {count(real['trained_test'])} test examples exactly; "
        f"train-memory conditioning gives {count(real['trained_test_with_train_memory'])}, and priming the generic provenance explanation gives "
        f"{count(real['trained_test_with_generic_explanation'])}. These excerpts lack actionable edit instructions, so this is not a fair test of general instruction following.", "",
        f"Explicit route dispatch passed {routing['dispatch']['correct']}/{routing['dispatch']['count']} declared cases. "
        "This is a deterministic dispatch check, **not** inferred routing accuracy or edit success:", "",
        "| Synthetic command | Engine | Exact edits |", "|---|---|---:|",
    ]
    for label, item in routing["synthetic_exact"].items():
        lines.append(f"| {label} | {item['engine']} | {item['exact']}/{item['count']} |")
    den = routing["substitution_denoising"]
    lines += ["", f"For {den['examples']} exact-byte-novel synthetic inputs with 10% substitutions, the corrupted input itself matched "
              f"{pct(den['corrupted_input_accuracy'])} of original bytes; the diffusion output matched {pct(den['output_accuracy_vs_original'])}. "
              "The denoiser mostly copies its input; it did not learn targeted replacements. Instruction-directed diffusion substitution raises "
              "`NotImplementedError` rather than silently ignoring the instruction. Same-length synthetic R tasks therefore fail under the requested route. "
              "Cognitive instruction priming is experimental and was not trained as general instruction following.", "",
              "## 3. Diffusion reconstruction and training defaults", "",
              f"`evaluation/evaluate.py`: clean {pct(rec)}; {over['exact_test_rows_in_train']}/{over['test_rows']} synthetic test rows also occur exactly in train. "
              f"The {novel['count']}-row subset excludes those exact duplicates, but not near duplicates. "
              f"10% position-wise original recovery: substitution {pct(ev['recovery_vs_original_real_bytes']['substitute']['0.1'])}, "
              f"insertion {pct(ev['recovery_vs_original_real_bytes']['insert']['0.1'])}, deletion {pct(ev['recovery_vs_original_real_bytes']['delete']['0.1'])}. "
              f"CPU one-pass inference: {ev['timing_ms_per_sample']:.3f} ms/sample.",
              "Soft-DTW weight and edit-corruption probability now both default to zero; the optional differentiable loss remains available. "
              "This changes future training, not the performance of any already-trained checkpoint. The earlier matched experiment does not establish "
              "that frozen target embeddings blocked gradients; backbone gradients were observed.", "",
              "## 4. Paired halting evidence", "",
              f"Diffusion: calibration, selection and test each used {h['protocol']['population_per_subset']} exact-byte-distinct synthetic sequences, "
              "with training-width padding, a last-timestep q-sampled input, the same initial state and matched per-step RNG. "
              f"A temperature fit on validation yielded held-out per-byte ECE {h['calibration']['test']['ece_10_bins']:.3f}; "
              "byte/timestep observations are correlated, and this is not unconditional generation. "
              f"The validation grid selected {'no' if h['selection']['policy'] is None else 'a'} policy under the strict no-per-sample-byte-loss criterion.",
              f"Test full schedule: {pct(test['full']['byte_accuracy'])} original-byte accuracy, {test['full']['exact']}/{test['full']['count']} exact, "
              f"{test['full']['mean_steps']:.1f} mean steps. An aggressive exploratory threshold used "
              f"{test['aggressive_005_mean']['mean_steps']:.1f} step, {pct(test['aggressive_005_mean']['byte_accuracy'])} aggregate byte accuracy, "
              f"but {test['aggressive_005_mean']['samples_with_fewer_correct_bytes_than_full']}/{test['aggressive_005_mean']['count']} "
              "samples lost correct bytes; both policies produced zero exact sequences. Paired production-loop behavior was checked against the trajectory simulator.",
              f"THINK: on {th['test_full']['count']} narrow synthetic tasks, the uncalibrated exploratory threshold {thought_policy['threshold'] if thought_policy else 'n/a'} "
              f"achieved {count(thought_policy['test']) if thought_policy else 'n/a'} with "
              f"{thought_policy['test']['mean_think_steps'] if thought_policy else 0:.1f} steps versus "
              f"{count(th['test_full'])} at {th['test_full']['mean_think_steps']:.1f} steps. "
              f"The existing default already uses {th['test_existing_default_one_step']['mean_think_steps']:.1f} step and "
              f"scores {count(th['test_existing_default_one_step'])}; there is no demonstrated saving over that default.",
              "Config-driven halting requires a matching calibration scope and an explicit threshold, but that check does not certify quality or guard against distribution shift. "
              "Both default config switches remain false; explicit threshold arguments are warning-only experimental overrides.", "",
              "## 5. Compatibility, cost and verification", "",
              "No learned parameters were added: diffusion **195,905**, included generator **344,408**, memory at capacity 256 **83,565**. "
              "Hashing/substring scans and metadata increase CPU/storage; alignment remains quadratic when explicitly enabled. "
              "Guarded hierarchical memory saves schema 4; unguarded saves remain schema 3. Guarded concept checkpoints use a wrapper so old loaders fail rather than silently discard the policy. "
              "New code still loads legacy unguarded checkpoints, but guarded workflows reject old slots without provenance. Optimizers are not migrated across architecture changes. "
              "Concept-bank slot counting/FIFO was fixed, changing previously broken multi-concept writes.",
    ]
    if verification:
        lines.append(f"Verification: **{verification.get('unit_tests', 'see verification.json')} unit tests**, "
                     "compilation, whitespace checks, tokenizer-import check, guarded ingestion and checkpoint reloads. "
                     "See [verification.json](phase6/verification.json).")
    lines += ["", "## Artifacts", "",
              "- [Clean real-edit metrics](phase6/real_traces/metrics.json) and [guarded memory](phase6/real_traces/memory.pt).",
              "- [Historical memory audit](phase6/legacy_audit.json).",
              "- [Full diffusion report](phase6/evaluation/report.md) and [metrics](phase6/evaluation/evaluation.json).",
              "- [Routing measurements](phase6/routing.json) and [paired halting measurements](phase6/halting.json).",
              "- [Historical Phase 5 diagnostics](diagnostic_report.md).", ""]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
