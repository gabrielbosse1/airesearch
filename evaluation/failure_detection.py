"""Classify measured failure modes from autonomous edits, denoising and retrieval."""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def counter(rows):
    return dict(collections.Counter(rows))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--edit-report", default="runs/edit_operation_analysis.json")
    p.add_argument("--alignment-report", default="runs/alignment/metrics.json")
    p.add_argument("--learning-report", default="runs/learning_curve.json")
    p.add_argument("--calibration-records", default="runs/calibration_predictions.npz")
    p.add_argument("--output", default="runs/failure_modes.md")
    p.add_argument("--json", default="runs/failure_detection.json")
    args = p.parse_args()
    edit = json.loads(Path(args.edit_report).read_text()); alignment = json.loads(Path(args.alignment_report).read_text())
    curve = json.loads(Path(args.learning_report).read_text()); arrays = np.load(args.calibration_records)
    real = edit["real_test"]["examples"]
    edit_failures = counter(row["failure"] for row in real)
    diffusion = {}
    for name, sample in (("baseline", alignment["baseline_test"]),
                         ("soft_dtw_retrained", alignment["arms"]["soft_dtw_retrained"]["test"])):
        diffusion[name] = {}
        for mode in ("clean", "insert", "delete"):
            rows = sample[mode]["examples"]
            classes = []
            for row in rows:
                similarity = 1 - row["edit_distance"]/max(row["length"], 1)
                position = row["position_matches"]/max(row["length"], 1)
                if row["exact"]:
                    classes.append("exact")
                elif position < 0.5 and similarity >= 0.75:
                    classes.append("shift_dominated_proxy")
                else:
                    classes.append("other_inexact")
            diffusion[name][mode] = {"count": len(rows), "classes": counter(classes)}
    calibration = {}
    for model in ("diffusion", "generator"):
        confidence = arrays[f"{model}_test_confidence"]; correct = arrays[f"{model}_test_correct"].astype(bool)
        high = confidence >= 0.9
        wrong = high & ~correct
        calibration[model] = {"bytes": len(confidence), "wrong": int((~correct).sum()),
                              "high_confidence_bytes": int(high.sum()), "high_confidence_wrong": int(wrong.sum()),
                              "high_confidence_error_fraction": float(wrong.sum()/high.sum()) if high.sum() else None,
                              "note": "per-byte counts are clustered within examples and conditions"}
    last = curve["history"][-1]["validation"]
    retrieval = {"count": last["retrieval_queries"], "matched_concept": int(sum(r["hit"] for r in last["retrieval_details"])),
                 "wrong_concept": int(sum(not r["hit"] for r in last["retrieval_details"]))}
    report = {"protocol": "Counts use distinct task populations, not additive across benchmarks; shift-dominated classification is a heuristic, not an inferred cause.",
              "real_edits": {"count": len(real), "classes": edit_failures}, "diffusion": diffusion,
              "confidence": calibration, "validation_retrieval": retrieval}
    output_json = Path(args.json); output_json.parent.mkdir(parents=True, exist_ok=True); output_json.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Phase 5 Failure Modes", "", "## Autonomous real edit failures", "",
             f"Held-out repository excerpts: **{len(real)}**. No host actions or target outputs were passed to the generator.", "",
             "| Classification | Cases |", "|---|---:|"]
    lines += [f"| {key.replace('_', ' ')} | {value} |" for key, value in edit_failures.items()]
    lines += ["", "## Diffusion recovery", "", "At 10% fixed-length insertion/deletion corruption, compared to the clean sequence. "
              "`shift dominated proxy` means position accuracy < 50% while Levenshtein similarity >= 75%; this heuristic does not prove causation.", "",
              "| Checkpoint | Mode | Samples | Exact | Shift-dominated proxy | Other inexact |", "|---|---|---:|---:|---:|---:|"]
    for model, modes in diffusion.items():
        for mode, row in modes.items():
            c = row["classes"]
            lines.append(f"| {model} | {mode} | {row['count']} | {c.get('exact',0)} | {c.get('shift_dominated_proxy',0)} | {c.get('other_inexact',0)} |")
    lines += ["", "## Overconfident byte predictions", "", "Scores >= 0.90 on the separate calibration test; bytes from the same sample/noise conditions are correlated.", "",
              "| Model and calibration scope | High-confidence bytes | Wrong among them | Error fraction |", "|---|---:|---:|---:|"]
    for model, row in calibration.items():
        fraction = "n/a" if row["high_confidence_error_fraction"] is None else f"{row['high_confidence_error_fraction']:.3%}"
        lines.append(f"| {model} | {row['high_confidence_bytes']} | {row['high_confidence_wrong']} | {fraction} |")
    lines += ["", "## Retrieval", "", f"On {retrieval['count']} unique-prefix validation queries, "
              f"{retrieval['wrong_concept']} top-1 memories had a different concept label. This is a metadata-label proxy, not a causal diagnosis.", "",
              "## Interpretation and next checks", "", "- Pointer/payload and action-count errors limit autonomous real edits; collect broader paired edits with explicit task instructions.",
              "- The shift proxy suggests testing length-aware, aligned decoders and corruption-specific supervision; do not equate high edit similarity with reconstruction of missing bytes.",
              "- Refit confidence after training and validate independently for each use; next-byte/denoising calibration does not certify THINK or ancestral stopping.",
              "- Retrieve/evict slots under task-grounded utility tests before trusting raw importance scores."]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text("\n".join(lines)+"\n")
    print(f"Wrote {args.output} and {args.json}: real edit classes {edit_failures}; validation retrieval errors {retrieval['wrong_concept']}/{retrieval['count']}.")


if __name__ == "__main__":
    main()
