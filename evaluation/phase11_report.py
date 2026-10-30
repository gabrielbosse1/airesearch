"""Build an evidence-led Phase 11 report from selected artifacts."""
from __future__ import annotations

import json
from pathlib import Path


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    benchmark = load("runs/phase11/benchmark.json")
    intent = load("runs/phase11/intent_metrics.json")
    response = load("runs/phase11/response_metrics.json")
    verify = load("runs/phase11/verification.json")
    previous = load("runs/phase10/training_metrics.json")["intent"]["test_no_history"]
    legacy = intent["test"]["phase10_development_test"]
    boundary, overall = benchmark["boundary_test"], benchmark["overall_intent_test"]
    retention, diversity = benchmark["retention_test"], benchmark["byte_diversity"]
    abl = retention["decoder_ablation"]
    safe = benchmark["safety_and_public_demo"]
    val = response["validation_exact_copy"]
    phase9 = load("runs/phase11/phase9_regression.json")
    demo = safe["user_blue_example"]
    samples = list(dict.fromkeys(row["text"] for row in diversity["samples"]))
    lines = [
        "# Phase 11 — Conversational Refinement & Multi-Turn Memory",
        "",
        "## Results",
        "",
        "| Measure | Result | Scope |",
        "|---|---:|---|",
        f"| Vague classification | {boundary['vague']['correct']}/{boundary['vague']['count']} ({boundary['vague']['accuracy']:.0%}) | 10 of 20 boundary prompts; >80% target |",
        f"| Specific classification | {boundary['specific']['correct']}/{boundary['specific']['count']} ({boundary['specific']['accuracy']:.0%}) | Other half of the same boundary set |",
        f"| Mixed boundary accuracy | {boundary['correct']}/{boundary['count']} ({boundary['accuracy']:.0%}) | Includes vague and specific prompts |",
        f"| Overall intent accuracy | {overall['correct']}/{overall['count']} ({overall['accuracy']:.1%}) | 70 synthetic prompts, seven balanced classes; >85% target |",
        f"| Historical Phase 10 development set | {legacy['correct']}/{legacy['count']} ({legacy['accuracy']:.1%}) | Down from {previous['correct']}/{previous['count']} ({previous['accuracy']:.1%}) before fine-tuning; not a pristine test |",
        f"| End-to-end two-turn retention | {retention['correct']}/{retention['count']} ({retention['accuracy']:.0%}) | Independent ephemeral user memories; >70% system target |",
        f"| Learned decoder within retention | {retention['generated_correct']}/{retention['count']} | Exact whole-value check; other {retention['fallback_correct']} successes used the grounded host fallback |",
        f"| Same-prompt response-byte diversity | {diversity['unique_byte_sequences']}/{diversity['sample_count']} distinct | Direct decoder, CHAT intent, temperature {diversity['sampling_temperature']}; {diversity['complete_valid_utf8']}/{diversity['sample_count']} complete UTF-8, not a quality guarantee |",
        f"| Full unittest regressions | {verify['tests_passed']}/{verify['tests_total']} | Includes Phase 9, Phase 10 and Phase 11; >=120 target |",
        "",
        "**The system meets the numerical retention target because it includes an exact-evidence fallback; the neural decoder alone does not meet 70%.** Classification and retention fixtures are small, synthetic, English-language examples. This is a bounded prototype, not demonstrated open-domain conversational competence.",
        "",
        f"**Important regression:** on the older Phase 10 development set, REMEMBER fell from {previous['per_class']['remember']['correct']}/{previous['per_class']['remember']['count']} to {legacy['per_class']['remember']['correct']}/{legacy['per_class']['remember']['count']}, and FORGET fell from {previous['per_class']['forget']['correct']}/{previous['per_class']['forget']['count']} to {legacy['per_class']['forget']['correct']}/{legacy['per_class']['forget']['count']}; CLARIFY improved from {previous['per_class']['clarify']['correct']}/{previous['per_class']['clarify']['count']} to {legacy['per_class']['clarify']['correct']}/{legacy['per_class']['clarify']['count']}. The set is template-heavy and was used during Phase 10 development. Accuracy cannot be assumed to improve across all intents.",
        "",
        "### Public blue example",
        "",
        f"- User: `My favorite color is blue.` → Assistant: `{demo['first']['response']}`",
        f"- User: `What is my favorite color?` → Assistant: `{demo['second']['response']}` (source: `{demo['second']['source']}`, retrieved episodes: {demo['second']['retrieved_episodes']}).",
        "- This particular example was inspected during development and is not counted in the ten fresh retention cases.",
        "",
        "### What the memory ablation actually shows",
        "",
        f"- With both retrieved vectors and raw episode bytes, the direct decoder produced the exact value in **{abl['correct_by_condition']['vectors_and_bytes']}/{abl['count']}** cases. With raw episode bytes but **no** vectors it produced **{abl['correct_by_condition']['bytes_only']}/{abl['count']}**; vectors alone: **{abl['correct_by_condition']['vectors_only']}/{abl['count']}**; neither: **{abl['correct_by_condition']['neither']}/{abl['count']}**.",
        f"- Removing the vectors while retaining episode bytes changed **{abl['outputs_changed_by_vectors_given_bytes']}/{abl['count']}** greedy outputs. The retrieved vectors are injected and change generator features (unit tested), but this ablation does **not** demonstrate that they drive these responses; byte-level history plus the host fallback accounts for observed retention.",
        "- The ablation uses gold RECALL intent to isolate the decoder. The end-to-end score instead uses the classifier and labels every exact read-only routing override and fallback.",
        "",
        "### Response diversity",
        "",
        f"- Same input, same intent: {diversity['unique_byte_sequences']} distinct byte sequences from {diversity['sample_count']} samples; byte entropy {diversity['byte_entropy_bits']:.3f} bits, distinct-1 {diversity['distinct_byte_1']:.4f}, distinct-2 {diversity['distinct_byte_2']:.4f}. These statistics are length- and sample-dependent.",
        f"- Greedy decoding across eight related CHAT inputs produced {diversity['greedy_distinct_inputs']['unique_byte_sequences']} unique outputs. The small model mostly reproduces familiar training-style greetings and sometimes emits malformed wording; valid UTF-8 is not semantic quality.",
        "- Distinct sampled outputs:",
    ]
    lines += [f"  - `{s.replace('`', chr(39))}`" for s in samples]
    lines += [
        "",
        "## Architecture and safeguards",
        "",
        f"- The generator has {benchmark['parameter_counts']['generator']:,} parameters; the separate autoregressive pointer/generator decoder has {benchmark['parameter_counts']['response_decoder']:,} (Phase 10 decoder: 62,273). Inputs, histories and outputs are raw UTF-8 bytes; EOS and segment identifiers are internal metadata, not output bytes. There is no external tokenizer.",
        "- The current-turn learned classifier runs before action selection. The session retrieves up to five latest scoped episodes using `MemoryModel.retrieve()` and passes their values as `memory_context` to the generator. It also passes actual current/episode bytes to the response decoder. Exact recall and narrowly anchored affirmative remember/forget commands may override an incorrect or low-confidence intent; results label those routes. The command overrides were added after the legacy development regression was inspected, not independently validated as a model improvement.",
        "- Destructive or mutating actions require narrow affirmative grammar; requested edits must be verified by the existing Phase 9 cognitive mechanism. For memory recall the generated value is checked against exact retrieved evidence; on mismatch or truncation, `grounded_fallback` returns the actual value. It is not neural generation and can miss paraphrases or facts outside the last five turns.",
        f"- The constructed checks passed {sum(bool(x['passed']) for x in safe.values())}/{len(safe)}: blue example, user isolation, explicit no-storage, negated edit, underspecified edit and forgetting. These cases are not a comprehensive safety test. A negated edit can still be classified EDIT at high confidence; the anchored side-effect grammar blocks it.",
        "",
        "## Privacy and compatibility",
        "",
        "- By default the demo keeps episodes in process only. **Adding `--memory PATH` now automatically writes full conversation turns to an unencrypted file.** The CLI prints a warning; `--no-episode-storage` disables implicit episode storage/retrieval but retains in-process history and permits explicitly remembered facts. A recognized no-storage turn is not recorded; session-wide no-storage also disables future session writes and local history. The English detector is limited; it is not a universal consent or privacy guarantee.",
        "- User IDs are namespaces, not authentication. Hosts must supply consent, access control, encryption, and cross-process synchronization. Atomic replacement rolls back ordinary failures; it is not crash-proof transactional storage or forensic erasure. When the episodic bank is full it recycles only the same user's oldest episode, never another user's.",
        "- Forgetting an exact fact also purges that user's entire live stored transcript to avoid retaining the detail in previous turns. It cannot erase backups, earlier checkpoints or filesystem remnants. Episodes are not promoted to semantic memory. The legacy demo remains available through `--legacy-phase10`.",
        f"- All {verify['phase9_source_tensors_unchanged']} Phase 9 source generator tensors remain identical. The three-way one-byte diagnostic remains exact: diffusion `{phase9['diffusion_native']['output_text']}`, natural cognition `{phase9['cognitive_natural_only']['output_text']}`, and explicit-request cognition `{phase9['cognitive_explicit_request']['output_text']}`. This example was in training; explicit request and stored replay are not autonomous generalization. Historical checkpoint hashes and real memory stayed unchanged during verification.",
        "",
        "## Training, held-out discipline and limitations",
        "",
        f"- Intent training preflight accepted {intent['preflight']['accepted']:,} examples and blocked {intent['preflight']['blocked']}; selected epoch {intent['selected_epoch']} using only the predeclared boundary, overall and prior validation sets. The final intent checkpoint was not selected on test.",
        f"- Response training preflight accepted {response['preflight']['accepted']:,} rows and blocked {response['preflight']['blocked']}. It used varied synthetic response styles, synthetic copying examples and training-only byte-alignment labels. Epoch {response['selected_epoch']} was selected by validation copying ({val['correct']}/{val['count']}) and then validation NLL ({response['validation_loss']:.4f}); held-out answers were never passed to inference.",
        "- The first decoder's ten retention cases were examined during development; they were replaced by ten new values protected **before** the copy-alignment run. The new test shares generic question wording and response styles with training, so it tests value retention in a narrow format, not fully novel dialogue. The first decoder and its inspected fixtures remain in `runs/phase11_initial/` as developmental evidence.",
        "- Exact and long-substring holdout guards cannot establish semantic novelty, audit hidden pretrained exposure, or serve as a privacy boundary. The generator's 0.55 threshold is not calibrated for deployment, auto-generated replies may be incorrect or malformed, the response decoder is deliberately small, and no evidence here overturns earlier 0/8 real uninstructed edit results.",
        "",
        "## Reproduction and artifacts",
        "",
        "```bash",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.conversational_eval_v2",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase11_verify",
        ".venv/bin/python -m demo.conversation --show-intent",
        "```",
        "",
        "Training commands and privacy details are in `AGENTS.md` and `README.md`. Detailed evidence: `runs/phase11/intent_metrics.json`, `runs/phase11/response_metrics.json`, `runs/phase11/benchmark.json`, `runs/phase11/phase9_regression.json`, `runs/phase11/verification.json`, and `runs/phase11/unittest.log`. Selected checkpoints: `runs/phase11/generator.pt` and `runs/phase11/response.pt`.",
    ]
    path = Path("runs/phase11_report.md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
