"""Build the Phase 10 report from recorded metrics and verification artifacts."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path("runs/phase10")
    training = json.loads((root / "training_metrics.json").read_text(encoding="utf-8"))
    benchmark = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
    verification = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    baseline = json.loads(Path("runs/phase10_baseline/training_metrics.json").read_text(encoding="utf-8"))
    validation = training["intent"]["validation"]
    development_test = training["intent"]["test_no_history"]
    initial_test = baseline["intent"]["test_no_history"]
    social = training["response"]["test_no_history"]
    challenge = benchmark["challenge"]
    latency = benchmark["latency"]
    dialogue = benchmark["dialogue"]
    safety = benchmark["safety"]
    params = training["parameters"]
    p = lambda number: f"{100 * number:.1f}%"
    lines = [
        "# Phase 10 — Conversational Foundation",
        "",
        "## Outcome",
        "",
        "Implemented and trained a raw-byte intent classifier, bounded social response decoder, "
        "and a conservative session layer with optional persistent user-scoped memory. "
        "This is an early research prototype, **not** a general conversational agent: "
        "it cannot reliably understand arbitrary requests, answer open-domain questions, "
        "or infer edits from unconstrained language.",
        "",
        "## What was built",
        "",
        "- `GeneratorConditioned.intent_classifier` is `Linear(d_model,d_model) → SiLU → Linear(d_model,7)`. "
        "The stable labels are chat, question, edit, remember, recall, forget and clarify. "
        "It combines frozen causal byte features with a new trainable raw-byte convolutional encoder; "
        "the current-user span is supplied separately from history. Explicit lengths preserve byte 0 and mask padding.",
        "- `ByteDialogueDecoder` generates social replies autoregressively over literal bytes with an internal EOS. "
        "The social training targets are three recurring canned strings; outputs are not evidence of "
        "general knowledge. When classified as a question, four exact reference questions are answered "
        "from a host table and other questions abstain; misclassification can choose another path. "
        "The decoder checkpoint is bound to the generator SHA-256 and expected attention-head count.",
        "- `ConversationSession` classifies before action selection. It keeps bounded, in-process recent turns; "
        "validates explicit commands before memory writes/deletes or edits; and asks for clarification "
        "when unsupported. Only explicit single-byte numeric replacements are routed through the Phase 9 "
        "learned cognitive policy. The host verifies that the model actually completed exactly that edit "
        "before returning a changed document; it does not overwrite the predicted bytes.",
        "- `ConversationMemory` stores explicitly requested facts as length-framed, exact raw-byte "
        "procedural key/value records. It supports separate user namespaces, cross-process reload, "
        "atomic file replacement and rollback on ordinary save errors, and deletion of every matching "
        "live version. It rejects a new fact when the bank is full rather than evicting another user's "
        "record through this API. Holdout guards audit the full explicit write request.",
        "",
        "## Training and independent checks",
        "",
        f"The final run used {training['data']['train_after_preflight']} training, "
        f"{training['data']['validation_inputs']} validation and {training['data']['test_inputs']} test messages; "
        f"{training['data']['train_blocked']} training messages were blocked by the byte holdout guard. "
        "Inputs and wording templates were split; prompts are synthetic and English-only. "
        f"The classifier and encoder selected epoch {training['intent']['selected_epoch']} by validation "
        "macro accuracy and then validation NLL; the decoder selected by validation response loss.",
        "",
        "| Evaluation | Result | Scope |",
        "|---|---:|---|",
        f"| Validation, mixed with/without a short history | {validation['correct']}/{validation['count']} "
        f"({p(validation['accuracy'])}); macro {p(validation['macro_accuracy'])} | Epoch selection |",
        f"| Synthetic template test, no history | {development_test['correct']}/{development_test['count']} "
        f"({p(development_test['accuracy'])}); macro {p(development_test['macro_accuracy'])} | "
        "Development holdout; head-only baseline test was inspected before changing architecture |",
        f"| Later, manually written challenge | {challenge['correct']}/{challenge['count']} "
        f"({p(challenge['accuracy'])}); macro {p(challenge['macro_accuracy'])} | Post-selection; small, not representative |",
        f"| Challenge: vague/clarify class | {challenge['per_class']['clarify']['correct']}/"
        f"{challenge['per_class']['clarify']['count']} | Important unmet classification capability |",
        f"| Action on those vague requests | {benchmark['vague_requests']['asked']}/"
        f"{benchmark['vague_requests']['count']} asked | Fallbacks, not correct classification |",
        f"| Social decoder on new test inputs | {social.get('complete',0)}/{social['count']} ended; "
        f"{social.get('valid_text',0)}/{social['count']} valid UTF-8; "
        f"{social.get('exact_seen_template',0)}/{social['count']} exact seen template | Outputs/targets are not novel |",
        f"| Side-effect checks | {safety['safe']}/{safety['count']} | Five specifically constructed negative/malformed cases, not a security certification |",
        "",
        f"The original head-only experiment reached {p(initial_test['macro_accuracy'])} macro accuracy on "
        "the template test; that result informed the addition of the byte encoder, so this test must "
        "not be presented as an untouched final architecture holdout. The later 35-case challenge was "
        "authored after model selection and was retrospectively checked for raw-byte overlap with "
        "training inputs, but may contain semantic similarities and is too small for deployment claims.",
        "",
        f"At the uncalibrated {challenge['threshold']:.2f} abstention threshold, challenge coverage was "
        f"{p(challenge['coverage_at_threshold'])} and covered accuracy {p(challenge['accuracy_when_covered'])}. "
        "High confidence does not imply correctness: `Do not remember my name is Sol.` was classified "
        "as remember with approximately 0.995 confidence, but the explicit-command check prevented a write.",
        "",
        "## Multi-turn behavior",
        "",
        "A single fresh synthetic session greeted the user, explicitly stored a hometown, reloaded "
        "an isolated memory file in a new session, recalled it for the original namespace, did not "
        "return it to a different namespace, asked about an ambiguous edit, completed a specified "
        "one-byte edit, and deleted the fact. This is one end-to-end scenario, not general proof of "
        "memory comprehension. The temporary test memory was discarded.",
        "",
        f"- Greeting: `{dialogue['greeting']['response']}` ({dialogue['greeting']['response_source']}).",
        f"- After reload: `{dialogue['recall_after_reload']['response']}`",
        f"- Edit events: `{dialogue['structured_edit']['edit_events']}`; exact output was verified, "
        "and the requested fields were explicitly parsed rather than inferred freely.",
        f"- Known reference question answered: {dialogue['known_reference_question_answered']}; "
        f"unknown question abstained: {dialogue['unknown_question_abstained']}.",
        "",
        "## Parameters, compatibility and verification",
        "",
        f"The Phase 9 generator has {params['phase9_generator']:,} parameters; the new encoder adds "
        f"{params['intent_encoder']:,} and head {params['intent_head']:,}, for "
        f"{params['generator_total']:,}. The separate decoder adds {params['response_decoder']:,} "
        f"({params['generator_total'] + params['response_decoder']:,} combined). "
        f"All {verification['phase9_source_tensors_unchanged']} Phase 9 checkpoint tensors were bit-identical. "
        "Entirely missing legacy intent components initialize deterministically and remain marked untrained; "
        "partially missing new checkpoints fail strict loading. Older software cannot strictly load the new "
        "weights; rebuild optimizer state and revalidate calibration after architectural changes.",
        "",
        f"The full regression suite passed **{verification['tests_passed']}/{verification['tests_total']}**. "
        "The Phase 9 `Hello Worl → Hello Word` diagnostic still succeeded in natural, explicitly "
        "conditioned and masked paths, but the example was in training and does not establish generalization. "
        f"Warm one-thread CPU intent-only latency over {latency['samples']} samples: "
        f"median {latency['median_ms']:.2f} ms, p95 {latency['p95_ms']:.2f} ms; excludes loading, "
        "memory I/O, generation and editing.",
        "",
        "## Privacy and operational limits",
        "",
        "Memory is ephemeral unless a path is supplied; saved checkpoint contents are **unencrypted**. "
        "Namespaces are not authentication, and the host must authenticate users, restrict file access, "
        "secure backups and protect sensitive data. Locks cover sessions sharing one wrapper only, not "
        "multiple processes; atomic replacement is not a power-loss durability guarantee. Deletion clears "
        "the current in-memory bank and new checkpoint, not old backups, disk remnants or OS caches. "
        "Only explicit `remember my ... is ...` facts persist—not full conversations or inferred personal "
        "details. Memory capacity is finite; there is no semantic recall of arbitrary conversations.",
        "",
        "The current context limit is 128 bytes, and only recent whole turns fitting that limit are included. "
        "Natural-language inputs must be UTF-8; edit documents may be arbitrary raw bytes. The classifier "
        "threshold is not calibrated for deployment, the learned models are narrow, and the bounded "
        "command grammars do not cover ordinary free-form tasks. The holdout guard cannot detect "
        "paraphrases, unknown datasets, opaque embeddings or private tensor mutation.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_conversation --intent-epochs 60 --response-epochs 100 --batch-size 128 --feature-batch-size 32 --intent-lr 0.003 --response-lr 0.003 --seed 101010 --threads 1 --output-dir runs/phase10",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase10_benchmark --samples 30 --threads 1",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase9_diagnostic --structured-generator runs/phase10/generator.pt --output runs/phase10/phase9_regression.json",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase10_verify",
        ".venv/bin/python -m evaluation.phase10_report",
        ".venv/bin/python -m demo.conversation --show-intent",
        "```",
        "",
        "Detailed evidence: `runs/phase10/training_metrics.json`, `runs/phase10/benchmark.json`, "
        "`runs/phase10/phase9_regression.json`, and `runs/phase10/verification.json`. The development "
        "head-only metrics are in `runs/phase10_baseline/training_metrics.json`.",
        "",
    ]
    Path("runs/phase10_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
