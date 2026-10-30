"""Build the evidence-led Phase 12 Markdown report from measured artifacts."""
from __future__ import annotations

import json
from pathlib import Path


def _read(name):
    return json.loads((Path("runs/phase12") / name).read_text(encoding="utf-8"))


def _yn(value):
    return "PASS" if value else "NOT MET"


def main():
    bench, memory, verify = _read("benchmark.json"), _read("memory_eval.json"), _read("verification.json")
    intent, reader, diversity = _read("intent_metrics.json"), _read("vector_reader_metrics.json"), _read("diverse_response_metrics.json")
    response, replay = _read("response_metrics.json"), _read("replay_metrics.json")
    short, long, cross = memory["short_context"], memory["long_context"], memory["cross_session"]
    fresh, old, old10, old11 = bench["fresh_intent_test"], bench["phase10_old_development"], bench["phase10_original_development"], bench["phase11_old_development"]
    varied, routed = bench["diversity"], bench["diversity"]["routed_session"]
    dev = varied["illustrative_routed_development"]
    binary = bench["raw_byte_diagnostic"]
    params = bench["parameter_counts"]
    cases = memory["fresh_final_cases"]
    safe = bench["safety_checks"]
    lines = [
        "# Phase 12 — Force Memory Usage + Prevent Forgetting",
        "",
        "## Executive result",
        "",
        "The short, long, and fresh-process demonstrations use a **trained neural reader over actual retrieved vectors**, with the earlier episode removed from the model's byte context. The representation is a narrowly parsed, reversible bit codec; it is not learned semantic compression. Replay improves an inspected historical comparison but does not prove forgetting is prevented. The diversity decoder meets the direct-decoder count, while the complete session **misses** it on the protected prompt because the classifier misroutes the greeting.",
        "",
        "| Acceptance check | Measurement | Status / scope |",
        "|---|---:|---|",
        f"| Cleared-context favorite-color example | `{short['normal']['text']}` | {_yn(short['correct'])}; actual vector reader, no answer fallback |",
        f"| Fresh structured vector facts | {cases['correct']}/{cases['count']} | {_yn(cases['correct'] == cases['count'])}; small synthetic statements |",
        f"| 500 filler turns | {long['filler_episodes_recorded']}/{long['filler_messages_sent']} recorded; answer `{long['answer']['text']}` | {_yn(long['correct'])}; episodic capacity {long['bank_episodic_capacity']} |",
        f"| New process / Alice works at ByteCorp | `{cross['child']['result']['text']}` | {_yn(cross['correct'])}; real checkpoint reload |",
        f"| Phase 10 accuracy >70% | {old['correct']}/{old['count']} ({old['accuracy']:.1%}) | {_yn(old['accuracy'] > .70)}; previously inspected development set |",
        f"| Fresh overall intent >85% | {fresh['correct']}/{fresh['count']} ({fresh['accuracy']:.1%}) | {_yn(fresh['accuracy'] > .85)}; balanced synthetic prompts |",
        f"| >10 distinct responses / 20 | direct {varied['unique_sequences']}/20; routed {routed['unique_sequences']}/20 | Direct decoder passes; complete session **does not** on this prompt |",
        f"| Malformed <10% | direct {varied['unverified_or_malformed_conservative']}/20 conservative proxy failures | Narrow template/structure proxy only; not a general quality test |",
        f"| Regression tests ≥147 | {verify['tests_passed']}/{verify['tests_total']} | {_yn(verify['tests_passed'] >= 147)} |",
        "",
        "## What forced memory actually means",
        "",
        "1. The observed statement is parsed by a narrow host grammar into a key and value. The value is encoded, at write time, into one to five 64-dimensional vectors: signed bits for up to eight bytes per slot, multiplied by a fixed orthogonal matrix. It is lossless, finite-capacity, **not encrypted**, and deliberately structured; this is not an opaque semantic summary.",
        "2. The observed episode is written into the existing episodic key-value bank. On a supported question, an exact owner/topic filter retrieves the vector slots. The generator gets only `U:` plus the current question and these vectors as `memory_context`; the separate learned reader gets the question features and the same vectors. The answer label is never passed into inference. The reader emits raw bytes and an internal EOS; a failed read abstains.",
        "3. Evaluation patches recent-transcript retrieval, explicit procedural fact lookup, the legacy episode-fact helper, and the contextual decoder to fail if called during the positive vector-only route. It also checks that every generator byte input is exactly the current question. No positive evidence-grounding or answer oracle corrects a wrong reader output.",
        "",
        f"- Original vector → **{short['normal']['text']}**. Zeroed vector → **{short['zeroed']['text']}**. Another user's manually substituted vector → **{short['swapped_other_user_vector']['text']}**. Restoring the vector → **{short['restored']['text']}**. A normal unrelated user did not retrieve the first user's fact. The swap is an intentional counterfactual test, not authorized cross-user retrieval.",
        f"- In the 500-turn run, the stored owner transcripts contained the target word before the final question: **{long['target_bytes_in_owner_transcripts_before_query']}**; another user's vector remained unchanged: **{long['other_user_vector_intact']}**. Filler messages went through `respond()` and were recorded, rather than merely incrementing a counter.",
        f"- In the fresh-process workplace case, the intent classifier predicted `{cross['child']['result']['intent']}` and the exact read-only grammar performed the explicitly labelled `{cross['child']['result']['routing']}` override. The expected value was used only for scoring after the child returned.",
        "- The public blue case has been inspected. The ten new cases are protected by source/target hash before Phase 12 training, but related question wordings and structures recur across prior data. Ten examples cannot establish generalization, and exact hash guards cannot detect paraphrases, hidden exposure or false provenance.",
        "",
        "## Raw-byte scope and limitations",
        "",
        f"- The codec can represent every byte 0–255. An exploratory direct-reader probe decoded **{binary['single_byte_correct']}/{binary['single_byte_count']} single-byte values**. However, it corrupted some multi-byte binary sequences:",
    ]
    for row in binary["multi_byte_sequences"]:
        lines.append(f"  - `{row['input_hex']}` → `{row['output_hex']}`: {'exact' if row['exact'] else 'incorrect'}.")
    lines += [
        "- This is not evidence of arbitrary binary fidelity. The conversational grammar accepts visible UTF-8 text; long values over 40 bytes are not silently truncated and do not enter vector recall. The fixed codec fingerprint and generator SHA-256 bind the selected reader; incompatible checkpoints fail to load.",
        "",
        "## Rehearsal and forgetting",
        "",
        f"- The intent trainer preflight accepted **{intent['preflight']['old_accepted']} old** and **{intent['preflight']['new_accepted']} new** synthetic examples, blocked {intent['preflight']['old_blocked'] + intent['preflight']['new_blocked']}, and registered {intent['preflight']['protected_hashes']} holdout hashes. Every optimizer batch had **{intent['mix_per_batch']['new']} new / {intent['mix_per_batch']['old']} old** rows ({intent['mix_per_batch']['fraction_old']:.2%} old) plus teacher distillation.",
        f"- Selected epoch {intent['selected_epoch']} using validation only. The feasibility threshold was raised from .82 to .85 after the first validation-only run, before inspecting the Phase 12 final test; this is validation tuning, not a fully preregistered selection procedure.",
        f"- The old, previously inspected Phase 10 development set went from **{old10['correct']}/{old10['count']}** (Phase 10) to **{old11['correct']}/{old11['count']}** (Phase 11) to **{old['correct']}/{old['count']}** (Phase 12). FORGET is **{old['per_class']['forget']['correct']}/{old['per_class']['forget']['count']}**, still below the original **{old10['per_class']['forget']['correct']}/{old10['per_class']['forget']['count']}**. The previously inspected Phase 11 boundary is {bench['phase11_boundary_development']['correct']}/{bench['phase11_boundary_development']['count']} (vague {bench['phase11_boundary_development']['vague']['correct']}/{bench['phase11_boundary_development']['vague']['count']}); Phase 11 was 19/20 and 9/10 respectively. Rehearsal is not a guarantee against regressions.",
        f"- `replay_multitask_phase12.pt` has **{replay['stored']}/{replay['capacity']}** guarded synthetic training records: 300 old intent, 300 new intent, 160 vector, 160 diverse-response, 80 contextual-response. It samples the training data because the buffer is bounded, and the live conversation path never automatically inserts user transcripts. Guarded task labels prevent accidental mixing of incompatible objectives. The 70/30 optimizer mix applies to intent training; the newly introduced reader/response tasks were **not all** trained with 30% replay.",
        "",
        "## Response diversity and quality",
        "",
        f"- The learned style-conditioned autoregressive byte decoder has {params['diverse_decoder']:,} parameters. It is trained with cross-entropy plus a cosine penalty on **predicted probability distributions**, not nondifferentiable sampled byte IDs. Thirty-two human-authored synthetic response styles recur in training.",
        f"- Protected prompt: `{varied['prompt']}`. Direct CHAT-conditioned sampling, with randomly sampled style and bytes at temperature {varied['sampling_temperature']}, gave **{varied['unique_sequences']}/{varied['samples']} distinct outputs**, {varied['complete_valid_utf8']}/{varied['samples']} complete UTF-8, and {varied['unverified_or_malformed_conservative']}/{varied['samples']} failing a deliberately strict structural/known-template proxy. This proxy rewards memorization; it is not a general grammar or semantic-quality score.",
        f"- The actual classifier labelled the same prompt `{routed['outputs'][0]['intent']}` and the session produced **{routed['unique_sequences']}/{routed['count']} distinct responses**, with **{routed['neural_diverse_responses']}** diverse-decoder replies. Thus the end-to-end diversity target fails for the protected prompt. An illustrative previously used `{dev['prompt']}` development prompt, selected after observing the failure, produced {dev['unique_sequences']}/20 distinct system replies; it is not a fresh test.",
        "- Direct samples (repetitions preserved):",
    ]
    lines.extend(f"  - `{row['text'].replace('`', chr(39))}`" for row in varied["outputs"])
    lines += [
        "",
        "## Safeguards, compatibility and costs",
        "",
        f"- {sum(x['passed'] for x in safe.values())}/{len(safe)} constructed safety checks passed: user isolation, no-storage wording, negated edit, underspecified edit, and forget/recall. These are not a comprehensive security evaluation. The English privacy pattern and intent confidence are limited; neither replaces host consent or authentication.",
        "- By default storage is ephemeral. Supplying `--memory PATH` automatically persists full user/assistant transcripts and **reversible** fact vectors in an **unencrypted** file. Hashes and rotation are not encryption; user IDs are namespaces, not identity verification. Use `--no-episode-storage` to disable implicit transcript/vector storage and retrieval. Explicit old-style remembered facts and local process history remain available in that mode.",
        "- `VectorConversationMemory` recycles only the same owner's ordinary episodes. It does not evict vector facts or other users' entries; full banks can refuse writes. Updates and forgetting are scoped and rolled back on ordinary save failure. `Forget my KEY` removes that fact; `Forget Alice's workplace` removes the corresponding workplace vector. Both purge the user's live transcript but not backups, other facts, old checkpoints or disk remnants. Process-local locks do not synchronize independent writers.",
        f"- Parameter counts: generator **{params['generator']:,}**, contextual response decoder **{params['contextual_decoder']:,}**, vector reader **{params['vector_reader']:,}**, diverse decoder **{params['diverse_decoder']:,}**. This adds separate models and storage; it does not replace the raw-byte backbone with a tokenized LLM. All {verify['phase9_source_tensors_unchanged']} original Phase 9 generator tensors remain unchanged, and the three-way edit diagnostic remains exact. Older Phase 10/11 paths are available via `--legacy-phase10` / `--legacy-phase11`.",
        f"- Vector-reader training preflight accepted {reader['preflight']['accepted']} and blocked {reader['preflight']['blocked']} synthetic examples; validation-only selected epoch {reader['selected_epoch']} ({reader['validation']['exact']}/{reader['validation']['count']} exact). Diverse response epoch {diversity['selected_epoch']} had {diversity['validation']['teacher_sequence_exact']}/{diversity['validation']['sequences']} teacher-forced validation sequences. The separate compatible contextual decoder selected epoch {response['selected_epoch']} with {response['validation_exact_copy']['correct']}/{response['validation_exact_copy']['count']} validation copy cases; it is not part of the vector-only positive answer.",
        "- Reader/response checkpoints are SHA-256-bound to their generator; vector slots record a codec fingerprint. Memory provenance checks require truthful raw sources and targets, but cannot audit opaque tensor contents. Training/validation data are synthetic and small, and prior real uninstructed editing remains 0/8.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_replay_intents",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_vector_memory",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_diverse_responses",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.train_contextual_responses --generator runs/phase12/generator.pt --phase12-guard --output-dir runs/phase12 --epochs 48 --seed 122012",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m training.finalize_replay",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.memory_eval",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase12_benchmark",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m evaluation.phase12_verify",
        ".venv/bin/python -m demo.conversation --show-intent",
        "```",
        "",
        "Evidence: `runs/phase12/{memory_eval,benchmark,verification,intent_metrics,vector_reader_metrics,diverse_response_metrics,response_metrics,replay_metrics}.json`, plus selected checkpoints and bounded replay snapshots in the same directory. `AGENTS.md`, `QWEN_CHAT_PROMPT.md`, and `README.md` describe the architecture, trade-offs and privacy implications.",
    ]
    path = Path("runs/phase12_report.md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
