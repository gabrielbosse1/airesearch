"""Evaluate vector-only short, 500-message and fresh-process conversation memory.

No answer or gold intent is supplied to a query. During recall, transcript and
procedural-fallback APIs are patched to fail if called; feature-input spies
verify that only the current question is supplied as bytes. This evaluates a
bounded structured fact codec and neural readout, not semantic understanding.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from data.phase12_eval_sets import memory_test_cases
from models.contextual_response import load_contextual_decoder
from models.conversation import ConversationTurn
from models.conversation_v3 import VectorConversationSession
from models.diverse_response import load_diverse_decoder
from models.episodic_conversation import _EPISODE_TYPE
from models.memory_model import MemoryModel
from models.vector_memory import VectorConversationMemory, load_vector_reader
from training.train_confidence import load_generator


def _load(args):
    model = load_generator(args.generator, "cpu", args.heads).eval()
    contextual = load_contextual_decoder(args.response, generator_path=args.generator)
    codec, reader = load_vector_reader(args.vector_reader, generator_path=args.generator)
    diverse = load_diverse_decoder(args.diverse_response, generator_path=args.generator)
    return model, contextual, codec, reader, diverse


def _session(components, user, *, path=None, memory=None, temperature=0.15):
    model, contextual, codec, reader, diverse = components
    if memory is None:
        bank = MemoryModel(d_model=model.d_model, capacity=192, n_heads=model.n_heads).eval()
        memory = VectorConversationMemory(bank, codec, path)
    return VectorConversationSession(model, contextual, memory, reader, diverse, user,
                                     max_history_turns=0, seed=12012, temperature=temperature), memory


def _result(result):
    return {"text": result.response.decode("utf-8", "replace"), "hex": result.response.hex(),
            "intent": result.intent, "action": result.action, "source": result.response_source,
            "routing": result.routing_reason, "vector_only": result.vector_only,
            "vector_count": result.retrieved_vector_count, "retrieved_episode_count": result.retrieved_episode_count,
            "episode_recorded": result.episode_recorded, "episode_persisted": result.episode_persisted}


def _forbidden(*_args, **_kwargs):
    raise AssertionError("a transcript, explicit fact or grounded fallback was consulted during vector-only recall")


def _query(session, question):
    session.history.clear()
    # No target/statement is passed into respond(). Assert both generator calls
    # use precisely the current question. Guard legacy paths even if future
    # refactors accidentally reintroduce them.
    with patch.object(session.memory, "recent", side_effect=_forbidden), \
         patch.object(session.memory, "recall", side_effect=_forbidden), \
         patch.object(session, "_episode_fact", side_effect=_forbidden), \
         patch.object(session.decoder, "generate", side_effect=_forbidden), \
         patch.object(session.model, "conversation_features", wraps=session.model.conversation_features) as features:
        result = session.respond(question)
    expected = b"U:" + question
    byte_inputs = [bytes(call.args[0][0].tolist()) for call in features.call_args_list]
    assert byte_inputs and all(payload == expected for payload in byte_inputs), "episode/other bytes entered context"
    actual_vectors = [call.kwargs.get("memory_context") for call in features.call_args_list if "memory_context" in call.kwargs]
    return result, {"question_only_in_all_feature_calls": True, "feature_calls": len(byte_inputs),
                    "vector_feature_calls": len(actual_vectors), "fallbacks_forbidden": True}


def _short(components):
    session, store = _session(components, b"alice")
    first = session.respond(b"My favorite color is blue")
    alternate, _ = _session(components, b"other-user", memory=store)
    other_first = alternate.respond(b"My favorite color is green")
    question = b"What is my favorite color?"
    other_before, _ = _session(components, b"unknown-user", memory=store)
    isolation, isolation_trace = _query(other_before, question)
    original = store.retrieve_fact(b"alice", question)
    swapped = store.retrieve_fact(b"other-user", question)
    assert original is not None and swapped is not None and len(original.slots) == len(swapped.slots) == 1
    answer, trace = _query(session, question)
    bank = store.memory.episodic
    saved = bank.values[list(original.slots)].clone()
    try:
        with torch.no_grad():
            bank.values[list(original.slots)] = 0
        zero, zero_trace = _query(session, question)
        with torch.no_grad():
            bank.values[list(original.slots)] = swapped.vectors.to(bank.values)
        changed, changed_trace = _query(session, question)
    finally:
        with torch.no_grad():
            bank.values[list(original.slots)] = saved
    restored, restored_trace = _query(session, question)
    return {"statement": "My favorite color is blue", "question": question.decode(),
            "first_recorded": first.episode_recorded, "other_statement_recorded": other_first.episode_recorded,
            "normal": _result(answer), "normal_trace": trace, "correct": answer.response == b"blue" and answer.vector_only,
            "zeroed": _result(zero), "zero_trace": zero_trace,
            "zero_changes_answer": zero.response != answer.response and zero.response != b"blue",
            "swapped_other_user_vector": _result(changed), "swapped_trace": changed_trace,
            "swap_changes_answer": changed.response != answer.response and changed.response != b"blue",
            "restored": _result(restored), "restored_trace": restored_trace,
            "restoration_recovers_answer": restored.response == b"blue",
            "unrelated_user": _result(isolation), "isolation_trace": isolation_trace,
            "isolation_passed": isolation.action == "ask" and isolation.response != b"blue"}


def _long(components):
    session, store = _session(components, b"long-user")
    protected, _ = _session(components, b"protected-user", memory=store)
    protected.respond(b"My hobby is pottery")
    other_before = store.retrieve_fact(b"protected-user", b"What is my hobby?")
    first = session.respond(b"My favorite color is blue")
    recorded = 0
    for i in range(500):
        # These are 500 actual distinct respond() calls, not a synthetic counter.
        result = session.respond(f"Filler message number {i:03d} about ordinary daily weather.".encode())
        recorded += int(result.episode_recorded)
    scope = store.scope(b"long-user")
    transcripts = [md for used, md in zip(store.memory.episodic.used.tolist(), store.memory.episodic.metadata)
                   if used and md.get("person") == scope and md.get("type") == _EPISODE_TYPE]
    target_in_transcript = any(b"blue" in md.get("user_bytes", b"").lower() or
                               b"blue" in md.get("assistant_bytes", b"").lower() for md in transcripts)
    answer, trace = _query(session, b"What is my favorite color?")
    other_after = store.retrieve_fact(b"protected-user", b"What is my hobby?")
    other_intact = (other_before is not None and other_after is not None and
                    torch.equal(other_before.vectors, other_after.vectors) and other_before.slots == other_after.slots)
    return {"filler_messages_sent": 500, "filler_episodes_recorded": recorded,
            "bank_episodic_capacity": store.memory.epi_cap,
            "bank_episodic_used": int(store.memory.episodic.used.sum()),
            "first_recorded": first.episode_recorded, "answer": _result(answer), "trace": trace,
            "target_bytes_in_owner_transcripts_before_query": target_in_transcript,
            "other_user_vector_intact": other_intact,
            "correct": first.episode_recorded and recorded == 500 and not target_in_transcript and
                       answer.response == b"blue" and answer.vector_only and other_intact}


def _fresh_cases(components):
    rows = []
    for i, case in enumerate(memory_test_cases()):
        session, _store = _session(components, f"case-{i}".encode())
        first = session.respond(case.statement)
        result, trace = _query(session, case.question)
        # Value is consulted only after the prediction is complete.
        rows.append({"statement": case.statement.decode(), "question": case.question.decode(),
                     "expected_for_scoring_only": case.value.decode(), "first_recorded": first.episode_recorded,
                     "result": _result(result), "trace": trace,
                     "correct": bool(first.episode_recorded and result.vector_only and result.response == case.value)})
    return {"count": len(rows), "correct": sum(row["correct"] for row in rows), "cases": rows}


def _cross_session(args, components):
    with tempfile.TemporaryDirectory(prefix="phase12-memory-") as directory:
        path = str(Path(directory) / "memory.pt")
        initial, _store = _session(components, b"same-user", path=path)
        first = initial.respond(b"Alice works at ByteCorp")
        if not first.episode_persisted:
            raise RuntimeError("cross-session setup did not persist the episode")
        # New interpreter, model and session; the child is passed no expected
        # answer or statement. Only its question, owner and memory path.
        cmd = [sys.executable, "-m", "evaluation.memory_eval", "--child", "--memory", path,
               "--user", "same-user", "--question", "Where does Alice work?",
               "--generator", args.generator, "--response", args.response, "--vector-reader", args.vector_reader,
               "--diverse-response", args.diverse_response, "--heads", str(args.heads)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        process = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env, timeout=90)
        child = json.loads(process.stdout)
    return {"statement": "Alice works at ByteCorp", "question": "Where does Alice work?",
            "fresh_interpreter": True, "persisted_setup": first.episode_persisted, "child": child,
            "correct": child["result"]["text"] == "ByteCorp" and child["result"]["vector_only"]}


def evaluate(args):
    torch.set_num_threads(1)
    components = _load(args)
    return {"scope": "structured owner-scoped vectors only; no transcript, gold answer or host fact fallback during recall",
            "short_context": _short(components), "fresh_final_cases": _fresh_cases(components),
            "long_context": _long(components), "cross_session": _cross_session(args, components)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="runs/phase12/generator.pt")
    parser.add_argument("--response", default="runs/phase12/response.pt")
    parser.add_argument("--vector-reader", default="runs/phase12/vector_reader.pt")
    parser.add_argument("--diverse-response", default="runs/phase12/diverse_response.pt")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--output", default="runs/phase12/memory_eval.json")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--memory")
    parser.add_argument("--user", default="same-user")
    parser.add_argument("--question", default="")
    args = parser.parse_args()
    if args.child:
        if not args.memory or not args.question:
            parser.error("child mode requires --memory and --question")
        torch.set_num_threads(1)
        session, _ = _session(_load(args), args.user.encode(), path=args.memory)
        result, trace = _query(session, args.question.encode())
        print(json.dumps({"result": _result(result), "trace": trace}))
        return
    report = evaluate(args)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"short": report["short_context"]["correct"], "fresh_cases": report["fresh_final_cases"]["correct"],
                      "fresh_count": report["fresh_final_cases"]["count"], "long": report["long_context"]["correct"],
                      "cross_session": report["cross_session"]["correct"], "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
