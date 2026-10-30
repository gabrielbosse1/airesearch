"""Phase 12 vector-only, replay, diversity, privacy and compatibility tests."""
from __future__ import annotations

import hashlib
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from data.conversation_examples import conversation_examples
from data.conversation_response_v2 import response_examples
from data.phase11_eval_sets import overall_intent_examples
from data.phase12_eval_sets import intent_test_examples, memory_test_cases
from data.vague_intent_pairs import hard_negative_examples
from models.contextual_response import ContextualByteDecoder
from models.conversation import ConversationTurn
from models.conversation_intents import ConversationIntent as I
from models.conversation_v3 import VectorConversationSession
from models.diverse_response import DiverseByteDecoder, distribution_diversity_loss, load_diverse_decoder
from models.generator_conditioned import GeneratorConditioned
from models.holdout import HoldoutGuard, HoldoutViolation
from models.memory_model import MemoryModel, TIER_EPISODIC
from models.procedural import byte_pattern_embedding
from models.vector_memory import (VectorConversationMemory, VectorFactCodec, VectorMemoryReader,
                                  load_vector_reader, question_topic, statement_fact)
from training.co_train import forced_memory_training_step, prepare_forced_memory_examples
from training.replay import ReplayBuffer, ReplayExample, mixed_examples


def bank(capacities=(16, 2, 4)):
    return MemoryModel(d_model=64, n_heads=4, tier_capacities=capacities, max_pattern_bytes=256).eval()


def store(capacities=(16, 2, 4), path=None):
    return VectorConversationMemory(bank(capacities), VectorFactCodec(), path)


def model():
    value = GeneratorConditioned(d_model=64, n_layers=1, n_heads=4, d_ff=64, max_len=128, dropout=0).eval()
    value.intent_trained.fill_(True)
    return value


def force(model_, intent):
    with torch.no_grad():
        for parameter in model_.intent_classifier.parameters():
            parameter.zero_()
        model_.intent_classifier[2].bias.fill_(-10)
        model_.intent_classifier[2].bias[int(intent)] = 10


def session(memory=None):
    model_ = model(); contextual = ContextualByteDecoder(64, hidden_dim=16, max_input_bytes=128).eval()
    contextual.response_trained.fill_(True)
    reader = VectorMemoryReader(64).eval(); reader.reader_trained.fill_(True)
    diverse = DiverseByteDecoder(64, hidden_dim=16, byte_dim=8, style_dim=8).eval(); diverse.response_trained.fill_(True)
    return VectorConversationSession(model_, contextual, memory or store(), reader, diverse, b"alice", seed=8)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_capacity_fifo_and_reproducible_guarded_roundtrip(self):
        guard = HoldoutGuard.from_records([{"input": b"protected test input"}])
        replay = ReplayBuffer(3, seed=42, guard=guard)
        replay.add([ReplayExample(bytes([i + 65]), 0, "old") for i in range(5)])
        self.assertEqual([x.message for x in replay.buffer], [b"C", b"D", b"E"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.pt"; replay.save(path)
            expected = replay.sample(3)
            loaded = ReplayBuffer.load(path, guard=guard)
            self.assertEqual(loaded.sample(3), expected)
            with self.assertRaises(HoldoutViolation):
                ReplayBuffer.load(path, guard=HoldoutGuard.from_records([{"input": b"different"}]))

    def test_whole_batch_preflight_prevents_partial_insertion(self):
        guard = HoldoutGuard.from_records([{"input": b"test example", "output": b"target"}])
        replay = ReplayBuffer(4, guard=guard); replay.add([ReplayExample(b"safe", 0, "old")])
        for rejected in (ReplayExample(b"test example", 0, "new"),
                         ReplayExample(b"other", 0, "new", "vector", b"target", (b"question",)),
                         ReplayExample(b"other", 0, "new", "vector", b"okay", (b"test example",))):
            with self.assertRaises(HoldoutViolation):
                replay.add([ReplayExample(b"ok", 0, "new"), rejected])
            self.assertEqual([x.message for x in replay.buffer], [b"safe"])
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(HoldoutViolation):
            ReplayBuffer().save(Path(tmp) / "no-guard.pt")

    def test_mixed_batch_is_70_30_and_task_scoped(self):
        guard = HoldoutGuard.from_records([]); replay = ReplayBuffer(100, seed=8, guard=guard)
        replay.add([ReplayExample(f"old-{i}".encode(), 0, "old") for i in range(50)])
        replay.add([ReplayExample(b"fact", 0, "old", "vector", b"value", (b"question",))])
        new = [ReplayExample(f"new-{i}".encode(), 0, "new") for i in range(10)]
        mixed = mixed_examples(new, replay, batch_size=100, rng=random.Random(2))
        self.assertEqual(sum(r.phase == "new" for r in mixed), 70)
        self.assertEqual(sum(r.phase == "old" for r in mixed), 30)
        self.assertEqual({r.task for r in mixed}, {"intent"})
        with self.assertRaises(ValueError):
            mixed_examples([ReplayExample(b"fresh", 0, "new", "vector", b"value")], replay,
                           batch_size=100, rng=random.Random(2))


class VectorMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_codec_covers_all_raw_byte_values_and_padding(self):
        codec = VectorFactCodec()
        for start in range(0, 256, 40):
            source = bytes(range(start, min(start + 40, 256)))
            vectors = codec.encode(source)
            decoded = (vectors @ codec.rotation.T).reshape(-1, 8)
            weights = 2 ** torch.arange(7, -1, -1)
            result = ((decoded[:len(source)] > 0).long() * weights).sum(dim=-1)
            self.assertEqual(bytes(result.tolist()), source)
            if len(source) % 8:
                self.assertTrue(torch.allclose(decoded[len(source)], torch.zeros(8), atol=1e-5))
        with self.assertRaises(ValueError):
            codec.encode(b"x" * 41)

    def test_writer_is_scoped_vector_only_and_uses_actual_retrieve(self):
        memory = store()
        memory.record_episode(b"alice", ConversationTurn(b"My favorite color is blue.", b"Noted."))
        with patch.object(memory.memory, "retrieve", wraps=memory.memory.retrieve) as retrieve:
            fact = memory.retrieve_fact(b"alice", b"What is my favorite color?")
        self.assertIsNotNone(fact); retrieve.assert_called_once()
        self.assertEqual(tuple(fact.vectors.shape), (1, 64))
        self.assertIsNone(memory.retrieve_fact(b"bob", b"What is my favorite color?"))
        md = memory.memory.episodic.metadata[fact.slots[0]]
        self.assertTrue(md["no_promote"])
        self.assertNotIn("user_bytes", md); self.assertNotIn("assistant_bytes", md)
        self.assertNotIn(b"blue", repr(md).encode())
        self.assertTrue(torch.equal(fact.vectors[0], memory.codec.encode(b"blue")[0]))

    def test_update_and_forget_remove_only_owned_matching_vectors(self):
        memory = store()
        memory.record_episode(b"alice", ConversationTurn(b"My color is blue.", b"Okay."))
        memory.record_episode(b"alice", ConversationTurn(b"My hobby is pottery.", b"Okay."))
        memory.record_episode(b"bob", ConversationTurn(b"My color is red.", b"Okay."))
        memory.record_episode(b"alice", ConversationTurn(b"My color is indigo.", b"Okay."))
        a = memory.retrieve_fact(b"alice", b"What is my color?")
        b = memory.retrieve_fact(b"bob", b"What is my color?")
        hobby = memory.retrieve_fact(b"alice", b"What is my hobby?")
        self.assertTrue(torch.equal(a.vectors, memory.codec.encode(b"indigo")))
        memory.forget_and_purge_history(b"alice", b"color")
        self.assertIsNone(memory.retrieve_fact(b"alice", b"What is my color?"))
        self.assertTrue(torch.equal(memory.retrieve_fact(b"bob", b"What is my color?").vectors, b.vectors))
        self.assertTrue(torch.equal(memory.retrieve_fact(b"alice", b"What is my hobby?").vectors, hobby.vectors))
        self.assertEqual(memory.recent(b"alice", b"q").turns, ())

    def test_capacity_recycles_only_same_owner_episodes_and_protects_facts(self):
        memory = store((6, 1, 1))
        memory.record_episode(b"alice", ConversationTurn(b"My color is blue.", b"ok"))
        memory.record_episode(b"bob", ConversationTurn(b"My color is red.", b"ok"))
        bob = memory.retrieve_fact(b"bob", b"What is my color?")
        for i in range(20):
            memory.record_episode(b"alice", ConversationTurn(f"filler {i}".encode(), b"ok"))
        self.assertTrue(torch.equal(bob.vectors, memory.retrieve_fact(b"bob", b"What is my color?").vectors))
        self.assertIsNotNone(memory.retrieve_fact(b"alice", b"What is my color?"))
        before = memory.memory.episodic.used.clone()
        with self.assertRaises(ValueError):
            memory.record_episode(b"outsider", ConversationTurn(b"My hobby is chess.", b"ok"))
        self.assertTrue(torch.equal(before, memory.memory.episodic.used))

    def test_transaction_rolls_back_episode_and_vector_on_save_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = store(path=Path(tmp) / "memory.pt")
            memory.record_episode(b"alice", ConversationTurn(b"My color is blue.", b"ok"))
            original = memory.retrieve_fact(b"alice", b"What is my color?")
            before_file = memory.path.read_bytes()
            with patch.object(memory, "_atomic_save", side_effect=OSError("disk")), self.assertRaises(OSError):
                memory.record_episode(b"alice", ConversationTurn(b"My color is green.", b"ok"))
            self.assertEqual(memory.path.read_bytes(), before_file)
            self.assertTrue(torch.equal(original.vectors, memory.retrieve_fact(b"alice", b"What is my color?").vectors))

    def test_guard_blocks_source_target_and_missing_vector_provenance(self):
        memory = store()
        memory.memory.set_holdout_guard(HoldoutGuard.from_records([{"input": b"My color is blue.", "output": b"blue"}]))
        before = memory.memory.episodic.used.clone()
        with self.assertRaises(HoldoutViolation):
            memory.record_episode(b"alice", ConversationTurn(b"My color is blue.", b"ok"))
        self.assertTrue(torch.equal(before, memory.memory.episodic.used))
        content = torch.zeros(64)
        with self.assertRaises(HoldoutViolation):
            memory.memory.write(content, {"type": "x"}, tier=TIER_EPISODIC,
                                raw_input=b"different source", raw_output=b"blue", value_override=content)
        self.assertTrue(torch.equal(before, memory.memory.episodic.used))
        with self.assertRaises(ValueError):
            memory.memory.write(content, {"type": "x"}, tier=TIER_EPISODIC, value_override=content)
        with self.assertRaises(ValueError):
            memory.memory.write(content, {"type": "x"}, tier=TIER_EPISODIC, raw_input=b"safe", value_override=content)

    def test_write_override_rejects_nonfinite_shapes_and_nonepisodic(self):
        memory = bank()
        for val in (torch.zeros(63), torch.full((64,), float("nan"))):
            with self.assertRaises(ValueError):
                memory.write(torch.zeros(64), {"type": "x"}, tier=TIER_EPISODIC, raw_input=b"source", raw_output=b"value", value_override=val)
        with self.assertRaises(ValueError):
            memory.write(torch.zeros(64), {"type": "x"}, tier="semantic", raw_input=b"source", raw_output=b"value", value_override=torch.zeros(64))
        self.assertEqual(int(memory.episodic.used.sum()), 0)

    def test_workplace_grammar_and_bounded_values(self):
        self.assertEqual(statement_fact(b"Alice works at ByteCorp."), (b"person:alice:workplace", b"ByteCorp"))
        self.assertEqual(question_topic(b"Where does Alice work?"), b"person:alice:workplace")
        self.assertIsNone(statement_fact(b"Do not remember my color is blue."))
        self.assertIsNone(question_topic(b"Please do not tell me my color?"))
        memory = store(); memory.record_episode(b"a", ConversationTurn(b"My color is " + b"x" * 41 + b".", b"ok"))
        self.assertIsNone(memory.retrieve_fact(b"a", b"What is my color?"))

    def test_reader_is_differentiable_through_vector_input(self):
        reader = VectorMemoryReader()
        vectors = torch.randn(2, 1, 64, requires_grad=True)
        context = torch.randn(2, 64, requires_grad=True)
        logits = reader(context, vectors)
        loss = -logits[0, 0, ord("a")] - logits[1, 0, ord("b")]
        loss.backward()
        self.assertGreater(float(vectors.grad.abs().sum()), 0)
        self.assertGreater(float(reader.projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(context.grad.abs().sum()), 0)

    def test_preparation_retrieves_vectors_and_training_receives_question_only(self):
        memory = store((16, 2, 4)); generator = model(); reader = VectorMemoryReader()
        examples = [{"statement": b"My color is blue.", "question": b"What is my color?", "target": b"blue"}]
        with patch.object(memory.memory, "retrieve", wraps=memory.memory.retrieve) as retrieve:
            prepared = prepare_forced_memory_examples(memory, examples)
        retrieve.assert_called_once()
        self.assertEqual(set(prepared[0]), {"question", "vectors", "target"})
        optimizer = torch.optim.Adam(reader.parameters(), lr=0.001)
        with patch.object(generator, "conversation_features", wraps=generator.conversation_features) as features:
            loss, stats = forced_memory_training_step(generator, memory, reader, prepared, optimizer)
        self.assertTrue(torch.isfinite(loss)); self.assertEqual(stats["episode_bytes_in_generator_input"], 0)
        self.assertEqual(bytes(features.call_args.args[0][0].tolist()), b"U:What is my color?")
        self.assertTrue(torch.equal(features.call_args.kwargs["memory_context"][0, 0], prepared[0]["vectors"][0]))


class VectorSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_vector_route_forbids_transcript_and_host_fallback(self):
        s = session(); force(s.model, I.CHAT)
        with patch.object(s.diverse_decoder, "generate", return_value=(b"Noted.", True, 0)):
            first = s.respond(b"My favorite color is blue.")
        self.assertTrue(first.episode_recorded); s.history.clear()
        # Constant output is a unit-test stub only; benchmark uses real weights.
        with patch.object(s.memory, "recent", side_effect=AssertionError("transcript")), \
             patch.object(s.memory, "recall", side_effect=AssertionError("fallback")), \
             patch.object(s, "_episode_fact", side_effect=AssertionError("fallback")), \
             patch.object(s.decoder, "generate", side_effect=AssertionError("fallback")), \
             patch.object(s.reader, "generate", return_value=(b"WRONG", True)), \
             patch.object(s.model, "conversation_features", wraps=s.model.conversation_features) as features:
            result = s.respond(b"What is my favorite color?")
        self.assertEqual(result.response, b"WRONG")  # no hidden evidence correction
        self.assertEqual(result.response_source, "vector_reader"); self.assertTrue(result.vector_only)
        self.assertEqual(result.retrieved_episode_count, 0); self.assertEqual(result.retrieved_vector_count, 1)
        self.assertTrue(all(bytes(call.args[0][0].tolist()) == b"U:What is my favorite color?" for call in features.call_args_list))

    def test_missing_vector_and_invalid_reader_fail_closed(self):
        s = session(); force(s.model, I.CHAT)
        self.assertEqual(s.respond(b"What is my color?").response_source, "vector_missing")
        with patch.object(s.diverse_decoder, "generate", return_value=(b"okay", True, 0)):
            s.respond(b"My color is blue.")
        with patch.object(s.reader, "generate", return_value=(b"", False)):
            result = s.respond(b"What is my color?")
        self.assertEqual(result.response_source, "vector_reader_failed"); self.assertEqual(result.action, "ask")

    def test_forget_purges_vectors_and_privacy_opt_out_disables_storage(self):
        s = session(); force(s.model, I.CHAT)
        with patch.object(s.diverse_decoder, "generate", return_value=(b"okay", True, 0)):
            s.respond(b"My color is blue.")
        self.assertIsNotNone(s.memory.retrieve_fact(b"alice", b"What is my color?"))
        self.assertEqual(s.respond(b"Forget my color.").action, "forget")
        self.assertIsNone(s.memory.retrieve_fact(b"alice", b"What is my color?"))
        result = s.respond(b"Don't store our conversation.")
        self.assertEqual(result.response_source, "privacy_guard"); self.assertFalse(s.record_episodes)
        with patch.object(s.diverse_decoder, "generate", return_value=(b"okay", True, 0)):
            private = s.respond(b"My hobby is chess.")
        self.assertFalse(private.episode_recorded)
        self.assertIsNone(s.memory.retrieve_fact(b"alice", b"What is my hobby?"))

    def test_workplace_forget_is_scoped_and_negation_does_not_delete(self):
        s = session(); force(s.model, I.CHAT)
        with patch.object(s.diverse_decoder, "generate", return_value=(b"okay", True, 0)):
            s.respond(b"Alice works at ByteCorp.")
            s.memory.record_episode(b"bob", ConversationTurn(b"Alice works at OtherCo.", b"ok"))
            s.respond(b"Do not forget Alice's workplace.")
        self.assertIsNotNone(s.memory.retrieve_fact(b"alice", b"Where does Alice work?"))
        result = s.respond(b"Forget Alice's workplace.")
        self.assertEqual(result.action, "forget")
        self.assertEqual(result.routing_reason, "exact_forget_workplace_command")
        self.assertFalse(result.episode_recorded)
        self.assertIsNone(s.memory.retrieve_fact(b"alice", b"Where does Alice work?"))
        self.assertIsNotNone(s.memory.retrieve_fact(b"bob", b"Where does Alice work?"))
        self.assertEqual(s.memory.recent(b"alice", b"query").turns, ())

    def test_runtime_conversation_never_populates_replay(self):
        s = session(); force(s.model, I.CHAT)
        with patch.object(ReplayBuffer, "add", side_effect=AssertionError("private replay")), \
             patch.object(s.diverse_decoder, "generate", return_value=(b"okay", True, 0)):
            result = s.respond(b"My private detail is secret.")
        self.assertTrue(result.episode_recorded)


class DiversityAndBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_diversity_loss_has_gradients_on_distributions(self):
        logits = torch.randn(2, 4, 6, 257, requires_grad=True)
        loss = distribution_diversity_loss(logits)
        loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertGreater(float(logits.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            distribution_diversity_loss(torch.randn(2, 6, 257))

    def test_reader_and_diverse_checkpoint_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = Path(tmp) / "generator.pt"; gen.write_bytes(b"one")
            digest = hashlib.sha256(gen.read_bytes()).hexdigest()
            codec = VectorFactCodec(); reader = VectorMemoryReader(); reader.reader_trained.fill_(True)
            reader_path = Path(tmp) / "reader.pt"
            torch.save({"schema_version": 1, "generator_sha256": digest, "context_dim": 64,
                        "codec_state_dict": codec.state_dict(), "codec_fingerprint": codec.fingerprint,
                        "reader_state_dict": reader.state_dict()}, reader_path)
            load_vector_reader(reader_path, generator_path=gen)
            decoder = DiverseByteDecoder(); decoder.response_trained.fill_(True)
            response_path = Path(tmp) / "response.pt"
            torch.save({"schema_version": 1, "generator_sha256": digest, "context_dim": 64, "num_styles": 32,
                        "hidden_dim": 96, "byte_dim": 48, "style_dim": 32, "state_dict": decoder.state_dict()}, response_path)
            load_diverse_decoder(response_path, generator_path=gen)
            gen.write_bytes(b"different")
            with self.assertRaises(ValueError): load_vector_reader(reader_path, generator_path=gen)
            with self.assertRaises(ValueError): load_diverse_decoder(response_path, generator_path=gen)

    def test_fresh_intent_and_memory_statements_are_disjoint_from_prior_fixtures(self):
        norm = lambda raw: b" ".join(raw.lower().rstrip(b".!? ").split())
        prior = set()
        for split in ("train", "validation", "test"):
            prior |= {norm(row.message) for row in conversation_examples(split)}
            prior |= {norm(msg) for msg, _ in hard_negative_examples(split)}
        for split in ("validation", "test"):
            prior |= {norm(msg) for msg, _ in overall_intent_examples(split)}
        prior |= {norm(row.message) for row in response_examples("train") if not row.history}
        self.assertFalse({norm(msg) for msg, _ in intent_test_examples()} & prior)
        old_statements = {norm(turn.user) for row in response_examples("train") for turn in row.history}
        self.assertFalse({norm(case.statement) for case in memory_test_cases()} & old_statements)
        self.assertEqual(len(intent_test_examples()), 70); self.assertEqual(len(memory_test_cases()), 10)


if __name__ == "__main__":
    unittest.main()
