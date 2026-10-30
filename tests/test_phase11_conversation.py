"""Scoped episodes, conditioned decoding, privacy and Phase 11 safety tests."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from data.conversation_examples import conversation_examples
from data.conversation_response_v2 import response_examples, retention_cases
from data.phase11_eval_sets import overall_intent_examples
from data.vague_intent_pairs import hard_negative_examples
from models.contextual_response import ContextualByteDecoder, load_contextual_decoder, pack_response_context
from models.conversation import ConversationTurn
from models.conversation_intents import ConversationIntent as I
from models.conversation_v2 import ContextualConversationSession
from models.episodic_conversation import EpisodicConversationMemory
from models.generator_conditioned import GeneratorConditioned
from models.holdout import HoldoutGuard, HoldoutViolation
from models.memory_model import MemoryModel


def model():
    value = GeneratorConditioned(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=128, dropout=0.0).eval()
    value.intent_trained.fill_(True)
    return value


def force(model_, intent):
    with torch.no_grad():
        for parameter in model_.intent_classifier.parameters():
            parameter.zero_()
        model_.intent_classifier[2].bias.fill_(-10)
        model_.intent_classifier[2].bias[int(intent)] = 10


def decoder():
    value = ContextualByteDecoder(16, hidden_dim=16, max_input_bytes=128).eval()
    value.response_trained.fill_(True)
    return value


def bank(*, capacities=(6, 2, 4)):
    return MemoryModel(d_model=16, n_heads=4, tier_capacities=capacities, max_pattern_bytes=256)


def memory(*, capacities=(6, 2, 4), path=None):
    return EpisodicConversationMemory(bank(capacities=capacities), path)


class Phase11EpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_retrieve_calls_memory_retrieve_and_keeps_latest_five(self):
        store = memory(capacities=(8, 2, 4))
        for i in range(7):
            store.record_episode(b"alice", ConversationTurn(f"turn {i}".encode(), b"ok"))
        store.record_episode(b"bob", ConversationTurn(b"private", b"secret"))
        with patch.object(store.memory, "retrieve", wraps=store.memory.retrieve) as retrieve:
            got = store.recent(b"alice", b"query", limit=5)
        retrieve.assert_called_once()
        self.assertEqual([t.user for t in got.turns], [b"turn 2", b"turn 3", b"turn 4", b"turn 5", b"turn 6"])
        self.assertEqual(tuple(got.vectors.shape), (5, 16))
        self.assertEqual(store.recent(b"bob", b"query").turns[0].user, b"private")
        self.assertEqual(store.recent(b"nobody", b"query").vectors.shape, (0, 16))

    def test_read_only_retrieval_has_no_access_side_effect(self):
        store = memory()
        store.record_episode(b"u", ConversationTurn(b"hello", b"hi"))
        counters = store.memory.episodic.access_counts.clone()
        tick = store.memory.session_tick
        store.recent(b"u", b"hi")
        self.assertTrue(torch.equal(counters, store.memory.episodic.access_counts))
        self.assertEqual(tick, store.memory.session_tick)
        with store.memory.read_only(), self.assertRaises(HoldoutViolation):
            store.record_episode(b"u", ConversationTurn(b"new", b"ok"))

    def test_full_bank_does_not_evict_other_user(self):
        store = memory(capacities=(2, 1, 1))
        store.record_episode(b"alice", ConversationTurn(b"a1", b"ok"))
        store.record_episode(b"bob", ConversationTurn(b"b1", b"ok"))
        with self.assertRaises(ValueError):
            store.record_episode(b"charlie", ConversationTurn(b"c1", b"ok"))
        store.record_episode(b"alice", ConversationTurn(b"a2", b"ok"))
        self.assertEqual(store.recent(b"alice", b"q").turns[0].user, b"a2")
        self.assertEqual(store.recent(b"bob", b"q").turns[0].user, b"b1")

    def test_persistence_reload_and_private_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.pt"
            store = memory(path=path)
            store.record_episode(b"alice", ConversationTurn(b"private", b"hi"))
            loaded = memory(path=path)
            self.assertEqual(loaded.recent(b"alice", b"q").turns, (ConversationTurn(b"private", b"hi"),))
            self.assertEqual(loaded.recent(b"bob", b"q").turns, ())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_failed_episode_save_and_purge_roll_back_every_affected_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.pt"
            store = memory(path=path)
            store.remember(b"alice", b"name", b"Ada", source=b"Remember my name is Ada.")
            store.record_episode(b"alice", ConversationTurn(b"My name is Ada.", b"ok"))
            original = path.read_bytes(); tick = store.memory.session_tick
            with patch.object(store, "_atomic_save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    store.record_episode(b"alice", ConversationTurn(b"new turn", b"reply"))
                with self.assertRaises(OSError):
                    store.forget_and_purge_history(b"alice", b"name")
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(store.memory.session_tick, tick)
            self.assertEqual(store.recall(b"alice", b"name"), b"Ada")
            self.assertEqual(len(store.recent(b"alice", b"q").turns), 1)

    def test_forget_purges_user_only_and_not_semantic_backups(self):
        store = memory()
        for user, value in ((b"alice", b"Ada"), (b"bob", b"Mia")):
            store.remember(user, b"name", value, source=b"explicit request")
            store.record_episode(user, ConversationTurn(b"My name is " + value + b".", b"noted"))
        facts, episodes = store.forget_and_purge_history(b"alice", b"name")
        self.assertEqual((facts, episodes), (1, 1))
        self.assertIsNone(store.recall(b"alice", b"name"))
        self.assertEqual(store.recent(b"alice", b"q").turns, ())
        self.assertEqual(store.recall(b"bob", b"name"), b"Mia")
        self.assertEqual(len(store.recent(b"bob", b"q").turns), 1)
        free = ~store.memory.episodic.used
        self.assertEqual(int(store.memory.episodic.values[free].abs().sum()), 0)

    def test_episode_is_not_consolidated_into_semantic_bank(self):
        store = memory()
        store.memory.promotion_threshold = 0.0
        store.record_episode(b"u", ConversationTurn(b"private fact", b"ok"))
        store.memory.episodic.access_counts[0] = 10
        result = store.memory.consolidate()
        self.assertEqual(result["promoted_to_semantic"], 0)
        self.assertEqual(int(store.memory.semantic.used.sum()), 0)

    def test_guard_checks_raw_episode_contents_before_any_write(self):
        protected = b"A protected conversational record"
        store = memory()
        store.memory.set_holdout_guard(HoldoutGuard.from_records([{"input": protected}]))
        with self.assertRaises(HoldoutViolation):
            store.record_episode(b"u", ConversationTurn(protected, b"hi"))
        with self.assertRaises(HoldoutViolation):
            store.record_episode(b"u", ConversationTurn(b"hi", protected))
        self.assertEqual(int(store.memory.episodic.used.sum()), 0)


class Phase11DecoderTests(unittest.TestCase):
    def test_input_packing_preserves_current_and_newest_whole_history(self):
        turns = [ConversationTurn(b"older", b"reply"), ConversationTurn(b"newer", b"reply")]
        payload, segments = pack_response_context(b"hi\x00", turns, max_bytes=24)
        self.assertTrue(payload.startswith(b"Q:hi\x00"))
        self.assertIn(b"newer", payload); self.assertNotIn(b"older", payload)
        self.assertEqual(len(payload), len(segments))
        self.assertEqual(segments[:5], bytes([0]) * 5)
        self.assertTrue(all(x == 1 for x in segments[5:]))
        with self.assertRaises(ValueError):
            pack_response_context(b"too long", (), max_bytes=4)

    def test_decoder_uses_byte_targets_copy_distribution_and_gradients(self):
        value = decoder()
        context = torch.randn(2, 16)
        intent = torch.tensor([int(I.CHAT), int(I.RECALL)])
        source = torch.tensor([[0, 255, 5], [2, 255, 0]])
        segments = torch.tensor([[0, 1, 1], [0, 1, 0]])
        lengths = torch.tensor([3, 2])
        targets = torch.tensor([[0, 255, 256], [255, 256, -100]])
        output = value(context, intent, source, segments, lengths, targets)
        self.assertEqual(output.shape, (2, 3, 257))
        self.assertTrue(torch.allclose(output.exp().sum(-1), torch.ones(2, 3), atol=1e-5))
        loss = torch.nn.functional.nll_loss(output.reshape(-1, 257), targets.flatten(), ignore_index=-100)
        loss.backward()
        self.assertGreater(float(value.source_encoder.weight_ih_l0.grad.abs().sum()), 0)
        self.assertGreater(float(value.copy_gate.weight.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            value(context, intent, source + 300, segments, lengths, targets)
        with self.assertRaises(ValueError):
            value(context, intent, source, segments, lengths, torch.tensor([[257], [1]]))

    def test_pointer_can_assign_probability_to_any_source_byte(self):
        value = decoder()
        with torch.no_grad():
            value.copy_gate.weight.zero_(); value.copy_gate.bias.fill_(30)
            value.query_projection.weight.zero_(); value.key_projection.weight.zero_()
        source = torch.tensor([[255, 255]])
        output = value(torch.zeros(1, 16), torch.tensor([int(I.RECALL)]), source, torch.ones_like(source),
                       torch.tensor([2]), torch.tensor([[255, 256]]))
        self.assertGreater(float(output.exp()[0, 0, 255]), 0.999)

    def test_decoder_checkpoint_requires_matching_generator_and_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = Path(tmp) / "gen.pt"; generator.write_bytes(b"generator")
            path = Path(tmp) / "response.pt"; value = decoder()
            torch.save({"schema_version": 2, "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
                        "generator_n_heads": 4, "context_dim": 16, "hidden_dim": 16,
                        "max_input_bytes": 128, "state_dict": value.state_dict()}, path)
            loaded = load_contextual_decoder(path, generator_path=generator)
            self.assertEqual(loaded.expected_generator_heads, 4)
            generator.write_bytes(b"different")
            with self.assertRaises(ValueError):
                load_contextual_decoder(path, generator_path=generator)

    def test_memory_mask_and_vectors_change_generator_features(self):
        value = model()
        ids = torch.tensor([[85, 58, 104, 105]])
        lengths = torch.tensor([4]); focus = torch.tensor([2])
        base = value.conversation_features(ids, lengths, focus_start=focus)
        vectors = torch.randn(1, 3, 16)
        masked = value.conversation_features(ids, lengths, focus_start=focus, memory_context=vectors,
                                             memory_mask=torch.zeros(1, 3, dtype=torch.bool))
        injected = value.conversation_features(ids, lengths, focus_start=focus, memory_context=vectors)
        self.assertTrue(torch.allclose(base, masked, atol=1e-6))
        self.assertGreater(float((base - injected).abs().max()), 1e-7)


class Phase11SessionTests(unittest.TestCase):
    def setUp(self):
        self.model, self.decoder, self.memory = model(), decoder(), memory()
        self.session = ContextualConversationSession(self.model, self.decoder, self.memory, b"alice")

    def test_implicit_fact_recall_and_counterfactual_uses_actual_episode(self):
        force(self.model, I.CHAT)
        with patch.object(self.decoder, "generate", return_value=(b"Thanks for sharing.", True)):
            first = self.session.respond(b"My favorite color is blue.")
        self.assertTrue(first.episode_recorded); self.assertFalse(first.episode_persisted)
        force(self.model, I.RECALL)
        # A wrong decoder value must not be trusted; fallback is sourced solely
        # from the retrieved actual message, never a supplied test target.
        with patch.object(self.decoder, "generate", return_value=(b"Your favorite color is red.", True)):
            second = self.session.respond(b"What is my favorite color?")
        self.assertEqual(second.response, b"You said blue.")
        self.assertEqual(second.response_source, "grounded_fallback")
        self.assertEqual(second.retrieved_episode_count, 1)
        other = ContextualConversationSession(self.model, self.decoder, self.memory, b"bob")
        self.assertEqual(other.respond(b"What is my favorite color?").action, "ask")
        force(self.model, I.CHAT)
        with patch.object(self.decoder, "generate", return_value=(b"Noted.", True)):
            other.respond(b"My favorite color is amber.")
        force(self.model, I.RECALL)
        with patch.object(self.decoder, "generate", return_value=(b"Wrong answer.", True)):
            self.assertIn(b"amber", other.respond(b"What is my favorite color?").response)

    def test_three_to_five_actual_vectors_are_injected_and_history_is_owner_scoped(self):
        for i in range(5):
            self.memory.record_episode(b"alice", ConversationTurn(f"old {i}".encode(), b"ok"))
        self.memory.record_episode(b"bob", ConversationTurn(b"bob secret", b"private"))
        expected = self.memory.recent(b"alice", b"hello").vectors
        force(self.model, I.CHAT)
        with patch.object(self.model, "conversation_features", wraps=self.model.conversation_features) as features:
            with patch.object(self.decoder, "generate", return_value=(b"Hello there!", True)):
                result = self.session.respond(b"hello")
        calls = [c for c in features.call_args_list if c.kwargs.get("memory_context") is not None]
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(calls[0].kwargs["memory_context"], expected.unsqueeze(0)))
        self.assertEqual(result.retrieved_episode_count, 5)

    def test_no_storage_request_overrides_classifier_and_is_not_kept_in_ram(self):
        force(self.model, I.REMEMBER)
        result = self.session.respond(b"Please don't remember my name is Secret.")
        self.assertEqual(result.response_source, "privacy_guard")
        self.assertFalse(result.episode_recorded)
        self.assertEqual(self.session.history, [])
        self.assertEqual(int(self.memory.memory.episodic.used.sum()), 0)
        self.assertEqual(int(self.memory.memory.procedural.used.sum()), 0)

    def test_session_wide_no_storage_applies_to_future_turns_and_explicit_facts(self):
        force(self.model, I.CHAT)
        with patch.object(self.decoder, "generate", return_value=(b"Noted.", True)):
            self.session.respond(b"Hello before opting out.")
        existing = int(self.memory.memory.episodic.used.sum())
        result = self.session.respond(b"Do not save anything I say.")
        self.assertFalse(result.episode_recorded)
        self.assertIn(b"Existing saved records remain", result.response)
        self.assertEqual(self.session.history, [])
        with patch.object(self.decoder, "generate", return_value=(b"Okay.", True)):
            future = self.session.respond(b"My password is secret.")
        self.assertFalse(future.episode_recorded)
        self.assertEqual(int(self.memory.memory.episodic.used.sum()), existing)
        self.assertEqual(self.session.history, [])
        force(self.model, I.REMEMBER)
        self.assertEqual(self.session.respond(b"Remember my name is Ada.").response_source, "privacy_guard")
        self.assertIsNone(self.memory.recall(b"alice", b"name"))

    def test_explicit_privacy_paraphrases_are_not_recorded(self):
        force(self.model, I.CHAT)
        for phrase in (b"I don't want you to save this secret.", b"I don't want this stored.",
                       b"Please avoid logging this private turn."):
            with self.subTest(phrase=phrase):
                store = memory()
                session = ContextualConversationSession(self.model, self.decoder, store, b"u")
                result = session.respond(phrase)
                self.assertEqual(result.response_source, "privacy_guard")
                self.assertFalse(result.episode_recorded)
                self.assertEqual(session.history, [])
                self.assertEqual(int(store.memory.episodic.used.sum()), 0)
        for phrase in (b"I don't want you to save anything I say.",
                       b"I don't want this conversation to be stored.", b"Stop logging our chats."):
            with self.subTest(phrase=phrase):
                store = memory()
                session = ContextualConversationSession(self.model, self.decoder, store, b"u")
                result = session.respond(phrase)
                self.assertEqual(result.response_source, "privacy_guard")
                self.assertFalse(session.record_episodes)
                self.assertEqual(session.history, [])

    def test_session_wide_no_storage_survives_nonfinite_classifier(self):
        with torch.no_grad():
            self.model.intent_classifier[2].bias.fill_(float("nan"))
        result = self.session.respond(b"Don't store our conversation.")
        self.assertEqual(result.response_source, "privacy_guard")
        self.assertFalse(self.session.record_episodes)
        self.assertTrue(self.session._privacy_disabled)
        self.assertEqual(self.session.history, [])
        self.assertEqual(int(self.memory.memory.episodic.used.sum()), 0)

    def test_disable_episodes_still_allows_only_in_process_history(self):
        session = ContextualConversationSession(self.model, self.decoder, self.memory, b"alice", record_episodes=False)
        force(self.model, I.CHAT)
        with patch.object(self.decoder, "generate", return_value=(b"Noted.", True)):
            result = session.respond(b"My hobby is knitting.")
        self.assertFalse(result.episode_recorded)
        self.assertEqual(int(self.memory.memory.episodic.used.sum()), 0)
        force(self.model, I.RECALL)
        with patch.object(self.decoder, "generate", return_value=(b"No answer.", True)):
            self.assertIn(b"knitting", session.respond(b"What is my hobby?").response)
        fresh = ContextualConversationSession(self.model, self.decoder, self.memory, b"alice", record_episodes=False)
        self.assertEqual(fresh.respond(b"What is my hobby?").action, "ask")

    def test_forget_clears_history_and_procedural_fact_without_recreating_episode(self):
        force(self.model, I.REMEMBER)
        with patch.object(self.decoder, "generate", return_value=(b"I will remember.", True)):
            self.assertEqual(self.session.respond(b"Remember my name is Ada.").action, "remember")
        force(self.model, I.FORGET)
        result = self.session.respond(b"Forget my name.")
        self.assertEqual(result.action, "forget")
        self.assertIn(b"conversation history", result.response)
        self.assertFalse(result.episode_recorded)
        self.assertEqual(self.memory.recent(b"alice", b"q").turns, ())
        self.assertIsNone(self.memory.recall(b"alice", b"name"))
        self.assertEqual(self.session.history, [])
        force(self.model, I.RECALL)
        self.assertEqual(self.session.respond(b"What is my name?").action, "ask")

    def test_exact_memory_commands_remain_available_when_intent_is_wrong(self):
        force(self.model, I.CHAT)
        with patch.object(self.decoder, "generate", return_value=(b"I'll remember that.", True)):
            remembered = self.session.respond(b"Remember my name is Ada.")
        self.assertEqual(remembered.intent, "chat")
        self.assertEqual(remembered.action, "remember")
        self.assertEqual(remembered.routing_reason, "exact_remember_command")
        self.assertEqual(self.memory.recall(b"alice", b"name"), b"Ada")
        with patch.object(self.decoder, "generate", return_value=(b"Okay.", True)):
            negated = self.session.respond(b"Do not forget my name.")
        self.assertNotEqual(negated.action, "forget")
        self.assertEqual(self.memory.recall(b"alice", b"name"), b"Ada")
        forgotten = self.session.respond(b"Forget my name.")
        self.assertEqual(forgotten.intent, "chat")
        self.assertEqual(forgotten.action, "forget")
        self.assertEqual(forgotten.routing_reason, "exact_forget_command")
        self.assertIsNone(self.memory.recall(b"alice", b"name"))
        self.assertEqual(self.memory.recent(b"alice", b"q").turns, ())

    def test_forgetting_episode_only_does_not_claim_a_separate_fact_was_removed(self):
        self.memory.record_episode(b"alice", ConversationTurn(b"Hello", b"Hi"))
        force(self.model, I.FORGET)
        result = self.session.respond(b"Forget my hometown.")
        self.assertEqual(result.action, "forget")
        self.assertIn(b"no separate saved fact matched", result.response)
        self.assertEqual(self.memory.recent(b"alice", b"q").turns, ())

    def test_no_action_from_underspecified_or_negated_edit(self):
        force(self.model, I.EDIT)
        with patch.object(self.session, "_do_edit") as edit:
            self.assertEqual(self.session.respond(b"Fix it", document=b"abc").action, "ask")
            self.assertEqual(self.session.respond(b"Do not change byte 1 to 33.", document=b"abc").action, "ask")
            edit.assert_not_called()

    def test_episode_capacity_failure_is_disclosed_not_misreported(self):
        store = memory(capacities=(1, 1, 1))
        store.record_episode(b"bob", ConversationTurn(b"private", b"ok"))
        session = ContextualConversationSession(self.model, self.decoder, store, b"alice")
        force(self.model, I.CHAT)
        with patch.object(self.decoder, "generate", return_value=(b"Hello.", True)):
            result = session.respond(b"Hello")
        self.assertFalse(result.episode_recorded)
        self.assertIn(b"could not save", result.response)
        self.assertEqual(store.recent(b"bob", b"q").turns[0].user, b"private")


class Phase11FixtureTests(unittest.TestCase):
    def test_boundary_and_overall_test_are_disjoint_from_train_and_validation(self):
        def norm(raw):
            return b" ".join(raw.lower().rstrip(b".!? ").split())
        train = {norm(m) for m, _ in hard_negative_examples("train")}
        train |= {norm(row.message) for row in conversation_examples("train")}
        validation = {norm(m) for m, _ in hard_negative_examples("validation") + overall_intent_examples("validation")}
        test = {norm(m) for m, _ in hard_negative_examples("test") + overall_intent_examples("test")}
        self.assertFalse(test & train); self.assertFalse(test & validation)
        self.assertEqual(len(hard_negative_examples("test")), 20)
        self.assertEqual(len(overall_intent_examples("test")), 70)

    def test_retention_statements_are_not_response_training_histories(self):
        train = {turn.user for row in response_examples("train") for turn in row.history}
        test = retention_cases()
        self.assertEqual(len(test), 10)
        self.assertFalse({case.statement for case in test} & train)
        # Common question wording and synthetic response styles may be shared.


if __name__ == "__main__":
    unittest.main()
