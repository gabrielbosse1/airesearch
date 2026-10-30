"""Mechanics, migration, safety and memory regression tests for Phase 10."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.conversation import ConversationMemory, ConversationSession, ConversationTurn, pack_conversation
from models.conversation_intents import ConversationIntent, NUM_INTENTS
from models.dialogue_response import ByteDialogueDecoder, load_dialogue_decoder
from models.generator_conditioned import GeneratorConditioned
from models.holdout import HoldoutGuard, HoldoutViolation
from models.memory_model import MemoryModel


def make_model() -> GeneratorConditioned:
    model = GeneratorConditioned(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=128, dropout=0.0).eval()
    model.intent_trained.fill_(True)
    return model


def force_intent(model: GeneratorConditioned, intent: ConversationIntent) -> None:
    with torch.no_grad():
        for parameter in model.intent_classifier.parameters():
            parameter.zero_()
        model.intent_classifier[2].bias.fill_(-10)
        model.intent_classifier[2].bias[int(intent)] = 10


def make_decoder() -> ByteDialogueDecoder:
    decoder = ByteDialogueDecoder(16).eval()
    decoder.response_trained.fill_(True)
    return decoder


def memory_model() -> MemoryModel:
    return MemoryModel(d_model=16, capacity=12, n_heads=4, max_pattern_bytes=256)


class Phase10HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_head_shape_raw_bytes_and_padding(self) -> None:
        model = make_model()
        raw = torch.tensor([[65, 0, 66]], dtype=torch.long)
        length = torch.tensor([3])
        padded = torch.tensor([[65, 0, 66, 99, 88]], dtype=torch.long)
        out = model.classify_intent(raw, length)
        self.assertEqual(tuple(out.shape), (1, NUM_INTENTS))
        self.assertTrue(torch.allclose(out, model.classify_intent(padded, length), atol=1e-5))
        with self.assertRaises(ValueError):
            model.classify_intent(torch.tensor([[256]]), torch.tensor([1]))
        with self.assertRaises(ValueError):
            model.classify_intent(raw, torch.tensor([0]))
        with self.assertRaises(ValueError):
            model.conversation_features(raw, length, focus_start=torch.tensor([3]))

    def test_encoder_and_head_receive_gradients(self) -> None:
        model = make_model()
        ids = torch.tensor([[97, 98, 99], [120, 121, 122]])
        labels = torch.tensor([int(ConversationIntent.CHAT), int(ConversationIntent.EDIT)])
        loss = torch.nn.functional.cross_entropy(model.classify_intent(ids, torch.tensor([3, 3])), labels)
        loss.backward()
        self.assertGreater(float(model.intent_encoder[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.intent_classifier[2].weight.grad.abs().sum()), 0)

    def test_full_legacy_migration_is_deterministic_and_partial_fails(self) -> None:
        original = make_model().state_dict()
        legacy = {k: v.clone() for k, v in original.items() if not k.startswith(("intent_encoder.", "intent_classifier.")) and k != "intent_trained"}
        first, second = make_model(), make_model()
        first.load_state_dict(legacy, strict=True)
        second.load_state_dict(legacy, strict=True)
        self.assertFalse(bool(first.intent_trained))
        for name in first.state_dict():
            if name.startswith(("intent_encoder.", "intent_classifier.")):
                self.assertTrue(torch.equal(first.state_dict()[name], second.state_dict()[name]), name)
            elif name != "intent_trained":
                self.assertTrue(torch.equal(first.state_dict()[name], legacy[name]), name)
        partial = {k: v.clone() for k, v in original.items()}
        del partial["intent_encoder.0.weight"]
        with self.assertRaises(RuntimeError):
            first.load_state_dict(partial, strict=True)
        partial = {k: v.clone() for k, v in original.items()}
        del partial["intent_trained"]
        with self.assertRaises(RuntimeError):
            first.load_state_dict(partial, strict=True)

    def test_decoder_uses_only_raw_bytes_and_internal_stop(self) -> None:
        decoder = make_decoder()
        context = torch.randn(2, 16)
        targets = torch.tensor([[0, 255, 256], [65, 256, -100]])
        logits = decoder(context, targets)
        self.assertEqual(tuple(logits.shape), (2, 3, 257))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 257), targets.reshape(-1), ignore_index=-100)
        loss.backward()
        self.assertGreater(float(decoder.byte_head.weight.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            decoder(context, torch.tensor([[257], [1]]))

    def test_decoder_checkpoint_requires_exact_generator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generator = Path(directory) / "generator.pt"
            response = Path(directory) / "response.pt"
            generator.write_bytes(b"trusted-generator")
            decoder = make_decoder()
            torch.save({"schema_version": 1, "context_dim": 16, "hidden_dim": 16,
                        "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
                        "generator_n_heads": 4, "state_dict": decoder.state_dict()}, response)
            loaded = load_dialogue_decoder(response, generator_path=generator)
            self.assertTrue(bool(loaded.response_trained))
            incompatible = make_model()
            incompatible.n_heads = 2
            with self.assertRaises(ValueError):
                ConversationSession(incompatible, loaded, ConversationMemory(memory_model()), b"user")
            generator.write_bytes(b"a different generator")
            with self.assertRaises(ValueError):
                load_dialogue_decoder(response, generator_path=generator)

    def test_untrained_head_or_decoder_fails_closed(self) -> None:
        model, decoder = make_model(), make_decoder()
        mem = ConversationMemory(memory_model())
        model.intent_trained.fill_(False)
        with self.assertRaises(RuntimeError):
            ConversationSession(model, decoder, mem, b"user")
        model.intent_trained.fill_(True); decoder.response_trained.fill_(False)
        with self.assertRaises(RuntimeError):
            ConversationSession(model, decoder, mem, b"user")


class Phase10MemoryTests(unittest.TestCase):
    def test_namespaces_persist_across_instances_and_are_not_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.pt"
            memory = ConversationMemory(memory_model(), path)
            memory.remember(b"ab", b"c", b"Ada", source=b"remember my c is Ada")
            memory.remember(b"a", b"bc", b"Mira", source=b"remember my bc is Mira")
            restored = ConversationMemory(memory_model(), path)
            self.assertEqual(restored.recall(b"ab", b"c"), b"Ada")
            self.assertEqual(restored.recall(b"a", b"bc"), b"Mira")
            self.assertIsNone(restored.recall(b"other", b"c"))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_forget_clears_all_versions_without_affecting_other_users(self) -> None:
        memory = ConversationMemory(memory_model())
        memory.remember(b"u1", b"name", b"Ada", source=b"remember my name is Ada")
        memory.remember(b"u1", b"name", b"Mira", source=b"remember my name is Mira")
        self.assertEqual(memory.recall(b"u1", b"name"), b"Mira")
        # A legacy bank may already contain duplicate versions; forget all.
        memory.memory.write_procedure(ConversationMemory._frame(b"u1", b"name"), b"Mira",
                                      torch.empty(0, 16), metadata={"type": "conversation_fact"})
        memory.remember(b"u2", b"name", b"Lina", source=b"remember my name is Lina")
        self.assertEqual(memory.forget(b"u1", b"name"), 2)
        self.assertIsNone(memory.recall(b"u1", b"name"))
        self.assertEqual(memory.recall(b"u2", b"name"), b"Lina")
        bank = memory.memory.procedural
        free = ~bank.used
        self.assertEqual(int(bank.input_bytes[free].sum()), 0)
        self.assertEqual(int(bank.output_bytes[free].sum()), 0)
        self.assertEqual(int(bank.trace_steps[free].abs().sum()), 0)
        with memory.memory.read_only():
            with self.assertRaises(HoldoutViolation):
                memory.forget(b"u2", b"name")

    def test_full_bank_rejects_new_fact_without_evicting_another_user(self) -> None:
        memory = ConversationMemory(MemoryModel(d_model=16, capacity=3, n_heads=4, max_pattern_bytes=256))
        memory.remember(b"u1", b"name", b"Ada", source=b"remember my name is Ada")
        with self.assertRaises(ValueError):
            memory.remember(b"u2", b"name", b"Lina", source=b"remember my name is Lina")
        self.assertEqual(memory.recall(b"u1", b"name"), b"Ada")
        memory.remember(b"u1", b"name", b"Mira", source=b"remember my name is Mira")
        self.assertEqual(memory.recall(b"u1", b"name"), b"Mira")

    def test_failed_atomic_save_rolls_back_bank_and_tick(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = ConversationMemory(memory_model(), Path(directory) / "memory.pt")
            with patch.object(memory, "_atomic_save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    memory.remember(b"u", b"name", b"Ada", source=b"remember my name is Ada")
            self.assertIsNone(memory.recall(b"u", b"name"))
            self.assertEqual(int(memory.memory.procedural.used.sum()), 0)
            self.assertEqual(memory.memory.session_tick, 0)
            memory.remember(b"u", b"name", b"Ada", source=b"remember my name is Ada")
            before = (Path(directory) / "memory.pt").read_bytes()
            with patch.object(memory, "_atomic_save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    memory.remember(b"u", b"name", b"Mira", source=b"remember my name is Mira")
            self.assertEqual(memory.recall(b"u", b"name"), b"Ada")
            self.assertEqual((Path(directory) / "memory.pt").read_bytes(), before)

    def test_guard_checks_full_explicit_request(self) -> None:
        raw = b"Remember my name is Mira."
        guarded = memory_model()
        guarded.set_holdout_guard(HoldoutGuard.from_records([{"input": raw}]))
        memory = ConversationMemory(guarded)
        with self.assertRaises(HoldoutViolation):
            memory.remember(b"u", b"name", b"Mira", source=raw)
        self.assertIsNone(memory.recall(b"u", b"name"))


class Phase10DialogueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = make_model()
        self.memory = ConversationMemory(memory_model())
        self.session = ConversationSession(self.model, make_decoder(), self.memory, b"alice")

    def test_recent_history_keeps_whole_turns_and_current_bytes(self) -> None:
        history = [ConversationTurn(b"older", b"answer"), ConversationTurn(b"newer", b"reply")]
        packed = pack_conversation(b"hello\x00", history, 32)
        self.assertTrue(packed.endswith(b"U:hello\x00"))
        self.assertNotIn(b"older", packed)
        self.assertIn(b"newer", packed)
        with self.assertRaises(ValueError):
            pack_conversation(b"too long", [], 4)

    def test_memory_conversation_and_user_isolation(self) -> None:
        force_intent(self.model, ConversationIntent.REMEMBER)
        result = self.session.respond(b"Remember my name is Ada.")
        self.assertEqual(result.action, "remember")
        force_intent(self.model, ConversationIntent.RECALL)
        result = self.session.respond(b"What is my name?")
        self.assertEqual(result.response, b"Your name is Ada.")
        other = ConversationSession(self.model, make_decoder(), self.memory, b"bob")
        self.assertEqual(other.respond(b"What is my name?").action, "ask")
        force_intent(self.model, ConversationIntent.FORGET)
        self.assertEqual(self.session.respond(b"Forget my name.").action, "forget")
        self.assertIsNone(self.memory.recall(b"alice", b"name"))

    def test_raw_binary_memory_is_not_emitted_as_invalid_chat_text(self) -> None:
        self.memory.remember(b"alice", b"name", b"\xff\x00", source=b"explicit raw value")
        force_intent(self.model, ConversationIntent.RECALL)
        result = self.session.respond(b"What is my name?")
        self.assertEqual(result.action, "ask")
        result.response.decode("utf-8")

    def test_known_questions_use_reference_and_unknown_questions_abstain(self) -> None:
        force_intent(self.model, ConversationIntent.QUESTION)
        known = self.session.respond(b"What is a byte?")
        self.assertEqual(known.action, "answer")
        self.assertEqual(known.response_source, "verified_reference")
        self.assertIn(b"eight bits", known.response)
        unknown = self.session.respond(b"Who wrote this mysterious book?")
        self.assertEqual(unknown.action, "ask")
        self.assertIn(b"don't have a reliable answer", unknown.response)

    def test_misclassification_cannot_authorize_negative_or_implicit_memory_write(self) -> None:
        force_intent(self.model, ConversationIntent.REMEMBER)
        for message in (b"Do not remember my name is Ada.", b"My name is Ada.", b"Remember my name",
                        "Remember my K is Ada.".encode("utf-8")):
            self.assertEqual(self.session.respond(message).action, "ask")
        self.assertIsNone(self.memory.recall(b"alice", b"name"))

    def test_edit_is_returned_only_after_model_result_matches_request(self) -> None:
        force_intent(self.model, ConversationIntent.EDIT)
        self.model.structured_edit_trained_kinds[2] = True
        invalid = SimpleNamespace(done=True, canvas=b"XYZ", events=())
        with patch.object(self.model, "cognitive_generate", return_value=invalid):
            outcome = self.session.respond(b"Replace byte 1 with 100.", document=b"abc")
        self.assertEqual(outcome.action, "ask")
        self.assertIsNone(outcome.edited_document)
        valid = SimpleNamespace(done=True, canvas=b"adc", events=())
        with patch.object(self.model, "cognitive_generate", return_value=valid) as run:
            outcome = self.session.respond(b"Replace byte 1 with 100.", document=b"abc")
        self.assertEqual(outcome.action, "edit")
        self.assertEqual(outcome.edited_document, b"adc")
        self.assertEqual(run.call_args.kwargs["structured_request"], bytes([1, 2, 100]))
        self.assertNotIn("controller", run.call_args.kwargs)

    def test_follow_up_confirmation_requires_explicit_request_and_document(self) -> None:
        force_intent(self.model, ConversationIntent.EDIT)
        self.model.structured_edit_trained_kinds[2] = True
        result = self.session.respond(b"Replace byte 1 with 100.")
        self.assertEqual(result.action, "ask")
        # A confirmation is an explicit authorization even if the head labels it
        # as clarification; classification still runs before this decision.
        force_intent(self.model, ConversationIntent.CLARIFY)
        with patch.object(self.model, "cognitive_generate", return_value=SimpleNamespace(done=True, canvas=b"adc", events=())):
            result = self.session.respond(b"Apply that edit.", document=b"abc")
        self.assertEqual(result.edited_document, b"adc")
        result = self.session.respond(b"Apply that edit.", document=b"abc")
        self.assertEqual(result.action, "ask")

    def test_invalid_and_oversized_text_fail_without_action(self) -> None:
        self.assertEqual(self.session.respond(b"\xff").action, "ask")
        self.assertEqual(self.session.respond(b"x" * 200).action, "ask")
        self.assertEqual(self.session.history, [])
        self.assertEqual(int(self.memory.memory.procedural.used.sum()), 0)


if __name__ == "__main__":
    unittest.main()
