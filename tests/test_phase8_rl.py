"""Regression tests for reasoning supervision, curriculum and score-function RL."""
from __future__ import annotations

import itertools
import math
import unittest
from unittest.mock import patch

import torch

from data.edit_curriculum import curriculum_examples
from data.ingest_real_traces import make_control_trace, optimal_edit_actions
from models.cognitive import CognitiveAction, CognitiveEvent, CognitiveResult, action_from_record, action_to_record
from models.generator_conditioned import GeneratorConditioned
from models.holdout import HoldoutGuard, HoldoutViolation
from models.memory_model import MemoryModel
from models.procedural import byte_pattern_embedding
from training.co_train import CURRICULUM_STAGES, co_train_step, collate, curriculum_stage, ingest_examples
from training.cognitive_train import cognitive_trace_supervision_loss, edit_intent_target, validate_action_trace
from training.rl_editing import (autonomous_rollout, byte_levenshtein, consistent_attention_math,
                                 destination_log_probability, editing_reward, evaluate_autonomous,
                                 hybrid_weights, preflight_training_examples,
                                 reinforce_loss, replay_action_log_probability)


def tiny() -> GeneratorConditioned:
    return GeneratorConditioned(d_model=16, n_layers=1, n_heads=2, d_ff=32, max_len=32,
                                dropout=0.0, scratchpad_slots=3, max_canvas_bytes=128).eval()


class ReasoningTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)
        self.g = tiny()

    def test_fixed_difference_and_post_think_mse_backpropagates(self) -> None:
        source, target = b"\x00ab", b"\x00ac"
        _, actions = make_control_trace(source, target)
        label = edit_intent_target(self.g, source, target)
        expected = (byte_pattern_embedding(target, 16, device=torch.device("cpu"))
                    - byte_pattern_embedding(source, 16, device=torch.device("cpu")))
        self.assertTrue(torch.allclose(label, expected))
        self.assertFalse(label.requires_grad)
        self.assertFalse(torch.equal(edit_intent_target(self.g, b"ab", b"ba"), torch.zeros(16)))
        trace, reasoning, count = cognitive_trace_supervision_loss(
            self.g, source, actions, target_bytes=target, return_reasoning=True,
        )
        self.assertEqual(count, 1)
        self.assertGreater(float(reasoning), 0)
        reasoning.backward()
        self.assertGreater(float(self.g.reasoning_predictor[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.g.scratchpad_thinker.weight_ih.grad.abs().sum()), 0)
        self.assertGreater(float(self.g.scratchpad_reader.weight_ih.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(trace))
        with self.assertRaises(ValueError):
            cognitive_trace_supervision_loss(self.g, source, actions, target_bytes=b"bad", return_reasoning=True)
        plain = cognitive_trace_supervision_loss(self.g, source, actions)
        self.assertIsInstance(plain, torch.Tensor)

    def test_no_think_has_zero_reasoning_loss(self) -> None:
        _, actions = make_control_trace(b"a", b"b")
        actions = [a for a in actions if a.token != self.g.THINK]
        _, reason, count = cognitive_trace_supervision_loss(self.g, b"a", actions, target_bytes=b"b", return_reasoning=True)
        self.assertEqual((float(reason), count), (0.0, 0))

    def test_old_checkpoint_migrates_but_partial_head_fails_strict(self) -> None:
        original = self.g.state_dict()
        old = {key: value.clone() for key, value in original.items() if not key.startswith("reasoning_predictor.")}
        restored = tiny()
        restored.load_state_dict(old, strict=True)
        for key, value in old.items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
        bad = {key: value.clone() for key, value in original.items() if key != "reasoning_predictor.2.bias"}
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(bad, strict=True)
        self.assertEqual(sum(p.numel() for p in self.g.reasoning_predictor.parameters()), 2 * (16 * 16 + 16))

    def test_cotraining_adds_half_reasoning_and_updates_predictor(self) -> None:
        memory = MemoryModel(d_model=16, capacity=16)
        source, target = b"ab", b"ac"
        _, actions = make_control_trace(source, target)
        batch = collate([{"input": source, "output": target, "instruction_bytes": b"replace b with c",
                          "trace_actions": [action_to_record(a) for a in actions]}])
        before = self.g.reasoning_predictor[0].weight.detach().clone()
        result = co_train_step(memory, self.g, batch, "cpu", torch.optim.Adam(memory.parameters(), lr=1e-3),
                               torch.optim.Adam(self.g.parameters(), lr=1e-3), cognitive_trace_weight=1.0)
        expected = (result["gen_loss"] - result["consistency_sim"] - result["retrieval_sim"]
                    + 0.1 * result["confidence_loss"] + result["cognitive_trace_loss"]
                    + 0.5 * result["reasoning_loss"])
        self.assertAlmostEqual(result["generator_total_loss"], expected, places=5)
        self.assertEqual(result["reasoning_think_events"], 1)
        self.assertFalse(torch.equal(before, self.g.reasoning_predictor[0].weight))

    def test_reasoning_supports_partially_labelled_batch(self) -> None:
        memory = MemoryModel(d_model=16, capacity=16)
        _, actions = make_control_trace(b"ab", b"ac")
        examples = [{"input": b"ab", "output": b"ac", "trace_actions": [action_to_record(a) for a in actions]},
                    {"input": b"xy", "output": b"xz"}]
        result = co_train_step(memory, self.g, collate(examples), "cpu",
                               torch.optim.Adam(memory.parameters(), lr=1e-3),
                               torch.optim.Adam(self.g.parameters(), lr=1e-3))
        self.assertEqual(result["reasoning_think_events"], 1)
        self.assertGreater(result["reasoning_loss"], 0)
        other_memory = MemoryModel(d_model=16, capacity=16)
        with self.assertRaises(ValueError):
            co_train_step(other_memory, self.g, collate(examples), "cpu",
                          torch.optim.Adam(other_memory.parameters(), lr=1e-3),
                          torch.optim.Adam(self.g.parameters(), lr=1e-3), cognitive_trace_weight=1.0)
        self.assertEqual(int(other_memory.procedural.used.sum()), 0)


class ReinforceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.g = tiny()

    def test_reward_raw_bytes_steps_empty_and_levenshtein(self) -> None:
        self.assertEqual(byte_levenshtein(b"\x00abc\xff", b"\x00acc\xff"), 1)
        self.assertEqual(byte_levenshtein(b"", b"\x00"), 1)
        events = (CognitiveEvent(CognitiveAction(self.g.THINK, think_steps=2), 2, None, thought_steps=2),
                  CognitiveEvent(CognitiveAction(self.g.EDIT_OP), 2, None))
        result = CognitiveResult(b"ab", True, "done", events, self.g.scratchpad, None, None)
        score = editing_reward(result, b"ac", self.g)
        self.assertAlmostEqual(float(score["reward"]), 0.5 - 0.03)
        empty = CognitiveResult(b"", True, "done", (), self.g.scratchpad, None, None)
        self.assertEqual(editing_reward(empty, b"", self.g)["reward"], 1.0)
        with self.assertRaises(ValueError):
            byte_levenshtein(b"a" * 100, b"b" * 100, max_cells=100)

    def test_levenshtein_matches_trace_reference_exhaustively(self) -> None:
        strings = [bytes(parts) for length in range(4) for parts in itertools.product((0, 255), repeat=length)]
        for source in strings:
            for target in strings:
                self.assertEqual(byte_levenshtein(source, target), optimal_edit_actions(source, target)[0])

    def test_exact_remap_probability_and_grad(self) -> None:
        logits = torch.tensor([0.1, -0.3, 0.2, 0.7, -0.9], requires_grad=True)
        probability = destination_log_probability(logits, 1, 4, 4, 0.7).exp()
        expected = torch.softmax(logits / 0.7, dim=0)[2:5].sum()
        self.assertTrue(torch.allclose(probability, expected))
        probability.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            destination_log_probability(logits, 1, 4, 2, 1.0)

    def test_sampled_sequence_log_probability_and_finite_difference(self) -> None:
        # LOAD all bytes, THINK, MOVE with an interior destination, REPLACE
        # with 0xff, SAVE, DONE. Record every actual sampler distribution.
        choices = iter([0, 0, 3, 2, 3, 3, 0, 3, 1, 3, 2, 0, 1, 0, 255, 1, 4])
        calls: list[tuple[torch.Tensor, int]] = []

        def choose(probs: torch.Tensor, num_samples: int, *args: object, **kwargs: object) -> torch.Tensor:
            value = next(choices)
            self.assertEqual(num_samples, 1)
            self.assertGreater(float(probs[value]), 0.0)
            calls.append((probs.detach().clone(), value))
            return torch.tensor([value], device=probs.device)

        with patch("torch.multinomial", side_effect=choose):
            result = self.g.cognitive_generate(b"\x00ab", max_steps=6, temperature=0.9)
        self.assertTrue(result.done)
        self.assertEqual(result.events[2].action.destination, 3)  # sampled 1 -> recorded 3
        self.assertEqual(result.events[3].action.data, b"\xff")
        expected = 0.0
        for i, (probs, selected) in enumerate(calls):
            # Interior destination 1 and 2 map to 3; all others remain distinct.
            probability = probs[1:4].sum() if i == 8 else probs[selected]
            expected += math.log(float(probability))
        logp, thoughts = replay_action_log_probability(self.g, b"\x00ab", result, temperature=0.9)
        self.assertAlmostEqual(float(logp), expected, places=4)
        self.assertEqual(len(thoughts), 1)
        logp.backward()
        analytic = float(self.g.edit_byte_head.bias.grad[255])
        self.assertNotEqual(analytic, 0.0)
        eps = 1e-3
        with torch.no_grad():
            self.g.edit_byte_head.bias[255] += eps
        plus, _ = replay_action_log_probability(self.g, b"\x00ab", result, temperature=0.9)
        with torch.no_grad():
            self.g.edit_byte_head.bias[255] -= 2 * eps
        minus, _ = replay_action_log_probability(self.g, b"\x00ab", result, temperature=0.9)
        self.assertAlmostEqual(analytic, float((plus - minus) / (2 * eps)), delta=0.01)
        with torch.no_grad(), self.assertRaises(RuntimeError):
            replay_action_log_probability(self.g, b"\x00ab", result, temperature=0.9)

    def test_rollout_is_differentiable_and_bounds_rejected(self) -> None:
        backend = getattr(torch.backends, "mha", None)
        if backend is not None and hasattr(backend, "get_fastpath_enabled"):
            previous = backend.get_fastpath_enabled()
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with consistent_attention_math():
                    self.assertFalse(backend.get_fastpath_enabled())
                    raise RuntimeError("intentional")
            self.assertEqual(backend.get_fastpath_enabled(), previous)
        torch.manual_seed(19)
        rollout = autonomous_rollout(self.g, b"abc", max_steps=12, temperature=0.8)
        self.assertTrue(rollout.log_probability.requires_grad)
        loss = reinforce_loss(rollout.log_probability, 0.7, baseline=0.2)
        self.assertAlmostEqual(float(loss), -0.5 * float(rollout.log_probability), places=5)
        loss.backward()
        self.assertGreater(float(self.g.head.weight.grad[self.g.LOAD_SPAN:].abs().sum()), 0)
        with self.assertRaises(ValueError):
            autonomous_rollout(self.g, b"a" * 120, max_steps=16, temperature=1.0)
        with self.assertRaises(ValueError):
            autonomous_rollout(self.g, b"a", max_steps=10, temperature=0.0)

    def test_evaluation_never_passes_target_or_target_budget(self) -> None:
        records = [{"id": "a", "input": b"one", "output": b"different"},
                   {"id": "b", "input": b"two", "output": b"x"}]
        original = self.g.cognitive_generate
        with patch.object(self.g, "cognitive_generate", wraps=original) as mocked:
            report = evaluate_autonomous(self.g, records, budget=7)
        self.assertEqual(report["count"], 2)
        for call in mocked.call_args_list:
            self.assertEqual(call.kwargs, {"max_steps": 7, "instruction_bytes": None})
            self.assertNotIn("controller", call.kwargs)
            self.assertNotIn("memory_model", call.kwargs)


class CurriculumTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)
        self.g = tiny()

    def test_schedule_and_annealing_boundaries(self) -> None:
        self.assertEqual([curriculum_stage(i, 10) for i in range(10)],
                         [stage for stage in CURRICULUM_STAGES for _ in range(2)])
        self.assertAlmostEqual(hybrid_weights(0, 11)[0], 0.8)
        self.assertAlmostEqual(hybrid_weights(0, 11)[1], 0.2)
        self.assertAlmostEqual(hybrid_weights(10, 11)[0], 0.2)
        self.assertAlmostEqual(hybrid_weights(10, 11)[1], 0.8)
        with self.assertRaises(ValueError):
            curriculum_stage(10, 10)

    def test_examples_disjoint_and_valid_edit_structures(self) -> None:
        train = curriculum_examples(split="train", count_per_stage=4)
        validation = curriculum_examples(split="validation", count_per_stage=2)
        train_bytes = {r[k] for r in train for k in ("input", "output")}
        val_bytes = {r[k] for r in validation for k in ("input", "output")}
        self.assertFalse(train_bytes & val_bytes)
        for record in train + validation:
            actions = [action_from_record(r) for r in record["trace_actions"]]
            self.assertEqual(validate_action_trace(self.g, record["input"], actions), record["output"])
            self.assertNotEqual(record["input"], record["output"])
            stage = record["curriculum_stage"]
            if stage == CURRICULUM_STAGES[0]:
                self.assertEqual(optimal_edit_actions(record["input"], record["output"])[0], 1)
                self.assertEqual(len(record["input"]), len(record["output"]))
            if stage == CURRICULUM_STAGES[1]:
                self.assertEqual(record["input"].replace(b"old", b"new"), record["output"])
            if stage == CURRICULUM_STAGES[2]:
                self.assertEqual(record["input"].replace(b"\nx=0\n", b"\ny=1\n"), record["output"])
            if stage == CURRICULUM_STAGES[3]:
                self.assertEqual(abs(record["input"].count(b"\n") - record["output"].count(b"\n")), 2)
            if stage == CURRICULUM_STAGES[4]:
                self.assertEqual(sorted(record["input"]), sorted(record["output"]))
                self.assertEqual(sum(a.kind == "move" for a in actions), 1)

    def test_holdout_preflight_checks_instruction_and_trace(self) -> None:
        train = curriculum_examples(split="train", count_per_stage=1)
        guard = HoldoutGuard.from_records([{"input": b"protected-target-123", "output": b"secret-result-456"}])
        train[1]["instruction_bytes"] = b"prefix protected-target-123 suffix"
        with self.assertRaises(HoldoutViolation):
            preflight_training_examples(self.g, train, guard)
        train[1]["instruction_bytes"] = b"safe"
        train[1]["split"] = "test"
        with self.assertRaises(ValueError):
            preflight_training_examples(self.g, train, guard)

    def test_curriculum_writes_only_current_stage_and_preflights_missing_stage(self) -> None:
        examples = curriculum_examples(split="train", count_per_stage=1)
        memory = MemoryModel(d_model=16, capacity=32)
        with self.assertRaises(ValueError):
            ingest_examples(memory, self.g, examples[:-1], "cpu", epochs=5, batch_size=1, curriculum=True)
        with self.assertRaises(ValueError):
            ingest_examples(memory, self.g, examples, "cpu", epochs=4, batch_size=1, curriculum=True)
        self.assertEqual(int(memory.procedural.used.sum()), 0)
        observed: list[tuple[int, bytes]] = []

        def fake_step(mem: MemoryModel, generator: GeneratorConditioned, batch: dict, *args: object, **kwargs: object) -> dict:
            value = bytes(batch["input_ids"][0][batch["input_mask"][0]].tolist())
            observed.append((int(mem.procedural.used.sum()), value))
            return {"gen_loss": 0.0, "retrieval_sim": 0.0, "consistency_sim": 0.0}

        with patch("training.co_train.co_train_step", side_effect=fake_step):
            logs = ingest_examples(memory, self.g, examples, "cpu", epochs=5, batch_size=1, curriculum=True)
        self.assertEqual([n for n, _ in observed], [1, 2, 3, 4, 5])
        self.assertEqual([value for _, value in observed], [r["input"] for r in examples])
        self.assertEqual([row["curriculum_stage"] for row in logs], list(CURRICULUM_STAGES))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
