from __future__ import annotations

import unittest
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from models.cognitive import CognitiveAction, apply_edit
from models.generator_conditioned import GeneratorConditioned, GeneratorState
from training.cognitive_train import cognitive_supervision_loss


class CognitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.g = GeneratorConditioned(d_model=16, n_layers=2, n_heads=4, d_ff=32,
                                      max_len=32, dropout=0.0, max_canvas_bytes=32)

    def test_vocabulary_and_parallel_state(self) -> None:
        g = self.g
        self.assertEqual([g.FUNC_CALL, g.FUNC_ARG, g.FUNC_END, g.LOAD_SPAN, g.SAVE_SPAN,
                          g.THINK, g.EDIT_OP, g.DONE], list(range(256, 264)))
        ids = torch.tensor([[0, 255, 259, 260, 261, 262, 263]])
        state = g(ids, return_state=True)
        self.assertIsInstance(state, GeneratorState)
        self.assertEqual(state.output.shape, (1, 7, 264))
        self.assertEqual(state.thought.shape, (1, 7, 16))
        self.assertEqual(state.confidence.shape, (1, 7))
        self.assertTrue(bool(((state.confidence >= 0) & (state.confidence <= 1)).all()))
        with self.assertRaises(ValueError):
            g(torch.tensor([[264]]))
        with self.assertRaises(ValueError):
            g(torch.tensor([[-1]]))

    def test_nonmonotonic_loop_and_raw_bytes(self) -> None:
        g = self.g
        A = CognitiveAction
        actions = iter([
            A(g.LOAD_SPAN, start=0, end=7),
            A(g.THINK, think_steps=3),
            A(g.EDIT_OP, start=3, kind="insert", data=b"!"),
            A(g.EDIT_OP, start=3, end=4, kind="delete"),
            A(g.EDIT_OP, start=4, end=7, kind="replace", data=b"\x00\xffx"),
            A(g.EDIT_OP, start=0, end=3, kind="move", destination=7),
            A(g.SAVE_SPAN), A(g.DONE),
        ])
        observations = []
        def controller(obs):
            observations.append(obs)
            return next(actions)
        result = g.cognitive_generate(b"abc def", controller=controller, max_steps=8)
        self.assertTrue(result.done)
        self.assertEqual(result.canvas, b" \x00\xffxabc")
        self.assertEqual(result.events[1].thought_steps, 3)
        self.assertEqual(observations[2].canvas, b"abc def")
        self.assertEqual(observations[6].canvas, b"abc def")  # not committed yet
        self.assertEqual(observations[7].canvas, b" \x00\xffxabc")
        self.assertFalse(torch.equal(observations[1].scratchpad, observations[2].scratchpad))
        self.assertEqual(int(torch.count_nonzero(g.scratchpad)), 0)  # isolated template
        self.assertNotIn("scratchpad", g.state_dict())

    def test_staged_edits_must_be_saved_and_exhaustion_is_visible(self) -> None:
        g = self.g
        actions = iter([CognitiveAction(g.LOAD_SPAN, start=1, end=3),
                        CognitiveAction(g.EDIT_OP, start=0, end=1, kind="replace", data=b"X")])
        result = g.cognitive_generate(b"abcd", controller=lambda _: next(actions), max_steps=2)
        self.assertFalse(result.done)
        self.assertEqual(result.halt_reason, "max_steps")
        self.assertEqual(result.canvas, b"abcd")
        self.assertEqual(result.pending_workspace, b"Xc")
        self.assertEqual(result.pending_span, (1, 3))
        actions = iter([CognitiveAction(g.LOAD_SPAN, start=0, end=1),
                        CognitiveAction(g.EDIT_OP, start=0, kind="insert", data=b"!"),
                        CognitiveAction(g.DONE)])
        with self.assertRaisesRegex(ValueError, "SAVE"):
            g.cognitive_generate(b"a", controller=lambda _: next(actions))

    def test_edits_validate_bounds_and_caps(self) -> None:
        g = self.g
        self.assertEqual(apply_edit(b"abcdef", CognitiveAction(g.EDIT_OP, kind="move", start=1, end=3, destination=6), max_length=9), b"adefbc")
        self.assertEqual(apply_edit(b"abcdef", CognitiveAction(g.EDIT_OP, kind="move", start=4, end=6, destination=0), max_length=9), b"efabcd")
        with self.assertRaises(ValueError):
            apply_edit(b"abcdef", CognitiveAction(g.EDIT_OP, kind="move", start=1, end=4, destination=2), max_length=9)
        with self.assertRaises(ValueError):
            apply_edit(b"a", CognitiveAction(g.EDIT_OP, kind="insert", start=-1, data=b"x"), max_length=2)
        with self.assertRaises(ValueError):
            apply_edit(b"a", CognitiveAction(g.EDIT_OP, kind="insert", start=1, data=b"xx"), max_length=2)
        with self.assertRaises(TypeError):
            apply_edit(b"a", CognitiveAction(g.EDIT_OP, kind="insert", start=0, data="x"), max_length=2)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            g.cognitive_generate(b"x" * 33)
        with self.assertRaises(ValueError):
            g.cognitive_generate(b"a", controller=lambda _: CognitiveAction(g.EDIT_OP, kind="delete", start=0, end=1))

    def test_auto_done_and_session_isolation(self) -> None:
        g = self.g
        with torch.no_grad():
            g.head.weight.zero_()
            g.head.bias.zero_()
            g.head.bias[g.DONE] = 30
        first = g.cognitive_generate(b"\x00\xff", max_steps=2)
        second = g.cognitive_generate(b"\x00\xff", max_steps=2)
        self.assertTrue(first.done and second.done)
        self.assertEqual(first.canvas, b"\x00\xff")
        self.assertTrue(torch.equal(first.scratchpad, second.scratchpad))
        self.assertEqual(first.events[0].action.token, g.DONE)

    def test_included_checkpoint_runs_each_synthetic_edit_autonomously(self) -> None:
        path = Path(__file__).resolve().parents[1] / "runs/dual_memory/generator.pt"
        self.assertTrue(path.exists())
        g = GeneratorConditioned(d_model=64, n_layers=2, n_heads=4, d_ff=128)
        g.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        for canvas, expected in ((b"I:abc", b"I:abc!"), (b"D:abc", b"D:bc"),
                                 (b"R:abc", b"R:?bc"), (b"M:abc", b"M:cab")):
            with self.subTest(canvas=canvas):
                result = g.cognitive_generate(canvas, max_steps=8)
                self.assertTrue(result.done)
                self.assertEqual(result.canvas, expected)
                self.assertEqual([e.action.token for e in result.events],
                                 [g.LOAD_SPAN, g.THINK, g.EDIT_OP, g.SAVE_SPAN, g.DONE])

    def test_latent_trace_and_supervised_policy_gradients(self) -> None:
        g = self.g
        scratch = g.read_scratchpad(g.scratchpad.clone(), torch.tensor([[0, 1, 255]]))
        trace = torch.randn(2, 16, requires_grad=True)
        scratch = g.replay_latent_trace(scratch, trace)
        thought = g.think_scratchpad(scratch, torch.randn(1, 16), steps=3)
        loss, terms = cognitive_supervision_loss(
            g, b"abc", thought.scratchpad,
            CognitiveAction(g.EDIT_OP, start=1, end=2, kind="replace", data=b"\x00"), workspace=b"abc",
        )
        self.assertEqual(set(terms), {"action", "start", "end", "kind", "byte"})
        loss.backward()
        for parameter in [g.scratchpad_reader.weight_ih, g.scratchpad_thinker.weight_ih,
                          g.pointer_start.weight, g.edit_byte_head.weight, trace]:
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.norm()), 0.0)

    def test_batched_adaptive_thought_skips_halted_samples(self) -> None:
        g = self.g
        class Confidence(nn.Module):
            def forward(self, features):
                return features[:, 0] * 10
            def probability(self, logits):
                return torch.sigmoid(logits)
            def warn_if_uncalibrated(self, expected_scope=None):
                pass
        class CountCell(nn.Module):
            def __init__(self, cell):
                super().__init__(); self.cell = cell; self.batch_sizes = []
            def forward(self, x, h):
                self.batch_sizes.append(x.shape[0]); return self.cell(x, h)
        g.confidence_head = Confidence()
        g.scratchpad_thinker = CountCell(g.scratchpad_thinker)
        context = torch.zeros(2, 16)
        context[:, 0] = torch.tensor([1.0, -1.0])
        result = g.think_scratchpad(g.scratchpad.expand(2, -1, -1).clone(), context, steps=4, confidence_threshold=0.9)
        self.assertEqual(result.steps.tolist(), [1, 4])
        self.assertEqual(g.scratchpad_thinker.batch_sizes, [16, 8, 8, 8])
        self.assertGreater(result.confidence[0].item(), 0.9)
        self.assertLess(result.confidence[1].item(), 0.1)

    def test_next_byte_calibration_does_not_silently_claim_thought_calibration(self) -> None:
        g = self.g
        g.confidence_head.calibrate(torch.tensor([-2., 2.]), torch.tensor([0., 1.]), scope="next_byte")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            g.think_scratchpad(g.scratchpad.clone(), torch.zeros(1, 16), steps=1, confidence_threshold=0.5)
        self.assertTrue(any("different task" in str(w.message) for w in caught))

    def test_uncalibrated_warning_and_bounds(self) -> None:
        g = self.g
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            g.think_scratchpad(g.scratchpad.clone(), torch.zeros(1, 16), steps=2, confidence_threshold=0.0)
        self.assertTrue(any("uncalibrated" in str(w.message) for w in caught))
        for invalid in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                g.think_scratchpad(g.scratchpad.clone(), torch.zeros(1, 16), confidence_threshold=invalid)
        with self.assertRaises(ValueError):
            g.think_scratchpad(g.scratchpad.clone(), torch.zeros(1, 16), steps=33)


if __name__ == "__main__":
    unittest.main()
