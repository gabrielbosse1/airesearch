import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.ingest_real_traces import ingest_real_traces, make_control_trace, optimal_edit_actions
from evaluation.calibrate_confidence import independent_observations, reliability, select_thresholds, write_config
from evaluation.diagnostics_common import chunk_context
from models.alignment import soft_dtw_alignment
from models.bytecore_diffusion import ByteCoreDiffusion
from models.cognitive import action_from_record, action_to_record
from models.generator_conditioned import GeneratorConditioned
from models.memory_model import MemoryModel
from training.co_train import co_train_step, collate, ingest_examples
from training.cognitive_train import cognitive_trace_supervision_loss, validate_action_trace
from training.train import corrupt_edit_batch


def reference_distance(a: bytes, b: bytes) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1, previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def reference_soft_dtw(a: torch.Tensor, b: torch.Tensor, gamma: float) -> torch.Tensor:
    n, m = len(a), len(b)
    matrix = [[a.new_tensor(float("inf")) for _ in range(m+1)] for _ in range(n+1)]
    matrix[0][0] = a.new_zeros(())
    for i in range(1, n+1):
        for j in range(1, m+1):
            prev = torch.stack([matrix[i-1][j], matrix[i][j-1], matrix[i-1][j-1]])
            matrix[i][j] = (a[i-1]-b[j-1]).square().mean() - gamma*torch.logsumexp(-prev/gamma, 0)
    return matrix[n][m]/max(n, m)


class RealTraceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.g = GeneratorConditioned(d_model=16, n_layers=1, n_heads=4, d_ff=32,
                                      max_len=32, max_canvas_bytes=128, dropout=0.0).eval()
        self.m = MemoryModel(d_model=16, capacity=32)

    def test_exact_distance_exhaustive_binary_and_control_trace(self):
        strings = [bytes(x) for n in range(4) for x in itertools.product([0, 255], repeat=n)]
        for a in strings:
            for b in strings:
                distance, actions = make_control_trace(a, b)
                self.assertEqual(distance, reference_distance(a, b))
                self.assertEqual(validate_action_trace(self.g, a, actions), b)
                self.assertEqual(len([x for x in actions if x.kind is not None]), distance)
                self.assertEqual([x.token for x in actions[:2]], [259, 261])
                self.assertEqual([x.token for x in actions[-2:]], [260, 263])
                self.assertEqual([action_from_record(action_to_record(x)) for x in actions], actions)
        self.assertEqual(optimal_edit_actions(b"kitten", b"sitting")[0], 3)
        with self.assertRaises(ValueError):
            optimal_edit_actions(b"abc", b"def", max_cells=4)

    def test_real_trace_co_training_and_persistence(self):
        rows = [{"id": "binary", "input": b"a\x00b", "output": b"a\xff\x00b", "explanation": b"binary",
                 "split": "train", "kind": "code", "source": "test"}]
        prepared, logs = ingest_real_traces(self.m, self.g, rows, epochs=1, batch_size=1, lr=1e-4)
        self.assertEqual(len(logs), 1)
        self.assertGreater(logs[0]["cognitive_trace_loss"], 0)
        self.assertEqual(self.m.execute_procedure(rows[0]["input"]).output, rows[0]["output"])
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "memory.pt")
            self.m.save(path)
            loaded = MemoryModel(d_model=16, capacity=32).load(path)
            result = loaded.execute_procedure(rows[0]["input"])
            self.assertTrue(torch.equal(result.latent_trace.cpu(), prepared[0]["latent_trace"]))
            self.assertEqual(result.explanation_bytes, b"binary")
            self.assertEqual(validate_action_trace(self.g, rows[0]["input"],
                [action_from_record(x) for x in result.metadata["trace_actions"]]), rows[0]["output"])

    def test_invalid_batch_rejected_before_persistent_write(self):
        _, actions = make_control_trace(b"a", b"b")
        valid = {"input": b"a", "output": b"b", "context": {},
                 "trace_actions": [action_to_record(a) for a in actions]}
        invalid = {"input": b"a", "output": b"c", "context": {},
                   "trace_actions": [action_to_record(a) for a in actions]}
        with self.assertRaises(ValueError):
            ingest_examples(self.m, self.g, [valid, invalid], "cpu", epochs=0)
        self.assertEqual(int(self.m.procedural.used.sum()), 0)
        opt_m = torch.optim.Adam(self.m.parameters()); opt_g = torch.optim.Adam(self.g.parameters())
        with self.assertRaises(ValueError):
            co_train_step(self.m, self.g, collate([invalid]), "cpu", opt_m, opt_g, cognitive_trace_weight=1.0)
        self.assertEqual(int(self.m.procedural.used.sum()), 0)

    def test_empty_target_and_action_gradients(self):
        _, actions = make_control_trace(b"\x00", b"")
        self.assertEqual(validate_action_trace(self.g, b"\x00", actions), b"")
        loss = cognitive_trace_supervision_loss(self.g, b"\x00", actions)
        loss.backward()
        self.assertGreater(float(self.g.edit_kind_head.weight.grad.abs().sum()), 0)
        record = {"id": "empty", "input": b"\x00", "output": b"", "explanation": b"",
                  "split": "train", "kind": "binary", "source": "test"}
        ingest_real_traces(self.m, self.g, [record], epochs=0)
        self.assertEqual(self.m.execute_procedure(b"\x00").output, b"")


class AlignmentTests(unittest.TestCase):
    def test_soft_dtw_reference_holey_masks_and_gradcheck(self):
        torch.manual_seed(9)
        a = torch.randn(2, 4, 3, dtype=torch.double, requires_grad=True)
        b = torch.randn(2, 5, 3, dtype=torch.double, requires_grad=True)
        am = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.bool)
        bm = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
        actual = soft_dtw_alignment(a, b, prediction_mask=am, target_mask=bm, gamma=.2)
        expected = (reference_soft_dtw(a[0][am[0]], b[0][bm[0]], .2)
                    + reference_soft_dtw(a[1][am[1]], b[1][bm[1]], .2))/2
        self.assertAlmostEqual(float(actual.detach()), float(expected.detach()), places=10)
        self.assertTrue(torch.autograd.gradcheck(lambda x, y: soft_dtw_alignment(x, y, prediction_mask=am, target_mask=bm, gamma=.2), (a, b), atol=1e-4))
        with self.assertRaises(ValueError):
            soft_dtw_alignment(a, b, prediction_mask=torch.zeros_like(am), target_mask=bm)
        with self.assertRaises(ValueError):
            soft_dtw_alignment(a, b, max_cells=1)

    def test_diffusion_loss_weight_and_finite_gradients(self):
        torch.manual_seed(10)
        m = ByteCoreDiffusion(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=20,
                              diffusion_timesteps=8, concept_dim=16, dropout=0.0)
        ids = torch.randint(0, 256, (2, 8)); clean = m.bytes_to_emb(ids)
        mask = torch.tensor([[1]*6+[0]*2, [1]*8], dtype=torch.bool)
        loss, details = m.compute_loss(clean, valid_mask=mask, corrupted_emb=clean.roll(1, dims=1),
                                       alignment_loss_weight=.25, confidence_weight=.1, return_details=True)
        self.assertTrue(torch.allclose(loss, details["denoising_mse"] + .25*details["alignment_loss"] + .1*details["confidence_bce"]))
        loss.backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()))
        with self.assertRaises(ValueError):
            m.compute_loss(clean, alignment_loss_weight=-.1)

    def test_edit_corruption_preserves_shape_and_input(self):
        import random
        ids = torch.tensor([[0, 1, 2, 3, 0, 0], [255, 0, 8, 9, 10, 11]])
        before = ids.clone()
        result = corrupt_edit_batch(ids, [4, 6], rng=random.Random(11), probability=1, fraction=.25)
        self.assertEqual(result.shape, ids.shape)
        self.assertTrue(torch.equal(ids, before))
        self.assertTrue(torch.equal(result[0, 4:], ids[0, 4:]))


class DiagnosticTests(unittest.TestCase):
    def test_thresholds_are_conservative_and_ties_are_grouped(self):
        x = np.linspace(.99, .01, 100); y = np.ones(100, dtype=bool)
        selected = select_thresholds(x, y)
        self.assertIsNotNone(selected["0.90"]["threshold"])
        self.assertIsNotNone(selected["0.95"]["threshold"])
        self.assertIsNone(selected["0.99"]["threshold"])
        tied = select_thresholds(np.array([.9, .9]), np.array([True, False]))
        self.assertIsNone(tied["0.90"]["threshold"])
        summary = reliability(np.array([0., 1.]), np.array([False, False]), np.array([1., 1.]))
        self.assertEqual(summary["mce"], 1.)

    def test_independent_sequence_observations(self):
        data = {"sample": np.array([0, 0, 1, 1]), "level": np.zeros(4, dtype=int),
                "position": np.array([0, 1, 0, 1]), "correct": np.array([1, 1, 1, 0], dtype=bool)}
        confidence = np.array([.8, .9, .7, .6])
        c, y = independent_observations(data, confidence, kind="diffusion", levels=1)
        np.testing.assert_allclose(c, [.8, .6]); self.assertEqual(y.tolist(), [True, False])
        c, y = independent_observations(data, confidence, kind="generator", levels=1)
        np.testing.assert_allclose(c, [.9, .6]); self.assertEqual(y.tolist(), [True, False])
        # Odd validation IDs must still sample all four noise conditions.
        data = {"sample": np.repeat([1, 3, 5, 7], 4), "level": np.tile(np.arange(4), 4),
                "position": np.zeros(16, dtype=int), "correct": np.ones(16, dtype=bool)}
        c, _ = independent_observations(data, np.tile(np.arange(4)/10, 4), kind="generator", levels=4, level_stride=2)
        np.testing.assert_allclose(c, [0., .1, .2, .3])

    def test_config_update_preserves_later_sections_and_is_idempotent(self):
        import yaml
        thresholds = {name: {target: {"threshold": None} for target in ("0.90", "0.95", "0.99")}
                      for name in ("diffusion", "generator")}
        legacy = ('base: 1\n\n# Phase 5 confidence thresholds (diagnostic-only; adaptive halting stays off).\n'
                  'phase5_calibration:\n  adaptive_halting_enabled: false\n  old: null\n\n'
                  '# unrelated comment\nother:\n  value: 42\n')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"; path.write_text(legacy)
            write_config(path, thresholds)
            first = path.read_text(); self.assertIn('# unrelated comment', first)
            self.assertEqual(yaml.safe_load(first)['other']['value'], 42)
            write_config(path, thresholds)
            self.assertEqual(path.read_text(), first)
            path.write_text('phase5_calibration:\n  user_owned: true\n')
            with self.assertRaises(ValueError):
                write_config(path, thresholds)
            self.assertIn('user_owned', path.read_text())

    def test_causal_context_and_side_effect_free_reads(self):
        torch.manual_seed(13)
        g = GeneratorConditioned(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=20,
                                  retro_chunk_size=4, dropout=0).eval()
        m = MemoryModel(d_model=16, capacity=32)
        m.write(torch.randn(16), {"type": "event"}, tier="episodic")
        x = torch.tensor([[1,2,3,4, 5,6,7,8, 9,10,11,12]])
        changed = x.clone(); changed[:, 4:8] = 99
        before = m.episodic.access_counts.clone()
        a, amask = chunk_context(g, m, x); b, bmask = chunk_context(g, m, changed)
        self.assertTrue(torch.equal(a[:, :2], b[:, :2]))
        self.assertTrue(torch.equal(amask, bmask))
        self.assertTrue(torch.equal(before, m.episodic.access_counts))
        self.assertFalse(bool(amask[:, 0].any()))


if __name__ == "__main__":
    unittest.main()
