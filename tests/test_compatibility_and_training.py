from __future__ import annotations

import math
import types
import unittest
import warnings
from collections import OrderedDict

import torch
import torch.nn.functional as F

from models.bytecore_diffusion import ByteCoreDiffusion
from models.diffusion_core import DiffusionCore
from models.epistemic import ConfidenceHead
from models.generator_conditioned import GeneratorConditioned
from models.memory_model import MemoryModel
from training.co_train import collate, co_train_step, ingest_examples


def legacy_generator_state(model: GeneratorConditioned, v1: bool = False) -> OrderedDict:
    new_prefixes = ("thought_init.", "thought_layers.", "confidence_head.", "scratchpad_reader.",
                    "scratchpad_thinker.", "think_context.", "think_norm.", "pointer_boundary",
                    "cognitive_state_emb.", "pointer_query.", "pointer_start.", "pointer_end.",
                    "pointer_destination.", "edit_kind_head.", "edit_byte_head.")
    state = OrderedDict()
    for key, value in model.state_dict().items():
        if key.startswith(new_prefixes) or (v1 and key.startswith("memory_adapters.")):
            continue
        state[key] = value.clone()
    state["control_emb.weight"] = state["control_emb.weight"][:3].clone()
    state["head.weight"] = state["head.weight"][:259].clone()
    state["head.bias"] = state["head.bias"][:259].clone()
    if v1:
        state["memory_proj.weight"] = model.memory_adapters["0"].prefix_proj.weight.detach().clone()
        state["memory_proj.bias"] = model.memory_adapters["0"].prefix_proj.bias.detach().clone()
        state["memory_gate.weight"] = torch.randn_like(state["memory_proj.weight"])
        state["memory_gate.bias"] = torch.randn_like(state["memory_proj.bias"])
    return state


class CompatibilityAndTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(12)

    def generator(self, mode: str = "hybrid") -> GeneratorConditioned:
        return GeneratorConditioned(d_model=16, n_layers=2, n_heads=4, d_ff=32, max_len=48,
                                    dropout=0.0, retro_chunk_size=16, injection_mode=mode)

    def test_generator_v2_migration_preserves_old_logits_and_strictness(self) -> None:
        original = self.generator().eval()
        legacy = legacy_generator_state(original)
        restored = self.generator().eval()
        restored.load_state_dict(legacy, strict=True)
        ids = torch.tensor([[0, 255, 256, 7, 258]])
        mem = torch.randn(1, 3, 16)
        with torch.no_grad():
            old_logits = original(ids, mem)[..., :259]
            new_logits = restored(ids, mem)[..., :259]
        self.assertTrue(torch.equal(old_logits, new_logits))
        self.assertEqual(tuple(restored.head.weight.shape), (264, 16))
        self.assertTrue(torch.equal(restored.control_emb.weight[:3], legacy["control_emb.weight"]))
        del legacy["blocks.0.linear1.bias"]
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(legacy, strict=True)

    def test_generator_v1_projection_migration(self) -> None:
        original = self.generator()
        legacy = legacy_generator_state(original, v1=True)
        restored = self.generator()
        restored.load_state_dict(legacy, strict=True)
        self.assertTrue(torch.equal(restored.memory_adapters["0"].prefix_proj.weight, legacy["memory_proj.weight"]))
        self.assertTrue(torch.equal(restored.head.weight[:259], legacy["head.weight"]))

    def test_current_checkpoint_does_not_hide_missing_fields(self) -> None:
        g = self.generator()
        state = OrderedDict(g.state_dict()); del state["confidence_head.net.1.weight"]
        with self.assertRaises(RuntimeError):
            self.generator().load_state_dict(state, strict=True)

    def test_live_retro_partial_chunk_never_leaks_future(self) -> None:
        class EchoMemory:
            def retrieve(self, query, top_k=4):
                val = query.unsqueeze(0) * 100
                return [{"value": val[0]}], val
        for mode in ("hybrid", "prefix", "cross_attn"):
            with self.subTest(mode=mode):
                g = self.generator(mode).eval()
                a = torch.randint(0, 256, (1, 33)); b = a.clone()
                b[:, 12:] = (b[:, 12:] + 37) % 256
                with torch.no_grad():
                    out_a = g(a, memory_model=EchoMemory())
                    out_b = g(b, memory_model=EchoMemory())
                self.assertEqual(float((out_a[:, :12] - out_b[:, :12]).abs().max()), 0.0)
                self.assertGreater(float((out_a[:, 16:] - out_b[:, 16:]).abs().max()), 0.0)

    def test_masked_memory_no_nan_and_no_gradient_to_masked_vectors(self) -> None:
        for mode in ("hybrid", "prefix", "cross_attn"):
            with self.subTest(mode=mode):
                g = self.generator(mode)
                x = torch.randint(0, 256, (2, 7))
                mem = torch.randn(2, 3, 16, requires_grad=True)
                mask = torch.tensor([[False, False, False], [True, False, False]])
                y = g(x, mem, memory_mask=mask)
                y.square().mean().backward()
                self.assertTrue(bool(torch.isfinite(y).all() and torch.isfinite(mem.grad).all()))
                self.assertEqual(float(mem.grad[0].abs().max()), 0.0)

    def test_fully_masked_rows_and_chunks_equal_no_memory(self) -> None:
        for mode in ("hybrid", "prefix", "cross_attn"):
            with self.subTest(mode=mode):
                g = self.generator(mode).eval()
                for adapter in g.memory_adapters.values():
                    adapter.cross_attn.out_proj.bias.data.fill_(0.4)
                    adapter.ffn[-2].bias.data.fill_(0.3)
                x = torch.randint(0, 256, (2, 7))
                mem = torch.randn(2, 3, 16)
                mask = torch.tensor([[False, False, False], [True, False, False]])
                with torch.no_grad():
                    mixed = g(x, mem, memory_mask=mask)
                    unconditioned = g(x[:1])
                    conditioned = g(x[1:], mem[1:], memory_mask=mask[1:])
                self.assertTrue(torch.allclose(mixed[:1], unconditioned, atol=1e-7))
                self.assertTrue(torch.allclose(mixed[1:], conditioned, atol=1e-7))
                x = torch.randint(0, 256, (1, 33))
                mem = torch.randn(1, 3, 2, 16)
                mask = torch.tensor([[[False, False], [True, True], [True, True]]])
                with torch.no_grad():
                    out = g(x, mem, memory_mask=mask, memory_chunk_size=16)
                    empty = g(x)
                self.assertTrue(torch.allclose(out[:, :16], empty[:, :16], atol=1e-7))

    def test_generate_masks_cognitive_controls(self) -> None:
        g = self.generator().eval()
        with torch.no_grad():
            g.head.weight.zero_(); g.head.bias.fill_(-100)
            g.head.bias[0] = 20; g.head.bias[g.DONE] = 100
        out, calls = g.generate(torch.tensor([255]), max_new=3, temperature=1)
        self.assertEqual(out.tolist(), [255, 0, 0, 0]); self.assertEqual(calls, [])
        with self.assertRaises(ValueError):
            g.generate(torch.ones(2, 3, dtype=torch.long))

    def test_function_calls_require_terminator_and_respect_model_token_budget(self) -> None:
        g = self.generator().eval()
        def run(tokens, budget):
            sequence = iter(tokens)
            calls = []
            def fake_forward(self, ids, *args, **kwargs):
                logits = torch.full((1, ids.shape[1], self.VOCAB_SIZE), -1000.0)
                logits[0, -1, next(sequence)] = 1000.0
                return logits
            g.forward = types.MethodType(fake_forward, g)
            result, names = g.generate(torch.tensor([65]), max_new=budget,
                                       func_handler=lambda name: calls.append(name) or "ok")
            return result, names, calls
        result, names, calls = run([g.FUNC_CALL, ord("x"), g.FUNC_END], 3)
        self.assertEqual(names, ["x"]); self.assertEqual(calls, ["x"])
        self.assertEqual(result[:4].tolist(), [65, g.FUNC_CALL, ord("x"), g.FUNC_END])
        result, names, calls = run([g.FUNC_CALL, ord("x")], 2)
        self.assertEqual(names, []); self.assertEqual(calls, [])
        self.assertEqual(result.tolist(), [65, g.FUNC_CALL, ord("x")])

    def test_diffusion_legacy_checkpoint_prediction_preserving(self) -> None:
        original = DiffusionCore(d_model=16, n_layers=2, n_heads=4, d_ff=32, max_len=12, dropout=0).eval()
        old = OrderedDict((k, v.clone()) for k, v in original.state_dict().items() if not k.startswith("confidence_head."))
        restored = DiffusionCore(d_model=16, n_layers=2, n_heads=4, d_ff=32, max_len=12, dropout=0).eval()
        restored.load_state_dict(old, strict=True)
        x, t = torch.randn(2, 7, 16), torch.tensor([0.1, 0.4])
        self.assertTrue(torch.equal(original(x, t), restored(x, t)))
        m = ByteCoreDiffusion(d_model=16, n_layers=2, n_heads=4, d_ff=32, max_len=12,
                              diffusion_timesteps=4, concept_dim=16, dropout=0)
        old_parent = OrderedDict((k, v.clone()) for k, v in m.state_dict().items() if not k.startswith("core.confidence_head."))
        m.load_state_dict(old_parent, strict=True)

    def test_confidence_training_and_calibration(self) -> None:
        m = ByteCoreDiffusion(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=12,
                              diffusion_timesteps=4, concept_dim=16, dropout=0)
        ids = torch.tensor([[0, 255, 1, 2], [3, 4, 0, 0]])
        mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
        loss, details = m.compute_loss(m.bytes_to_emb(ids), valid_mask=mask, return_details=True)
        self.assertGreater(float(details["confidence_bce"].detach()), 0)
        loss.backward()
        self.assertGreater(float(m.core.confidence_head.net[-1].weight.grad.norm()), 0)
        head = ConfidenceHead(4, 4)
        logits = torch.tensor([-4., 4., -3., 3., 2., -2.])
        targets = torch.tensor([0., 1., 1., 0., 1., 0.])
        before = F.binary_cross_entropy_with_logits(logits, targets)
        temp = head.calibrate(logits, targets)
        after = F.binary_cross_entropy_with_logits(logits / temp, targets)
        self.assertLessEqual(float(after), float(before) + 1e-6)
        self.assertTrue(bool(head.calibrated))
        head.train()
        self.assertFalse(bool(head.calibrated))
        with self.assertRaises(ValueError):
            head.calibrate(logits, torch.ones_like(targets))

    def test_diffusion_adaptive_halting_reduces_active_batch(self) -> None:
        m = ByteCoreDiffusion(d_model=8, n_layers=1, n_heads=2, d_ff=16, max_len=10,
                              diffusion_timesteps=4, concept_dim=8, dropout=0).eval()
        call_sizes = []
        def fake(self, xt, t_idx, concept, guidance_scale=1.0, *, return_confidence=False):
            call_sizes.append(xt.shape[0])
            mean = torch.zeros_like(xt); var = torch.zeros(xt.shape[0], 1, 1)
            pred = torch.full_like(xt, 7.0)
            if return_confidence:
                conf = concept[:, :1].expand(xt.shape[0], xt.shape[1])
                return mean, var, pred, conf
            return mean, var, pred
        m.p_mean_variance = types.MethodType(fake, m)
        concept = torch.zeros(2, 8); concept[0, 0] = 0.99; concept[1, 0] = 0.01
        with warnings.catch_warnings(record=True):
            out, stats = m.p_sample_loop((2, 3, 8), concept, confidence_threshold=0.9, return_stats=True)
        self.assertEqual(stats.steps.tolist(), [1, 4])
        self.assertEqual(stats.halted_early.tolist(), [True, False])
        self.assertEqual(call_sizes, [2, 1, 1, 1])
        self.assertTrue(torch.equal(out[0], torch.full_like(out[0], 7.0)))
        call_sizes.clear()
        _, stats = m.p_sample_loop((2, 3, 8), concept, return_stats=True)
        self.assertEqual(stats.steps.tolist(), [4, 4])
        self.assertEqual(call_sizes, [2, 2, 2, 2])

    def test_denoising_calibration_warns_for_ancestral_sampling(self) -> None:
        m = ByteCoreDiffusion(d_model=8, n_layers=1, n_heads=2, d_ff=16, max_len=10,
                              diffusion_timesteps=4, concept_dim=8, dropout=0).eval()
        m.core.confidence_head.calibrated.fill_(True)
        m.core.confidence_head.calibration_scope.fill_(m.core.confidence_head.SCOPES["diffusion_denoising"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m.generate(1, 3, confidence_threshold=0.5)
        self.assertTrue(any("different task" in str(w.message) for w in caught))

    def test_cotraining_handles_nul_and_updates_confidence(self) -> None:
        g = self.generator(); m = MemoryModel(d_model=16, capacity=24)
        examples = [
            {"input": [0, 1, 2], "output": [0, 255], "context": {"type": "skill:a"}},
            {"input": [4], "output": [5], "context": {"type": "skill:b"}},
        ]
        batch = collate(examples)
        self.assertTrue(bool(batch["input_mask"][0, 0]))
        self.assertTrue(bool(batch["output_mask"][0, 0]))
        self.assertFalse(bool(batch["output_mask"][1, 1]))
        old = g.confidence_head.net[-1].weight.detach().clone()
        g.confidence_head.calibrated.fill_(True)
        g.eval()
        result = co_train_step(m, g, batch, "cpu", torch.optim.Adam(m.parameters(), lr=1e-3),
                               torch.optim.Adam(g.parameters(), lr=1e-3))
        self.assertTrue(math.isfinite(result["confidence_loss"]))
        self.assertFalse(bool(g.confidence_head.calibrated))
        self.assertFalse(torch.equal(old, g.confidence_head.net[-1].weight))
        self.assertGreater(float(m.query_encoder[0].weight.grad.norm()), 0)
        self.assertGreater(float(m.importance_scorer.raw_w_error.grad.abs()), 0)
        self.assertGreater(float(m.value_encoder[0].weight.grad.norm()), 0)

    def test_ingestion_validates_all_examples_before_writing(self) -> None:
        g = self.generator(); m = MemoryModel(d_model=16, capacity=24, max_pattern_bytes=3)
        examples = [{"input": [1], "output": [2], "context": {}},
                    {"input": [1], "output": [1, 2, 3, 4], "context": {}}]
        with self.assertRaises(ValueError):
            ingest_examples(m, g, examples, device="cpu", epochs=0)
        self.assertEqual(m.tier_stats()["procedural_used"], 0)

    def test_ingestion_installs_replayable_skills(self) -> None:
        g = self.generator(); m = MemoryModel(d_model=16, capacity=24)
        examples = [{"input": [0, 255], "output": [255, 0], "context": {"type": "skill:bytes"}}]
        ingest_examples(m, g, examples, device="cpu", epochs=1, batch_size=1)
        result = m.execute_procedure(b"\x00\xff")
        self.assertEqual(result.output, b"\xff\x00")
        self.assertEqual(result.latent_trace.shape, (2, 16))


if __name__ == "__main__":
    unittest.main()
