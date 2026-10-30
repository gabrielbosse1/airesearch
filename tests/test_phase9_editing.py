"""Regression tests for explicit cognitive edit frames and native mask diffusion."""
from __future__ import annotations

import copy
import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from models.bytecore_diffusion import ByteCoreDiffusion
from models.cognitive import CognitiveAction, apply_edit
from models.cognitive_arguments import decode_edit_arguments, encode_edit_arguments
from models.edit_router import EditRequest, EditRouter
from models.generator_conditioned import CognitivePolicyOutput, GeneratorConditioned
from models.holdout import HoldoutGuard, HoldoutViolation
from data.positional_edits import replacement_record
from training.cognitive_train import cognitive_supervision_loss
from training.rl_editing import autonomous_rollout, preflight_training_examples
from training.train_masked_diffusion import load_model, train as train_masked


class EditArgumentTests(unittest.TestCase):
    def test_short_and_extended_frames_round_trip(self):
        examples = [
            (CognitiveAction(262, start=9, end=10, kind="replace", data=b"d"), bytes([9, 2, 100])),
            (CognitiveAction(262, start=255, kind="insert", data=b"\x00"), bytes([255, 0, 0])),
            (CognitiveAction(262, start=256, end=257, kind="delete"), bytes([0, 129, 0, 1])),
            (CognitiveAction(262, start=300, end=301, kind="replace", data=b"\xff"), bytes([44, 130, 255, 1])),
            (CognitiveAction(262, start=1, end=3, kind="move", destination=8), bytes([1, 67, 0, 2, 8])),
        ]
        for action, frame in examples:
            with self.subTest(action=action):
                self.assertEqual(encode_edit_arguments(action), frame)
                decoded = decode_edit_arguments(frame)
                self.assertEqual(decoded, action)

    def test_random_operations_keep_host_edit_semantics(self):
        rng = random.Random(912)
        source = bytes(rng.randrange(256) for _ in range(400))
        for _ in range(400):
            start = rng.randrange(401)
            kind = rng.choice(("insert", "delete", "replace", "move"))
            end = rng.randrange(start, 401)
            data = bytes(rng.randrange(256) for _ in range(rng.randrange(6))) if kind in ("insert", "replace") else b""
            destinations = list(range(start + 1)) + list(range(end, 401))
            dest = rng.choice(destinations) if kind == "move" else None
            action = CognitiveAction(262, start=start, end=None if kind == "insert" else end,
                                     kind=kind, data=data, destination=dest)
            decoded = decode_edit_arguments(encode_edit_arguments(action))
            self.assertEqual(apply_edit(source, action, max_length=1000),
                             apply_edit(source, decoded, max_length=1000))

    def test_codec_rejects_malformed_or_ambiguous_frames(self):
        bad = [b"", b"\x01\x02", b"\x01\x06x", b"\x01\x82x", b"\x01\x82x\x00",
               b"\x01\x42x\x01", b"\x01\x20x\x01", b"\x01\x01x", b"\x01\x03\x00",
               b"\x01\x22\x00\x00extra", b"\x01\x82x\x81\x00", b"\x01\x02x\x00"]
        for frame in bad:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                decode_edit_arguments(frame)
        with self.assertRaises(ValueError):
            decode_edit_arguments(bytes([0, 130, 100, 1]), max_position=255)
        with self.assertRaises(ValueError):
            encode_edit_arguments(CognitiveAction(262, start=2, end=5, kind="move", destination=3))

    def test_untrained_pointer_can_distinguish_long_positions(self):
        g = GeneratorConditioned(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=16)
        positions = g._pointer_positions(310, device=torch.device("cpu"))
        self.assertTrue(torch.equal(positions[:16], g.pos_emb(torch.arange(16))))
        self.assertFalse(torch.equal(positions[300], positions[301]))

    def test_structured_request_migrates_fails_closed_and_is_not_forced(self):
        g = GeneratorConditioned(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=32, dropout=0.).eval()
        original = g.state_dict()
        legacy = {key: value.clone() for key, value in original.items()
                  if not key.startswith("request_") and not key.startswith("structured_edit_trained")}
        restored = GeneratorConditioned(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=32, dropout=0.).eval()
        with torch.no_grad():
            restored.request_value_proj.weight.fill_(9)
            restored.structured_edit_trained.fill_(True)
        restored.load_state_dict(legacy, strict=True)
        self.assertFalse(bool(restored.structured_edit_trained))
        self.assertEqual(restored.request_value_proj.weight.abs().sum().item(), 0)
        request = bytes([2, 2, 100])
        with self.assertRaises(RuntimeError):
            restored.cognitive_generate(b"abc", structured_request=request)
        restored.mark_structured_edit_trained()
        with self.assertRaises(RuntimeError):
            restored.cognitive_generate(b"abc", structured_request=bytes([1, 1, 0]))
        scratch = restored.scratchpad.clone()
        old = restored.cognitive_policy(b"abc", scratch, loaded=True)
        conditioned = restored.cognitive_policy(b"abc", scratch, loaded=True, structured_request=request)
        self.assertTrue(torch.equal(old.start_logits, conditioned.start_logits))
        self.assertTrue(torch.equal(old.byte_logits, conditioned.byte_logits))
        # The request does not replace logits or edit output by itself.
        self.assertTrue(torch.equal(old.kind_logits, conditioned.kind_logits))
        partial = copy.deepcopy(restored.state_dict()); del partial["request_end_proj.weight"]
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(partial, strict=True)
        partial = copy.deepcopy(restored.state_dict()); del partial["structured_edit_trained_kinds"]
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(partial, strict=True)
        partial = {k: v for k, v in restored.state_dict().items() if not k.startswith("request_")}
        with self.assertRaises(ValueError):
            restored.load_state_dict(partial, strict=True)

    def test_structured_argument_gradients_and_long_coordinates(self):
        g = GeneratorConditioned(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=32, dropout=0.).eval()
        source = b"x" * 300
        edit = CognitiveAction(g.EDIT_OP, start=299, end=300, kind="replace", data=b"z")
        frame = encode_edit_arguments(edit)
        self.assertGreater(len(frame), 3)
        loss, _ = cognitive_supervision_loss(g, source, g.scratchpad.clone(), edit, workspace=source,
                                              last_token=g.THINK, structured_request=frame)
        loss.backward()
        for name in ("request_start_proj", "request_end_proj", "request_value_proj", "request_kind_proj"):
            self.assertGreater(getattr(g, name).weight.grad.abs().sum().item(), 0, name)
        code = g._request_positions(torch.tensor([299, 300]))
        self.assertFalse(torch.equal(code[0], code[1]))

    def test_rl_sampler_replays_structured_request(self):
        g = GeneratorConditioned(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=32, dropout=0.).eval()
        g.mark_structured_edit_trained()
        torch.manual_seed(71)
        rollout = autonomous_rollout(g, b"abcd", max_steps=5, temperature=.8,
                                     structured_request=bytes([2, 2, 100]))
        self.assertTrue(torch.isfinite(rollout.log_probability))
        (-rollout.log_probability).backward()
        self.assertTrue(any(p.grad is not None for p in g.parameters()))

    def test_preflight_audits_structured_request_bytes(self):
        g = GeneratorConditioned(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=32)
        example = replacement_record(b"abc", 1, ord("x"), identifier="train", split="train")
        guard = HoldoutGuard.from_records([{"input": example["structured_request"]}])
        with self.assertRaises(HoldoutViolation):
            preflight_training_examples(g, [example], guard)

    def test_autonomous_actions_are_serialized_and_parsed_before_edit(self):
        g = GeneratorConditioned(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=32).eval()
        def logits(size, index):
            result = torch.full((size,), -100.0); result[index] = 100.0
            return result
        def policy(self, canvas, scratch, *, pointer_bytes=None, loaded=False, dirty=False, last_token=None, **kwargs):
            token = {None: self.LOAD_SPAN, self.LOAD_SPAN: self.EDIT_OP,
                     self.EDIT_OP: self.SAVE_SPAN, self.SAVE_SPAN: self.DONE}[last_token]
            pointer = canvas if pointer_bytes is None else pointer_bytes
            start = 9 if token == self.EDIT_OP else 0
            end = 10 if token in (self.LOAD_SPAN, self.EDIT_OP) else len(pointer)
            return CognitivePolicyOutput(logits(5, token - self.LOAD_SPAN), logits(4, 2), logits(256, 100),
                                         logits(len(pointer) + 1, start), logits(len(pointer) + 1, end),
                                         logits(len(pointer) + 1, 0), torch.zeros(self.d_model), torch.tensor(0.))
        g.cognitive_policy = types.MethodType(policy, g)
        with patch("models.generator_conditioned.decode_edit_arguments", wraps=decode_edit_arguments) as parser:
            result = g.cognitive_generate(b"Hello Worl", max_steps=8)
        self.assertTrue(result.done)
        self.assertEqual(result.canvas, b"Hello Word")
        edits = [event for event in result.events if event.action.token == g.EDIT_OP]
        self.assertEqual(edits[0].argument_bytes, bytes([9, 2, 100]))
        self.assertEqual(edits[0].control_and_arguments, (g.EDIT_OP, 9, 2, 100))
        parser.assert_called_once()


class MaskedDiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.m = ByteCoreDiffusion(d_model=16, n_layers=1, n_heads=4, d_ff=32,
                                   max_len=20, diffusion_timesteps=4, concept_dim=16, dropout=0.).eval()
        self.ids = torch.tensor([[0, 2, 3, 4, 255], [10, 11, 12, 13, 14]])
        self.mask = torch.tensor([[0, 1, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
        self.request = torch.tensor([[250, 100, 249, 0, 248], [90, 91, 92, 93, 94]])

    def test_legacy_checkpoint_migrates_and_incomplete_head_fails(self):
        state = self.m.state_dict()
        x = self.m.bytes_to_emb(self.ids)
        t = torch.zeros(2)
        expected = self.m.core(x, t).detach()
        legacy = {k: v.clone() for k, v in state.items() if k not in
                  {"mask_edit_trained", "core.edit_mask_embedding", "core.edit_value_proj.weight"}}
        with torch.no_grad():
            self.m.core.edit_mask_embedding.fill_(3)
            self.m.core.edit_value_proj.weight.fill_(2)
            self.m.mask_edit_trained.fill_(True)
        self.m.load_state_dict(legacy, strict=True)
        self.assertFalse(bool(self.m.mask_edit_trained))
        self.assertEqual(self.m.core.edit_mask_embedding.abs().sum().item(), 0)
        self.assertEqual(self.m.core.edit_value_proj.weight.abs().sum().item(), 0)
        self.assertTrue(torch.equal(expected, self.m.core(x, t)))
        broken = copy.deepcopy(state)
        broken.pop("core.edit_value_proj.weight")
        with self.assertRaises(RuntimeError):
            self.m.load_state_dict(broken, strict=True)
        missing_head = {k: v for k, v in state.items() if k not in
                        {"core.edit_mask_embedding", "core.edit_value_proj.weight"}}
        with self.assertRaises(ValueError):
            self.m.load_state_dict(missing_head, strict=True)

    def test_mask_is_conditioned_and_outside_requests_are_invisible(self):
        x = self.m.bytes_to_emb(self.ids)
        with torch.no_grad():
            self.m.core.edit_value_proj.weight.copy_(torch.eye(16))
        t = torch.zeros(2)
        a = self.m.core(x, t, edit_mask=self.mask, requested_byte_ids=self.request)
        changed = self.request.clone(); changed[~self.mask] = 1
        b = self.m.core(x, t, edit_mask=self.mask, requested_byte_ids=changed)
        self.assertTrue(torch.equal(a, b))
        changed[0, 1] = 101
        c = self.m.core(x, t, edit_mask=self.mask, requested_byte_ids=changed)
        self.assertFalse(torch.equal(a[0, 1], c[0, 1]))

    def test_sampler_clamps_every_step_but_never_overwrites_masked_prediction(self):
        self.m.mark_mask_edit_trained()
        source = self.m.bytes_to_emb(self.ids)
        calls = []
        def fake(model, xt, t, concept, guidance_scale=1., *, return_confidence=False, edit_mask=None, requested_byte_ids=None):
            calls.append((xt.clone(), edit_mask.clone(), requested_byte_ids.clone()))
            self.assertTrue(torch.equal(xt[~edit_mask], source[:1][~edit_mask]))
            # Intentionally predict 7 despite requested values of 100 and 0.
            pred = model.bytes_to_emb(torch.full(xt.shape[:2], 7, dtype=torch.long))
            var = torch.zeros((xt.shape[0], 1, 1))
            return (pred, var, pred, torch.full(xt.shape[:2], .5)) if return_confidence else (pred, var, pred)
        self.m.p_mean_variance = types.MethodType(fake, self.m)
        output, stats = self.m.masked_edit(self.ids, self.mask, self.request, return_stats=True)
        self.assertEqual(len(calls), 4)
        self.assertEqual(stats.steps.tolist(), [4, 0])
        self.assertTrue(torch.equal(output[~self.mask], self.ids[~self.mask]))
        self.assertEqual(output[0, 1].item(), 7)
        self.assertEqual(output[0, 3].item(), 7)
        self.assertNotEqual(output[0, 1].item(), self.request[0, 1].item())

    def test_untrained_zero_and_invalid_masks(self):
        zeros = torch.zeros_like(self.mask)
        self.assertTrue(torch.equal(self.m.masked_edit(self.ids, zeros, self.request), self.ids))
        with self.assertRaises(RuntimeError):
            self.m.masked_edit(self.ids, self.mask, self.request)
        with self.assertRaises(ValueError):
            self.m.masked_edit(self.ids, self.mask.float() * 2, self.request, allow_untrained_mask=True)
        with self.assertRaises(ValueError):
            self.m.masked_edit(self.ids, self.mask[:, :2], self.request, allow_untrained_mask=True)
        with self.assertRaises(TypeError):
            self.m.masked_edit(self.ids, self.mask, self.request.float(), allow_untrained_mask=True)
        with self.assertRaises(ValueError):
            self.m.p_sample_loop(tuple(self.m.bytes_to_emb(self.ids).shape), edit_mask=self.mask,
                                 allow_untrained_mask=True)

    def test_masked_loss_only_supervises_requested_valid_positions(self):
        source = self.m.bytes_to_emb(self.ids)
        target_ids = torch.where(self.mask, self.request, self.ids)
        target = self.m.bytes_to_emb(target_ids)
        torch.manual_seed(55)
        loss = self.m.compute_loss(target, source_emb=source, edit_mask=self.mask,
                                   requested_byte_ids=self.request, confidence_weight=0)
        loss.backward()
        self.assertGreater(self.m.core.edit_value_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.m.core.edit_mask_embedding.grad.abs().sum().item(), 0)
        changed_target = target.clone(); changed_target[~self.mask] = -55
        changed_request = self.request.clone(); changed_request[~self.mask] = 77
        torch.manual_seed(55)
        loss2 = self.m.compute_loss(changed_target, source_emb=source, edit_mask=self.mask,
                                    requested_byte_ids=changed_request, confidence_weight=0)
        self.assertTrue(torch.allclose(loss.detach(), loss2.detach()))
        with self.assertRaises(ValueError):
            self.m.compute_loss(target, source_emb=source, edit_mask=self.mask, valid_mask=~self.mask)
        with self.assertRaises(ValueError):
            self.m.compute_loss(target, edit_mask=torch.zeros_like(self.mask))
        torch.manual_seed(1)
        _, details = self.m.compute_loss(target, random_edit_mask_probability=.3, return_details=True)
        self.assertGreaterEqual(int(details["masked_positions"]), 2)

    def test_mask_training_preserves_guarded_concept_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base = root / "base"; output = root / "run" / "diffusion"
            processed = root / "processed"; processed.mkdir()
            for split in ("val", "test"):
                (processed / f"{split}.jsonl").write_text('{"bytes": [1, 2, 3]}\n', encoding="utf-8")
            settings = {"vocab_size": 256, "d_model": 16, "n_layers": 1, "n_heads": 4, "d_ff": 32,
                        "max_len": 130, "diffusion_timesteps": 2, "concept_capacity": 4,
                        "concept_dim": 16, "dropout": 0.0}
            config = root / "config.yaml"; config.write_text(yaml.safe_dump({"model": settings}), encoding="utf-8")
            original = ByteCoreDiffusion(**settings, device="cpu").eval()
            original.bank.set_holdout_guard(HoldoutGuard.from_records([]))
            original.save(str(base))
            before = (base / "concept_bank.pt").read_bytes()
            args = types.SimpleNamespace(config=str(config), runs_dir=str(base), pairs="data/real_edit_pairs.jsonl",
                                         processed_dir=str(processed), output_dir=str(output), steps=1,
                                         train_per_length=1, validation_per_length=1, test_per_length=1,
                                         batch_size=2, eval_every=1, lr=.003, seed=6106, threads=1)
            report = train_masked(args)
            self.assertEqual(report["training_updates"], 1)
            self.assertEqual((base / "concept_bank.pt").read_bytes(), before)
            restored = load_model(str(config), str(output))
            self.assertIsNotNone(restored.bank.heldout_guard)
            restored.bank.audit_holdouts()
            self.assertTrue(bool(restored.mask_edit_trained))

    def test_router_explicit_request_and_free_form_rejection(self):
        self.m.mark_mask_edit_trained()
        g = GeneratorConditioned(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=20)
        router = EditRouter(self.m, g, diffusion_pad_to=None)
        result = router.execute(EditRequest(b"abcd", "replace", length_preserving=True,
                                            edit_mask=b"\x00\x01\x00\x00", replacement_bytes=b"\x00z\x00\x00"))
        self.assertEqual((result.engine, result.reason, len(result.output)), ("diffusion", "mask_conditioned_prediction", 4))
        self.assertEqual(result.output[0:1] + result.output[2:], b"acd")
        with self.assertRaises(NotImplementedError):
            router.execute(EditRequest(b"abcd", "replace", length_preserving=True, instruction=b"change b to z"))
        with self.assertRaises(ValueError):
            router.execute(EditRequest(b"abcd", "replace", length_preserving=True, edit_mask=b"\x02" * 4))


if __name__ == "__main__":
    unittest.main()
