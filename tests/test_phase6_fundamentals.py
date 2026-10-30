import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from data.ingest_real_traces import ingest_real_traces
from models.bytecore_diffusion import ByteCoreDiffusion
from models.cognitive import CognitiveAction, action_to_record
from models.concept_bank import ConceptBank
from models.edit_router import EditRequest, EditRouter, route_edit
from models.generator_conditioned import GeneratorConditioned
from models.holdout import HoldoutGuard, HoldoutViolation, content_hash, pair_hash
from models.memory_model import MemoryModel
from training.co_train import co_train_step, collate, ingest_examples


class HoldoutGuardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.held = {"input": b"held-out-source-12345", "output": b"held-out-target-abcde"}
        self.guard = HoldoutGuard.from_records([self.held])
        self.m = MemoryModel(d_model=16, capacity=32)
        self.m.set_holdout_guard(self.guard)
        self.g = GeneratorConditioned(d_model=16, n_layers=1, n_heads=4, d_ff=32,
                                      max_len=32, max_canvas_bytes=128, dropout=0.0).eval()

    def test_hashes_are_framed_and_process_stable(self):
        self.assertNotEqual(pair_hash(b"a", b"bc"), pair_hash(b"ab", b"c"))
        self.assertNotEqual(content_hash(b"a"), pair_hash(b"a", b""))
        script = "from models.holdout import pair_hash; print(pair_hash(b'a', b'bc'))"
        outputs = []
        for seed in ("1", "123"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            outputs.append(subprocess.check_output([sys.executable, "-c", script], env=env, text=True).strip())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], pair_hash(b"a", b"bc"))

    def test_guard_blocks_role_swaps_substrings_metadata_hex_and_opaque_writes(self):
        trace = torch.zeros(1, 16)
        for source, target, metadata in [
            (self.held["input"], b"safe", {}),
            (b"safe", self.held["output"], {}),
            (self.held["output"], b"safe", {}),
            (b"prefix-" + self.held["input"] + b"-suffix", b"safe", {}),
            (b"safe", b"ok", {"nested": {"leak_hex": self.held["output"].hex()}}),
            (b"safe", b"ok", {"split": "test"}),
        ]:
            with self.assertRaises(HoldoutViolation):
                self.m.write_procedure(source, target, trace, metadata=metadata)
        with self.assertRaises(HoldoutViolation):
            self.m.write(torch.randn(16), {"type": "event"})
        with self.assertRaises(HoldoutViolation):
            self.m.write(torch.randn(16), {"type": "event"}, raw_input=self.held["input"])
        with self.assertRaises(HoldoutViolation):
            self.m.write_fact({"subject": self.held["output"], "predicate": b"is", "object": b"x"})
        self.assertEqual(self.m.session_tick, 0)
        self.assertEqual(self.m.audit_holdouts()["checked_slots"], 0)

    def test_unprotected_explicit_split_and_low_level_bypass_rejected(self):
        unguarded = MemoryModel(d_model=16, capacity=32)
        with self.assertRaises(HoldoutViolation):
            unguarded.write_procedure(b"x", b"y", torch.zeros(1, 16), metadata={"split": "validation"})
        z = torch.zeros(16)
        with self.assertRaises(HoldoutViolation):
            self.m.episodic.write_slot(z, z, {}, 0, 0.0, self.m.importance_scorer)
        with self.assertRaises(HoldoutViolation):
            self.m.semantic.write_or_merge(z, z, {}, 0, 0.0, self.m.importance_scorer)
        with self.assertRaises(HoldoutViolation):
            self.m.procedural.write_skill(z, z, z.unsqueeze(0), 1, {}, 0, 0.0, self.m.importance_scorer)

    def test_ingestion_and_cotrain_validate_whole_batch_before_mutation(self):
        good = {"input": b"training", "output": b"result", "context": {"split": "train"}}
        bad = {"input": self.held["input"], "output": b"result", "context": {"split": "train"}}
        before = {k: v.clone() for k, v in self.g.state_dict().items()}
        with self.assertRaises(HoldoutViolation):
            ingest_examples(self.m, self.g, [good, bad], "cpu", epochs=0)
        self.assertEqual(self.m.session_tick, 0)
        om = torch.optim.Adam(self.m.parameters()); og = torch.optim.Adam(self.g.parameters())
        with self.assertRaises(HoldoutViolation):
            co_train_step(self.m, self.g, collate([good, bad]), "cpu", om, og)
        self.assertEqual(self.m.session_tick, 0)
        self.assertFalse(self.g.training)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in self.g.state_dict().items()))
        records = [{"id": "1", "input": b"train", "output": b"ok", "explanation": b"",
                    "split": "train", "kind": "binary", "source": "x"},
                   {"id": "2", "input": self.held["input"], "output": self.held["output"], "explanation": b"",
                    "split": "test", "kind": "binary", "source": "x"}]
        with self.assertRaises(HoldoutViolation):
            ingest_real_traces(self.m, self.g, records)
        self.assertEqual(self.m.session_tick, 0)

    def test_temporary_trace_payload_is_scanned_before_any_batch_write(self):
        good_trace = [CognitiveAction(self.g.LOAD_SPAN, start=0, end=5), CognitiveAction(self.g.DONE)]
        bad_trace = [CognitiveAction(self.g.LOAD_SPAN, start=0, end=4)]
        bad_trace += [CognitiveAction(self.g.EDIT_OP, start=i, kind="insert", data=bytes([byte]))
                      for i, byte in enumerate(self.held["output"])]
        bad_trace += [CognitiveAction(self.g.EDIT_OP, start=0, end=len(self.held["output"]), kind="delete"),
                      CognitiveAction(self.g.SAVE_SPAN), CognitiveAction(self.g.DONE)]
        good = {"input": b"first", "output": b"first", "context": {"split": "train"},
                "trace_actions": [action_to_record(x) for x in good_trace]}
        bad = {"input": b"safe", "output": b"safe", "context": {"split": "train"},
               "trace_actions": [action_to_record(x) for x in bad_trace]}
        with self.assertRaises(HoldoutViolation):
            ingest_examples(self.m, self.g, [good, bad], "cpu", epochs=0)
        self.assertEqual(self.m.session_tick, 0)
        with self.assertRaises(HoldoutViolation):
            self.m.write_procedure(b"safe", b"safe", torch.zeros(1, 16),
                                   metadata={"trace_actions": bad["trace_actions"]})
        self.assertEqual(self.m.session_tick, 0)
        om = torch.optim.Adam(self.m.parameters()); og = torch.optim.Adam(self.g.parameters())
        with self.assertRaises(HoldoutViolation):
            co_train_step(self.m, self.g, collate([good, bad]), "cpu", om, og, cognitive_trace_weight=1.0)
        self.assertEqual(self.m.session_tick, 0)

    def test_guarded_roundtrip_audit_and_read_only(self):
        self.m.write_procedure(b"safe source", b"safe target", torch.zeros(1, 16), metadata={"split": "train"})
        self.m.write(torch.zeros(16), {"type": "event"}, raw_input=b"another safe source")
        self.m.consolidate()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "memory.pt")
            self.m.save(path)
            self.assertEqual(torch.load(path, weights_only=True)["schema_version"], 4)
            loaded = MemoryModel(d_model=16, capacity=32).load(path)
            self.assertIsNotNone(loaded.heldout_guard)
            self.assertEqual(loaded.audit_holdouts()["failed_slots"], 0)
            before = {k: v.clone() for k, v in loaded.state_dict().items()}
            with loaded.read_only():
                loaded.execute_procedure(b"safe source")
                loaded.retrieve(torch.zeros(16))
                with self.assertRaises(HoldoutViolation):
                    loaded.write(torch.zeros(16), {}, raw_input=b"safe")
                with self.assertRaises(HoldoutViolation):
                    loaded.consolidate()
            self.assertTrue(all(torch.equal(before[k], v) for k, v in loaded.state_dict().items()))
            with self.assertRaises(HoldoutViolation):
                loaded.write_procedure(self.held["input"], b"new", torch.zeros(1, 16))
            other = HoldoutGuard.from_records([{"input": b"different", "output": b"target"}])
            other_memory = MemoryModel(d_model=16, capacity=32)
            other_memory.set_holdout_guard(other)
            with self.assertRaises(HoldoutViolation):
                other_memory.load(path)

    def test_old_checkpoint_cannot_enter_guarded_memory_or_mutate_it(self):
        old = MemoryModel(d_model=16, capacity=32)
        old.write_procedure(self.held["input"], self.held["output"], torch.zeros(1, 16))
        with self.assertRaises(HoldoutViolation):
            old.set_holdout_guard(self.guard)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "old.pt"); old.save(path)
            before = {k: v.clone() for k, v in self.m.state_dict().items()}
            with self.assertRaises(HoldoutViolation):
                self.m.load(path)
            self.assertTrue(all(torch.equal(before[k], v) for k, v in self.m.state_dict().items()))
            with self.assertRaises(HoldoutViolation):
                self.m.load_state_dict(old.state_dict())
            with self.assertRaises(HoldoutViolation):
                self.m.episodic.load_state_dict(old.episodic.state_dict())

    def test_guarded_diffusion_checkpoint_cannot_drop_policy(self):
        def build():
            return ByteCoreDiffusion(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=32,
                                      diffusion_timesteps=8, concept_dim=16, dropout=0.0)
        model = build()
        model.bank.set_holdout_guard(self.guard)
        model.write_concept("safe", b"safe example")
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            saved = torch.load(str(Path(tmp) / "concept_bank.pt"), weights_only=True)
            self.assertIn("guarded_state", saved)
            loaded = build().load(tmp)
            self.assertIsNotNone(loaded.bank.heldout_guard)
            with self.assertRaises(HoldoutViolation):
                loaded.write_concept("bad", self.held["input"])
            with self.assertRaises(HoldoutViolation):
                loaded.load_state_dict(model.state_dict())
            with self.assertRaises(HoldoutViolation):
                loaded.bank.load_state_dict(model.bank.state_dict())
            loaded.load(tmp)  # the audited two-file loader remains valid

    def test_concept_guard_and_multiple_slots(self):
        bank = ConceptBank(capacity=2, d=16)
        bank.set_holdout_guard(self.guard)
        v = torch.zeros(16)
        with self.assertRaises(HoldoutViolation):
            bank.write("bad", v, v)
        with self.assertRaises(HoldoutViolation):
            bank.write("bad", v, v, raw_input=self.held["input"])
        bank.write("one", v, v, raw_input=b"one")
        bank.write("two", v + 1, v + 1, raw_input=b"two")
        self.assertEqual(int(bank.used.sum()), 2)
        self.assertNotEqual(bank.name_to_idx["one"], bank.name_to_idx["two"])
        bank.write("three", v + 2, v + 2, raw_input=b"three")
        self.assertNotIn("one", bank.name_to_idx)
        self.assertEqual(len(bank.name_to_idx), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "bank.pt"); bank.save(path)
            restored = ConceptBank(capacity=2, d=16).load(path)
            self.assertIsNotNone(restored.heldout_guard)
            with self.assertRaises(HoldoutViolation):
                restored.write("bad", v, v, raw_input=self.held["output"])
        model = ByteCoreDiffusion(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=20,
                                  diffusion_timesteps=8, concept_dim=16, dropout=0.0)
        model.bank.set_holdout_guard(self.guard)
        with self.assertRaises(HoldoutViolation):
            model.write_concept("bad", list(self.held["input"]))
        self.assertEqual(int(model.bank.used.sum()), 0)


class EditAndHaltingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.d = ByteCoreDiffusion(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_len=32,
                                   diffusion_timesteps=8, concept_dim=16, dropout=0.0).eval()
        self.g = GeneratorConditioned(d_model=16, n_layers=1, n_heads=4, d_ff=32,
                                      max_len=32, max_canvas_bytes=128, dropout=0.0).eval()

    def test_explicit_route_and_no_silent_instruction_loss(self):
        for op in ("insert", "delete", "move"):
            self.assertEqual(route_edit(EditRequest(b"abc", op)), "cognitive")
        for op in ("denoise", "substitute"):
            self.assertEqual(route_edit(EditRequest(b"abc", op)), "diffusion")
        self.assertEqual(route_edit(EditRequest(b"abc", "replace", True)), "diffusion")
        self.assertEqual(route_edit(EditRequest(b"abc", "replace", False)), "cognitive")
        with self.assertRaises(ValueError):
            route_edit(EditRequest(b"abc", "replace"))
        router = EditRouter(self.d, self.g, diffusion_pad_to=30)
        result = router.execute(EditRequest(b"abc", "substitute"))
        self.assertEqual(result.engine, "diffusion")
        self.assertEqual(len(result.output), 3)
        with self.assertRaises(NotImplementedError):
            router.execute(EditRequest(b"abc", "substitute", instruction=b"replace a with z"))

    def test_conditioned_initial_state_and_thought_steps(self):
        initial = torch.randn(1, 3, 16)
        torch.manual_seed(31)
        a, stats_a = self.d.p_sample_loop(tuple(initial.shape), guidance_scale=1.0, initial_xt=initial, return_stats=True)
        torch.manual_seed(31)
        b, stats_b = self.d.p_sample_loop(tuple(initial.shape), guidance_scale=1.0, initial_xt=initial,
                                         confidence_threshold=1.0, return_stats=True)
        self.assertTrue(torch.equal(a, b)); self.assertTrue(torch.equal(stats_a.steps, stats_b.steps))
        actions = [CognitiveAction(self.g.LOAD_SPAN, start=0, end=3),
                   CognitiveAction(self.g.THINK, think_steps=4), CognitiveAction(self.g.DONE)]
        result = self.g.cognitive_generate(b"abc", controller=lambda obs: actions[obs.step],
                                           instruction_bytes=b"inspect", max_steps=3, autonomous_think_steps=4)
        self.assertTrue(result.done)
        self.assertEqual(result.canvas, b"abc")
        self.assertEqual(result.events[1].thought_steps, 4)

    def test_configured_halting_requires_matching_calibration(self):
        disabled = {"enabled": False, "confidence_threshold": None}
        torch.manual_seed(8)
        _, stats = self.d.generate(1, 3, guidance_scale=1.0, return_stats=True, adaptive_config=disabled)
        self.assertEqual(int(stats.steps[0]), self.d.diffusion_timesteps)
        enabled = {"enabled": True, "confidence_threshold": 1.0, "min_steps": 1}
        with self.assertRaises(ValueError):
            self.d.generate(1, 3, adaptive_config=enabled)
        with self.assertRaises(ValueError):
            self.g.cognitive_generate(b"abc", adaptive_config={"enabled": True, "confidence_threshold": .9})
        self.d.core.confidence_head.calibrated.fill_(True)
        self.d.core.confidence_head.calibration_scope.fill_(self.d.core.confidence_head.SCOPES["diffusion_sampling"])
        _, stats = self.d.generate(1, 3, guidance_scale=1.0, return_stats=True, adaptive_config=enabled)
        self.assertEqual(int(stats.steps[0]), self.d.diffusion_timesteps)
        self.g.confidence_head.calibrated.fill_(True)
        self.g.confidence_head.calibration_scope.fill_(self.g.confidence_head.SCOPES["thought"])
        actions = [CognitiveAction(self.g.LOAD_SPAN, start=0, end=3),
                   CognitiveAction(self.g.THINK, think_steps=4), CognitiveAction(self.g.DONE)]
        configured = {"enabled": True, "confidence_threshold": 1.0,
                      "min_thought_steps": 1, "max_thought_steps": 4}
        result = self.g.cognitive_generate(b"abc", controller=lambda obs: actions[obs.step],
                                           max_steps=3, adaptive_config=configured)
        self.assertEqual(result.events[1].thought_steps, 4)
        with self.assertRaises(ValueError):
            self.g.cognitive_generate(b"abc", confidence_threshold=.9, adaptive_config=configured)

    def test_empty_targets_are_protected_without_blocking_empty_metadata(self):
        guard = HoldoutGuard.from_records([{"input": b"nonempty-heldout", "output": b""}])
        guard.validate(b"safe", b"other", metadata={"explanation": b"", "data_hex": ""}, extra=(b"",))
        with self.assertRaises(HoldoutViolation):
            guard.validate(b"safe", b"")

    def test_alignment_defaults_are_off(self):
        config = yaml.safe_load(Path("configs/default.yaml").read_text())
        self.assertEqual(config["training"]["alignment_loss_weight"], 0.0)
        self.assertEqual(config["training"]["edit_noise_probability"], 0.0)


if __name__ == "__main__":
    unittest.main()
