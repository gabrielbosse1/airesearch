from __future__ import annotations

import os
import tempfile
import unittest
from collections import OrderedDict

import torch

from models.contradiction_graph import Fact
from models.cognitive import CognitiveAction
from models.generator_conditioned import GeneratorConditioned
from models.memory_model import MemoryModel
from training.graph_train import graph_consistency_loss


class StructuredMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.m = MemoryModel(d_model=16, capacity=24, max_trace_steps=4,
                             max_pattern_bytes=16, max_explanation_bytes=16)

    def test_exact_procedure_replay_and_latent_order(self) -> None:
        m = self.m
        trace = torch.arange(48, dtype=torch.float32).reshape(3, 16)
        slot = m.write_procedure(b"\x00\xffin", b"\xff\x00out", trace, b"\x00why", metadata={"name": "raw"})
        seen = []
        result = m.execute_procedure(b"\x00\xffin", latent_step_handler=seen.append)
        self.assertEqual(result.slot, slot)
        self.assertEqual(result.output, b"\xff\x00out")
        self.assertEqual(result.explanation_bytes, b"\x00why")
        self.assertTrue(torch.equal(result.latent_trace, trace))
        self.assertTrue(torch.equal(torch.stack(seen), trace))
        read = m.read_procedure(slot)
        self.assertEqual(read.input_pattern, b"\x00\xffin")
        self.assertEqual(read.metadata["name"], "raw")
        with self.assertRaises(LookupError):
            m.execute_procedure(b"almost")
        with self.assertRaises(LookupError):
            m.execute_procedure(b"almost", slot=slot)

    def test_procedure_trace_primes_cognitive_scratchpad(self) -> None:
        trace = torch.randn(2, 16)
        self.m.write_procedure(b"in", b"\\x00out\\xff", trace)
        execution = self.m.execute_procedure(b"in")
        g = GeneratorConditioned(d_model=16, n_layers=1, n_heads=4, d_ff=32, dropout=0)
        result = g.cognitive_generate(execution.output, initial_trace=execution.latent_trace,
                                      controller=lambda _: CognitiveAction(g.DONE))
        self.assertEqual(result.canvas, b"\\x00out\\xff")
        self.assertGreater(float(result.scratchpad.norm()), 0.0)
        self.assertEqual(int(torch.count_nonzero(g.scratchpad)), 0)

    def test_empty_duplicate_and_bounds(self) -> None:
        m = self.m
        m.write_procedure(b"", b"first", torch.empty(0, 16))
        newest = m.write_procedure(b"", b"second", torch.empty(0, 16))
        result = m.execute_procedure(b"")
        self.assertEqual(result.slot, newest)
        self.assertEqual(result.output, b"second")
        self.assertEqual(result.latent_trace.shape, (0, 16))
        with self.assertRaises(ValueError):
            m.write_procedure(b"x" * 17, b"", torch.zeros(1, 16))
        with self.assertRaises(ValueError):
            m.write_procedure(b"", b"", torch.zeros(5, 16))
        with self.assertRaises(ValueError):
            m.write_procedure(b"", b"", torch.full((1, 16), float("nan")))
        with self.assertRaises(TypeError):
            m.write_procedure("x", b"", torch.zeros(1, 16))  # type: ignore[arg-type]

    def test_eviction_clears_old_executable_data(self) -> None:
        m = MemoryModel(d_model=16, tier_capacities=(1, 1, 1), max_pattern_bytes=16)
        old = m.write_procedure(b"secret", b"old", torch.ones(1, 16))
        new = m.write(torch.randn(16), {"type": "skill:new"})
        self.assertEqual(old, new)
        with self.assertRaises(LookupError):
            m.execute_procedure(b"secret")
        with self.assertRaises(LookupError):
            m.read_procedure(new)
        self.assertEqual(int(m.procedural.input_bytes.sum()), 0)

    def test_schema3_roundtrip_retains_procedure_and_graph(self) -> None:
        m = self.m
        slot = m.write_procedure(b"\x00\xff", b"yes", torch.randn(2, 16), b"why")
        m.write_fact(Fact(b"a", b"same_as", b"b"))
        m.write_fact(Fact(b"b", b"color", b"red", exclusive=True, valid_from=5, valid_to=10))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "memory.pt")
            m.save(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["schema_version"], 3)
            restored = MemoryModel(d_model=16, capacity=24, max_trace_steps=4,
                                   max_pattern_bytes=16, max_explanation_bytes=16).load(path)
            self.assertEqual(restored.execute_procedure(b"\x00\xff").slot, slot)
            self.assertEqual(restored.explain_contradiction(Fact(b"a", b"color", b"blue"), as_of=7).status, "conflict")
            self.assertEqual(restored.explain_contradiction(Fact(b"a", b"color", b"blue"), as_of=12).status, "unknown")

    def test_v2_pair_scorer_and_buffers_migrate_strictly(self) -> None:
        m = self.m
        v2 = OrderedDict()
        new_proc = {"structured", "input_bytes", "output_bytes", "explanation_bytes", "input_lens", "output_lens", "explanation_lens"}
        for key, value in m.state_dict().items():
            if key.startswith("procedural.") and key.split(".", 1)[1] in new_proc:
                continue
            if key.startswith("contradiction_scorer.pair_scorer."):
                key = key.replace("contradiction_scorer.pair_scorer.", "contradiction_scorer.")
            elif key.startswith("contradiction_scorer."):
                continue
            v2[key] = value.clone()
        restored = MemoryModel(d_model=16, capacity=24, max_pattern_bytes=16, max_explanation_bytes=16)
        restored.load_state_dict(v2, strict=True)
        self.assertTrue(torch.equal(restored.contradiction_scorer.pair_scorer[0].weight, m.contradiction_scorer.pair_scorer[0].weight))
        self.assertFalse(bool(restored.procedural.structured.any()))
        del v2["key_encoder.0.weight"]
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(v2, strict=True)

    def test_v1_flat_checkpoint_migration(self) -> None:
        m = self.m
        state = {name: getattr(m, name).state_dict() for name in ("key_encoder", "value_encoder", "query_encoder", "context_proj", "person_linker")}
        state["contradiction_scorer"] = m.contradiction_scorer.pair_scorer.state_dict()
        state.update(keys=torch.randn(2, 16), values=torch.randn(2, 16),
                     used=torch.tensor([True, True]), access_counts=torch.tensor([2, 1]),
                     last_access=torch.tensor([1, 2]), metadata=[{"type": "fact"}, {"type": "skill:test"}], session_tick=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v1.pt"); torch.save(state, path)
            restored = MemoryModel(d_model=16, capacity=24, max_pattern_bytes=16, max_explanation_bytes=16).load(path)
            self.assertEqual(restored.tier_stats()["semantic_slots_used"], 1)
            self.assertEqual(restored.tier_stats()["procedural_used"], 1)
            self.assertTrue(torch.equal(restored.contradiction_scorer.pair_scorer[0].weight, m.contradiction_scorer.pair_scorer[0].weight))

    def test_direct_positive_negative_and_exclusivity(self) -> None:
        m = self.m
        m.write_fact(Fact(b"alice", b"lives", b"paris", exclusive=True, valid_from=0, valid_to=10))
        m.write_fact(Fact(b"alice", b"likes", b"coffee"))
        self.assertEqual(m.explain_contradiction(Fact(b"alice", b"lives", b"paris"), as_of=5).status, "support")
        self.assertEqual(m.explain_contradiction(Fact(b"alice", b"lives", b"paris", polarity=False), as_of=5).status, "conflict")
        self.assertEqual(m.explain_contradiction(Fact(b"alice", b"lives", b"rome"), as_of=5).status, "conflict")
        self.assertEqual(m.explain_contradiction(Fact(b"alice", b"lives", b"rome"), as_of=10).status, "unknown")
        self.assertEqual(m.explain_contradiction(Fact(b"alice", b"likes", b"tea")).status, "unknown")
        self.assertEqual(m.explain_contradiction(Fact(b"bob", b"lives", b"rome"), as_of=5).status, "unknown")

    def test_alias_paths_respect_hops_and_temporal_intersections(self) -> None:
        m = self.m
        a = m.write_fact(Fact(b"a", b"same_as", b"b", valid_from=0, valid_to=20))
        b = m.write_fact(Fact(b"b", b"same_as", b"c", valid_from=2, valid_to=12))
        c = m.write_fact(Fact(b"c", b"color", b"red", exclusive=True, valid_from=5, valid_to=15))
        claim = Fact(b"a", b"color", b"blue")
        report = m.explain_contradiction(claim, as_of=7, max_hops=3)
        self.assertEqual(report.status, "conflict")
        self.assertEqual(report.evidence[0].path_slots, (a, b, c))
        self.assertEqual((report.evidence[0].valid_from, report.evidence[0].valid_to), (5.0, 12.0))
        self.assertEqual(m.explain_contradiction(claim, as_of=12).status, "unknown")
        self.assertEqual(m.explain_contradiction(claim, as_of=7, max_hops=2).status, "unknown")
        self.assertEqual(m.explain_contradiction(Fact(b"a", b"color", b"blue", valid_from=12, valid_to=14)).status, "unknown")
        self.assertTrue(m.explain_contradiction(claim, as_of=7, max_hops=3, max_states=1).truncated)

    def test_transitive_inference_is_explicit(self) -> None:
        m = self.m
        a = m.write_fact(Fact(b"ann", b"ancestor", b"beth", transitive=True))
        b = m.write_fact(Fact(b"beth", b"ancestor", b"cory", transitive=True))
        claim = Fact(b"ann", b"ancestor", b"cory", polarity=False)
        report = m.explain_contradiction(claim, max_hops=2)
        self.assertEqual(report.status, "conflict")
        self.assertIn((a, b), [e.path_slots for e in report.evidence])
        m2 = MemoryModel(d_model=16, capacity=12)
        m2.write_fact(Fact(b"ann", b"knows", b"beth"))
        m2.write_fact(Fact(b"beth", b"knows", b"cory"))
        self.assertEqual(m2.explain_contradiction(Fact(b"ann", b"knows", b"cory", polarity=False)).status, "unknown")

    def test_graph_message_passing_gradients(self) -> None:
        m = self.m
        with torch.no_grad():
            m.contradiction_scorer.message_gate.fill_(0.5)
            m.contradiction_scorer.time_gate.fill_(0.5)
        query = torch.randn(16, requires_grad=True)
        values = torch.randn(3, 16, requires_grad=True)
        adjacency = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        age = torch.tensor([1., 5., 10.])
        out = m.score_graph(query, values, adjacency, age, hops=2)
        no_links = m.score_graph(query, values, torch.zeros_like(adjacency), age, hops=2)
        self.assertFalse(torch.allclose(out, no_links))
        out.sum().backward()
        for p in (query, values, m.contradiction_scorer.message_mlp[0].weight,
                  m.contradiction_scorer.time_mlp[0].weight, m.contradiction_scorer.pair_scorer[0].weight):
            self.assertIsNotNone(p.grad)
            self.assertGreater(float(p.grad.norm()), 0.0)

    def test_supervised_graph_loss_trains_reasoner_and_query_encoder(self) -> None:
        m = self.m
        values = torch.randn(3, 16)
        adjacency = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        loss, scores = graph_consistency_loss(m, torch.randn(16), values, adjacency,
                                               torch.tensor([1., 2., 3.]), torch.tensor([1., 0., 1.]))
        self.assertEqual(tuple(scores.shape), (3,))
        loss.backward()
        self.assertGreater(float(m.value_encoder[0].weight.grad.norm()), 0.0)
        self.assertGreater(float(m.contradiction_scorer.message_gate.grad.abs()), 0.0)
        self.assertGreater(float(m.contradiction_scorer.time_gate.grad.abs()), 0.0)
        with self.assertRaises(ValueError):
            graph_consistency_loss(m, torch.randn(16), values, adjacency, torch.ones(3), torch.tensor([2., 0., 1.]))

    def test_fact_validation(self) -> None:
        with self.assertRaises(ValueError):
            Fact(b"a", b"p", b"b", valid_from=10, valid_to=10)
        with self.assertRaises(ValueError):
            Fact(b"a", b"p", b"b", polarity="false")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.m.explain_contradiction(Fact(b"a", b"p", b"b"), as_of=float("nan"))


if __name__ == "__main__":
    unittest.main()
