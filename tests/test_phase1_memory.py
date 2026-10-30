from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from models.generator_conditioned import GeneratorConditioned
from models.memory_model import MemoryModel


class HierarchicalMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(21)
        self.m = MemoryModel(d_model=16, capacity=24, promotion_threshold=0.5)

    def test_importance_surprise_and_frequency(self) -> None:
        scorer = self.m.importance_scorer
        values = torch.ones(3, 16)
        scores = scorer(torch.tensor([0., 3., 3.]), torch.tensor([0., 0., 20.]),
                        torch.zeros(3), values)
        self.assertLess(float(scores[0].detach()), float(scores[1].detach()))
        self.assertLess(float(scores[1].detach()), float(scores[2].detach()))

    def test_recent_episode_outranks_old_episode(self) -> None:
        m = self.m
        key = torch.ones(16)
        m.episodic.write_slot(key, torch.ones(16), {"id": "old"}, 0, 0.5, m.importance_scorer)
        m.episodic.write_slot(key, torch.ones(16), {"id": "new"}, 200, 0.5, m.importance_scorer)
        m.session_tick = 200
        entries, vectors = m.retrieve(torch.ones(16), top_k=2, tier_filter="episodic")
        self.assertEqual(entries[0]["metadata"]["id"], "new")
        self.assertGreater(entries[0]["weight"], entries[1]["weight"])
        self.assertEqual(tuple(vectors.shape), (2, 16))

    def test_semantic_exemplar_read_preserves_separate_modes(self) -> None:
        m = self.m
        a = torch.zeros(16); a[0] = 1
        b = torch.zeros(16); b[1] = 1
        m.semantic.write_or_merge(a * 30, a, {}, 0, 0.5, m.importance_scorer, cluster_key="modes")
        m.semantic.write_or_merge(b * 30, b, {}, 1, 0.5, m.importance_scorer, cluster_key="modes")
        self.assertEqual(int(m.semantic.exemplar_used.sum()), 2)
        _, _, values, _ = m.semantic.score_active(a.unsqueeze(0) * 20, 1, m.importance_scorer, 0.35)
        actual = F.cosine_similarity(values[0, 0].unsqueeze(0), a.unsqueeze(0)).item()
        centroid = F.cosine_similarity(((a + b) / 2).unsqueeze(0), a.unsqueeze(0)).item()
        self.assertGreater(actual, 0.99)
        self.assertGreater(actual, centroid + 0.2)

    def test_promotion_and_person_filters(self) -> None:
        m = self.m
        slot = m.write(torch.ones(16), {"person": "Alice", "type": "intro"}, prediction_error=4.0)
        m.write(torch.ones(16), {"person": "Bob", "type": "intro"})
        m.episodic.access_counts[slot] = 4
        stats = m.consolidate()
        self.assertGreaterEqual(stats["promoted_to_semantic"], 1)
        entries, vecs = m.retrieve(torch.ones(16), top_k=100, person_filter="Alice")
        self.assertTrue(all(e["metadata"].get("person") == "Alice" for e in entries))
        self.assertEqual(len(entries), vecs.shape[0])
        absent, empty = m.retrieve(torch.ones(16), person_filter="Nobody")
        self.assertEqual(absent, []); self.assertEqual(tuple(empty.shape), (0, 16))

    def test_live_chunked_retrieval_trains_memory_in_every_mode(self) -> None:
        for mode in ("hybrid", "prefix", "cross_attn"):
            with self.subTest(mode=mode):
                memory = MemoryModel(d_model=16, capacity=24)
                for i in range(7):
                    memory.write(torch.randn(16), {"type": "fact" if i % 2 else "intro"}, prediction_error=float(i))
                gen = GeneratorConditioned(d_model=16, n_layers=2, n_heads=4, d_ff=32, max_len=48,
                                           retro_chunk_size=16, injection_mode=mode, dropout=0)
                ids = torch.randint(0, 256, (2, 33))
                logits = gen(ids, memory_model=memory)
                F.cross_entropy(logits.reshape(-1, 264), torch.randint(0, 256, (66,))).backward()
                self.assertGreater(float(memory.query_encoder[0].weight.grad.norm()), 0.0)
                self.assertGreater(float(memory.tier_router.weight.grad.norm()), 0.0)
                self.assertGreater(float(memory.importance_scorer.raw_w_error.grad.abs()), 0.0)

    def test_multivector_end_to_end_gradients(self) -> None:
        m = self.m
        for i in range(6):
            m.write(torch.randn(16), {"type": "intro" if i % 2 else "fact"}, prediction_error=float(i + 1))
        g = GeneratorConditioned(d_model=16, n_layers=2, n_heads=4, d_ff=32, dropout=0)
        ids = torch.randint(0, 256, (2, 6))
        query = g.byte_emb(ids).mean(dim=1)
        vectors, weights, mask = m.retrieve_differentiable(query, top_k=4)
        self.assertEqual(tuple(vectors.shape), (2, 4, 16))
        self.assertEqual(tuple(weights.shape), (2, 4))
        logits = g(ids, vectors, memory_mask=mask)
        F.cross_entropy(logits.reshape(-1, 264), torch.randint(0, 256, (12,))).backward()
        parameters = [m.query_encoder[0].weight, m.tier_router.weight,
                      m.importance_scorer.raw_w_error,
                      g.memory_adapters["0"].cross_attn.q_proj.weight,
                      g.memory_adapters["0"].cross_attn.null_key]
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
