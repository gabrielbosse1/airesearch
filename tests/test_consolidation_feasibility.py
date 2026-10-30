import ast
import unittest
from data.consolidation_examples import greeting_train, greeting_test, session_test, RESPONSE_VARIATIONS
from data.phase13_feasibility import routing_examples, chunk_examples
from models.conversation_intents import ConversationIntent
from evaluation.phase13_eval import distance


class ConsolidationAndFeasibilityTests(unittest.TestCase):
    def test_session_fixture_sizes(self):
        self.assertGreaterEqual(len(greeting_train()), 50)
        self.assertEqual(len(greeting_test()), 20)
        self.assertEqual(len(session_test()), 50)
        self.assertEqual({x[1] for x in session_test()}, set(ConversationIntent))
        self.assertTrue(all(len(v) >= 10 for v in RESPONSE_VARIATIONS.values()))

    def test_history_pairs_and_disjoint_prompts(self):
        train, valid = routing_examples('train'), routing_examples('validation')
        self.assertEqual((len(train), len(valid)), (50, 20))
        self.assertFalse({x[0] for x in train} & {x[0] for x in valid})
        for rows in (train, valid):
            for i in range(0, len(rows), 2):
                self.assertEqual(rows[i][0], rows[i+1][0])
                self.assertNotEqual(rows[i][1], rows[i+1][1])
                self.assertNotEqual(rows[i][2], rows[i+1][2])

    def test_chunk_fixtures_are_valid_and_lossless(self):
        train, valid = chunk_examples('train'), chunk_examples('validation')
        self.assertEqual((len(train), len(valid)), (20, 5))
        self.assertFalse({x[0] for x in train} & {x[0] for x in valid})
        for _, target, events in train + valid:
            self.assertEqual(len(target), 120)
            ast.parse(target)
            chunks = [x for kind, x in events if kind == 'GENERATE_CHUNK']
            self.assertEqual([len(x) for x in chunks], [30] * 4)
            self.assertEqual(b''.join(chunks), target)
            self.assertEqual(distance(target, target), 0)

if __name__ == '__main__': unittest.main()
