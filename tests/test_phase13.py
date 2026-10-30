import unittest
import torch
from unittest.mock import patch
from data.phase13_examples import contextual_examples, chunk_training_events
from models.chunked_generation import generate_in_chunks
from models.chunked_decoder import generate_diverse_in_chunks
from models.diverse_response import DiverseByteDecoder
from models.conversation import ConversationTurn, pack_conversation


class Phase13Tests(unittest.TestCase):
    def test_context_pairs_and_bounds(self):
        examples = contextual_examples('test')
        self.assertEqual(len(examples), 20)
        for prompt in {x.prompt for x in examples}:
            self.assertEqual(len({x.intent for x in examples if x.prompt == prompt}), 2)
        message = b'a' * 30
        packed = pack_conversation(message, [ConversationTurn(b'x' * 300, b'y')], 64)
        self.assertEqual(packed, b'U:' + message)

    def test_chunk_continuation_and_done(self):
        source = b'abcdefg'
        calls = []
        def generate(prefix, budget):
            calls.append(prefix)
            return source[len(prefix):len(prefix)+budget], len(prefix) + budget >= len(source)
        out = generate_in_chunks(generate, lambda p, c: True, target_length=20, chunk_size=3)
        self.assertEqual(out.output, source)
        self.assertEqual(calls, [b'', b'abc', b'abcdef'])
        self.assertEqual(out.halt_reason, 'done')
        self.assertEqual([x[0] for x in out.events], ['GENERATE_CHUNK', 'THINK'] * 3 + ['DONE'])

    def test_reject_no_progress_and_target(self):
        rejected = generate_in_chunks(lambda p, n: (b'a', False), lambda p, c: False, target_length=2)
        self.assertEqual((rejected.output, rejected.halt_reason), (b'', 'rejected'))
        self.assertEqual(generate_in_chunks(lambda p, n: (b'', False), lambda p, c: True, target_length=2).halt_reason, 'no_progress')
        self.assertEqual(generate_in_chunks(lambda p, n: (b'a' * n, False), lambda p, c: True, target_length=3, chunk_size=2).output, b'aaa')
        with self.assertRaises(ValueError):
            generate_in_chunks(lambda p, n: (b'abcdef', False), lambda p, c: True, target_length=2)

    def test_diverse_prefix_matches_single_pass_greedy(self):
        torch.manual_seed(42)
        decoder = DiverseByteDecoder(context_dim=8, num_styles=2, hidden_dim=8, byte_dim=4, style_dim=4).eval()
        context = torch.randn(1, 8)
        single, _, _ = decoder.generate(context, style=0, max_bytes=30, sample=False)
        chunked = generate_diverse_in_chunks(decoder, context, lambda p, c: True, style=0, target_length=30, chunk_size=7)
        self.assertEqual(single, chunked.output)

    def test_session_routes_with_bounded_history_and_current_focus(self):
        from tests.test_phase12_memory_replay import session
        s = session()
        s.record_episodes = False
        s.history = [ConversationTurn(b'earlier', b'ok'), ConversationTurn(b'recent', b'yes')]
        with patch.object(s.model, 'conversation_features', wraps=s.model.conversation_features) as spy:
            s.respond(b'Thanks!')
        calls = spy.call_args_list
        inputs = [(bytes(c.args[0][0].tolist()), c.kwargs['focus_start'].item()) for c in calls]
        self.assertIn((b'U:earlier\nA:ok\nU:recent\nA:yes\nU:Thanks!', 32), inputs)
        self.assertIn((b'U:Thanks!', 2), inputs)

    def test_training_event_bytes_are_lossless(self):
        data = b'abc\x00def' * 20
        events = chunk_training_events(data, 13)
        self.assertEqual(b''.join(value for kind, value in events if kind == 'GENERATE_CHUNK'), data)
        self.assertEqual(events[-1][0], 'DONE')

if __name__ == '__main__':
    unittest.main()
