"""Phase 15 tests: tool vocabulary, frame safety, sandbox bounds and the loop."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

from data.tool_use_examples import debugging_tasks, routing_prompts, tool_examples
from demo.tool_agent import ToolAgent
from models.generator_conditioned import GeneratorConditioned
from models.sandbox import ByteSandbox, looks_failed
from models.tool_use import (MAX_ARGUMENT_BYTES, ROUTE_NAMES, TOOL_NAMES, ToolArgumentDecoder,
                             ToolCall, ToolCallPolicy, ToolFrameError, ToolFrameTruncated,
                             encode_tool_call, encode_tool_result, find_tool_call,
                             load_tool_policy, pack_tool_request, parse_tool_call, sanitize_result)
from training.train_tool_use import argument_tensors, batch_loss, cache_rows


def small_generator() -> GeneratorConditioned:
    return GeneratorConditioned(d_model=16, n_layers=2, n_heads=4, d_ff=32, max_len=48,
                                dropout=0.0, retro_chunk_size=16)


class ToolVocabularyTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1501)
        self.g = small_generator()

    def test_control_ids_are_internal_and_vocabulary_grew(self) -> None:
        g = self.g
        self.assertEqual((g.TOOL_CALL, g.TOOL_ARG_START, g.TOOL_ARG_END, g.TOOL_RESULT), (264, 265, 266, 267))
        self.assertEqual(g.VOCAB_SIZE, 268)
        self.assertEqual(g.control_emb.num_embeddings, 12)
        self.assertEqual(g.head.weight.shape[0], 268)
        self.assertGreaterEqual(min(g.TOOL_CALL, g.TOOL_ARG_START, g.TOOL_ARG_END, g.TOOL_RESULT), 256)
        with self.assertRaises(ValueError):
            g(torch.tensor([[268]]))

    def test_pre_tool_checkpoint_migrates_without_changing_old_predictions(self) -> None:
        g = self.g.eval()
        legacy = OrderedDict(g.state_dict())
        legacy["control_emb.weight"] = legacy["control_emb.weight"][:8].clone()
        legacy["head.weight"] = legacy["head.weight"][:264].clone()
        legacy["head.bias"] = legacy["head.bias"][:264].clone()
        restored = small_generator().eval()
        restored.load_state_dict(legacy, strict=True)
        ids = torch.tensor([[0, 255, 256, 7, 258, 262, 263]])
        with torch.no_grad():
            old = g(ids)[..., :264]
            new = restored(ids)[..., :264]
        self.assertTrue(torch.equal(old, new))
        # Untrained tool controls must never win an argmax on a legacy model.
        with torch.no_grad():
            tool_logits = restored(ids)[0, -1, 264:]
        self.assertTrue(bool((tool_logits < old[0, -1].max()).all()))
        # The caller's tensors are never mutated by migration.
        self.assertEqual(legacy["head.bias"].shape[0], 264)
        self.assertEqual(legacy["control_emb.weight"].shape[0], 8)

    def test_migration_is_deterministic(self) -> None:
        g = self.g.eval()
        legacy = OrderedDict(g.state_dict())
        legacy["head.weight"] = legacy["head.weight"][:264].clone()
        legacy["head.bias"] = legacy["head.bias"][:264].clone()
        legacy["control_emb.weight"] = legacy["control_emb.weight"][:8].clone()
        first, second = small_generator().eval(), small_generator().eval()
        first.load_state_dict(OrderedDict(legacy), strict=True)
        second.load_state_dict(OrderedDict(legacy), strict=True)
        self.assertTrue(torch.equal(first.head.bias, second.head.bias))
        self.assertTrue(torch.equal(first.control_emb.weight, second.control_emb.weight))

    def test_shrinking_or_corrupt_vocabulary_still_fails_strict(self) -> None:
        g = self.g
        state = OrderedDict(g.state_dict())
        del state["head.bias"]
        with self.assertRaises(RuntimeError):
            small_generator().load_state_dict(state, strict=True)


class ToolFrameTests(unittest.TestCase):
    def test_round_trip_and_token_ids(self) -> None:
        call = ToolCall(b"exec_python", b"print(5*5)")
        frame = encode_tool_call(call)
        self.assertEqual(frame[0], 264)
        self.assertEqual(frame[12], 265)
        self.assertEqual(frame[-1], 266)
        self.assertTrue(all(0 <= value <= 255 for value in frame[1:12] + frame[13:-1]))
        self.assertEqual(parse_tool_call(frame), call)
        self.assertEqual(find_tool_call([3, 9, 7] + frame + [1])[0], call)

    def test_incomplete_frames_are_never_dispatchable(self) -> None:
        with self.assertRaises(ToolFrameTruncated):
            parse_tool_call([264] + list(b"exec_python") + [265] + list(b"print(1"))
        with self.assertRaises(ToolFrameError):
            parse_tool_call([264] + list(b"exec_python"))
        self.assertIsNone(find_tool_call([264] + list(b"exec_python") + [265] + list(b"print(1")))

    def test_control_bytes_inside_arguments_are_rejected(self) -> None:
        forged = [264] + list(b"exec_python") + [265] + list(b"print(1)") + [264] + list(b"exec_python") + [266]
        with self.assertRaises(ToolFrameError):
            parse_tool_call(forged)

    def test_unknown_tool_and_out_of_range_ids_are_rejected(self) -> None:
        with self.assertRaises(ToolFrameError):
            parse_tool_call([264] + list(b"rm_rf") + [265] + list(b"/") + [266])
        with self.assertRaises(ToolFrameError):
            parse_tool_call([264, 300, 265, 266])
        with self.assertRaises(ToolFrameError):
            parse_tool_call([264] + list(b"exec_python") + [265] + list(b"x" * (MAX_ARGUMENT_BYTES + 1)) + [266])
        with self.assertRaises(ToolFrameError):
            ToolCall(b"exec_python", b"x" * (MAX_ARGUMENT_BYTES + 1))

    def test_result_frames_carry_only_raw_bytes(self) -> None:
        encoded = encode_tool_result(b"25\n")
        self.assertEqual(encoded[0], 267)
        self.assertTrue(all(0 <= value <= 255 for value in encoded[1:]))
        self.assertEqual(encode_tool_result(b"x" * 10000)[1:], list(b"x" * 4096))

    def test_packing_keeps_the_informative_tail(self) -> None:
        request = b"Run this: print(1)"
        packed, focus = pack_tool_request(request, None, max_len=48)
        self.assertEqual(packed, b"T:" + request)
        self.assertEqual(focus, len(b"T:"))
        long_result = b"HEAD" + b"x" * 200 + b"SyntaxError: invalid syntax"
        packed_with_result, _ = pack_tool_request(request, long_result, max_len=128)
        self.assertLessEqual(len(packed_with_result), 128)
        self.assertTrue(packed_with_result.endswith(b"SyntaxError: invalid syntax"))
        self.assertTrue(packed_with_result.startswith(b"T:"))
        with self.assertRaises(ValueError):
            pack_tool_request(b"", None, max_len=48)
        with self.assertRaises(ValueError):
            pack_tool_request(b"x" * 200, None, max_len=48)

    def test_sanitize_keeps_tail_and_marks_truncation(self) -> None:
        self.assertEqual(sanitize_result(b"short", limit=16), b"short")
        trimmed = sanitize_result(b"y" * 100, limit=16)
        self.assertTrue(trimmed.startswith(b"..."))
        self.assertEqual(len(trimmed), 16)


class SandboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp(prefix="phase15-sandbox-")
        self.sandbox = ByteSandbox(self.directory, timeout=5.0)

    def tearDown(self) -> None:
        self.sandbox.close()

    def test_execute_python_returns_stdout_and_stderr(self) -> None:
        output = self.sandbox.call(b"exec_python", b"print(5*5)")
        self.assertIn(b"25", output)
        self.assertIn(b"---STDERR---", output)
        failed = self.sandbox.call(b"exec_python", b"pritn('hi')")
        self.assertIn(b"NameError", failed)

    def test_timeout_is_enforced(self) -> None:
        sandbox = ByteSandbox(self.directory, timeout=1.0)
        self.assertEqual(sandbox.call(b"exec_python", b"while True: pass"), b"ERROR: Execution timed out.")

    def test_paths_cannot_escape_the_root(self) -> None:
        for argument in (b"../outside.txt", b"/etc/hostname", b"a/../../b.txt"):
            self.assertIn(b"ERROR", self.sandbox.call(b"read_file", argument))
            self.assertIn(b"ERROR", self.sandbox.call(b"write_file", argument + b"\nhi"))
        self.assertTrue(Path(self.directory).is_dir())

    def test_write_then_read_round_trip(self) -> None:
        self.assertIn(b"OK: wrote", self.sandbox.call(b"write_file", b"notes.txt\nhello sandbox"))
        self.assertEqual(self.sandbox.call(b"read_file", b"notes.txt"), b"hello sandbox")
        self.assertIn(b"ERROR", self.sandbox.call(b"read_file", b"missing.txt"))

    def test_unknown_tools_and_disabled_sandbox_refuse(self) -> None:
        self.assertIn(b"ERROR", self.sandbox.call(b"rm_everything", b"/"))
        disabled = ByteSandbox(self.directory, enabled=False)
        self.assertEqual(disabled.call(b"exec_python", b"print(1)"), b"ERROR: sandbox execution is disabled.")

    def test_code_runs_inside_the_sandbox_directory(self) -> None:
        output = self.sandbox.call(b"exec_python", b"import os\nopen('made.txt','w').write('x')\nprint(sorted(os.listdir()))")
        self.assertIn(b"made.txt", output)
        self.assertTrue((Path(self.directory) / "made.txt").exists())


class ToolHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1502)
        self.generator = small_generator()
        self.policy = ToolCallPolicy(self.generator)

    def test_heads_accept_and_reject_shapes(self) -> None:
        policy = self.policy
        context, memory, mask, ids = policy.hidden_states([b"Calculate 5 * 5."], [None])
        self.assertEqual(context.shape, (1, self.generator.d_model))
        route = torch.tensor([1])
        previous = torch.tensor([[ToolArgumentDecoder.START, 32]])
        logits = policy.argument_decoder(context, route, previous, memory, mask, ids)
        self.assertEqual(logits.shape, (1, 2, 257))
        with self.assertRaises(ValueError):
            policy.argument_decoder(context, torch.tensor([99]), previous, memory, mask, ids)
        with self.assertRaises(ValueError):
            policy.route_head(context[:, :4])

    def test_generation_emits_a_parseable_frame(self) -> None:
        generation = self.policy.generate(b"Calculate 5 * 5.", threshold=0.0)
        self.assertIn(generation.route, range(len(ROUTE_NAMES)))
        if generation.call is not None:
            self.assertEqual(parse_tool_call(generation.frame), generation.call)
            self.assertIn(generation.call.tool, TOOL_NAMES)

    def test_training_step_updates_only_the_new_heads(self) -> None:
        from data.tool_use_examples import ToolExample

        rows = [ToolExample(b"Calculate 2 * 2.", b"exec_python", b"print(2*2)", None),
                ToolExample(b"Calculate 3 * 3.", b"exec_python", b"print(3*3)", None)]
        cached = cache_rows(self.policy, rows)
        tensors = argument_tensors(rows, torch.device("cpu"))
        before = {key: value.clone() for key, value in self.generator.state_dict().items()}
        head_before = [p.detach().clone() for p in self.policy.parameters()]
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.05)
        loss, metrics = batch_loss(self.policy, torch.tensor([0, 1]), cached, tensors)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["count"], 2)
        self.assertTrue(any(not torch.equal(before_value, after) for before_value, after
                            in zip(head_before, self.policy.parameters())))
        for key, value in self.generator.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_checkpoint_binding_and_trained_flag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase15-ckpt-") as directory:
            generator_path = Path(directory) / "generator.pt"
            torch.save(self.generator.state_dict(), generator_path)
            policy_path = Path(directory) / "tool_policy.pt"
            self.policy.save(policy_path, generator_path=generator_path)
            with self.assertRaises(RuntimeError):
                load_tool_policy(policy_path, generator=self.generator, generator_path=generator_path)
            self.policy.route_head.tool_trained.fill_(True)
            self.policy.argument_decoder.tool_trained.fill_(True)
            self.policy.save(policy_path, generator_path=generator_path)
            restored = load_tool_policy(policy_path, generator=self.generator, generator_path=generator_path)
            self.assertTrue(restored.trained)
            other = Path(directory) / "other.pt"
            torch.save(small_generator().state_dict(), other)
            with self.assertRaises(RuntimeError):
                load_tool_policy(policy_path, generator=self.generator, generator_path=other)


class ToolLoopTests(unittest.TestCase):
    """End-to-end loop with a scripted head: the host must refuse unsafe frames."""

    class ScriptedPolicy:
        def __init__(self, script):
            self.script = list(script)
            self.seen_results = []

        def generate(self, request, result=None, *, threshold=0.0, max_bytes=MAX_ARGUMENT_BYTES):
            from models.tool_use import ToolGeneration

            self.seen_results.append(result)
            item = self.script.pop(0) if self.script else None
            if item is None:
                return ToolGeneration(route=0, route_confidence=1.0, call=None, frame=(), complete=True)
            tool, argument, complete = item
            call = ToolCall(tool, argument) if complete else None
            frame = tuple(encode_tool_call(ToolCall(tool, argument))) if complete else (264,)
            return ToolGeneration(route=1 if complete else 0, route_confidence=0.9,
                                  call=call, frame=frame, complete=complete)

    def _agent(self, script, **kwargs) -> ToolAgent:
        directory = tempfile.mkdtemp(prefix="phase15-loop-")
        self.addCleanup(shutil.rmtree, directory, True)
        sandbox = ByteSandbox(directory, timeout=5.0)
        return ToolAgent(self.ScriptedPolicy(script), sandbox, **kwargs)

    def test_successful_first_call_stops_and_reports_output(self) -> None:
        agent = self._agent([(b"exec_python", b"print(6*7)", True)])
        turn = agent.respond(b"Calculate 6 * 7.")
        self.assertEqual(turn.rounds, 1)
        self.assertEqual(turn.response, b"42")
        self.assertEqual(turn.calls, [b"exec_python"])
        self.assertFalse(turn.repaired)

    def test_failure_is_injected_and_a_correction_runs(self) -> None:
        agent = self._agent([(b"exec_python", b"pritn('hi')", True),
                             (b"exec_python", b"print('hi')", True)])
        turn = agent.respond(b"Run this and fix it: pritn('hi')")
        self.assertEqual(turn.rounds, 2)
        self.assertTrue(turn.repaired)
        self.assertEqual(turn.response, b"hi")
        # The second routing decision saw the sandbox's error bytes.
        seen = agent.policy.seen_results[1]
        self.assertIsNotNone(seen)
        self.assertIn(b"NameError", seen)
        # The failed round's bytes are what the head reads as [TOOL_RESULT].
        self.assertIn(b"NameError", turn.results[0])
        self.assertTrue(looks_failed(turn.results[0]))
        self.assertEqual(agent.injected_context(), sanitize_result(turn.results[-1]))

    def test_truncated_frame_is_never_executed(self) -> None:
        agent = self._agent([(b"exec_python", b"print('nope')", False)])
        turn = agent.respond(b"Calculate 1 + 1.")
        self.assertEqual(turn.rounds, 0)
        self.assertEqual(turn.tool_calls, 0)
        self.assertIn(b"no tool", turn.response)

    def test_max_rounds_bounds_the_loop(self) -> None:
        agent = self._agent([(b"exec_python", b"pritn(1)", True)] * 5, max_rounds=2)
        turn = agent.respond(b"Run and fix: pritn(1)")
        self.assertEqual(turn.rounds, 2)
        self.assertTrue(turn.response.startswith(b"The sandbox reported an error"))


class DatasetTests(unittest.TestCase):
    def test_splits_are_instance_disjoint_and_large_enough(self) -> None:
        train = {row.request for row in tool_examples("train")}
        test = {row.request for row in tool_examples("test")}
        self.assertFalse(train & test)
        self.assertGreaterEqual(len(train), 200)
        self.assertEqual(len(routing_prompts("test")), 20)
        self.assertEqual(len(debugging_tasks("test")), 5)

    def test_repair_rows_use_real_sandbox_output(self) -> None:
        with ByteSandbox(timeout=5.0) as sandbox:
            rows = tool_examples("train", sandbox)
        with_result = [row for row in rows if row.result is not None]
        self.assertTrue(with_result)
        self.assertTrue(any(b"Error" in (row.result or b"") for row in with_result))
        self.assertTrue(any(row.tool == b"" for row in rows))


if __name__ == "__main__":
    unittest.main()
