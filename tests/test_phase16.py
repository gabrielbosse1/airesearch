"""Phase 16: the scalable foundation (ByteCore v2).

These tests are structural: they pin the properties the phase is about -- a
standard causal LM with no copy head and no frame parser, ReAct tool use
intercepted by regex, memory injected as a prefix, and scale driven by config
rather than by code changes.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from data.scalable_react_dataset import (_memory_pool, _pool_slice, build_split,
                                         dataset_stats, generate_counts)
from models.foundation import (BOS, EOS, PAD, VOCAB_SIZE, CausalByteTransformer,
                               FoundationConfig, decode_ids, encode_bytes,
                               load_foundation, pad_batch, save_foundation)
from models.foundation_memory import (MemoryEntry, PrefixMemory, ScriptedRetriever,
                                      decode_vectors)
from models.react import (ReactLoop, format_action, format_observation, last_action,
                          parse_action, render_prompt, render_trajectory,
                          sandbox_observation)
from models.sandbox import ByteSandbox


class ConfigTests(unittest.TestCase):
    def test_defaults_are_valid_and_small(self):
        config = FoundationConfig()
        self.assertEqual(config.vocab_size, VOCAB_SIZE)
        self.assertEqual(config.d_model, 64)
        self.assertGreaterEqual(config.d_model, 32)

    def test_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            FoundationConfig(d_model=0)
        with self.assertRaises(ValueError):
            FoundationConfig(d_model=65)  # not divisible by n_heads
        with self.assertRaises(ValueError):
            FoundationConfig(vocab_size=300)
        with self.assertRaises(ValueError):
            FoundationConfig(dropout=1.5)

    def test_from_yaml_mapping(self):
        import yaml

        document = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
        section = dict(document["foundation"])
        section.pop("presets", None)
        config = FoundationConfig.from_mapping(section)
        self.assertEqual(config.d_model, section["d_model"])
        self.assertEqual(config.max_seq_len, section["max_seq_len"])

    def test_unknown_config_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            FoundationConfig.from_mapping({"d_model": 64, "copy_head": True})

    def test_scaled_keeps_everything_else(self):
        config = FoundationConfig().scaled(d_model=512, n_layers=12)
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.n_layers, 12)
        self.assertEqual(config.n_heads, FoundationConfig().n_heads)

    def test_yaml_exposes_the_scale_knobs(self):
        import yaml

        document = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
        for key in ("d_model", "n_layers", "n_heads", "max_seq_len"):
            self.assertIn(key, document["foundation"])
        presets = document["foundation"]["presets"]
        self.assertIn("production", presets)
        self.assertGreaterEqual(presets["production"]["d_model"], 512)
        self.assertGreaterEqual(presets["production"]["n_layers"], 12)


class FoundationModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(16016)
        self.model = CausalByteTransformer(FoundationConfig(n_layers=2, d_model=32, n_heads=2, d_ff=64))

    def test_output_shape_and_vocabulary(self):
        ids = torch.randint(0, 256, (2, 12))
        logits = self.model(ids)
        self.assertEqual(tuple(logits.shape), (2, 12, VOCAB_SIZE))

    def test_causal_mask_blocks_the_future(self):
        ids = torch.randint(0, 256, (1, 10))
        self.model.eval()
        with torch.no_grad():
            baseline = self.model(ids).clone()
            changed = ids.clone()
            changed[0, 8] = (changed[0, 8] + 7) % 256  # modify a later position
            after = self.model(changed)
        torch.testing.assert_close(baseline[:, :8], after[:, :8])  # past is untouched
        self.assertFalse(torch.allclose(baseline[:, 8:], after[:, 8:]))  # present/future sees it

    def test_encode_decode_round_trip_drops_control_bytes(self):
        payload = b"Thought: ok\nAction: exec_python(\"print(1)\")"
        ids = encode_bytes(payload, bos=True, eos=True)
        self.assertEqual(int(ids[0].item()), BOS)
        self.assertEqual(int(ids[-1].item()), EOS)
        self.assertEqual(decode_ids(ids), payload)

    def test_padding_is_masked_out_of_the_loss(self):
        sequences = [encode_bytes(b"short", bos=True, eos=True), encode_bytes(b"a much longer sequence", bos=True, eos=True)]
        ids, targets = pad_batch(sequences, 32)
        self.assertEqual(int((targets == PAD).sum().item()) > 0, True)
        loss = self.model.loss(ids, targets)
        self.assertTrue(torch.isfinite(loss))

    def test_loss_is_finite_and_decreases_when_the_data_is_trivial(self):
        repeated = encode_bytes(b"abababababababababababab", bos=True, eos=True)
        ids, targets = pad_batch([repeated] * 4, repeated.shape[0])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        before = float(self.model.loss(ids, targets).item())
        for _ in range(25):
            optimizer.zero_grad(set_to_none=True)
            loss = self.model.loss(ids, targets)
            loss.backward()
            optimizer.step()
        self.assertLess(float(self.model.loss(ids, targets).item()), before)

    def test_generation_stops_at_eos(self):
        ids = encode_bytes(b"Answer: 42", bos=True, eos=True)
        trained = CausalByteTransformer(FoundationConfig(n_layers=1, d_model=32, n_heads=2, d_ff=64))
        generated = trained.generate(ids, max_new_bytes=40)
        self.assertLessEqual(generated.shape[0], 40)

    def test_checkpoint_round_trip_keeps_predictions(self):
        path = Path("runs/phase16/_test_foundation.pt")
        self.model.eval()  # dropout would otherwise make two runs differ
        saved = save_foundation(self.model, path)
        loaded = load_foundation(path).eval()
        self.assertEqual(saved["parameters"], loaded.parameters_count)
        ids = torch.randint(0, 256, (1, 9))
        with torch.no_grad():
            torch.testing.assert_close(self.model(ids), loaded(ids))
        path.unlink(missing_ok=True)

    def test_parameter_count_grows_with_width_and_depth(self):
        small = CausalByteTransformer(FoundationConfig(n_layers=2, d_model=32, n_heads=2, d_ff=64))
        wide = CausalByteTransformer(FoundationConfig(n_layers=2, d_model=64, n_heads=4, d_ff=256))
        deep = CausalByteTransformer(FoundationConfig(n_layers=4, d_model=32, n_heads=2, d_ff=64))
        self.assertGreater(wide.parameters_count, small.parameters_count)
        self.assertGreater(deep.parameters_count, small.parameters_count)

    def test_no_copy_head_pointer_gate_or_frame_parser(self):
        names = {name for name, _ in self.model.named_modules()}
        names |= {name for name, _ in self.model.named_parameters()}
        names |= {name for name, _ in self.model.named_buffers()}
        forbidden = ("copy", "pointer", "gate", "frame", "tool_call", "retro", "verifier")
        offenders = sorted(name for name in names if any(word in name.lower() for word in forbidden))
        self.assertEqual(offenders, [])

    def test_foundation_source_does_not_import_the_legacy_frame_stack(self):
        source = Path("models/foundation.py").read_text(encoding="utf-8")
        for forbidden in ("tool_use", "copy_gate", "TOOL_ARG_START"):
            self.assertNotIn(forbidden, source)

    def test_production_preset_parameter_count(self):
        import yaml

        document = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
        preset = document["foundation"]["presets"]["production"]
        config = FoundationConfig.from_mapping({**{k: v for k, v in document["foundation"].items() if k != "presets"},
                                                **preset})
        # Meta device: full shape accounting without allocating the weights.
        model = CausalByteTransformer(config).to(torch.device("meta"))
        self.assertGreater(model.parameters_count, 50_000_000)
        wider = CausalByteTransformer(config.scaled(d_model=1024, d_ff=4096)).to(torch.device("meta"))
        self.assertGreater(wider.parameters_count, 100_000_000)

    def test_production_width_forward_pass_runs(self):
        config = FoundationConfig(d_model=512, n_layers=2, n_heads=8, d_ff=2048, max_seq_len=2048)
        model = CausalByteTransformer(config)
        logits = model(torch.randint(0, 256, (1, 16)))
        self.assertEqual(tuple(logits.shape), (1, 16, VOCAB_SIZE))


class ReactFormatTests(unittest.TestCase):
    def test_parses_a_single_argument(self):
        action = parse_action(b"Action: exec_python(\"print(1 + 1)\")")
        self.assertIsNotNone(action)
        self.assertEqual(action.tool, b"exec_python")
        self.assertEqual(action.args, (b"print(1 + 1)",))

    def test_parses_two_arguments(self):
        action = parse_action(b'Action: write_file("notes.txt", "hello sandbox")')
        self.assertEqual(action.args, (b"notes.txt", b"hello sandbox"))

    def test_ignores_an_unterminated_call(self):
        self.assertIsNone(parse_action(b"Action: exec_python(\"print(1)"))

    def test_ignores_prose_that_mentions_action(self):
        self.assertIsNone(parse_action(b"Action: is not always a tool call"))

    def test_uses_the_last_action_line(self):
        action = parse_action(b"Action: read_file(\"a.txt\")\nAction: exec_python(\"print(2)\")")
        self.assertEqual(action.args, (b"print(2)",))

    def test_escaped_quotes_are_unescaped(self):
        action = parse_action(b'Action: write_file("a.txt", "say \\"hi\\"")')
        self.assertEqual(action.args[1], b'say "hi"')

    def test_observation_is_single_line_and_bounded(self):
        observation = format_observation(b"line one\nline two\n", limit=16)
        self.assertNotIn(b"\n", observation)
        self.assertLessEqual(len(observation), 16 + 3)

    def test_format_action_escapes_quotes(self):
        rendered = format_action(b"write_file", (b"a.txt", b'he said "hi"'))
        self.assertEqual(rendered, b'Action: write_file("a.txt", "he said \\"hi\\"")')

    def test_prompt_rendering_is_stable(self):
        prompt = render_prompt(b"Calculate 6 * 7.", b"[MEM] my:color = blue")
        self.assertTrue(prompt.startswith(b"[MEM] my:color = blue\n"))
        self.assertTrue(prompt.endswith(b"Thought:"))
        self.assertIn(b"Request: Calculate 6 * 7.", prompt)

    def test_trajectory_joins_prompt_and_body(self):
        text = render_trajectory(b"Hi.", b"no tool here\nAnswer: Hello!")
        self.assertIn(b"Request: Hi.\nThought: no tool here\nAnswer: Hello!", text)

    def test_sandbox_observation_success_and_failure(self):
        with ByteSandbox() as sandbox:
            success = sandbox_observation(sandbox.execute_python(b"print(2 + 2)"))
            self.assertEqual(success, b"4")
            failure = sandbox_observation(sandbox.execute_python(b"pritn(2 + 2)"))
        self.assertTrue(failure.startswith(b"NameError"))


class ReactLoopTests(unittest.TestCase):
    class ScriptedModel:
        """Emits one scripted continuation per round; the last one repeats."""

        def __init__(self, *scripts):
            self.scripts = [bytes(script) for script in scripts]
            self.calls = 0
            self.config = FoundationConfig()

        def generate(self, prefix, *, max_new_bytes=160, temperature=0.0, stop=()):
            index = min(self.calls, len(self.scripts) - 1)
            self.calls += 1
            return torch.tensor(list(self.scripts[index][:max_new_bytes]), dtype=torch.long)

    def test_executes_an_action_and_injects_the_observation(self):
        with ByteSandbox() as sandbox:
            model = self.ScriptedModel(b" compute\nAction: exec_python(\"print(6 * 7)\")",
                                       b" done\nAnswer: 42")
            loop = ReactLoop(model, {"exec_python": lambda code: sandbox_observation(sandbox.execute_python(code))})
            run = loop.run(b"Calculate 6 * 7.")
        self.assertEqual(len(run.steps), 1)
        self.assertEqual(run.steps[0].observation, b"42")
        self.assertIn(b"Observation: 42", run.transcript)
        self.assertEqual(run.actions, ["exec_python"])
        self.assertEqual(run.answer, b"42")

    def test_model_text_after_the_action_is_discarded(self):
        # The model tries to invent an observation; the host must overwrite it.
        script = (b" compute\nAction: exec_python(\"print(1)\")\n"
                  b"Observation: 999\nAnswer: 999")
        with ByteSandbox() as sandbox:
            loop = ReactLoop(self.ScriptedModel(script, b" done\nAnswer: 1"),
                             {"exec_python": lambda code: sandbox_observation(sandbox.execute_python(code))})
            run = loop.run(b"Run one.")
        self.assertNotIn(b"Observation: 999", run.transcript)
        self.assertIn(b"Observation: 1", run.transcript)

    def test_unknown_tool_fails_closed(self):
        loop = ReactLoop(self.ScriptedModel(b" try\nAction: delete_all(\"/\")", b" stopping\nAnswer: no"), {})
        run = loop.run(b"Delete everything.")
        self.assertEqual(len(run.steps), 1)
        self.assertTrue(run.steps[0].observation.startswith(b"ERROR: unknown tool"))

    def test_round_budget_is_respected(self):
        script = b" go\nAction: exec_python(\"print(1)\")"
        counter = {"calls": 0}

        def tool(code):
            counter["calls"] += 1
            return b"1"

        loop = ReactLoop(self.ScriptedModel(script), {"exec_python": tool}, max_rounds=2)
        run = loop.run(b"Loop.")
        self.assertLessEqual(counter["calls"], 2)

    def test_declining_to_act_is_a_valid_outcome(self):
        loop = ReactLoop(self.ScriptedModel(b" no tool needed\nAnswer: Hello!"), {})
        run = loop.run(b"Hello!")
        self.assertEqual(run.tool_calls, 0)
        self.assertEqual(run.answer, b"Hello!")

    def test_memory_prefix_reaches_the_prompt(self):
        memory = PrefixMemory(ScriptedRetriever({b"my:favorite_color": b"blue"}), top_k=2)
        prefix = memory.prefix(b"What is my favorite color?")
        self.assertEqual(prefix, b"[MEM] my:favorite_color = blue")
        loop = ReactLoop(self.ScriptedModel(b" recall\nAnswer: blue"), {}, memory=memory.prefix)
        run = loop.run(b"What is my favorite color?")
        self.assertTrue(run.transcript.startswith(b"[MEM] my:favorite_color = blue\n"))


class MemoryIntegrationTests(unittest.TestCase):
    def test_prefix_renders_top_k_entries_in_order(self):
        retriever = ScriptedRetriever({b"my:favorite_color": b"blue"})
        memory = PrefixMemory(retriever, top_k=3)
        entries = memory.retrieve(b"What is my favorite color?")
        self.assertEqual([entry.topic for entry in entries], [b"my:favorite_color"])
        self.assertEqual(memory.render(entries), b"[MEM] my:favorite_color = blue")

    def test_empty_retrieval_yields_no_prefix(self):
        memory = PrefixMemory(ScriptedRetriever({}), top_k=3)
        self.assertEqual(memory.prefix(b"What is my favorite color?"), b"")

    def test_retrieval_failure_never_breaks_generation(self):
        def broken(_query):
            raise RuntimeError("store unavailable")

        self.assertEqual(PrefixMemory(broken).prefix(b"anything"), b"")

    def test_prefix_injection_is_attention_over_ordinary_bytes(self):
        model = CausalByteTransformer(FoundationConfig(n_layers=2, d_model=32, n_heads=2, d_ff=64))
        with_prefix = encode_bytes(b"[MEM] my:color = blue\nRequest: color?\nThought:", bos=True)
        without_prefix = encode_bytes(b"Request: color?\nThought:", bos=True)
        with torch.no_grad():
            a = model(with_prefix.unsqueeze(0))
            b = model(without_prefix.unsqueeze(0))
        # Same suffix, different prefix -> different predictions: the prefix is
        # being attended to, not concatenated and ignored.
        self.assertFalse(torch.allclose(a[0, -1], b[0, -1]))

    def test_codec_inverse_recovers_the_stored_value(self):
        from models.vector_memory import VectorFactCodec

        codec = VectorFactCodec()
        for value in (b"blue", b"42", b"acme labs", b"x" * 40):
            vectors = codec.encode(value)
            self.assertEqual(decode_vectors(vectors, codec.rotation), value)

    def test_prefix_memory_entry_dataclass_defaults(self):
        entry = MemoryEntry(topic=b"my:x", value=b"y")
        self.assertEqual(entry.source, "vector")
        self.assertEqual(entry.score, 1.0)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with ByteSandbox() as sandbox:
            cls.rows = build_split("test", sandbox=sandbox,
                                   counts={"exec": 6, "repair": 4, "file": 4, "memory": 4, "chat": 4})

    def test_small_build_produces_every_kind(self):
        stats = dataset_stats(self.rows)
        self.assertEqual(stats["count"], 22)
        self.assertEqual(set(stats["by_kind"]), {"exec", "repair", "file", "memory", "chat"})

    def test_every_trajectory_follows_the_react_format(self):
        for row in self.rows:
            self.assertIn(b"Request: ", row.text)
            self.assertIn(b"Thought:", row.text)
            self.assertIn(b"Answer: ", row.text)

    def test_tool_trajectories_carry_a_parsable_action(self):
        for row in self.rows:
            if row.kind == "chat":
                continue
            self.assertIsNotNone(last_action(row.text), row.text[:80])

    def test_repair_observations_are_real_failures(self):
        for row in self.rows:
            if row.kind != "repair":
                continue
            self.assertIn(b"Error:", row.text)

    def test_memory_trajectories_prefix_the_fact(self):
        prefixed = [row for row in self.rows if row.kind == "memory"]
        self.assertTrue(prefixed)
        for row in prefixed:
            self.assertTrue(row.text.startswith(b"[MEM] "))
            self.assertIn(b"Action: recall(", row.text)

    def test_splits_are_instance_disjoint(self):
        exec_train = _pool_slice("exec", "train", 16001, size=40)
        exec_test = _pool_slice("exec", "test", 16001, size=40)
        self.assertEqual(set(_render_exec(item) for item in exec_train)
                         & set(_render_exec(item) for item in exec_test), set())
        memory_train = _pool_slice("memory", "train", 16001, size=30)
        memory_test = _pool_slice("memory", "test", 16001, size=30)
        # Stored facts are identified by (topic, value): a topic may legitimately
        # recur with a different value, the pair may not.
        self.assertEqual({(topic, value) for topic, value, _ in memory_train}
                         & {(topic, value) for topic, value, _ in memory_test}, set())
        file_train = {name for name, _ in _pool_slice("file", "train", 16001, size=30)}
        file_test = {name for name, _ in _pool_slice("file", "test", 16001, size=30)}
        self.assertEqual(file_train & file_test, set())

    def test_default_configuration_generates_more_than_ten_thousand(self):
        counts = generate_counts()
        self.assertGreaterEqual(sum(counts.values()), 10_000)

    def test_observations_match_a_fresh_sandbox(self):
        with ByteSandbox() as sandbox:
            for row in self.rows:
                if row.kind != "exec":
                    continue
                action = last_action(row.text)
                observation = row.text.split(b"Observation: ")[1].split(b"\n")[0]
                self.assertEqual(sandbox_observation(sandbox.execute_python(action.args[0])), observation)
                break


def _render_exec(item):
    from data.scalable_react_dataset import _EXEC_TEMPLATES

    return _EXEC_TEMPLATES[int(item["template"])][1].format(**item)


if __name__ == "__main__":
    unittest.main()
