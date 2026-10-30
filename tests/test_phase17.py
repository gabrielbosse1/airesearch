"""Phase 17: real GitHub data -- collection, conversion, mixture, grounding."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from data.github_to_react import (CorpusIndex, copyable_report, github_to_react,
                                  tool_explain_commit, tool_review_pr,
                                  tool_summarize_readme, verify)
from data.mixed_react_dataset import (GITHUB_KINDS, load_github,
                                      load_trajectories, split_github)
from evaluation.github_eval import (build_tools, grounding_facts,
                                    observation_only_facts)
from models.foundation import FoundationConfig, encode_bytes
from models.react import ReactLoop, last_action

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "github_raw" / "manifest.json"
STORIES = ROOT / "data" / "github_raw" / "github_stories.jsonl"
REACT = ROOT / "data" / "github_raw" / "github_react.jsonl"
MIXED = ROOT / "runs" / "phase17"

_INDEX_CACHE: dict[str, CorpusIndex] = {}


def corpus() -> CorpusIndex:
    """Load the collected corpus once per test session (7.6 MB, ~2 s)."""
    if "index" not in _INDEX_CACHE:
        _INDEX_CACHE["index"] = CorpusIndex.from_jsonl(STORIES)
    return _INDEX_CACHE["index"]


def require_collected() -> None:
    if not STORIES.exists() or not MANIFEST.exists():
        raise unittest.SkipTest("collected GitHub corpus is not present in this checkout")


def require_converted() -> list[dict]:
    if not REACT.exists():
        raise unittest.SkipTest("data/github_raw/github_react.jsonl has not been generated")
    with REACT.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require_mixed() -> None:
    if not (MIXED / "react_train.jsonl").exists():
        raise unittest.SkipTest("the mixed dataset has not been built")


class ScriptedModel:
    """Emit a fixed continuation, so loop tests need no trained weights."""

    def __init__(self, script: bytes) -> None:
        self.script = script
        self.config = FoundationConfig()

    def generate(self, prefix, *, max_new_bytes: int = 200, temperature: float = 0.0,
                 stop: tuple[int, ...] = ()):
        return torch.tensor(list(self.script[:max_new_bytes]), dtype=torch.long)


# ------------------------------------------------------------------ collection
class CollectedCorpusTests(unittest.TestCase):
    def test_manifest_records_provenance(self):
        require_collected()
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertGreaterEqual(manifest["records"], 4000)
        self.assertEqual(manifest["by_type"]["readme"], 10)
        self.assertTrue(manifest["authenticated"])
        self.assertEqual(len(manifest["licenses"]), len(manifest["repositories"]))
        self.assertIn("scrubbing", manifest)

    def test_every_record_carries_its_licence(self):
        require_collected()
        with STORIES.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertTrue(all(row.get("license") for row in rows))
        self.assertTrue(all(row.get("source") for row in rows))

    def test_corpus_index_loads_every_kind(self):
        require_collected()
        index = corpus()
        self.assertGreater(len(index.issues), 100)
        self.assertGreater(len(index.prs), 500)
        self.assertGreater(len(index.commits), 500)
        self.assertEqual(len(index.readmes), 10)


# ------------------------------------------------------------------ tools
class RealToolTests(unittest.TestCase):
    def test_tools_report_what_they_found(self):
        index = corpus()
        commit = index.commits[0]
        observation = tool_explain_commit(index, commit["sha"])
        self.assertIn(commit["sha"].encode(), observation)
        self.assertIn(commit["author"].encode(), observation)

    def test_tools_say_so_when_they_find_nothing(self):
        index = corpus()
        self.assertTrue(tool_explain_commit(index, "0" * 12).startswith(b"no commit"))
        self.assertTrue(tool_review_pr(index, "nobody/nothing#1").startswith(b"no pull request"))
        self.assertTrue(tool_summarize_readme(index, "nobody/nothing").startswith(b"no README"))

    def test_search_is_a_real_lexical_match(self):
        index = corpus()
        hits = index.search_issues("memory leak", k=3)
        self.assertTrue(hits)
        for score, _record in hits:
            self.assertGreater(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_readme_statistics_are_computed_from_the_document(self):
        index = corpus()
        repo = sorted(index.readmes)[0]
        content = index.readmes[repo]["content"]
        observation = tool_summarize_readme(index, repo).decode()
        self.assertIn(f"{len(content)} characters", observation)


# ------------------------------------------------------------------ conversion
class ConversionTests(unittest.TestCase):
    def test_observations_are_reproducible(self):
        rows = require_converted()
        index = corpus()
        result = verify(rows, index, count=40)
        self.assertEqual(result["mismatches"], 0)
        self.assertEqual(result["checked"], 40)

    def test_every_argument_is_recoverable_from_the_request(self):
        rows = require_converted()
        report = copyable_report(rows)
        self.assertEqual(report["fraction"], 1.0)

    def test_no_example_exceeds_the_rendering_budget(self):
        rows = require_converted()
        self.assertLessEqual(max(len(row["text"].encode("latin-1")) for row in rows), 768)

    def test_converter_is_deterministic(self):
        """Rebuilding from the same corpus reproduces the same bytes."""
        require_collected()
        rows = require_converted()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rebuilt.jsonl"
            rebuilt = github_to_react(str(STORIES), output_file=str(output))
            self.assertEqual([row["text"] for row in rebuilt],
                             [row["text"] for row in rows])


# ------------------------------------------------------------------ mixture
class MixtureTests(unittest.TestCase):
    def test_held_out_records_are_absent_from_training(self):
        require_converted()
        require_mixed()
        splits = split_github(load_github(REACT))
        train_keys = {self._key(row) for row in splits["train"]}
        test_keys = {self._key(row) for row in splits["test"]}
        self.assertFalse(train_keys & test_keys)
        self.assertGreater(len(test_keys), 100)

    def test_splits_cover_every_record_exactly_once(self):
        require_converted()
        rows = load_github(REACT)
        splits = split_github(rows)
        total = sum(len(split) for split in splits.values())
        self.assertEqual(total, len(rows))

    def test_split_is_stable_across_calls(self):
        require_converted()
        rows = load_github(REACT)
        first = split_github(rows)
        second = split_github(rows)
        self.assertEqual([row.text for row in first["test"]],
                         [row.text for row in second["test"]])

    def test_training_file_contains_no_held_out_text(self):
        require_mixed()
        held_out = {row.text for row in load_trajectories(MIXED / "github_heldout.jsonl")}
        train = {row.text for row in load_trajectories(MIXED / "react_train.jsonl")}
        self.assertFalse(held_out & train)

    def test_mixture_keeps_the_synthetic_kinds(self):
        require_mixed()
        rows = load_trajectories(MIXED / "react_train.jsonl")
        kinds = {row.kind for row in rows}
        self.assertTrue({"exec", "file", "chat", "repair", "memory"} <= kinds)
        self.assertTrue(kinds & set(GITHUB_KINDS))

    @staticmethod
    def _key(row) -> bytes:
        call = last_action(row.text)
        return call.args[0] if call and call.args else row.request


# ------------------------------------------------------------------ grounding
class GroundingTests(unittest.TestCase):
    def test_facts_come_from_the_observation_only(self):
        index = corpus()
        commit = index.commits[0]
        observation = tool_explain_commit(index, commit["sha"])
        facts = grounding_facts(b"explain_commit", observation)
        self.assertTrue(facts)
        request = b"Explain commit " + commit["sha"].encode() + b": nothing else"
        self.assertEqual(observation_only_facts(b"explain_commit", observation, request), facts)

    def test_a_request_that_already_states_the_fact_is_not_grounding(self):
        observation = b"commit abc by Ada Lovelace on 1852-12-10: notes"
        facts = grounding_facts(b"explain_commit", observation)
        self.assertIn(b"Ada Lovelace", facts)
        self.assertEqual(
            observation_only_facts(b"explain_commit", observation,
                                   b"who wrote it, Ada Lovelace, on 1852-12-10"), [])

    def test_every_held_out_observation_carries_at_least_one_fact(self):
        require_converted()
        require_mixed()
        rows = load_trajectories(MIXED / "github_heldout.jsonl")
        missing = 0
        for row in rows:
            action = last_action(row.text)
            observation = row.text.split(b"Observation: ", 1)[-1].split(b"\n", 1)[0]
            if not observation_only_facts(action.tool, observation, row.request):
                missing += 1
        self.assertEqual(missing, 0, f"{missing} held-out rows have no observation-only fact")


# ------------------------------------------------------------------ loop
class RealToolLoopTests(unittest.TestCase):
    def test_the_loop_runs_a_real_tool_and_keeps_its_bytes(self):
        index = corpus()
        commit = index.commits[0]
        script = (b" I will look it up.\nAction: explain_commit(\"" + commit["sha"].encode()
                  + b"\")\nAnswer: done.")
        loop = ReactLoop(ScriptedModel(script), build_tools(index), max_rounds=1)
        run = loop.run(b"Explain commit " + commit["sha"].encode() + b": please")
        self.assertEqual(len(run.steps), 1)
        self.assertEqual(run.steps[0].observation.decode(),
                         tool_explain_commit(index, commit["sha"]).decode())

    def test_an_answer_the_model_wrote_after_the_action_is_discarded(self):
        index = corpus()
        commit = index.commits[1]
        script = (b"Action: explain_commit(\"" + commit["sha"].encode()
                  + b"\")\nObservation: I invented this\nAnswer: from the model")
        loop = ReactLoop(ScriptedModel(script), build_tools(index), max_rounds=1)
        run = loop.run(b"Explain commit " + commit["sha"].encode() + b": please")
        self.assertNotIn(b"I invented this", run.transcript)

    def test_an_unknown_tool_fails_closed(self):
        index = corpus()
        loop = ReactLoop(ScriptedModel(b"Action: fetch_everything(\"x\")"),
                         build_tools(index), max_rounds=1)
        run = loop.run(b"do something")
        self.assertIn(b"unknown tool", run.steps[0].observation)

    def test_sequences_encode_as_raw_bytes(self):
        """A real trajectory is stored as bytes and encoded as bytes -- no tokens."""
        raw = require_converted()[0]["text"].encode("latin-1")
        ids = encode_bytes(raw, bos=True, eos=True)
        body = ids[1:-1]  # without <bos> and <eos>
        self.assertEqual(len(body), len(raw))
        self.assertEqual(bytes(body.tolist()), raw)


if __name__ == "__main__":
    unittest.main()
