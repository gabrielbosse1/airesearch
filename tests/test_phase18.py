"""Phase 18: reference-based arguments, grounded refusal, balanced batches."""
from __future__ import annotations

import unittest
from pathlib import Path

import torch

from data.phase18_corpus import build_refusals, mark_request
from data.github_to_react import CorpusIndex, mark_entities
from data.mixed_react_dataset import load_trajectories
from models.foundation import (REF_END, REF_START, VOCAB_SIZE, FoundationConfig,
                               decode_ids, encode_bytes)
from models.react import (REF_OPEN, UnresolvedReference, mark,
                          reference_spans, reference_tools, resolve_arguments,
                          resolve_reference, strip_references)
from training.train_foundation import (answer_span, balanced_batches,
                                       is_failure_observation, refusal_weights)

ROOT = Path(__file__).resolve().parent.parent
STORIES = ROOT / "data" / "github_raw" / "github_stories.jsonl"
REACT = ROOT / "data" / "github_raw" / "github_react.jsonl"
P18 = ROOT / "runs" / "phase18"

_INDEX_CACHE: dict[str, CorpusIndex] = {}


def corpus() -> CorpusIndex:
    if "index" not in _INDEX_CACHE:
        _INDEX_CACHE["index"] = CorpusIndex.from_jsonl(STORIES)
    return _INDEX_CACHE["index"]


def require_corpus() -> None:
    if not STORIES.exists():
        raise unittest.SkipTest("the collected GitHub corpus is not present")


def require_phase18() -> None:
    if not (P18 / "react_train.jsonl").exists():
        raise unittest.SkipTest("the Phase 18 corpus has not been built")


# ------------------------------------------------------------------ tokens
class ReferenceTokenTests(unittest.TestCase):
    def test_two_control_ids_were_added(self):
        self.assertEqual((REF_START, REF_END), (259, 260))
        self.assertEqual(VOCAB_SIZE, 261)

    def test_markers_round_trip_through_the_codec(self):
        raw = b"commit " + mark(b"00c3ce0a55e5") + b" changed things"
        ids = encode_bytes(raw, bos=True, eos=True)
        self.assertIn(REF_START, ids.tolist())
        self.assertIn(REF_END, ids.tolist())
        self.assertEqual(decode_ids(ids), raw)

    def test_plain_bytes_are_untouched(self):
        raw = bytes(range(256))
        self.assertEqual(decode_ids(encode_bytes(raw, bos=True, eos=True)), raw)

    def test_the_model_accepts_the_wider_vocabulary(self):
        from models.foundation import CausalByteTransformer
        model = CausalByteTransformer(FoundationConfig(vocab_size=VOCAB_SIZE, max_seq_len=64))
        self.assertEqual(model.config.vocab_size, 261)
        ids = encode_bytes(b"a" + mark(b"abc123") + b"b", bos=True)[:64]
        self.assertEqual(int(model(ids.unsqueeze(0)).shape[-1]), 261)

    def test_a_stale_vocabulary_is_rejected(self):
        with self.assertRaises(ValueError):
            FoundationConfig(vocab_size=259)


# --------------------------------------------------------------- resolution
class ReferenceResolutionTests(unittest.TestCase):
    def setUp(self):
        self.request = (b"Request: Explain commit " + mark(b"00c3ce0a55e5")
                        + b": Update coverage to 7.17.0")

    def test_spans_are_found_in_order(self):
        self.assertEqual(reference_spans(self.request), [b"00c3ce0a55e5"])

    def test_an_index_resolves_to_the_span(self):
        self.assertEqual(resolve_reference(b"1", self.request), b"00c3ce0a55e5")

    def test_the_literal_span_resolves_to_itself(self):
        self.assertEqual(resolve_reference(b"00c3ce0a55e5", self.request), b"00c3ce0a55e5")

    def test_an_unknown_identifier_fails_closed(self):
        with self.assertRaises(UnresolvedReference):
            resolve_reference(b"deadbeefcafe", self.request)

    def test_an_out_of_range_index_fails_closed(self):
        with self.assertRaises(UnresolvedReference):
            resolve_reference(b"9", self.request)

    def test_resolution_is_against_the_request_not_the_action(self):
        """The whole safety property: only what the user typed can be passed on."""
        action = resolve_arguments((b"[REF]1[/REF]",), self.request)
        self.assertEqual(action, (b"00c3ce0a55e5",))
        with self.assertRaises(UnresolvedReference):
            resolve_arguments((b"[REF]1f4b2c9d8e7a[/REF]",), self.request)

    def test_the_wrapped_tool_receives_resolved_bytes(self):
        seen: list[bytes] = []

        def handler(identifier: bytes) -> bytes:
            seen.append(identifier)
            return b"commit " + identifier

        tools = reference_tools({b"explain_commit": handler}, self.request)
        self.assertIn(b"00c3ce0a55e5", tools[b"explain_commit"](b"[REF]1[/REF]"))
        self.assertEqual(seen, [b"00c3ce0a55e5"])

    def test_an_unresolved_reference_becomes_an_error_observation(self):
        tools = reference_tools({b"explain_commit": lambda _: b"should not run"}, self.request)
        self.assertTrue(tools[b"explain_commit"](b"[REF]9[/REF]").startswith(b"ERROR"))

    def test_markers_strip_to_the_underlying_bytes(self):
        self.assertEqual(strip_references(self.request),
                         b"Request: Explain commit 00c3ce0a55e5: Update coverage to 7.17.0")

    def test_mark_request_marks_the_longest_entity_first(self):
        # The shorter sha is a prefix of the longer one; marking it first would
        # cut the longer identifier in half.
        marked = mark_request(b"sha 00c3ce0a55e5 and 00c3ce0a55e5abc",
                              [b"00c3ce0a55e5", b"00c3ce0a55e5abc"])
        spans = reference_spans(marked)
        self.assertEqual(sorted(spans), [b"00c3ce0a55e5", b"00c3ce0a55e5abc"])
        self.assertEqual(strip_references(marked),
                         b"sha 00c3ce0a55e5 and 00c3ce0a55e5abc")


# ------------------------------------------------------------------ corpus
class Phase18CorpusTests(unittest.TestCase):
    def test_real_rows_carry_a_marked_request(self):
        require_phase18()
        rows = [row for row in load_trajectories(P18 / "react_train.jsonl")
                if row.kind.startswith("github_") and "refuse" not in row.kind]
        self.assertTrue(rows)
        for row in rows[:50]:
            self.assertIn(REF_OPEN, row.request)
            self.assertIn(b'[REF]1[/REF]', row.body)

    def test_every_real_observation_is_still_a_real_tool_result(self):
        """Refitting must not have turned real observations into text."""
        require_corpus()
        require_phase18()
        index = corpus()
        rows = [row for row in load_trajectories(P18 / "github_heldout.jsonl")
                if not row.kind.startswith("github_refuse")]
        checked = 0
        for row in rows:
            entity = reference_spans(row.request)
            if not entity:
                continue
            action = row.body.split(b"Action: ", 1)[1].split(b"\n", 1)[0]
            name = action[:action.index(b"(")]
            observation = row.text.split(b"Observation: ", 1)[1].split(b"\n", 1)[0]
            expected = _call(name, entity[0], index)
            self.assertEqual(observation, b" ".join(expected.split()),
                             f"stale observation for {entity[0]!r}")
            checked += 1
        self.assertGreater(checked, 50)

    def test_refusals_are_real_lookups_that_really_fail(self):
        require_corpus()
        rows = build_refusals(corpus(), count=40)
        self.assertTrue(rows)
        for row in rows[:20]:
            observation = row.text.split(b"Observation: ", 1)[1].split(b"\n", 1)[0]
            failed = observation.startswith(b"no ") or observation.startswith(b"ERROR")
            self.assertTrue(failed, observation[:80])

    def test_refusals_never_invent_an_identifier_in_the_answer(self):
        require_corpus()
        for row in build_refusals(corpus(), count=60):
            answer = row.text.split(b"Answer: ", 1)[1]
            self.assertTrue(any(phrase in answer.lower() for phrase in
                                (b"could not find", b"could not resolve")), answer)

    def test_the_corpus_holds_more_than_a_thousand_refusals(self):
        require_phase18()
        count = sum(1 for row in load_trajectories(P18 / "react_train.jsonl")
                    if row.kind.startswith("github_refuse"))
        self.assertGreater(count, 1000)

    def test_the_corpus_matches_what_the_loop_will_emit(self):
        """``ReactLoop`` appends ``Thought:`` after every observation.

        Trajectories that go straight from ``Observation:`` to ``Answer:`` are
        off-distribution exactly where the answer is produced -- this was
        measured as 0/10 answers before the bridging line was added, and it is
        the kind of thing that silently survives a rebuild.
        """
        require_phase18()
        rows = [row for row in load_trajectories(P18 / "react_train.jsonl")
                if row.kind.startswith("github_")]
        checked = 0
        for row in rows:
            if b"Observation: " not in row.text or b"Answer: " not in row.text:
                continue
            between = row.text.split(b"Observation: ", 1)[1].split(b"Answer: ", 1)[0]
            self.assertIn(b"Thought:", between,
                          "no bridging Thought between Observation and Answer")
            checked += 1
        self.assertGreater(checked, 1000)

    def test_almost_no_row_is_truncated_by_the_context_window(self):
        require_phase18()
        rows = load_trajectories(P18 / "react_train.jsonl")
        over = sum(1 for row in rows if len(row.text) > 768)
        self.assertLess(over / max(len(rows), 1), 0.001,
                        f"{over} rows exceed max_seq_len 768 and would be cut mid-answer")

    def test_refusal_kinds_are_all_present(self):
        require_phase18()
        kinds = {row.kind for row in load_trajectories(P18 / "react_train.jsonl")
                 if row.kind.startswith("github_refuse")}
        self.assertTrue({"github_refuse_commit", "github_refuse_pr",
                         "github_refuse_issue", "github_refuse_readme",
                         "github_refuse_reference"} <= kinds)


def _call(name: bytes, identifier: bytes, index: CorpusIndex) -> bytes:
    from data.github_to_react import (tool_explain_commit, tool_review_pr,
                                      tool_search_issues, tool_summarize_readme)
    argument = identifier.decode("utf-8", "replace")
    return {b"explain_commit": tool_explain_commit, b"review_pr": tool_review_pr,
            b"search_issues": tool_search_issues,
            b"summarize_readme": tool_summarize_readme}[name](index, argument)


# ----------------------------------------------------------------- training
class RefusalWeightingTests(unittest.TestCase):
    def test_a_failure_observation_is_detected(self):
        self.assertTrue(is_failure_observation(
            b"Thought: x\nObservation: no commit abc in the corpus\nAnswer: none"))
        self.assertTrue(is_failure_observation(
            b"Thought: x\nObservation: ERROR: unresolved reference\nAnswer: none"))
        self.assertFalse(is_failure_observation(
            b"Thought: x\nObservation: commit abc by Ada on 2026-01-01: error handling\nAnswer: ok"))

    def test_a_subject_that_mentions_error_is_not_a_failure(self):
        self.assertFalse(is_failure_observation(
            b"Thought: x\nObservation: commit abc by Ada: fix an error path\nAnswer: ok"))

    def test_the_answer_line_is_the_only_weighted_span(self):
        text = (b"Request: x\nThought: y\nAction: t(\"[REF]1[/REF]\")\n"
                b"Observation: no commit abc in the corpus\nAnswer: I could not find that commit.")
        ids = encode_bytes(text, bos=True, eos=True)
        weights = refusal_weights(text, len(ids))
        self.assertAlmostEqual(float(weights.max()), 2.0)
        begin, end = answer_span(text)
        hot = (weights > 1.0).nonzero().flatten().tolist()
        self.assertTrue(hot)
        self.assertGreaterEqual(min(hot), begin - 2)
        self.assertLessEqual(max(hot), end)

    def test_a_success_trajectory_is_flat(self):
        text = (b"Request: x\nThought: y\nAction: t(\"[REF]1[/REF]\")\n"
                b"Observation: commit abc by Ada on 2026-01-01: a change\nAnswer: found it.")
        ids = encode_bytes(text, bos=True, eos=True)
        weights = refusal_weights(text, len(ids))
        self.assertAlmostEqual(float(weights.max()), 1.0)

    def test_weights_are_zero_on_padding(self):
        short = torch.ones(5)
        long = torch.ones(9)
        ids, targets, weights = _pad([short.long(), long.long()], [short, long], 9)
        self.assertEqual(float(weights[0, 5:].sum()), 0.0)
        self.assertGreater(float(weights[1].sum()), 0.0)


def _pad(sequences, weights, length):
    from training.train_foundation import pad_batch_with_weights
    return pad_batch_with_weights(sequences, weights, length)


class BalancedBatchTests(unittest.TestCase):
    def test_every_batch_holds_one_sub_batch_per_group(self):
        sequences = [torch.full((10 + i % 7,), 5, dtype=torch.long) for i in range(200)]
        groups = [list(range(0, 100)), list(range(100, 200))]
        import random
        seen = 0
        for sub_batches in balanced_batches(groups, [0.5, 0.5], sequences, 32,
                                            random.Random(0), 4096):
            self.assertEqual(len(sub_batches), 2)
            for ids, targets, weights in sub_batches:
                self.assertGreater(ids.shape[0], 0)
                self.assertEqual(weights.shape, targets.shape)
            seen += 1
        self.assertGreater(seen, 1)

    def test_each_sub_batch_is_bounded_by_its_share_of_the_budget(self):
        sequences = [torch.full((50,), 5, dtype=torch.long) for _ in range(64)]
        groups = [list(range(0, 32)), list(range(32, 64))]
        import random
        for sub_batches in balanced_batches(groups, [0.5, 0.5], sequences, 64,
                                            random.Random(0), 1000):
            # Half of a 1000-token budget, rows of 50 -> at most 10 rows each.
            for ids, _targets, _weights in sub_batches:
                self.assertLessEqual(int(ids.numel()), 500 + 50)

    def test_padding_is_not_inflated_by_the_other_group(self):
        """Short rows must not be padded out to the long group's length."""
        short = [torch.full((20,), 5, dtype=torch.long) for _ in range(40)]
        long = [torch.full((300,), 5, dtype=torch.long) for _ in range(40)]
        sequences = short + long
        groups = [list(range(0, 40)), list(range(40, 80))]
        import random
        for sub_batches in balanced_batches(groups, [0.5, 0.5], sequences, 64,
                                            random.Random(0), 4096):
            widths = sorted(int(ids.shape[1]) for ids, _t, _w in sub_batches)
            self.assertLess(widths[0], 100)   # the short group stays short
            self.assertGreater(widths[-1], 100)

    def test_shares_must_match_groups(self):
        sequences = [torch.full((5,), 5, dtype=torch.long)]
        import random
        with self.assertRaises(ValueError):
            list(balanced_batches([[0]], [0.5, 0.5], sequences, 4, random.Random(0), 100))


class RunTimeMarkingTest(unittest.TestCase):
    """A raw request has to be marked before the model ever sees it."""

    def test_a_commit_sha_is_marked(self):
        marked = mark_entities(b"Explain commit 5c06b141f3b0: Fix the async close")
        self.assertEqual(reference_spans(marked), [b"5c06b141f3b0"])

    def test_a_pull_request_reference_is_marked_as_one_span(self):
        """``owner/repo`` must not be marked inside ``owner/repo#123``."""
        marked = mark_entities(b"Review pull request denoland/deno#36798")
        self.assertEqual(reference_spans(marked), [b"denoland/deno#36798"])

    def test_a_repository_name_does_not_swallow_the_full_stop(self):
        marked = mark_entities(b"Summarize the project python/cpython.")
        self.assertEqual(reference_spans(marked), [b"python/cpython"])

    def test_the_issue_query_is_marked(self):
        marked = mark_entities(b"Find issues similar to: Reusable workflow for CV")
        self.assertEqual(reference_spans(marked), [b"Reusable workflow for CV"])

    def test_an_english_word_of_hex_letters_is_not_a_sha(self):
        """``decade`` is valid hex; marking it would shift the real indices."""
        self.assertEqual(reference_spans(mark_entities(b"Tell me about the decade-long effort")), [])

    def test_a_request_that_names_nothing_is_left_alone(self):
        self.assertEqual(mark_entities(b"What is the weather?"), b"What is the weather?")

    def test_marking_is_idempotent(self):
        once = mark_entities(b"Review pull request denoland/deno#36798")
        self.assertEqual(mark_entities(once), once)


if __name__ == "__main__":
    unittest.main()
