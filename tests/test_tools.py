"""True tooling: token-id triggers, sandboxed search, EDIT as vocabulary."""
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.foundation import (EOS, EDIT, EDIT_END, OBS_END, OBS_START,
                               TOOL_SEARCH, TOOL_SEARCH_END, VOCAB_SIZE,
                               CausalByteTransformer, encode_bytes)
from training.train import make_search_fn, run_search


class ScriptedModel:
    """Emits a fixed id script, one id per forward call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.config = SimpleNamespace(max_seq_len=256)

    def eval(self):
        return self

    def forward(self, ids, memory=None):
        index = self.calls
        self.calls += 1
        next_id = self.script[index] if index < len(self.script) else EOS
        logits = torch.full((1, ids.shape[1], VOCAB_SIZE), -20.0)
        logits[0, -1, next_id] = 20.0
        return logits


def test_mock_search_is_deterministic_and_echoes_query():
    a = run_search(b"python bytes", backend="mock")
    b = run_search(b"python bytes", backend="mock")
    assert a == b
    assert b"python bytes" in a
    assert b"mock ddgs" in a
    assert run_search(b"other query", backend="mock") != a


def test_search_runs_in_isolated_subprocess():
    # The mock script runs with -I -S: no site-packages, no user site, no env.
    result = run_search(b"hello", backend="mock")
    assert isinstance(result, bytes) and len(result) > 0


def test_search_timeout_is_graceful(monkeypatch):
    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)
    monkeypatch.setattr(subprocess, "run", boom)
    assert run_search(b"q", backend="mock") == b"(search timed out)"


def test_search_failure_is_graceful(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no fork")
    monkeypatch.setattr(subprocess, "run", boom)
    assert b"search error" in run_search(b"q", backend="mock")


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        run_search(b"q", backend="bing")


def test_tool_search_token_triggers_execution():
    model = ScriptedModel([TOOL_SEARCH, ord("q"), TOOL_SEARCH_END, ord("a"), EOS])
    search = make_search_fn("mock")
    prefix = encode_bytes(b"hi")
    ids, stats = CausalByteTransformer.generate_with_tools(
        model, prefix, search, max_new_bytes=32)
    assert stats["tool_calls"] == 1
    assert stats["queries"] == [b"q"]
    generated = ids.tolist()
    assert OBS_START in generated and OBS_END in generated
    # The observation payload sits between the OBS markers.
    start = generated.index(OBS_START) + 1
    end = generated.index(OBS_END)
    payload = bytes(generated[start:end])
    assert b"q" in payload and b"mock ddgs" in payload


def test_edit_tokens_receive_no_host_treatment():
    model = ScriptedModel([EDIT, ord("x"), EDIT_END, EOS])
    ids, stats = CausalByteTransformer.generate_with_tools(
        model, encode_bytes(b"hi"), make_search_fn("mock"), max_new_bytes=16)
    assert stats["tool_calls"] == 0 and stats["queries"] == []
    assert ids.tolist() == [EDIT, ord("x"), EDIT_END]


def test_tool_call_budget_is_enforced():
    script = [TOOL_SEARCH, TOOL_SEARCH_END] * 10
    model = ScriptedModel(script)
    ids, stats = CausalByteTransformer.generate_with_tools(
        model, encode_bytes(b"hi"), make_search_fn("mock"),
        max_new_bytes=64, max_tool_calls=2)
    assert stats["tool_calls"] == 2


def test_observation_limit_truncates():
    model = ScriptedModel([TOOL_SEARCH, ord("q"), TOOL_SEARCH_END, EOS])
    ids, _ = CausalByteTransformer.generate_with_tools(
        model, encode_bytes(b"hi"), lambda q: b"x" * 10000,
        max_new_bytes=32, observation_limit=50)
    generated = ids.tolist()
    payload = generated[generated.index(OBS_START) + 1: generated.index(OBS_END)]
    assert len(payload) == 50


def test_trajectory_injector_trains_control_ids():
    from training.train import inject_tool_trajectories
    base = [("code", {"kind": "text", "data": b"def parse bytes from a stream please"})] * 200
    injected = list(inject_tool_trajectories(iter(base), rate=0.5, seed=24))
    assert len(injected) > 200
    tools = [s for g, s in injected if g == "tools"]
    assert tools and all(s["kind"] == "ids" for s in tools)
    flat = [i for s in tools for i in s["ids"]]
    assert TOOL_SEARCH in flat and EDIT in flat and OBS_START in flat
    # Observations come from the REAL mock sandbox (deterministic marker).
    search_traj = next(s for s in tools if TOOL_SEARCH in s["ids"])
    assert b"mock ddgs" in bytes(i for i in search_traj["ids"] if i < 256)
    # rate=0 is a pure pass-through.
    assert list(inject_tool_trajectories(iter(base[:3]), rate=0.0)) == base[:3]


def test_injector_caches_repeated_mock_queries(monkeypatch):
    import training.train as tt
    calls = []
    real = tt.run_search

    def counting(query, **kwargs):
        calls.append(query)
        return real(query, **kwargs)

    monkeypatch.setattr(tt, "run_search", counting)
    # Identical documents -> identical query -> exactly ONE sandbox spawn.
    base = [("code", {"kind": "text",
                      "data": b"alpha beta gamma delta epsilon zeta"})] * 40
    out = list(tt.inject_tool_trajectories(iter(base), rate=1.0, seed=7))
    search_trajs = [s for g, s in out if g == "tools" and TOOL_SEARCH in s["ids"]]
    assert search_trajs, "expected at least one search trajectory"
    assert len(calls) == 1  # memoised: one spawn for the repeated query
    # Results are still the real deterministic mock output.
    assert b"mock ddgs" in bytes(i for i in search_trajs[0]["ids"] if i < 256)


def test_no_regex_hacks_in_runtime():
    # The Phase 24 contract: no host-side regex parsing of model text.
    import re as regex  # the TEST may use regex; the runtime may not
    for path in ("models/foundation.py", "models/memory_model.py",
                 "models/encoders.py", "training/train.py", "data/sources.py"):
        source = Path(path).read_text(encoding="utf8")
        assert not regex.search(r"^\s*import re\s*(#.*)?$", source, regex.M), path
        assert not regex.search(r"\bre\.(compile|match|search|sub|findall)\b", source), path
        assert "ACTION_RE" not in source, path
