"""Phase 24 cleanup contract: the repo stays stripped to the bone."""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

REQUIRED = [
    "models/foundation.py", "models/memory_model.py", "models/encoders.py",
    "training/train.py", "data/sources.py",
    "configs/t4_scaled.yaml", "pretrain_4h.ipynb",
    "README.md", "ARCHITECTURE_AUDIT.md", "AGENTS.md",
    "requirements.txt", ".gitignore",
    "models/__init__.py", "training/__init__.py", "data/__init__.py",
    "tests/__init__.py",
]

FORBIDDEN = [
    "QWEN_CHAT_PROMPT.md", "configs/default.yaml",
    "data/phase22_sources.py", "evaluation",
    "models/contradiction_graph.py", "models/foundation_memory.py",
    "models/holdout.py", "models/procedural.py", "models/react.py",
    "models/sandbox.py", "notebook", "runs/phase22",
    "training/phase22_common.py", "training/phase22_reward.py",
    "training/phase22_train.py",
    "tests/test_phase22.py", "tests/test_react_runtime.py",
]


@pytest.mark.parametrize("path", REQUIRED)
def test_required_files_exist(path):
    assert (REPO / path).exists(), path


@pytest.mark.parametrize("path", FORBIDDEN)
def test_legacy_paths_are_gone(path):
    assert not (REPO / path).exists(), path


def test_no_legacy_phase_scripts_anywhere():
    for path in REPO.rglob("*.py"):
        if ".git" in path.parts:
            continue
        assert "phase22" not in path.name, path
        assert "phase2" not in path.name or "phase" not in path.name, path


def test_no_other_training_scripts():
    scripts = sorted(p.name for p in (REPO / "training").glob("*.py"))
    assert scripts == ["__init__.py", "train.py"]


def test_no_host_side_regex_parsers_in_models():
    import re as regex
    for path in (REPO / "models").glob("*.py"):
        source = path.read_text(encoding="utf8")
        assert "ACTION_RE" not in source
        assert "QUOTED_ARG_RE" not in source
        assert "REF_SPAN_RE" not in source
        assert not regex.search(r"^\s*import re\s*(#.*)?$", source, regex.M)
        assert not regex.search(r"\bre\.(compile|match|search|sub|findall)\b", source)
