"""The single Colab notebook: valid nbformat, parseable cells, T4-ready."""
import ast
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "pretrain_4h.ipynb"


def test_exactly_one_notebook_in_repo():
    notebooks = [p for p in REPO.rglob("*.ipynb")
                 if ".git" not in p.parts and "node_modules" not in p.parts]
    assert notebooks == [NOTEBOOK]


def test_notebook_is_valid_nbformat():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) >= 8


def test_all_code_cells_are_valid_python():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source)  # no magics: pure Python, parseable


def test_notebook_targets_the_right_run():
    source = NOTEBOOK.read_text(encoding="utf8")
    assert "gabrielbosse9/airesearch" in source          # HF checkpoint target
    assert "t4_scaled.yaml" in source                    # production preset
    assert "training.train" in source                    # the ONE script
    assert "220" in source                               # minute budget
    assert "checkpoint-minutes" in source and "30" in source
    assert "T4" in source
    assert "fp16" in source or "float16" in source       # T4 precision fallback
    assert "HF_TOKEN" in source


def test_notebook_has_no_legacy_phase_references():
    source = NOTEBOOK.read_text(encoding="utf8")
    for legacy in ("phase22", "phase1_", "01_setup", "02_distill", "03_rl"):
        assert legacy not in source
