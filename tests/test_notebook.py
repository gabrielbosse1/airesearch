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


def test_notebook_audits_the_ptbr_mix_before_training():
    """24.1.2: the mix verification cell must run BEFORE the 4-hour launch."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    code = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    audit = next(src for src in code if "audit_mixture" in src)
    train = next(src for src in code if "training.train" in src)
    assert code.index(audit) < code.index(train), "audit must precede the run"
    assert "format_mixture_report" in audit
    assert "mix_audit_batches" in audit and "mix_audit_tolerance" in audit
    assert "max_seconds" in audit                      # cannot hang the notebook
    assert "quotas_ok" in audit and "ptbr" in audit    # hard gate on 50% PT-BR
    source = NOTEBOOK.read_text(encoding="utf8")
    assert "50/25/15/10" in source and "PT-BR" in source
    assert "40/30/20/10" not in source                 # legacy mix is gone


def test_notebook_verifies_zero_disk_streaming_before_training():
    """24.2.3: the Zero-Disk cell runs right after data init, before training."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    code = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    audit = next(src for src in code if "audit_mixture" in src)
    zero = next(src for src in code if "disk_usage" in src)
    train = next(src for src in code if "training.train" in src)
    # Right after data initialization (the audit cell builds the mixture)...
    assert code.index(zero) == code.index(audit) + 1
    # ...and strictly before the 4-hour launch.
    assert code.index(zero) < code.index(train)
    # 1) prints + asserts the dataset TYPE (must be IterableDataset)
    assert "IterableDataset" in zero and "isinstance" in zero
    assert "open_stream" in zero and "type(stream)" in zero
    # 2) disk usage before/after the first 10 streamed batches
    assert "range(10)" in zero and "next_batch" in zero
    assert "before_mb" in zero and "after_mb" in zero
    # 3) asserts the growth is negligible
    assert "growth_mb" in zero and "MAX_GROWTH_MB" in zero
    assert "assert growth_mb <= MAX_GROWTH_MB" in zero
    # the iterable-safe shuffle contract is visible in the cell output
    assert "buffer_size" in zero


def test_notebook_has_no_legacy_phase_references():
    source = NOTEBOOK.read_text(encoding="utf8")
    for legacy in ("phase22", "phase1_", "01_setup", "02_distill", "03_rl"):
        assert legacy not in source
