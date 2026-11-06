"""The single Colab notebook: valid nbformat, parseable cells, T4-ready."""
import ast
import json
from pathlib import Path

import nbformat
import pytest

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
    assert "gabrielbosse9/airesearch2" in source         # HF checkpoint target
    assert "t4_scaled.yaml" in source                    # production preset
    assert "training.train" in source                    # the ONE script
    assert "220" in source                               # minute budget
    assert "checkpoint-minutes" in source and "10" in source
    assert "upload-minutes" in source and "5" in source
    assert "T4" in source
    assert "fp16" in source or "float16" in source       # T4 precision fallback
    assert "HF_TOKEN" in source
    assert "userdata.get" in source and "getpass" in source
    assert "arena/b49dc41e-airesearch" in source      # delivery provenance
    assert "arena/6adc4370-airesearch" not in source
    nb = nbformat.read(NOTEBOOK, as_version=4)
    clone = next(cell.source for cell in nb.cells
                 if cell.cell_type == "code" and "REPO_URL" in cell.source)
    assert 'os.environ.get("AIRESEARCH_BRANCH", "main")' in clone
    assert "debug_bundle.zip" in source and "sync_run_folder" in source


def test_notebook_audits_the_ptbr_mix_before_training():
    """24.1.2: the mix verification cell must run BEFORE the 4-hour launch."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    code = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    audit = next(src for src in code if "audit_mixture" in src)
    train = next(src for src in code if "--hf-repo" in src and "Popen" in src)
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
    train = next(src for src in code if "--hf-repo" in src and "Popen" in src)
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


def test_notebook_launch_captures_console_and_syncs_even_on_failure():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    launch = next(cell.source for cell in nb.cells
                  if cell.cell_type == "code" and "--hf-repo" in cell.source and "Popen" in cell.source)
    assert "notebook_launch.log" in launch
    assert "stderr=subprocess.STDOUT" in launch
    assert "finally:" in launch and "sync_run_folder" in launch
    assert "strict=False" in launch  # diagnostics must not mask the original failure
    assert "launch_exit.json" in launch


def test_notebook_contains_operator_instructions_and_debug_fetch():
    source = NOTEBOOK.read_text(encoding="utf8")
    assert "https://colab.research.google.com/github/gabrielbosse1/airesearch/blob/main/pretrain_4h.ipynb" in source
    assert "Runtime → Run all" in source
    assert "Notebook access" in source and "write" in source
    assert "Disconnected?" in source and "checkpoint_latest.pt" in source

    nb = nbformat.read(NOTEBOOK, as_version=4)
    code = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    fetch = next(src for src in code if "FETCH_DEBUG_BUNDLE" in src)
    assert "runs/latest.json" in fetch and "debug_bundle.zip" in fetch
    assert "hf_hub_download" in fetch and "remote_run_path" in fetch
    assert "files.download" in fetch


def test_notebook_monitors_sigkill_and_releases_preflight_ram():
    source = NOTEBOOK.read_text(encoding="utf8")
    assert "Restart session" in source and "SIGKILL" in source
    assert "buffer_size=1000" in source and "100 decoded rows" in source
    nb = nbformat.read(NOTEBOOK, as_version=4)
    code = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    gpu = next(src for src in code if src.startswith("# 1) GPU check"))
    assert "nvidia-smi" in gpu and "torch.cuda" not in gpu  # no parent CUDA context
    zero = next(src for src in code if "disk_usage" in src)
    assert "del mix, stream, report" in zero and "gc.collect()" in zero
    launch = next(src for src in code if "--hf-repo" in src and "Popen" in src)
    for marker in ("threading.Thread", "notebook_monitor.jsonl", "host_memory_snapshot",
                   "explain_child_exit", "notebook_failure.json", "make_debug_bundle",
                   "publish_monitor_pointer", "monitor_stop.set()", "monitor.join"):
        assert marker in launch, marker
    assert "notebook_upload_seconds" in launch and "notebook_monitor_seconds" in launch


def test_debug_fetch_prefers_local_bundle_after_failed_launch():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    fetch = next(cell.source for cell in nb.cells
                 if cell.cell_type == "code" and "FETCH_DEBUG_BUNDLE" in cell.source)
    assert 'globals().get("RUN_DIR")' in fetch
    assert "using local debug bundle" in fetch and "using remote debug bundle" in fetch
    assert "raise SystemExit" not in fetch


def test_launch_monitor_retains_real_sigkill_bundle_with_mocked_hub(tmp_path, monkeypatch):
    """Execute the notebook launch cell against a child that SIGKILLs itself.

    This is a local signal-path test, not a Colab OOM measurement. Every Hub
    operation is mocked; the parent must still write/download a debug bundle.
    """
    import os
    import training.train as train

    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = next(cell.source for cell in nb.cells
                  if cell.cell_type == "code" and "--hf-repo" in cell.source and "Popen" in cell.source)
    begin = source.index("cmd = [")
    end = source.index('print("launching:"', begin)
    child = "import os,signal,time; print('child started', flush=True); time.sleep(0.25); os.kill(os.getpid(), signal.SIGKILL)"
    source = source[:begin] + f"cmd = [sys.executable, '-c', {child!r}]\n" + source[end:]
    syncs, pointers = [], []

    def sync(*args, **kwargs):
        syncs.append((args, kwargs))
        return {"uploaded": True, "path_in_repo": args[2]}

    def upload(path, repo, remote, **kwargs):
        pointers.append(remote)
        return {"uploaded": True, "path_in_repo": remote}

    monkeypatch.setattr(train, "sync_run_folder", sync)
    monkeypatch.setattr(train, "hf_upload", upload)
    namespace = {
        "config": {"runtime": {"notebook_monitor_seconds": 0.03,
                               "notebook_upload_seconds": 0.03, "min_host_available_gb": 0}},
        "RUN_DIR": tmp_path, "RUN_ID": "sigkill-test", "HF_REPO": "user/repo",
        "os": os, "write_json": train.write_json, "utc_stamp": train.utc_stamp,
        "redact_secrets": train.redact_secrets, "resolve_hf_token": lambda: "hf_private_token",
        "hf_upload": upload, "sync_run_folder": sync,
        "publish_preflight": lambda *a, **k: {"uploaded": True},
    }
    with pytest.raises(AssertionError, match="training failed"):
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    exit_record = json.loads((tmp_path / "launch_exit.json").read_text())
    assert exit_record["exit_code"] == -9 and exit_record["signal_name"] == "SIGKILL"
    assert exit_record["confirmed_oom_kill"] is False  # no invented Colab evidence
    assert (tmp_path / "notebook_failure.json").exists()
    assert (tmp_path / "failure.json").exists()
    assert (tmp_path / "debug_bundle.zip").exists()
    heartbeats = [json.loads(line) for line in (tmp_path / "notebook_monitor.jsonl").read_text().splitlines()]
    assert any(row["event"] == "heartbeat" for row in heartbeats)
    assert syncs and "runs/latest.json" in pointers


def test_notebook_preflights_native_precision_without_parent_cuda_context():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    code = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    gpu = next(src for src in code if src.startswith("# 1) GPU check"))
    assert "compute_cap" in gpu and "torch.cuda" not in gpu
    config_cell = next(src for src in code if src.startswith("# 5) Configuration"))
    assert "resolve_precision" in config_cell and "precision_plan" in config_cell
    assert 'publish_preflight("precision"' in config_cell
    assert "bf16 emulation must not be treated as native" in config_cell
    assert "CUDA precision preflight failed" in config_cell
    assert "loss_backward" in config_cell
    assert "mixed-precision loss/backward preflight: PASS" in config_cell


def test_notebook_precision_child_executes_loss_backward_offline(capsys):
    """Run the notebook's embedded child script on CPU with fp16 autocast.

    The CUDA device strings and autocast context are substituted for an
    offline CPU execution. This validates the child script, JSON contract and
    real model loss/backward path without a GPU, Hub access or a parent CUDA
    context; it is not a T4 kernel/VRAM measurement.
    """
    nb = nbformat.read(NOTEBOOK, as_version=4)
    config_cell = next(cell.source for cell in nb.cells
                       if cell.cell_type == "code" and cell.source.startswith("# 5) Configuration"))
    tree = ast.parse(config_cell)
    assignment = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "precision_code"
                              for target in node.targets))
    child = ast.literal_eval(assignment.value)
    assert "scaler.scale(loss).backward()" in child
    assert "confidence_logits" not in child  # the model owns the numerically safe loss
    child = child.replace('"cuda"', '"cpu"')
    child = child.replace("plan.autocast_context()", 'torch.autocast("cpu", dtype=torch.float16)')
    namespace = {"__name__": "__main__"}
    exec(compile(child, str(NOTEBOOK) + ":precision_code", "exec"), namespace)
    output = capsys.readouterr().out
    report = json.loads(output.strip().splitlines()[-1])
    assert report["precision"]["resolved"] == "float32"  # CPU PrecisionPlan
    assert report["loss_backward"]["passed"] is True
    assert report["loss_backward"]["device"] == "cpu"
    assert report["loss_backward"]["model_scale"] == "tiny"
    assert report["loss_backward"]["gradient_checkpointing"] is True
    assert report["loss_backward"]["loss"] > 0.0
    assert "confidence" in report["loss_backward"]["paths"]


def _notebook_auth_cell():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    return next(cell.source for cell in nb.cells
                if cell.cell_type == "code" and cell.source.startswith("# 4) Hugging Face auth"))


def test_notebook_auth_normalizes_colab_dict_secret_before_env(monkeypatch, tmp_path, capsys):
    import os
    import sys
    import types
    import getpass
    import training.train as train

    source = _notebook_auth_cell()
    assert "normalize_hf_token" in source
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    os.environ["BYTECORE_RUN_ID"] = "auth-test"

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    userdata = types.ModuleType("google.colab.userdata")
    userdata.get = lambda name: {"value": "hf_private_token"}
    colab.userdata = userdata
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", userdata)
    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: pytest.fail("valid wrapped secret must not prompt"))
    calls = []

    def verify(repo_id, *, token=None, run_dir=None):
        calls.append((repo_id, token, run_dir))
        return {"repo_id": repo_id, "user": "tester", "write_verified": True}

    monkeypatch.setattr(train, "verify_hf_access", verify)
    exec(compile(source, str(NOTEBOOK) + ":auth", "exec"), {"__name__": "__main__"})
    assert os.environ["HF_TOKEN"] == "hf_private_token"
    assert calls and calls[0][1] == "hf_private_token"
    assert "hf_private_token" not in capsys.readouterr().out


def test_notebook_auth_falls_back_to_hidden_input_for_unsupported_secret(monkeypatch, tmp_path):
    import os
    import sys
    import types
    import getpass
    import training.train as train

    source = _notebook_auth_cell()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    os.environ["BYTECORE_RUN_ID"] = "auth-test"

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    userdata = types.ModuleType("google.colab.userdata")
    userdata.get = lambda name: {"error": "SecretNotFoundError"}
    colab.userdata = userdata
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", userdata)
    monkeypatch.setattr(getpass, "getpass", lambda *a, **k: "hf_fallback_token")
    monkeypatch.setattr(train, "verify_hf_access",
                        lambda repo_id, *, token=None, run_dir=None: {"repo_id": repo_id, "user": "tester"})
    exec(compile(source, str(NOTEBOOK) + ":auth", "exec"), {"__name__": "__main__"})
    assert os.environ["HF_TOKEN"] == "hf_fallback_token"


def test_notebook_documents_bounded_audio_probe():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    config_cell = next(cell.source for cell in nb.cells
                       if cell.cell_type == "code" and cell.source.startswith("# 5) Configuration"))
    assert "audio encoder probe bound:" in config_cell
    assert "max_audio_seconds" in config_cell
    assert "16000" in config_cell
    source = NOTEBOOK.read_text(encoding="utf8")
    assert "Phase 25.7" in source
    assert "Trainer.max_audio_samples" in source
