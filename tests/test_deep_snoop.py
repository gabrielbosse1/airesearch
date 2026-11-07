"""Offline regression tests for the Phase 26 checkpoint snoop."""

from pathlib import Path

import nbformat
import torch
import yaml

from evaluation.deep_snoop import (
    ProbeRecorder, _syntax_check_python, architecture_report, build_model,
    generate_trace, load_payload, select_checkpoint_file,
)
from models.foundation import CausalByteTransformer, FoundationConfig
from models.memory_model import SkillBankView


def tiny_model():
    cfg = FoundationConfig(
        d_model=16, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=32,
        max_seq_len=64, dropout=0.0, memory_dim=32,
        refinement_steps=2, refinement_rank=8, max_thought_steps=2,
        attention_type="sliding_window", window_size=8, sliding_chunk_size=4,
        memory_cross_attention=True,
    )
    return CausalByteTransformer(cfg).eval()


def test_checkpoint_picker_prefers_latest_then_final_then_highest_step():
    assert select_checkpoint_file(["checkpoints/step_80.pt", "checkpoints/checkpoint_latest.pt"]) == \
        "checkpoints/checkpoint_latest.pt"
    assert select_checkpoint_file(["checkpoints/checkpoint_final.pt", "checkpoint_step_120.pt"]) == \
        "checkpoints/checkpoint_final.pt"
    assert select_checkpoint_file(["checkpoint_step_90.pt", "checkpoint_step_120.pt"]) == \
        "checkpoint_step_120.pt"


def test_production_architecture_report_shows_actual_tiny_shapes():
    model = tiny_model()
    report = architecture_report(model)
    assert report["parameters"] == model.parameters_count
    assert report["production_parameter_count_match"] is False
    assert report["layers"][0]["query_weight"] == [16, 16]
    assert report["layers"][0]["skill_glu_weight"] == [32, 32]
    assert report["layers"][0]["cross_v_weight"] == [16, 32]
    assert report["global_shapes"]["thought_slots"] == [1, 2, 16]
    assert report["expected_config_checks"]["window_size"]["match"] is False


def test_production_checkpoint_geometry_is_verified_on_meta():
    repo = Path(__file__).resolve().parents[1]
    config = FoundationConfig.from_mapping(yaml.safe_load(
        (repo / "configs/t4_scaled.yaml").read_text(encoding="utf-8"))["model"])
    with torch.device("meta"):
        model = CausalByteTransformer(config)
    report = architecture_report(model)
    assert report["parameters"] == 366_316_833
    assert report["production_config_match"] is True
    assert report["global_shapes"]["thought_slots"] == [1, 4, 1024]
    assert report["layers"][0]["cross_v_weight"] == [1024, 8192]
    assert report["layers"][7]["skill_glu_weight"] == [2048, 8192]


def test_safe_mmap_checkpoint_load_and_meta_assignment(tmp_path):
    model = tiny_model()
    path = tmp_path / "checkpoint_latest.pt"
    torch.save({"schema_version": 7, "config": model.config.to_dict(),
                "model": model.state_dict(), "step": 42}, path)
    payload = load_payload(path)
    restored = build_model(payload, torch.device("cpu"))
    assert restored.config == model.config
    assert restored.parameters_count == model.parameters_count
    ids = torch.randint(0, 256, (1, 7))
    with torch.no_grad():
        assert torch.allclose(model(ids), restored(ids))


def test_probe_records_thought_ffn_and_attention_with_memory():
    torch.manual_seed(4)
    model = tiny_model()
    memory = SkillBankView(torch.randn(3, 16), torch.randn(3, 32), torch.ones(3, dtype=torch.bool))
    with ProbeRecorder(model) as recorder:
        result = generate_trace(model, "hi", recorder, memory=memory,
                                max_new_bytes=2, target_slot=1)
    assert len(result["thought_trace"]) > 0
    assert result["thought_buffer_l2_mean"] > 0
    assert result["adaptive"]["buffer_gate"] == 1.0
    assert "ffn_layer_0_output_l2" in result["activation_summary"]
    assert result["attention_last_generated_position"]["0"]["attended_key_count"] > 0
    cross = result["memory_attention_last_generated_position"]["0"]
    assert cross["slots"] == 3
    assert 0 <= cross["injected_fact_weight"] <= 1
    assert cross["injected_fact_rank"] in (1, 2, 3)


def test_notebook_documents_optional_phase26_snoop():
    repo = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(repo / "pretrain_4h.ipynb", as_version=4)
    assert "Phase 26" in notebook.cells[-2].source
    assert "gabrielbosse9/airesearch2" in notebook.cells[-2].source
    assert "RUN_DEEP_SNOOP = False" in notebook.cells[-1].source
    assert "evaluation/deep_snoop.py" in notebook.cells[-1].source


def test_python_check_is_syntax_only():
    good = _syntax_check_python("def quicksort(arr):\n    return arr")
    bad = _syntax_check_python("def quicksort(arr)\n    return arr")
    assert good["compiles"] is True
    assert bad["compiles"] is False
    assert "SyntaxError" in bad["error"]
