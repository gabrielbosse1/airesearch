"""Phase 22 regression tests: scaled architecture, real-data pipeline, RL, eval.

Everything here runs offline on CPU. No HF download, no GPU, no VRAM claim.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "t4_scaled.yaml"

from data.phase22_sources import (  # noqa: E402
    HF_SEARCH_QUERIES, LICENSE_NOTES, SourceSpec, _self_test as sources_self_test,
    conversation_to_bytes, minecraft_phase_coverage, minecraft_row_to_react,
    strict_utf8, weighted_mixture,
)
from evaluation.final_agent_eval import (  # noqa: E402
    NOVEL_MINECRAFT_GOAL, build_cases, evaluate, oracle_agent, score_case,
)
from models.foundation import (  # noqa: E402
    CausalByteTransformer, FoundationConfig, RMSNorm, SwiGLUFFN,
    apply_rotary, rotary_tables,
)
from training.phase22_common import (  # noqa: E402
    BytePacker, PrecisionPlan, estimate_vram, load_checkpoint, load_config,
    make_grad_scaler, resolve_precision, save_checkpoint, save_deploy_copy,
    vram_fill_ladder,
)
from training.phase22_reward import (  # noqa: E402
    DRAFT_CLOSE, DRAFT_OPEN, PAST_CLOSE, PAST_OPEN, WEIGHTS, EditCommand,
    apply_edit_commands, compute_reward, dominance_report, encode_edit,
    parse_temporal_trajectory,
)


def tiny_config(**overrides):
    values = dict(n_layers=1, d_model=32, n_heads=4, n_kv_heads=2, d_ff=64,
                  max_seq_len=64, dropout=0.0, norm_type="rmsnorm",
                  ffn_type="swiglu", position_type="rope", refinement_steps=2,
                  refinement_rank=8, gradient_checkpointing=False)
    values.update(overrides)
    return FoundationConfig(**values)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
def test_legacy_defaults_are_bit_identical_to_phase16():
    config = FoundationConfig(n_layers=2, d_model=32, n_heads=2, d_ff=64)
    assert config.norm_type == "layernorm"
    assert config.ffn_type == "gelu"
    assert config.position_type == "learned"
    assert config.n_kv_heads == 0
    assert config.refinement_steps == 0
    model = CausalByteTransformer(config)
    assert model.parameters_count == 42309
    assert model.position_embedding is not None
    assert model.refinement is None
    assert isinstance(model.blocks[0].norm_1, torch.nn.LayerNorm)
    assert model.blocks[0].attention.n_kv_heads == 2


def test_t4_scaled_config_targets_150_to_200m():
    config = load_config(CONFIG_PATH)
    model = config["model"]
    assert model["d_model"] == 1024
    assert model["n_layers"] == 12
    assert model["n_heads"] == 8
    assert model["n_kv_heads"] == 2
    assert model["gradient_checkpointing"] is True
    assert model["norm_type"] == "rmsnorm"
    assert model["ffn_type"] == "swiglu"
    assert model["position_type"] == "rope"
    with torch.device("meta"):
        count = CausalByteTransformer(FoundationConfig.from_mapping(model)).parameters_count
    assert 150_000_000 <= count <= 200_000_000
    assert count == 183_830_791


def test_scaled_components_are_instantiated_without_legacy_names():
    model = CausalByteTransformer(tiny_config())
    assert isinstance(model.blocks[0].norm_1, RMSNorm)
    assert isinstance(model.blocks[0].feed_forward, SwiGLUFFN)
    assert model.position_embedding is None       # RoPE, no learned table
    assert model.refinement.steps == 2
    assert model.blocks[0].attention.n_kv_heads == 2
    assert model.blocks[0].attention.repeats == 2
    forbidden = ("copy", "pointer", "gate", "frame", "tool_call", "retro", "verifier")
    names = [name for name, _ in model.named_modules()]
    names += [name for name, _ in model.named_parameters()]
    names += [name for name, _ in model.named_buffers()]
    for name in names:
        assert not any(word in name.lower() for word in forbidden), name


def test_rope_tables_and_relative_shift_invariance():
    cos, sin = rotary_tables(16, 8, 10000.0, torch.device("cpu"), torch.float32)
    assert tuple(cos.shape) == (1, 1, 16, 8)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    m, n = 2, 7
    rotated_q = apply_rotary(q, cos[:, :, m:m + 1], sin[:, :, m:m + 1])
    rotated_k = apply_rotary(k, cos[:, :, n:n + 1], sin[:, :, n:n + 1])
    shifted_k = apply_rotary(k, cos[:, :, n - m:n - m + 1], sin[:, :, n - m:n - m + 1])
    lhs = (rotated_q * rotated_k).sum()
    rhs = (q * shifted_k).sum()   # R_0 q == q
    assert torch.allclose(lhs, rhs, atol=1e-5), (lhs, rhs)


def test_pre_norm_swiglu_gqa_refinement_gradients_are_finite():
    torch.manual_seed(22)
    model = CausalByteTransformer(tiny_config(gradient_checkpointing=True)).train()
    ids = torch.randint(0, 256, (2, 12))
    loss = model.loss(ids)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.refinement.step_scales.grad is not None
    assert float(model.refinement.step_scales.grad.abs().sum()) > 0
    assert all(torch.isfinite(p.grad).all().item()
               for p in model.parameters() if p.grad is not None)


def test_refinement_is_identity_at_initialisation():
    torch.manual_seed(22)
    plain = CausalByteTransformer(tiny_config(refinement_steps=0)).eval()
    refined = CausalByteTransformer(tiny_config(refinement_steps=3)).eval()
    # Copy the shared backbone so only the loop differs.
    refined.load_state_dict(
        {k: v for k, v in plain.state_dict().items()}, strict=False)
    ids = torch.randint(0, 256, (1, 9))
    with torch.no_grad():
        torch.testing.assert_close(plain(ids), refined(ids), atol=1e-6, rtol=1e-6)


def test_config_validation_rejects_bad_upgrades():
    with pytest.raises(ValueError):
        FoundationConfig(n_kv_heads=3)                 # does not divide n_heads=4
    with pytest.raises(ValueError):
        FoundationConfig(norm_type="batchnorm")
    with pytest.raises(ValueError):
        FoundationConfig(position_type="rope", d_model=28, n_heads=4)  # odd head_dim
    with pytest.raises(ValueError):
        FoundationConfig(refinement_steps=-1)


# ---------------------------------------------------------------------------
# Precision + VRAM
# ---------------------------------------------------------------------------
def test_precision_falls_back_to_fp16_on_turing_t4():
    with (mock.patch("torch.cuda.is_bf16_supported", return_value=False),
          mock.patch("torch.cuda.get_device_capability", return_value=(7, 5))):
        plan = resolve_precision(torch.device("cuda"), "bfloat16", "auto")
    assert plan.resolved == "float16"
    assert plan.bf16_supported is False
    assert plan.scaler_required is True
    assert plan.fallback is True
    assert plan.capability == (7, 5)
    assert "sm_75" in plan.reason


def test_precision_uses_bf16_on_ampere():
    with (mock.patch("torch.cuda.is_bf16_supported", return_value=True),
          mock.patch("torch.cuda.get_device_capability", return_value=(8, 0))):
        plan = resolve_precision(torch.device("cuda"), "bfloat16", "auto")
    assert plan.resolved == "bfloat16"
    assert plan.scaler_required is False
    assert plan.fallback is False


def test_cpu_precision_is_explicitly_unmeasured():
    plan = resolve_precision(torch.device("cpu"), "bfloat16", "auto")
    assert plan.resolved == "float32"
    assert plan.autocast is False
    assert make_grad_scaler(plan) is None


def test_vram_config_targets_10_to_12gb_and_ladder_is_honest():
    config = load_config(CONFIG_PATH)
    assert config["training"]["requested_precision"] == "bfloat16"
    assert config["training"]["micro_batch_size"] == 4
    assert config["training"]["grad_accum"] == 8
    assert config["model"]["gradient_checkpointing"] is True
    assert config["vram"]["target_min_gb"] == 10.0
    assert config["vram"]["target_max_gb"] == 12.0
    estimate = estimate_vram(config, batch_size=4, seq_len=1024)
    assert estimate["parameters"] == 183_830_791
    assert estimate["estimate_only"] is True
    assert estimate["estimated_total_gb"] < config["vram"]["device_total_gb"]
    ladder = vram_fill_ladder(config, device=torch.device("cpu"))
    assert ladder["measured"] is False
    assert ladder["target_met"] is False
    assert ladder["target_band_gb"] == [10.0, 12.0]
    assert ladder["defaults"]["micro_batch_size"] == 4
    assert any(t["estimated_gb"] >= 10.0 for t in ladder["trials"])
    assert all(t["measured"] is False for t in ladder["trials"])


# ---------------------------------------------------------------------------
# Lossless packing + resume
# ---------------------------------------------------------------------------
def test_byte_packer_is_lossless_across_utf8_boundaries_and_resume():
    docs = ["Olá, mundo! ".encode(), "cobblestone → gerador ".encode(),
            "日本語のテキスト ".encode(), b"a" * 200] * 25
    packer = BytePacker(37)
    items = []
    for doc in docs:
        items += packer.feed(doc, trainable_prefix=5)
    expected = b"".join(docs)
    assert packer.decode_all(items) == expected
    # No emitted sequence may end inside a UTF-8 continuation byte.
    for ids, weights in items:
        assert tuple(ids.shape) == tuple(weights.shape)
        assert int(ids.max()) < 261
    restored = BytePacker(37)
    restored.load_state_dict(packer.state_dict())
    assert restored.carry == packer.carry
    assert restored.carry_labels == packer.carry_labels
    assert restored.decode_all(items) == expected
    assert 0.0 in items[0][1].tolist() and 1.0 in items[0][1].tolist()


def test_strict_utf8_rejects_bad_rows_without_replacement():
    assert strict_utf8("Olá") == "Olá".encode()
    assert strict_utf8("bad\x00nul") is None
    assert strict_utf8("lone\ud800") is None


def test_checkpoint_roundtrip_restores_everything():
    model = CausalByteTransformer(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "checkpoint_phase1.pt"
        info = save_checkpoint(path, model, phase=1, step=11, optimizer=optimizer,
                               extra={"packer": BytePacker(32).state_dict()})
        assert info["bytes"] > 0
        fresh = CausalByteTransformer(tiny_config())
        fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
        restored = load_checkpoint(path, fresh, optimizer=fresh_optimizer,
                                   expected_phase=1)
        assert restored["step"] == 11
        assert restored["rng_restored"] is True
        assert restored["optimizer_restored"] is True
        assert all(torch.equal(a, b) for a, b in zip(
            model.state_dict().values(), restored["model"].state_dict().values()))
        deploy = save_deploy_copy(Path(folder) / "deploy_fp16.pt", path)
        assert deploy["bytes"] < info["bytes"]
        with pytest.raises(ValueError):
            load_checkpoint(path, expected_phase=2)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
def test_reward_weights_are_exactly_as_requested():
    assert WEIGHTS.correct_final == 1.0
    assert WEIGHTS.per_edit_token == -0.5
    assert WEIGHTS.verified_edit_fix == 0.8
    assert WEIGHTS.wrong_guess_without_search < -1.0
    assert WEIGHTS.search_and_use > 1.0
    assert WEIGHTS.max_efficient_edits == 2


def test_search_beats_guess_and_efficient_edit_beats_thrashing():
    report = dominance_report()
    assert report["all_satisfied"] is True
    assert report["ordering"]["search_gt_guess"] is True
    assert report["ordering"]["search_gt_lucky_guess"] is True
    assert report["ordering"]["efficient_edit_gt_thrashing"] is True
    assert report["ordering"]["past_action_is_immutable"] is True
    assert report["rewards"]["search_and_use"]["total"] > report["rewards"]["wrong_guess"]["total"]
    assert report["rewards"]["efficient_edit"]["total"] > report["rewards"]["thrashing"]["total"]
    assert report["rewards"]["efficient_edit"]["verified_fix"] is True
    assert report["rewards"]["efficient_edit"]["edits"] == 1


def test_temporal_markers_and_lossless_edit_commands():
    draft = b"agua + lava produzem obsidiana"
    offset = draft.find(b"obsidiana")
    command = encode_edit(at=offset, delete=9, insert=b"pedregulho")
    edits = parse_temporal_trajectory(DRAFT_OPEN + command + DRAFT_CLOSE).draft_edits
    assert len(edits) == 1
    assert apply_edit_commands(draft, edits) == b"agua + lava produzem pedregulho"
    illegal = PAST_OPEN + command + PAST_CLOSE + DRAFT_OPEN + command + DRAFT_CLOSE
    parsed = parse_temporal_trajectory(illegal)
    assert parsed.illegal_edits == 1
    case = {"expected_answer": b"pedregulho", "initial_draft": draft,
            "expected_draft": b"agua + lava produzem pedregulho",
            "search_required": False}
    clean = compute_reward(DRAFT_OPEN + command + DRAFT_CLOSE + b"\nAnswer: pedregulho\n", case)
    dirty = compute_reward(illegal + b"\nAnswer: pedregulho\n", case)
    assert clean.total > dirty.total
    assert clean.flags["temporal_violation"] is False
    assert dirty.flags["temporal_violation"] is True


# ---------------------------------------------------------------------------
# Real dataset registry
# ---------------------------------------------------------------------------
def test_dataset_registry_has_verified_ids_and_licences():
    report = sources_self_test()
    assert report["react_labels_present"] is True
    assert report["no_fabricated_observation"] is True
    assert len(HF_SEARCH_QUERIES) >= 10
    assert "NON-COMMERCIAL" in LICENSE_NOTES["HuggingFaceH4/no_robots"]
    assert "distilabel" in LICENSE_NOTES["argilla/distilabel-sft-easy"]
    python_spec = SourceSpec.from_mapping(
        {"id": "bigcode/the-stack", "data_dir": "data/python"})
    assert python_spec.load_kwargs()["data_dir"] == "data/python"
    assert python_spec.load_kwargs()["streaming"] is True
    pt_spec = SourceSpec.from_mapping(
        {"id": "HuggingFaceFW/fineweb-2", "name": "por_Latn"})
    assert pt_spec.load_kwargs()["name"] == "por_Latn"


def test_config_phase1_and_phase2_sources_are_real_hf_ids():
    config = load_config(CONFIG_PATH)
    ids = {spec["id"] for group in config["data"]["phase1"] for spec in group["sources"]}
    ids |= {spec["id"] for group in config["data"]["phase2"] for spec in group["sources"]}
    assert "bigcode/the-stack" in ids
    assert "TucanoBR/GigaVerbo" in ids
    assert "HuggingFaceFW/fineweb-edu" in ids
    assert "open-web-math/open-web-math" in ids
    assert "HuggingFaceH4/no_robots" in ids
    assert "argilla/distilabel-sft-easy" in ids
    assert "TESS-Computer/minecraft-vla-stage3" in ids
    assert "bigcode/the-stack-v2" not in ids          # gated, rejected
    assert "oscar-corpus/OSCAR-2301" not in ids       # suspended, rejected
    schedule = weighted_mixture(config["data"]["phase1"])
    assert len(schedule) >= 20
    assert all(s.streaming for s in schedule)


def test_minecraft_converter_does_not_invent_observations():
    row = {"actions": ["explore biome", "mine cobblestone", "place cobblestone"]}
    transcript = minecraft_row_to_react(row, provenance="reactor-team/minecraft-vpt-mp4")
    assert transcript is not None
    assert b"[source: no textual observation]" in transcript
    assert b"Ladder: explore -> gather -> craft -> build" in transcript
    assert transcript.count(b"Action:") == 3
    coverage = minecraft_phase_coverage([transcript])
    assert coverage["gather"] >= 1 and coverage["build"] >= 1
    assert coverage["craft"] < 2   # metadata line must not inflate coverage
    chat = conversation_to_bytes({"messages": [
        {"role": "user", "content": "Olá"}, {"role": "assistant", "content": "Oi!"}]})
    assert chat and "Olá".encode() in chat and b"assistant: Oi!" in chat


# ---------------------------------------------------------------------------
# Notebooks, docs, evaluation, smoke test
# ---------------------------------------------------------------------------
def test_three_notebooks_are_valid_resumable_and_compile():
    nbformat = pytest.importorskip("nbformat")
    names = ["01_setup_and_pretrain.ipynb", "02_distill_and_align.ipynb",
             "03_rl_and_agent.ipynb"]
    checkpoints = ["checkpoint_phase1.pt", "checkpoint_phase2.pt", "final_model.pt"]
    for name, checkpoint in zip(names, checkpoints):
        path = ROOT / "notebook" / name
        assert path.is_file()
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        sources = "\n".join(cell.source for cell in nb.cells)
        assert checkpoint in sources, name
        assert "resume" in sources.lower() or "RESUME" in sources, name
        code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
        assert len(code_cells) >= 6
        for index, cell in enumerate(code_cells):
            compile(cell.source, f"{name}:cell{index}", "exec")
    sources2 = (ROOT / "notebook" / names[1]).read_text(encoding="utf8")
    sources3 = (ROOT / "notebook" / names[2]).read_text(encoding="utf8")
    assert "init_from_phase=1" in sources2
    assert "init_from_phase=2" in sources3


def test_docs_have_exactly_one_phase22_section_with_required_terms():
    for name in ("AGENTS.md", "QWEN_CHAT_PROMPT.md", "README.md"):
        text = (ROOT / name).read_text(encoding="utf8")
        assert text.count("## Phase 22:") == 1, name
        assert text.count("## Phase 21:") == 1, name
        for term in ("SwiGLU", "RMSNorm", "RoPE", "GQA", "Iterative Refinement",
                     "checkpoint_phase1.pt", "final_model.pt", "bfloat16"):
            assert term in text, (name, term)


def test_final_eval_oracle_passes_but_is_not_a_model_claim():
    report = evaluate(root=ROOT)
    assert report["oracle_mode"] is True
    assert report["cases_passed"] == 4
    assert report["all_passed"] is True
    assert report["decontamination"]["status"] in ("unverified", "clean")
    assert report["decontamination"]["shards_available"] is False
    names = {case.name for case in build_cases()}
    assert names == {"minecraft_novel_goal", "search_required",
                     "one_or_two_edits", "memory_with_distractor"}
    assert NOVEL_MINECRAFT_GOAL == b"Build a cobblestone generator"


def test_final_eval_rejects_a_guessing_trajectory():
    case = next(c for c in build_cases() if c.name == "search_required")
    guess = case.prompt + b"Answer: eu acho que sao 42 downloads\n"
    result = score_case(case, guess)
    assert result.passed is False
    assert result.checks["used_search_tool"] is False
    assert result.checks["used_observation"] is False
    assert result.checks["search_result_used"] is False
    oracle = score_case(case, oracle_agent(case))
    assert oracle.passed is True


def test_benchmark_artefacts_are_honest_when_present():
    path = ROOT / "runs" / "phase22" / "benchmark.json"
    if not path.is_file():
        pytest.skip("run `python -m evaluation.phase22_benchmark` to generate reports")
    payload = json.loads(path.read_text(encoding="utf8"))
    assert payload["architecture"]["parameter_breakdown"]["exact_match"] is True
    assert payload["parameters"]["scaled"] == 183_830_791
    assert payload["vram"]["measured_vram_gb"] is None
    assert payload["vram"]["fill_ladder_measured"] is False
    assert payload["final_eval"]["oracle_mode"] is True
    assert payload["dataset_inventory"]["no_downloads_in_sandbox"] is True
    assert payload["precision"]["resolved"] == "float32"  # CPU sandbox
    inventory = ROOT / "runs" / "phase22" / "dataset_inventory.json"
    assert inventory.is_file()
    ids = {item["id"] for item in json.loads(inventory.read_text(encoding="utf8"))["phase1"]}
    assert "TucanoBR/GigaVerbo" in ids


def test_phase22_cli_self_tests_all_pass():
    with tempfile.TemporaryDirectory() as folder:
        commands = [
            [sys.executable, "-m", "data.phase22_sources", "--self-test"],
            [sys.executable, "-m", "training.phase22_reward", "--self-test"],
            [sys.executable, "-m", "training.phase22_train", "--self-test"],
            [sys.executable, "-m", "evaluation.final_agent_eval", "--self-test",
             "--out", str(Path(folder) / "final_eval.json")],
        ]
        for command in commands:
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                       timeout=180)
            assert completed.returncode == 0, (command, completed.stderr[-1500:])
            if command[-1] == "--self-test":
                payload = json.loads(completed.stdout)
                assert "all_satisfied" not in payload or payload["all_satisfied"] is True
                assert "measured_vram" not in payload or payload["measured_vram"] is False
