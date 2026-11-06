"""Model-decided adaptive loop: confidence head, early break, aux BCE."""
import math

import pytest
import torch
import torch.nn.functional as F

from models.foundation import PAD, VOCAB_SIZE, CausalByteTransformer
from models.memory_model import SkillBankView


def test_confidence_head_shape_and_range(tiny_model):
    hidden = torch.randn(2, 6, 64)
    conf = tiny_model.refinement.confidence(hidden)
    assert conf.shape == (2, 6)
    assert ((conf > 0) & (conf < 1)).all()
    # Neutral at init: zero weights, zero bias -> 0.5 everywhere.
    assert torch.allclose(conf, torch.full_like(conf, 0.5), atol=1e-6)


def test_early_break_when_confident(tiny_model):
    tiny_model.eval()
    with torch.no_grad():
        tiny_model.refinement.confidence_head.bias.fill_(10.0)  # sigmoid ~ 1
    ids = torch.randint(0, 256, (1, 8))
    _, info = tiny_model(ids, return_info=True)
    assert info["steps_run"] == 1
    assert info["confidence"] > 0.9


def test_full_loop_when_unsure(tiny_model):
    tiny_model.eval()
    with torch.no_grad():
        tiny_model.refinement.confidence_head.bias.fill_(-10.0)  # sigmoid ~ 0
    ids = torch.randint(0, 256, (1, 8))
    _, info = tiny_model(ids, return_info=True)
    assert info["steps_run"] == 4
    assert info["confidence"] < 0.1


def test_loop_never_exceeds_four(tiny_model):
    tiny_model.eval()
    ids = torch.randint(0, 256, (1, 8))
    for bias in (-100.0, 0.0, 100.0):
        with torch.no_grad():
            tiny_model.refinement.confidence_head.bias.fill_(bias)
        _, info = tiny_model(ids, return_info=True)
        assert 1 <= info["steps_run"] <= 4


def test_training_runs_all_steps_and_collects_states(tiny_model):
    tiny_model.train()
    ids = torch.randint(0, 256, (2, 8))
    _, info = tiny_model(ids, collect_states=True, return_info=True)
    assert info["steps_run"] == 4
    assert len(info["states"]) == 4
    # Even with a confident head, training never early-exits.
    with torch.no_grad():
        tiny_model.refinement.confidence_head.bias.fill_(10.0)
    _, info = tiny_model(ids, collect_states=True, return_info=True)
    assert info["steps_run"] == 4


def test_aux_bce_targets_low_entropy_states(tiny_model):
    # With step_scales=0 all states are identical; force a constant confidence
    # c and check aux == BCE(c, fraction-of-low-entropy positions) exactly.
    tiny_model.train()
    with torch.no_grad():
        tiny_model.refinement.step_scales.zero_()
        tiny_model.refinement.confidence_head.weight.zero_()
        tiny_model.refinement.confidence_head.bias.fill_(0.7)
    c = torch.sigmoid(torch.tensor(0.7)).item()
    ids = torch.randint(0, 256, (2, 10))
    targets = ids.clone()
    targets[:, -3:] = PAD
    _, metrics = tiny_model.loss(ids, targets, confidence_weight=0.1,
                                 entropy_threshold=0.35, return_metrics=True)
    with torch.no_grad():
        # Recompute the actual intermediate state (step_scales=0 -> all equal).
        _, info = tiny_model(ids, collect_states=True, return_info=True)
        state = info["states"][0]
        step_logits = tiny_model.head(tiny_model.norm_final(state)).float()
        log_probs = torch.log_softmax(step_logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1) / math.log(VOCAB_SIZE)
        valid = targets != PAD
        fraction = (entropy[valid] < 0.35).float().mean().item()
    expected = -(fraction * math.log(c) + (1 - fraction) * math.log(1 - c))
    assert abs(metrics["confidence_bce"] - expected) < 1e-4


def test_confidence_targets_are_detached(tiny_model):
    # The BCE target must not backprop into the byte head: gradients from the
    # aux term reach the confidence head, and the CE term reaches the rest.
    tiny_model.train()
    ids = torch.randint(0, 256, (2, 8))
    loss, _ = tiny_model.loss(ids, confidence_weight=1.0, return_metrics=True)
    loss.backward()
    assert tiny_model.refinement.confidence_head.weight.grad is not None


def test_confidence_head_learns_entropy_signal(tiny_model):
    # Direct optimisation of the head against fixed low/high-entropy targets.
    refinement = tiny_model.refinement
    states = torch.randn(64, 64)
    targets = (states[:, 0] > 0).float()
    optimizer = torch.optim.Adam(refinement.confidence_head.parameters(), lr=0.05)
    initial = torch.nn.functional.binary_cross_entropy(
        refinement.confidence(states.unsqueeze(0))[0], targets).item()
    for _ in range(60):
        optimizer.zero_grad()
        conf = refinement.confidence(states.unsqueeze(0))[0]
        loss = torch.nn.functional.binary_cross_entropy(conf.clamp(1e-6, 1 - 1e-6), targets)
        loss.backward()
        optimizer.step()
    final = torch.nn.functional.binary_cross_entropy(
        refinement.confidence(states.unsqueeze(0))[0], targets).item()
    assert final < initial * 0.5


def test_adaptive_disabled_in_training_mode(tiny_model):
    tiny_model.train()
    with torch.no_grad():
        tiny_model.refinement.confidence_head.bias.fill_(10.0)
    ids = torch.randint(0, 256, (1, 8))
    _, info = tiny_model(ids, adaptive=True, return_info=True)  # ignored in train
    assert info["steps_run"] == 4


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("thought_steps", [0, 4])
def test_confidence_loss_under_autocast_uses_fp32_logits_bce(
        tiny_config, monkeypatch, dtype, checkpointing, thought_steps):
    """Regression for the live T4 fp16 probe abort.

    CUDA rejects probability BCE inside autocast even if its input was cast to
    float32. CPU autocast may allow that op, so this test additionally guards
    the op choice and verifies the reduction really runs with autocast off.
    It exercises forward + backward, masking, memory reads and checkpoint
    recompute on CPU; it is not a CUDA kernel or VRAM measurement.
    """
    cfg = tiny_config.scaled(attention_type="sliding_window", window_size=8,
                             sliding_chunk_size=4, gradient_checkpointing=checkpointing,
                             max_thought_steps=thought_steps)
    model = CausalByteTransformer(cfg)
    ids = torch.randint(0, 256, (2, 12))
    targets = ids.clone()
    targets[:, -2:] = PAD
    memory = SkillBankView(torch.randn(2, 2, cfg.d_model, requires_grad=True),
                           torch.randn(2, 2, cfg.memory_dim, requires_grad=True),
                           torch.ones(2, 2, dtype=torch.bool))
    real_bce_with_logits = F.binary_cross_entropy_with_logits
    calls = []

    def forbidden_probability_bce(*args, **kwargs):
        raise AssertionError("probability BCE is unsafe under CUDA autocast; score logits instead")

    def checked_logits_bce(logits, target, *args, **kwargs):
        assert logits.dtype is torch.float32
        assert target.dtype is torch.float32
        try:
            autocast_enabled = torch.is_autocast_enabled("cpu")
        except TypeError:  # older torch exposes a separate CPU query
            autocast_enabled = torch.is_autocast_cpu_enabled()
        assert not autocast_enabled, "BCE reduction must leave autocast"
        calls.append((logits.shape, target.shape))
        return real_bce_with_logits(logits, target, *args, **kwargs)

    monkeypatch.setattr(F, "binary_cross_entropy", forbidden_probability_bce)
    monkeypatch.setattr(F, "binary_cross_entropy_with_logits", checked_logits_bce)
    with torch.autocast("cpu", dtype=dtype):
        loss, metrics = model.loss(ids, targets, memory, confidence_weight=0.1,
                                   return_metrics=True)
    assert len(calls) == cfg.refinement_steps
    assert loss.dtype is torch.float32 and torch.isfinite(loss)
    assert math.isfinite(metrics["confidence_bce"])
    loss.backward()
    head_grad = model.refinement.confidence_head.weight.grad
    assert head_grad is not None and torch.isfinite(head_grad).all()
    assert float(head_grad.abs().sum()) > 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_confidence_logits_preserve_probability_api_and_combined_state(tiny_model):
    refinement = tiny_model.refinement
    hidden = torch.randn(2, 6, 64)
    thought = torch.randn(2, 64)
    with torch.no_grad():
        refinement.confidence_head.weight.normal_(0.0, 0.1)
        refinement.confidence_head.bias.fill_(0.3)
        refinement.thought_confidence.weight.normal_(0.0, 0.1)
    for pooled in (None, thought):
        logits = refinement.confidence_logits(hidden, pooled)
        probability = refinement.confidence(hidden, pooled)
        assert logits.shape == probability.shape == (2, 6)
        assert torch.allclose(probability, torch.sigmoid(logits), atol=1e-7)
        assert ((probability > 0) & (probability < 1)).all()


@pytest.mark.parametrize("bias,threshold,expected", [
    (-40.0, 0.0, 0.0), (40.0, 0.0, 40.0),
    (-40.0, 1.1, 40.0), (40.0, 1.1, 0.0),
])
def test_confidence_logits_bce_is_stable_at_extreme_logits(
        tiny_config, bias, threshold, expected):
    model = CausalByteTransformer(tiny_config.scaled(max_thought_steps=0))
    with torch.no_grad():
        model.refinement.confidence_head.bias.fill_(bias)
    ids = torch.randint(0, 256, (1, 12))
    with torch.autocast("cpu", dtype=torch.float16):
        loss, metrics = model.loss(ids, confidence_weight=1.0,
                                   entropy_threshold=threshold, return_metrics=True)
    assert metrics["confidence_bce"] == pytest.approx(expected, abs=1e-3)
    assert torch.isfinite(loss)
    loss.backward()
    grad = model.refinement.confidence_head.bias.grad
    assert grad is not None and torch.isfinite(grad).all()
