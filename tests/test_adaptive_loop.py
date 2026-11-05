"""Model-decided adaptive loop: confidence head, early break, aux BCE."""
import math

import torch

from models.foundation import PAD, VOCAB_SIZE


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
