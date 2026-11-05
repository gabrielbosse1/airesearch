"""The Gated Skill Adapter: exact GLU math, shapes, masking, gradients."""
import pytest
import torch

from models.foundation import (CausalByteTransformer, FoundationConfig,
                               GatedSkillAdapter)
from models.memory_model import SkillBankView, stack_views


def test_glu_exact_math_production_widths():
    torch.manual_seed(0)
    adapter = GatedSkillAdapter(d_model=1024, memory_dim=8192)
    retrieved = torch.randn(2, 3, 8192)
    out = adapter.glu_gate(retrieved)
    assert out.shape == (2, 3, 1024)
    projected = adapter.glu(retrieved)
    assert projected.shape == (2, 3, 2048)
    information, filter_half = projected.chunk(2, dim=-1)
    expected = information * torch.sigmoid(filter_half)
    assert torch.allclose(out, expected, atol=1e-6)


def test_retrieve_shapes_global_and_batched():
    adapter = GatedSkillAdapter(64, 128)
    hidden = torch.randn(2, 5, 64)
    keys = torch.randn(4, 64)
    values = torch.randn(4, 128)
    mask = torch.ones(4, dtype=torch.bool)
    assert adapter.retrieve(hidden, keys, values, mask).shape == (2, 5, 128)
    bk = torch.randn(2, 4, 64)
    bv = torch.randn(2, 4, 128)
    bm = torch.ones(2, 4, dtype=torch.bool)
    assert adapter.retrieve(hidden, bk, bv, bm).shape == (2, 5, 128)


def test_empty_memory_is_exact_zero():
    adapter = GatedSkillAdapter(64, 128)
    hidden = torch.randn(2, 5, 64)
    empty = SkillBankView(torch.zeros(0, 64), torch.zeros(0, 128),
                          torch.zeros(0, dtype=torch.bool))
    assert adapter(hidden, empty).abs().max().item() == 0.0
    assert adapter(hidden, None).abs().max().item() == 0.0


def test_fully_masked_rows_retrieve_zero_not_nan():
    adapter = GatedSkillAdapter(64, 128)
    adapter.scale.data.fill_(1.0)
    hidden = torch.randn(2, 5, 64)
    keys = torch.randn(2, 3, 64)
    values = torch.randn(2, 3, 128)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    view = SkillBankView(keys, values, mask)
    out = adapter(hidden, view)
    assert torch.isfinite(out).all()
    assert out[1].abs().max().item() == 0.0
    assert out[0].abs().max().item() > 0.0


def test_scale_zero_init_means_identity_injection():
    adapter = GatedSkillAdapter(64, 128)
    assert adapter.scale.item() == 0.0
    view = SkillBankView(torch.randn(3, 64), torch.randn(3, 128),
                         torch.ones(3, dtype=torch.bool))
    assert adapter(torch.randn(1, 4, 64), view).abs().max().item() == 0.0


def test_high_fidelity_key_matching_retrieves_correct_value():
    # Production widths, one position: a query aligned with key 0 must pull
    # value 0 (softmax mass ~1), and the GLU output must match glu(value 0).
    torch.manual_seed(0)
    adapter = GatedSkillAdapter(1024, 8192)
    adapter.scale.data.fill_(1.0)
    with torch.no_grad():
        adapter.query.weight.copy_(torch.eye(1024))
    keys = torch.zeros(2, 1024)
    keys[0, 0] = 100.0
    keys[1, 1] = 100.0
    values = torch.randn(2, 8192)
    hidden = torch.zeros(1, 1, 1024)
    hidden[0, 0, 0] = 100.0
    mask = torch.ones(2, dtype=torch.bool)
    retrieved = adapter.retrieve(hidden, keys, values, mask)
    assert torch.allclose(retrieved[0, 0], values[0], atol=1e-3)
    out = adapter(hidden, SkillBankView(keys, values, mask))
    assert torch.allclose(out[0, 0], adapter.glu_gate(values[0:1])[0], atol=1e-3)


def test_gradients_flow_to_projectors_through_view():
    from models.encoders import SkillKeyProjector, SkillProjector
    torch.manual_seed(0)
    adapter = GatedSkillAdapter(64, 128)
    adapter.scale.data.fill_(1.0)
    vp = SkillProjector(32, 128)
    kp = SkillKeyProjector(32, 64)
    features = torch.randn(2, 32)
    keys, values = kp(features), vp(features)
    # TWO slots per row: with a single slot softmax == 1 is constant and the
    # key projector mathematically gets zero gradient. Distractors fix that.
    distractor_k = torch.randn(2, 1, 64)
    distractor_v = torch.randn(2, 1, 128)
    view = SkillBankView(torch.cat([keys.unsqueeze(1), distractor_k], dim=1),
                         torch.cat([values.unsqueeze(1), distractor_v], dim=1),
                         torch.ones(2, 2, dtype=torch.bool))
    hidden = torch.randn(2, 4, 64)
    adapter(hidden, view).sum().backward()
    assert vp.projection.weight.grad.abs().sum().item() > 0
    assert kp.projection.weight.grad.abs().sum().item() > 0
    assert adapter.glu.weight.grad.abs().sum().item() > 0


def test_wrong_value_width_rejected():
    adapter = GatedSkillAdapter(64, 128)
    with pytest.raises(ValueError):
        adapter.glu_gate(torch.randn(1, 1, 256))


def test_injection_in_all_eight_production_layers():
    import yaml
    with open("configs/t4_scaled.yaml", "r", encoding="utf8") as handle:
        cfg = FoundationConfig.from_mapping(yaml.safe_load(handle)["model"])
    assert cfg.n_layers == 8 and cfg.max_seq_len == 4096 and cfg.memory_dim == 8192
    # Meta device: full production geometry, zero memory cost.
    with torch.device("meta"):
        model = CausalByteTransformer(cfg)
    adapters = [b.skill_adapter for b in model.blocks if b.has_skills]
    assert len(adapters) == 8
    for adapter in adapters:
        assert adapter.d_model == 1024 and adapter.memory_dim == 8192
        assert adapter.glu.out_features == 2048
    # Exact analytic parameter count (audited in ARCHITECTURE_AUDIT.md).
    per_layer = (1024                      # norm_1
                 + 2624000                 # GQA attention (q,k,v,o + biases)
                 + 1024                    # norm_2
                 + 3 * 1024 * 4096         # SwiGLU
                 + 1024                    # skill_norm
                 + 1024 * 1024 + 8192 * 2048 + 1)  # adapter q, glu, scale
    expected = (8 * per_layer
                + 265 * 1024               # token embedding
                + 1024 + 3 * 1024 * 256 + 4 + 1025  # refinement + conf head
                # Phase 24.4 latent thought buffer (S=4 slots, D=1024):
                + 4 * 1024                 # learned slot embeddings
                + 2 * (4 * 1024 * 1024 + 4 * 1024)  # cross-MHAs: in_proj 3D^2+3D, out D^2+D
                + 4 * 1024                 # cross-attention pre-norms
                + 2                        # ReZero gates (buffer, context)
                + 1024 + 1025              # thought_norm + thought_confidence head
                + 1024                     # norm_final
                + 1024 * 265 + 265)        # byte head
    assert model.parameters_count == expected == 274_025_753


def test_stack_views_broadcasts_global_to_batched():
    global_view = SkillBankView(torch.randn(2, 64), torch.randn(2, 128),
                                torch.ones(2, dtype=torch.bool))
    local = SkillBankView(torch.randn(3, 1, 64), torch.randn(3, 1, 128),
                          torch.ones(3, 1, dtype=torch.bool))
    merged = stack_views(global_view, local)
    assert merged.batched and merged.keys.shape == (3, 3, 64)
    assert merged.values.shape == (3, 3, 128)
    assert stack_views(None, None) is None
