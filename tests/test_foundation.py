"""Bottom-up regression tests for the Phase 22 byte-level foundation.

Every test here pins one mathematical statement from ``ARCHITECTURE_AUDIT.md``
(Level 1 primitives and Level 2 blocks) so that the document cannot drift away
from the code:

* Level 1 -- RMSNorm, SwiGLU, RoPE, GQA, linear maps.
* Level 2 -- pre-norm residual block, iterative refinement, byte packing.
* Level 3 -- the assembled ``CausalByteTransformer`` and its parameter budget.

No network, no GPU, no claims about quality: shapes, formulas and invariants.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from models.foundation import (
    BOS, EOS, PAD, REF_CLOSE, REF_END, REF_OPEN, REF_START, VOCAB_SIZE,
    CausalByteTransformer, FoundationConfig, IterativeRefinement, RMSNorm,
    SwiGLUFFN, apply_rotary, decode_ids, encode_bytes, load_foundation,
    make_norm, pad_batch, repeat_kv, rotary_tables, rotate_half,
    save_foundation,
)


def tiny_config(**overrides) -> FoundationConfig:
    """The scaled stack (RMSNorm/SwiGLU/RoPE/GQA/refinement) at test width."""
    values = dict(d_model=16, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=32,
                  max_seq_len=32, dropout=0.0, norm_type="rmsnorm",
                  ffn_type="swiglu", position_type="rope", refinement_steps=2,
                  refinement_rank=4)
    values.update(overrides)
    return FoundationConfig(**values)


def legacy_config(**overrides) -> FoundationConfig:
    """The Phase 16 default: LayerNorm/GELU/learned positions/MHA/no loop."""
    values = dict(d_model=16, n_layers=1, n_heads=4, d_ff=32, max_seq_len=32,
                  dropout=0.0)
    values.update(overrides)
    return FoundationConfig(**values)


# ---------------------------------------------------------------------------
# Level 1 -- base primitives
# ---------------------------------------------------------------------------
def test_rmsnorm_matches_the_closed_form():
    torch.manual_seed(0)
    norm = RMSNorm(8)
    weight = torch.randn(8)
    with torch.no_grad():
        norm.weight.copy_(weight)
    x = torch.randn(3, 5, 8)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight
    assert torch.allclose(norm(x), expected, atol=1e-6)
    # No mean subtraction and no bias: the only parameter is the gain.
    assert norm.weight.shape == (8,)
    assert [name for name, _ in norm.named_parameters()] == ["weight"]


def test_rmsnorm_is_scale_invariant_and_preserves_dtype():
    norm = RMSNorm(8)
    x = torch.randn(2, 2, 8, dtype=torch.float64)
    out = norm(x)
    assert out.dtype == torch.float64
    # Scale invariance holds up to the eps in the denominator (1e-5 here).
    assert torch.allclose(norm(3.0 * x), out, rtol=2e-3, atol=2e-3)


def test_layernorm_default_is_kept_for_historical_checkpoints():
    config = legacy_config()
    assert isinstance(make_norm(config), torch.nn.LayerNorm)
    assert isinstance(make_norm(tiny_config()), RMSNorm)


def test_swiglu_matches_the_closed_form_and_has_no_biases():
    config = tiny_config()
    ffn = SwiGLUFFN(config)
    assert all(module.bias is None for module in (ffn.w_value, ffn.w_branch, ffn.w_out))
    x = torch.randn(2, 3, config.d_model)
    expected = ffn.w_out(F.silu(ffn.w_value(x)) * ffn.w_branch(x))
    assert torch.allclose(ffn(x), expected, atol=1e-6)
    # 3 * d * d_ff parameters per layer, no biases.
    assert sum(p.numel() for p in ffn.parameters()) == 3 * config.d_model * config.d_ff


def test_rotary_table_frequencies_follow_theta_to_the_minus_2i_over_d():
    head_dim, theta = 8, 10_000.0
    cos, sin = rotary_tables(5, head_dim, theta, torch.device("cpu"), torch.float32)
    assert cos.shape == sin.shape == (1, 1, 5, head_dim)
    half = head_dim // 2
    for position in (0, 3):
        angle_0 = float(position) * 1.0
        angle_1 = float(position) * theta ** (-1.0 / half)
        assert math.isclose(float(cos[0, 0, position, 0]), math.cos(angle_0), rel_tol=1e-6)
        assert math.isclose(float(sin[0, 0, position, 1]), math.sin(angle_1), rel_tol=1e-6)
        # The table is duplicated so ``rotate_half`` can pair (d, d + half).
        assert torch.equal(cos[..., :half], cos[..., half:])


def test_rotate_half_swaps_halves_with_sign_and_rope_preserves_norm():
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    expected = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])
    assert torch.equal(rotate_half(x), expected)
    random = torch.randn(2, 2, 6, 8)
    cos, sin = rotary_tables(6, 8, 10_000.0, torch.device("cpu"), torch.float32)
    rotated = apply_rotary(random, cos, sin)
    assert torch.allclose(rotated.norm(dim=-1), random.norm(dim=-1), atol=1e-5)


def test_rope_dot_product_depends_only_on_the_relative_offset():
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)
    cos, sin = rotary_tables(16, 8, 10_000.0, torch.device("cpu"), torch.float32)

    def dot(p: int, n: int) -> float:
        q_p = apply_rotary(q, cos[:, :, p:p + 1], sin[:, :, p:p + 1])
        k_n = apply_rotary(k, cos[:, :, n:n + 1], sin[:, :, n:n + 1])
        return float((q_p * k_n).sum())

    assert math.isclose(dot(2, 7), dot(9, 14), rel_tol=1e-5)
    assert not math.isclose(dot(2, 7), dot(2, 8), rel_tol=1e-6)


def test_repeat_kv_shape_and_values():
    hidden = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).view(2, 2, 3, 4)
    repeated = repeat_kv(hidden, 3)
    assert repeated.shape == (2, 6, 3, 4)
    for head in range(6):
        assert torch.equal(repeated[:, head], hidden[:, head // 3])
    assert repeat_kv(hidden, 1) is hidden


def test_gqa_matches_explicit_kv_repetition():
    torch.manual_seed(0)
    from models.foundation import CausalSelfAttention  # local import keeps the test explicit

    config = tiny_config(n_layers=1)
    attention = CausalSelfAttention(config)
    attention.eval()
    hidden = torch.randn(2, 5, config.d_model)
    with torch.no_grad():
        got = attention(hidden)

        batch, length = hidden.shape[0], hidden.shape[1]
        head_dim = config.d_model // config.n_heads
        q = attention.query(hidden).view(batch, length, config.n_heads, head_dim).transpose(1, 2)
        k = attention.key(hidden).view(batch, length, config.n_kv_heads, head_dim).transpose(1, 2)
        v = attention.value(hidden).view(batch, length, config.n_kv_heads, head_dim).transpose(1, 2)
        cos, sin = rotary_tables(length, head_dim, config.rope_theta, hidden.device, q.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        repeats = config.n_heads // config.n_kv_heads
        assert repeats == 2
        k_rep, v_rep = repeat_kv(k, repeats), repeat_kv(v, repeats)
        scores = (q @ k_rep.transpose(-2, -1)) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        expected = attention.output(
            (weights @ v_rep).transpose(1, 2).contiguous().view(batch, length, config.d_model))
    assert got.shape == (2, 5, config.d_model)
    assert torch.allclose(got, expected, atol=1e-5)


def test_gqa_projections_are_a_fraction_of_mha():
    config = tiny_config(d_model=32, n_heads=8, n_kv_heads=2, n_layers=1)
    from models.foundation import CausalSelfAttention

    gqa = CausalSelfAttention(config)
    mha = CausalSelfAttention(tiny_config(d_model=32, n_heads=8, n_kv_heads=0, n_layers=1))
    gqa_kv = gqa.key.weight.numel() + gqa.value.weight.numel()
    mha_kv = mha.key.weight.numel() + mha.value.weight.numel()
    assert gqa_kv * 4 == mha_kv  # n_kv_heads=2 of n_heads=8
    assert gqa.query.weight.shape == mha.query.weight.shape


def test_attention_is_causal():
    torch.manual_seed(0)
    model = CausalByteTransformer(tiny_config())
    model.eval()
    prefix = torch.randint(0, 256, (1, 6))
    other = torch.randint(0, 256, (1, 6))
    with torch.no_grad():
        left = model(prefix)
        right = model(torch.cat((prefix[:, :4], other[:, 4:]), dim=1))
    assert torch.allclose(left[:, :4], right[:, :4], atol=1e-6)
    assert not torch.allclose(left[:, 5], right[:, 5])


# ---------------------------------------------------------------------------
# Level 2 -- blocks
# ---------------------------------------------------------------------------
def test_block_is_pre_norm_with_two_residuals():
    torch.manual_seed(0)
    config = tiny_config(n_layers=1)
    model = CausalByteTransformer(config)
    block = model.blocks[0]
    hidden = torch.randn(1, 4, config.d_model)
    mask = model.causal_mask
    with torch.no_grad():
        got = block(hidden, mask)
        first = hidden + block.attention(block.norm_1(hidden), mask)
        expected = first + block.feed_forward(block.norm_2(first))
    assert got.shape == hidden.shape
    assert torch.allclose(got, expected, atol=1e-6)


def test_iterative_refinement_is_weight_tied_and_identity_at_initialisation():
    config = tiny_config(refinement_steps=2, refinement_rank=4)
    loop = IterativeRefinement(config)
    assert loop.step_scales.shape == (2,)
    assert torch.count_nonzero(loop.step_scales) == 0
    hidden = torch.randn(2, 3, config.d_model)
    assert torch.equal(loop(hidden), hidden)
    # The cell is shared across passes: parameter count is independent of steps.
    params_two = sum(p.numel() for p in IterativeRefinement(config).parameters())
    params_four = sum(p.numel() for p in
                      IterativeRefinement(tiny_config(refinement_steps=4)).parameters())
    assert params_four - params_two == 2  # only step_scales grows
    with torch.no_grad():
        loop.step_scales.fill_(1.0)
    assert not torch.allclose(loop(hidden), hidden)


def test_iterative_refinement_applies_a_low_rank_bottleneck():
    config = tiny_config(refinement_steps=3, refinement_rank=4)
    loop = IterativeRefinement(config)
    assert loop.down.weight.shape == (4, config.d_model)
    assert loop.branch.weight.shape == (4, config.d_model)
    assert loop.up.weight.shape == (config.d_model, 4)
    # h_(t+1) = h_(t) + alpha_t * Cell(RMSNorm(h_(t))) applied ``steps`` times.
    hidden = torch.randn(1, 2, config.d_model)
    with torch.no_grad():
        loop.step_scales.copy_(torch.tensor([0.5, 0.25, 0.125]))
        expected = hidden
        for alpha in (0.5, 0.25, 0.125):
            expected = expected + alpha * loop.cell(expected)
        got = loop(hidden)
    assert torch.allclose(got, expected, atol=1e-6)


def test_refinement_runs_inside_the_forward_pass_before_the_byte_head():
    config = tiny_config(refinement_steps=2)
    model = CausalByteTransformer(config)
    with torch.no_grad():
        model.refinement.step_scales.fill_(0.5)
        ids = torch.tensor([[BOS, 104, 105]])
        logits = model(ids)
        hidden = model.token_embedding(ids)
        for block in model.blocks:
            hidden = block(hidden, model.causal_mask)
        hidden = model.refinement(hidden)
        expected = model.head(model.norm_final(hidden))
    assert torch.allclose(logits, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Level 3 -- the assembled model
# ---------------------------------------------------------------------------
def test_forward_returns_vocab_logits_for_every_position():
    config = tiny_config(n_layers=3)
    model = CausalByteTransformer(config)
    ids = torch.randint(0, 256, (2, 7))
    logits = model(ids)
    assert logits.shape == (2, 7, VOCAB_SIZE) == (2, 7, 261)
    assert logits.dtype == torch.float32


def test_forward_rejects_bad_shapes_ids_and_lengths():
    model = CausalByteTransformer(tiny_config(max_seq_len=8))
    with pytest.raises(ValueError):
        model(torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.full((1, 4), VOCAB_SIZE, dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 9, dtype=torch.long))


def test_loss_matches_manual_cross_entropy_and_ignores_pad():
    torch.manual_seed(0)
    model = CausalByteTransformer(tiny_config(max_seq_len=8))
    ids = torch.tensor([[BOS, 100, 101, PAD, PAD]])
    logits = model(ids)
    expected = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB_SIZE),
                               ids[:, 1:].reshape(-1), ignore_index=PAD)
    assert torch.allclose(model.loss(ids), expected, atol=1e-6)
    # The two PAD targets are ignored: the loss equals the 3 supervised bytes.
    supervised = F.cross_entropy(model(ids[:, :4])[:, :3].reshape(-1, VOCAB_SIZE),
                                 ids[:, 1:4].reshape(-1), ignore_index=PAD)
    assert torch.allclose(model.loss(ids), supervised, atol=1e-6)


def test_pad_batch_shifts_targets_by_one_and_masks_padding():
    sequences = [torch.tensor([BOS, 10, 11]), torch.tensor([BOS, 20])]
    ids, targets = pad_batch(sequences, 4)
    assert ids.tolist() == [[BOS, 10, 11, PAD], [BOS, 20, PAD, PAD]]
    assert targets.tolist() == [[10, 11, PAD, PAD], [20, PAD, PAD, PAD]]
    assert int((ids == PAD).sum()) == 3 and int((targets == PAD).sum()) == 5


def test_encode_decode_roundtrip_is_byte_exact_and_drops_controls():
    payload = b"print('ola')\n" + bytes(range(256))
    ids = encode_bytes(payload, bos=True, eos=True)
    assert ids[0].item() == BOS and ids[-1].item() == EOS
    assert decode_ids(ids) == payload
    assert REF_START not in ids.tolist()
    marked = encode_bytes(b"see " + REF_OPEN + b"abc" + REF_CLOSE)
    expected = list(b"see ") + [REF_START] + list(b"abc") + [REF_END]
    assert marked.tolist()[1:] == expected
    assert decode_ids(marked) == b"see " + REF_OPEN + b"abc" + REF_CLOSE


def test_reference_markers_are_the_same_bytes_everywhere():
    from models import react

    assert react.REF_OPEN == REF_OPEN
    assert react.REF_CLOSE == REF_CLOSE


def test_generate_stops_at_eos_and_respects_the_byte_budget():
    model = CausalByteTransformer(tiny_config(max_seq_len=16))
    prefix = torch.tensor([BOS, 65])
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(-1e4)
        model.head.bias[ord("x")] = 1e4
    assert decode_ids(model.generate(prefix, max_new_bytes=5)) == b"xxxxx"
    with torch.no_grad():
        model.head.bias[ord("x")] = -1e4
        model.head.bias[EOS] = 1e4
    assert model.generate(prefix, max_new_bytes=5).numel() == 0


def test_checkpoint_roundtrip_restores_config_and_weights(tmp_path):
    config = tiny_config(n_layers=2)
    model = CausalByteTransformer(config)
    path = tmp_path / "foundation.pt"
    meta = save_foundation(model, path, extra={"note": "unit-test"})
    assert meta["parameters"] == model.parameters_count
    assert len(meta["sha256"]) == 64
    restored = load_foundation(path)
    assert restored.config == config
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)
    ids = torch.randint(0, 256, (1, 5))
    with torch.no_grad():
        assert torch.equal(model(ids), restored(ids))


def test_load_rejects_foreign_checkpoints(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"schema_version": 1, "config": {}, "state_dict": {}}, path)
    with pytest.raises(ValueError):
        load_foundation(path)


def test_gradient_checkpointing_matches_plain_backward():
    torch.manual_seed(0)
    plain = CausalByteTransformer(tiny_config(n_layers=2, gradient_checkpointing=False))
    checked = CausalByteTransformer(tiny_config(n_layers=2, gradient_checkpointing=True))
    checked.load_state_dict(plain.state_dict())
    ids = torch.randint(0, 256, (2, 8))
    plain.train(), checked.train()
    plain.loss(ids).backward()
    checked.loss(ids).backward()
    for left, right in zip(plain.parameters(), checked.parameters()):
        assert torch.allclose(left.grad, right.grad, atol=1e-5)


def test_parameter_budget_of_the_phase22_config_is_exact():
    production = FoundationConfig(d_model=1024, n_layers=12, n_heads=8, n_kv_heads=2,
                                  d_ff=4096, max_seq_len=2048, dropout=0.0,
                                  norm_type="rmsnorm", ffn_type="swiglu",
                                  position_type="rope", refinement_steps=2,
                                  refinement_rank=256, norm_eps=1e-6,
                                  gradient_checkpointing=True)
    model = CausalByteTransformer(production)
    total = sum(p.numel() for p in model.parameters())
    assert total == 183_830_791
    per_block = 15_208_960
    assert sum(p.numel() for b in model.blocks for p in b.parameters()) == 12 * per_block
    assert sum(p.numel() for p in model.refinement.parameters()) == 3 * 1024 * 256 + 1024 + 2
    assert model.token_embedding.weight.numel() == 261 * 1024
    assert model.head.weight.numel() + model.head.bias.numel() == 261 * 1024 + 261


def test_rope_configuration_has_no_learned_position_table():
    rope_model = CausalByteTransformer(tiny_config(position_type="rope"))
    learned = CausalByteTransformer(tiny_config(position_type="learned"))
    assert rope_model.position_embedding is None
    assert learned.position_embedding is not None
    rope_params = sum(p.numel() for p in rope_model.parameters())
    learned_params = sum(p.numel() for p in learned.parameters())
    assert learned_params - rope_params == 32 * 16


def test_no_legacy_mechanism_survives_in_the_state_dict():
    model = CausalByteTransformer(tiny_config())
    forbidden = ("gate", "pointer", "copy", "diffusion", "concept", "memory_bank")
    names = " ".join(model.state_dict()).lower()
    for word in forbidden:
        assert word not in names


def test_config_validation_rejects_inconsistent_stacks():
    with pytest.raises(ValueError):
        FoundationConfig(d_model=10, n_heads=4)
    with pytest.raises(ValueError):
        FoundationConfig(d_model=32, n_heads=4, n_kv_heads=3)
    with pytest.raises(ValueError):
        FoundationConfig(vocab_size=256)
    with pytest.raises(ValueError):
        FoundationConfig(position_type="rope", d_model=36, n_heads=4)  # odd head_dim
    with pytest.raises(ValueError):
        FoundationConfig.from_mapping({"d_model": 32, "unknown": 1})
