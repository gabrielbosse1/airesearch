"""Foundation: vocabulary purity, causality, wide stack, checkpoint I/O."""
import pytest
import torch

from models.foundation import (BOS, EOS, PAD, TOOL_SEARCH, TOOL_SEARCH_END,
                               OBS_START, OBS_END, EDIT, EDIT_END, VOCAB_SIZE,
                               CausalByteTransformer, FoundationConfig,
                               decode_ids, encode_bytes, load_foundation,
                               pad_batch, save_foundation)


def test_vocabulary_layout():
    assert VOCAB_SIZE == 265
    assert (BOS, EOS, PAD) == (256, 257, 258)
    assert (TOOL_SEARCH, TOOL_SEARCH_END) == (259, 260)
    assert (OBS_START, OBS_END) == (261, 262)
    assert (EDIT, EDIT_END) == (263, 264)


def test_encode_is_pure_byte_mapping():
    # The literal TEXT '[TOOL_SEARCH]' must stay ordinary bytes: only control
    # id 259 means a tool call. No scanning, no regex, no marker parsing.
    data = b"[TOOL_SEARCH]hello[/TOOL_SEARCH][EDIT]x[/EDIT]"
    ids = encode_bytes(data, bos=False)
    assert ids.tolist() == list(data)
    assert decode_ids(ids) == data


def test_encode_decode_roundtrip_utf8():
    data = "olá mundo,ação çãõ 🚀".encode("utf8")
    ids = encode_bytes(data, bos=True, eos=True)
    assert int(ids[0]) == BOS and int(ids[-1]) == EOS
    assert decode_ids(ids) == data


def test_decode_keep_controls_renders_names():
    ids = torch.tensor([BOS, TOOL_SEARCH, ord("q"), TOOL_SEARCH_END, EDIT, EOS])
    assert decode_ids(ids) == b"q"
    assert decode_ids(ids, keep_controls=True) == b"[TOOL_SEARCH]q[/TOOL_SEARCH][EDIT]"


def test_config_rejects_unknown_and_invalid():
    with pytest.raises(ValueError):
        FoundationConfig.from_mapping({"not_a_key": 1})
    with pytest.raises(ValueError):
        FoundationConfig(vocab_size=261)
    with pytest.raises(ValueError):
        FoundationConfig(d_model=64, n_heads=5)
    with pytest.raises(ValueError):
        FoundationConfig(memory_dim=-1)
    with pytest.raises(ValueError):
        FoundationConfig(confidence_threshold=0.0)
    with pytest.raises(ValueError):
        FoundationConfig(confidence_threshold=1.5)


def test_logits_shape_and_range(tiny_model):
    ids = torch.randint(0, VOCAB_SIZE, (2, 16))
    logits = tiny_model(ids)
    assert logits.shape == (2, 16, VOCAB_SIZE)
    assert torch.isfinite(logits).all()


def test_causality_future_tokens_do_not_leak(tiny_model):
    tiny_model.eval()
    ids = torch.randint(0, 256, (1, 12))
    changed = ids.clone()
    changed[0, 6:] = torch.randint(0, 256, (6,))
    with torch.no_grad():
        a = tiny_model(ids)
        b = tiny_model(changed)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-5)
    assert not torch.allclose(a[:, 6:], b[:, 6:], atol=1e-3)


def test_gqa_and_rope_paths_run():
    cfg = FoundationConfig(d_model=64, n_layers=1, n_heads=8, n_kv_heads=2,
                           d_ff=128, max_seq_len=64, position_type="rope",
                           ffn_type="swiglu", norm_type="rmsnorm", dropout=0.0)
    model = CausalByteTransformer(cfg)
    assert model(torch.randint(0, 256, (1, 8))).shape == (1, 8, VOCAB_SIZE)


def test_layernorm_gelu_learned_position_path():
    cfg = FoundationConfig(d_model=64, n_layers=1, n_heads=4, d_ff=128,
                           max_seq_len=64, norm_type="layernorm", ffn_type="gelu",
                           position_type="learned", dropout=0.0)
    model = CausalByteTransformer(cfg)
    assert model.position_embedding is not None
    assert model(torch.randint(0, 256, (1, 8))).shape == (1, 8, VOCAB_SIZE)


def test_pad_batch_targets_shifted_and_masked():
    seqs = [torch.tensor([BOS, 1, 2, EOS]), torch.tensor([BOS, 3])]
    ids, targets = pad_batch(seqs, 6)
    assert ids[0].tolist() == [BOS, 1, 2, EOS, PAD, PAD]
    assert targets[0].tolist() == [1, 2, EOS, PAD, PAD, PAD]
    assert targets[1].tolist() == [3, PAD, PAD, PAD, PAD, PAD]


def test_loss_ignores_padding(tiny_model):
    ids = torch.full((1, 8), PAD, dtype=torch.long)
    ids[0, :4] = torch.tensor([BOS, 10, 20, 30])
    targets = torch.full((1, 8), PAD, dtype=torch.long)
    targets[0, :3] = torch.tensor([10, 20, 30])
    loss = tiny_model.loss(ids, targets, confidence_weight=0.0)
    assert torch.isfinite(loss)


def test_gradient_checkpointing_matches_plain(tiny_config):
    torch.manual_seed(0)
    plain = CausalByteTransformer(tiny_config.scaled(gradient_checkpointing=False))
    torch.manual_seed(0)
    checked = CausalByteTransformer(tiny_config.scaled(gradient_checkpointing=True))
    ids = torch.randint(0, 256, (2, 12))
    l1 = plain.loss(ids, confidence_weight=0.0)
    l2 = checked.loss(ids, confidence_weight=0.0)
    assert torch.allclose(l1, l2, atol=1e-5)
    l1.backward()
    l2.backward()
    for p1, p2 in zip(plain.parameters(), checked.parameters()):
        if p1.grad is not None:
            assert torch.allclose(p1.grad, p2.grad, atol=1e-5)


def test_checkpoint_roundtrip(tmp_path, tiny_model):
    ids = torch.randint(0, 256, (1, 8))
    tiny_model.eval()
    with torch.no_grad():
        before = tiny_model(ids)
    info = save_foundation(tiny_model, tmp_path / "m.pt")
    assert info["parameters"] == tiny_model.parameters_count
    restored = load_foundation(tmp_path / "m.pt")
    restored.eval()
    with torch.no_grad():
        after = restored(ids)
    assert torch.allclose(before, after, atol=1e-6)


def test_generate_stops_at_eos(tiny_model):
    # Force EOS immediately by biasing the head.
    with torch.no_grad():
        tiny_model.head.bias.fill_(-10.0)
        tiny_model.head.bias[EOS] = 10.0
    out = tiny_model.generate(encode_bytes(b"hi"), max_new_bytes=8)
    assert out.numel() == 0


def test_refinement_identity_at_init(tiny_model):
    # step_scales are zero: refinement must not change the hidden stream.
    ids = torch.randint(0, 256, (1, 8))
    tiny_model.eval()
    with torch.no_grad():
        with_ref = tiny_model(ids)
        tiny_model.refinement = None
        without = tiny_model(ids)
    assert torch.allclose(with_ref, without, atol=1e-6)
