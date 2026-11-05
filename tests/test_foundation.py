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


# ---------------------------------------------------------------------------
# Phase 24.4: Latent Thought Matrix (isolated scratchpad inside the loop)
# ---------------------------------------------------------------------------
def _open_thought_gates(model):
    """Open the ReZero gates (same convention as the skill-adapter tests)."""
    thoughts = model.refinement.thoughts
    with torch.no_grad():
        thoughts.buffer_gate.fill_(1.0)
        thoughts.context_gate.fill_(1.0)
        thoughts.slot_embeddings.normal_(0.0, 0.02)
    return thoughts


def test_latent_thought_buffer_shape_and_update(tiny_model):
    thoughts = _open_thought_gates(tiny_model)
    assert thoughts.capacity == 4                       # matches the loop count
    assert thoughts.slot_embeddings.shape == (1, 4, 64)
    tiny_model.train()
    ids = torch.randint(0, 256, (2, 16))
    _, info = tiny_model(ids, collect_states=True, return_info=True)
    buffer = thoughts.last_buffer
    assert buffer.shape == (2, 4, 64)                   # (B, max_thought_steps, D)
    assert info["thought_steps"] == 4 and info["thought_norm"] > 0.0
    assert not buffer.requires_grad                     # telemetry is detached
    # Every slot is written and the recurrence ADVANCES: slot t != slot t-1.
    norms = [float(buffer[0, t].norm()) for t in range(4)]
    assert all(n > 0 for n in norms)
    for t in range(1, 4):
        assert not torch.allclose(buffer[:, t], buffer[:, t - 1], atol=1e-4)
    # The buffer is a pure function of the forward: a new forward re-thinks.
    _, info2 = tiny_model(ids, return_info=True)
    assert info2["thought_steps"] == 4


def test_thought_buffer_isolated_from_lm_head(tiny_model):
    """24.4.3/24.4.5: the buffer updates during the forward, but the byte
    head only ever sees the FINAL refined CONTEXT hidden states."""
    thoughts = _open_thought_gates(tiny_model)
    tiny_model.train()
    captured = []
    handle = tiny_model.head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0]))
    ids = torch.randint(0, 256, (2, 12))
    logits, info = tiny_model(ids, collect_states=True, return_info=True)
    handle.remove()
    # ONE head call in a training forward, on context-shaped input only.
    assert len(captured) == 1
    head_input = captured[0]
    assert head_input.shape == (2, 12, 64)              # (B, T, D): no thought rows
    # It is EXACTLY norm_final(final context state) -- nothing else mixed in.
    final_state = info["states"][-1]
    assert torch.equal(head_input, tiny_model.norm_final(final_state))
    assert logits.shape == (2, 12, 265)
    # The scratchpad really participated: gradients reach it THROUGH the
    # context cross-attention (never through a direct head path).
    assert float(thoughts.last_buffer.abs().sum()) > 0
    loss = tiny_model.loss(ids, confidence_weight=0.1)
    loss.backward()
    assert thoughts.buffer_gate.grad is not None and thoughts.buffer_gate.grad.abs() > 0
    assert thoughts.context_gate.grad is not None and thoughts.context_gate.grad.abs() > 0
    assert thoughts.slot_embeddings.grad is not None
    assert float(thoughts.slot_embeddings.grad.abs().sum()) > 0
    assert thoughts.read_context.in_proj_weight.grad is not None


def test_thought_writeback_is_causal(tiny_model):
    """The buffer summarises the WHOLE window, so only the frontier position
    may read it back: earlier logits must be byte-for-byte future-proof."""
    _open_thought_gates(tiny_model)
    tiny_model.eval()
    ids = torch.randint(0, 256, (2, 16))
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 1) % 256               # perturb the last byte
    with torch.no_grad():
        a = tiny_model(ids)
        b = tiny_model(ids2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6), "future leaked through the buffer"
    assert not torch.allclose(a[:, -1], b[:, -1], atol=1e-6), "frontier ignores the context"


def test_thought_zero_init_matches_legacy_architecture(tiny_config):
    """ReZero gates + zero slots + zero thought-confidence: a fresh 24.4 model
    is numerically identical to the legacy (max_thought_steps=0) model."""
    torch.manual_seed(24)
    legacy = CausalByteTransformer(tiny_config.scaled(max_thought_steps=0))
    torch.manual_seed(24)
    fresh = CausalByteTransformer(tiny_config.scaled(max_thought_steps=4))
    assert legacy.refinement.thoughts is None
    missing, unexpected = fresh.load_state_dict(legacy.state_dict(), strict=False)
    assert not unexpected and all("thought" in key for key in missing)
    assert fresh.parameters_count > legacy.parameters_count
    legacy.eval()
    fresh.eval()
    ids = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        assert torch.allclose(legacy(ids), fresh(ids), atol=1e-7)
        # Confidence is neutral w.r.t. the buffer at init (zero-init term).
        hidden = torch.randn(2, 12, 64)
        base = fresh.refinement.confidence(hidden)
        combined = fresh.refinement.confidence(hidden, torch.randn(2, 64))
        assert torch.allclose(base, combined, atol=1e-7)


def test_confidence_evaluates_combined_context_and_thought(tiny_model):
    """24.4.4: an uncertain buffer keeps the loop running; a confident one
    lets it break -- the head reads context AND scratchpad."""
    refinement = tiny_model.refinement
    hidden = torch.randn(1, 8, 64)
    thought = torch.randn(1, 64)
    base = refinement.confidence(hidden)
    assert torch.allclose(base, torch.full_like(base, 0.5), atol=1e-6)
    with torch.no_grad():
        refinement.thought_confidence.bias.fill_(10.0)   # buffer says "converged"
    confident = refinement.confidence(hidden, thought)
    assert bool((confident > 0.99).all())
    assert torch.allclose(refinement.confidence(hidden), base, atol=1e-6)  # context-only path
    tiny_model.eval()
    ids = torch.randint(0, 256, (1, 8))
    _, info = tiny_model(ids, return_info=True)
    assert info["steps_run"] == 1                        # early break on combined state
    assert info["thought_steps"] == 1                    # only slot 0 was written
    buffer = refinement.thoughts.last_buffer
    assert buffer.shape == (1, 4, 64)                    # unwritten slots keep their init
    with torch.no_grad():
        refinement.thought_confidence.bias.fill_(-10.0)  # buffer says "uncertain"
    _, info = tiny_model(ids, return_info=True)
    assert info["steps_run"] == 4                        # loop keeps thinking


def test_max_thought_steps_config_validation(tiny_config):
    import pytest
    with pytest.raises(ValueError):
        tiny_config.scaled(max_thought_steps=-1)
    cfg = FoundationConfig.from_mapping({"max_thought_steps": 2, "refinement_steps": 4})
    assert cfg.max_thought_steps == 2
    model = CausalByteTransformer(cfg)
    assert model.refinement.thoughts.capacity == 2       # capacity <= loop count
    with pytest.raises(ValueError):
        FoundationConfig.from_mapping({"max_thought_steps": "four"})


def test_thought_gates_bootstrap_no_deadlock(tiny_model):
    """Regression: double-zero ReZero deadlocked the scratchpad (zero slots
    -> zero inject -> zero gradient on both gates, observed as norm 0.0 in
    the smoke probe). Asymmetric init must bootstrap: the write gate and
    thought-confidence receive gradient at step 0, the rest from step 1."""
    tiny_model.train()
    thoughts = tiny_model.refinement.thoughts
    assert float(thoughts.context_gate.detach()) == 0.0   # write side closed at init
    assert float(thoughts.buffer_gate.detach()) == 1.0    # read side open at init
    params = list(thoughts.parameters()) + list(tiny_model.refinement.thought_confidence.parameters())
    optimizer = torch.optim.AdamW(params, lr=0.05)
    ids = torch.randint(0, 256, (2, 12))
    tiny_model.loss(ids, confidence_weight=0.1).backward()
    assert thoughts.context_gate.grad is not None and float(thoughts.context_gate.grad.abs()) > 0
    assert float(tiny_model.refinement.thought_confidence.weight.grad.abs().sum()) > 0
    optimizer.step()
    optimizer.zero_grad()
    assert float(thoughts.context_gate.detach()) != 0.0   # the write gate opened
    tiny_model.loss(ids, confidence_weight=0.1).backward()
    assert float(thoughts.slot_embeddings.grad.abs().sum()) > 0
    assert float(thoughts.read_context.in_proj_weight.grad.abs().sum()) > 0
    assert float(thoughts.write_back.in_proj_weight.grad.abs().sum()) > 0
