"""Multimodal ingestion: frozen encoders, 8192 projection, codec, agnosticism."""
import pytest
import torch
from torch import nn

from models.encoders import (FrozenEncoder, MultimodalIngestor, SkillKeyProjector,
                             SkillProjector, TextSkillProjector, bytes_to_matrix,
                             matrix_to_bytes, pooled_text_embedding,
                             sample_video_frames)
from models.foundation import PAD
from models.memory_model import ProceduralSkillBank


def test_skill_projector_maps_to_8192():
    for encoder_dim in (768, 2048, 512):
        projector = SkillProjector(encoder_dim, 8192)
        out = projector(torch.randn(2, encoder_dim))
        assert out.shape == (2, 8192)
    key_proj = SkillKeyProjector(768, 1024)
    assert key_proj(torch.randn(2, 768)).shape == (2, 1024)


def test_projector_rejects_wrong_width():
    with pytest.raises(ValueError):
        SkillProjector(768, 8192)(torch.randn(2, 512))


def test_frozen_encoder_never_trains():
    backbone = nn.Sequential(nn.Linear(16, 32), nn.BatchNorm1d(32), nn.Dropout(0.5))
    encoder = FrozenEncoder(backbone, 32)
    assert all(not p.requires_grad for p in encoder.parameters())
    encoder.train()  # parent flips to train...
    assert not encoder.backbone.training  # ...backbone stays eval
    out = encoder(torch.randn(4, 16))
    assert out.shape == (4, 32) and not out.requires_grad


def test_frozen_encoder_pools_sequence_features():
    class SeqBackbone(nn.Module):
        def forward(self, x):
            return x.unsqueeze(1).expand(-1, 5, -1)  # (B, T, D)
    encoder = FrozenEncoder(SeqBackbone(), 16)
    assert encoder(torch.randn(2, 16)).shape == (2, 16)


def test_matrix_agnosticism_swap_encoder_keep_core():
    # Two encoders of DIFFERENT widths feed the SAME bank + adapter geometry.
    from models.foundation import GatedSkillAdapter
    bank = ProceduralSkillBank(capacity=4, key_dim=64, value_dim=128)
    adapter = GatedSkillAdapter(64, 128)
    adapter.scale.data.fill_(1.0)
    for width in (32, 96):
        ingestor = MultimodalIngestor(None, memory_dim=128, key_dim=64, encoder_dim=width)
        key, value = ingestor.project_features(torch.randn(1, width))
        assert key.shape == (1, 64) and value.shape == (1, 128)
        bank.write(key[0], value[0], modality="image")
    out = adapter(torch.randn(1, 3, 64), bank.view())
    assert out.shape == (1, 3, 64) and torch.isfinite(out).all()


def test_ingestor_writes_bank_without_gradients():
    ingestor = MultimodalIngestor(None, memory_dim=128, key_dim=64, encoder_dim=32)
    bank = ProceduralSkillBank(capacity=2, key_dim=64, value_dim=128)
    # projector-only ingestor: use project_features + bank.write directly
    key, value = ingestor.project_features(torch.randn(1, 32))
    bank.write(key[0], value[0], modality="audio")
    assert len(bank) == 1 and not bank.values.requires_grad


def test_batch_view_shape_and_mask():
    ingestor = MultimodalIngestor(None, memory_dim=128, key_dim=64, encoder_dim=32)
    keys, values = ingestor.project_features(torch.randn(3, 32))
    view = ingestor.batch_view(keys, values)
    assert view.batched and view.keys.shape == (3, 1, 64)
    assert view.mask.all()


def test_codec_roundtrip_bounded_error():
    # For |x| <= 1.5: tanh(x) <= 0.905, d(atanh)/du <= 5.53, uint8 step
    # 1/127.5 -> worst-case roundtrip error <= 0.5/127.5 * 5.53 ~ 0.022.
    vector = torch.linspace(-1.5, 1.5, 768)
    data = matrix_to_bytes(vector)
    assert isinstance(data, bytes) and len(data) == 768
    restored = bytes_to_matrix(data)
    assert (restored - vector).abs().max().item() < 0.03
    assert bytes_to_matrix(matrix_to_bytes(torch.zeros(16))).shape == (16,)
    # Extreme tails stay bounded by the codec's max_abs clamp.
    extreme = torch.tensor([-50.0, 50.0])
    assert bytes_to_matrix(matrix_to_bytes(extreme)).abs().max().item() <= 3.0


def test_codec_is_deterministic():
    vector = torch.linspace(-2, 2, 64)
    assert matrix_to_bytes(vector) == matrix_to_bytes(vector.clone())


def test_text_skill_projector_and_pooling(tiny_model):
    projector = TextSkillProjector(64, 128, 64)
    ids = torch.tensor([[5, 6, 7, PAD, PAD]])
    pooled = pooled_text_embedding(tiny_model, ids, PAD)
    assert pooled.shape == (1, 64) and pooled.requires_grad
    key, value = projector(pooled)
    assert key.shape == (1, 64) and value.shape == (1, 128)
    value.sum().backward()
    assert projector.value.projection.weight.grad.abs().sum().item() > 0


def test_video_frame_sampling():
    frames = torch.randn(20, 3, 8, 8)
    subset = sample_video_frames(frames, max_frames=8)
    assert subset.shape == (8, 3, 8, 8)
    ingestor = MultimodalIngestor(None, memory_dim=128, key_dim=64, encoder_dim=32)
    keys, values = ingestor.project_features(torch.randn(8, 32))
    assert keys.mean(dim=0, keepdim=True).shape == (1, 64)


import importlib.util

_HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None


@pytest.mark.skipif(not _HAS_TORCHVISION, reason="torchvision not installed")
def test_real_resnet_frozen_random_weights():
    # pretrained=False: random weights, NO download. Shape/freeze contract only.
    from models.encoders import build_encoder
    encoder = build_encoder("resnet50", pretrained=False)
    assert encoder.output_dim == 2048
    assert all(not p.requires_grad for p in encoder.parameters())
    out = encoder(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 2048) and not out.requires_grad
