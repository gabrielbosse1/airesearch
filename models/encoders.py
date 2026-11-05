"""Multimodal skill ingestion: frozen encoders -> 8192-dim skill matrices.

The core model never sees pixels or waveforms. A FROZEN pre-trained encoder
(ResNet/ViT for images and video frames, Wav2Vec2 for audio) produces a
feature vector; a trainable ``SkillProjector`` (``Linear(encoder_dim, 8192)``)
maps it into the skill space and a ``SkillKeyProjector``
(``Linear(encoder_dim, 1024)``) produces the retrieval key. The pair is
written to the ``ProceduralSkillBank`` (gradient-free) or handed to the model
as a batch-local, differentiable ``SkillBankView`` during training.

Matrix agnosticism: swap ResNet (2048-dim) for ViT (768-dim) and only the
projector changes. The bank, the adapter and the core model are untouched --
they only ever read 1024-dim keys and 8192-dim values.

Bidirectional byte prediction uses a deterministic uint8 codec
(``matrix_to_bytes`` / ``bytes_to_matrix``): the *image/audio byte payload* is
the serialised skill matrix (tanh-companded uint8), NOT raw pixels. Text->media
training predicts those codec bytes conditioned on a TEXT skill matrix;
media->text training predicts caption bytes conditioned on the media matrix.
Both directions use the same byte vocabulary and the same cross-entropy.

Heavy backbones (torchvision / torchaudio) are imported lazily so the CPU
test-suite runs without them; ``pretrained=False`` builds random-weight
backbones for offline shape tests (never used for real ingestion).
"""
from __future__ import annotations

import contextlib
import io
import os
from typing import Any, Callable

import torch
from torch import nn

from .memory_model import KEY_DIM, VALUE_DIM, ProceduralSkillBank, SkillBankView

# ---------------------------------------------------------------------------
# Projectors (the ONLY trainable part of ingestion)
# ---------------------------------------------------------------------------
class SkillProjector(nn.Module):
    """``Linear(encoder_dim, 8192)`` -- encoder features -> skill value."""

    def __init__(self, encoder_dim: int, memory_dim: int = VALUE_DIM) -> None:
        super().__init__()
        if encoder_dim < 1 or memory_dim < 1:
            raise ValueError("encoder_dim and memory_dim must be positive")
        self.encoder_dim = int(encoder_dim)
        self.memory_dim = int(memory_dim)
        self.projection = nn.Linear(encoder_dim, memory_dim)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.encoder_dim:
            raise ValueError(f"expected features of width {self.encoder_dim}, "
                             f"got {features.shape[-1]}")
        return self.projection(features)


class SkillKeyProjector(nn.Module):
    """``Linear(encoder_dim, 1024)`` -- encoder features -> retrieval key."""

    def __init__(self, encoder_dim: int, key_dim: int = KEY_DIM) -> None:
        super().__init__()
        self.encoder_dim = int(encoder_dim)
        self.key_dim = int(key_dim)
        self.projection = nn.Linear(encoder_dim, key_dim)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class TextSkillProjector(nn.Module):
    """Pooled core-model embeddings (1024) -> (key 1024, value 8192).

    Used for the text->media direction: the caption's mean embedding becomes
    a TEXT skill matrix that conditions the prediction of media codec bytes.
    """

    def __init__(self, d_model: int = KEY_DIM, memory_dim: int = VALUE_DIM,
                 key_dim: int = KEY_DIM) -> None:
        super().__init__()
        self.value = SkillProjector(d_model, memory_dim)
        self.key = SkillKeyProjector(d_model, key_dim)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key(pooled), self.value(pooled)


# ---------------------------------------------------------------------------
# Deterministic uint8 matrix codec ("media bytes")
# ---------------------------------------------------------------------------
def matrix_to_bytes(vector: torch.Tensor) -> bytes:
    """Serialise a feature vector as uint8 bytes via tanh companding.

    ``byte = round((tanh(x) + 1) * 127.5)``. Deterministic, bounded, and
    invertible up to uint8 quantisation by :func:`bytes_to_matrix`.
    """
    if vector.ndim != 1:
        raise ValueError("matrix_to_bytes expects a 1-D feature vector")
    companded = torch.tanh(vector.detach().float())
    quantised = torch.round((companded + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return bytes(quantised.tolist())


def bytes_to_matrix(data: bytes, *, max_abs: float = 3.0) -> torch.Tensor:
    """Inverse codec: uint8 bytes -> approximate feature vector (float32)."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    values = torch.tensor(list(data), dtype=torch.float32) / 127.5 - 1.0
    values = values.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.atanh(values).clamp(-max_abs, max_abs)


# ---------------------------------------------------------------------------
# Frozen encoder wrapper
# ---------------------------------------------------------------------------
class FrozenEncoder(nn.Module):
    """Wraps any backbone as a frozen feature extractor.

    * ``requires_grad_(False)`` on every backbone parameter;
    * ``train()`` is overridden so the backbone stays in eval mode (no
      dropout / batchnorm drift) even when the parent module trains;
    * forward runs under ``torch.no_grad()`` -- features are constants.
    """

    def __init__(self, backbone: nn.Module, output_dim: int,
                 feature_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.backbone = backbone
        self.output_dim = int(output_dim)
        self.feature_fn = feature_fn
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenEncoder":
        super().train(mode)
        self.backbone.eval()  # frozen means frozen
        return self

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # T4 preset may store frozen inference-only weights in fp16 to save
        # VRAM. Autocast bridges fp32 raw inputs; returned features are fp32 so
        # trainable projectors never see a dtype mismatch outside autocast.
        parameter = next(self.backbone.parameters(), None)
        low_precision = (inputs.device.type == "cuda" and parameter is not None
                         and parameter.dtype in (torch.float16, torch.bfloat16))
        context = (torch.autocast(device_type=inputs.device.type, dtype=parameter.dtype)
                   if low_precision else contextlib.nullcontext())
        with context:
            if self.feature_fn is not None:
                features = self.feature_fn(self.backbone, inputs)
            else:
                features = self.backbone(inputs)
        if features.ndim > 2:  # e.g. (B, T, D) audio frames -> mean pool
            features = features.mean(dim=tuple(range(1, features.ndim - 1)))
        if features.shape[-1] != self.output_dim:
            raise ValueError(f"backbone produced width {features.shape[-1]}, "
                             f"expected {self.output_dim}")
        return features.float()


def _resnet_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
    x = backbone.conv1(images)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    x = backbone.avgpool(x)
    return torch.flatten(x, 1)


def build_encoder(name: str, *, pretrained: bool = True) -> FrozenEncoder:
    """Registry of frozen backbones. Lazy imports keep CPU tests light.

    image: ``resnet50`` (2048) | ``vit_b_16`` (768)
    audio: ``wav2vec2`` (768)
    video: ``video_resnet50`` -- frames sampled and mean-pooled by the caller
           through the same image encoder (matrix agnosticism in action).
    """
    name = name.lower()
    if name in ("resnet50", "video_resnet50"):
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        return FrozenEncoder(resnet50(weights=weights), 2048, _resnet_features)
    if name == "vit_b_16":
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        backbone = vit_b_16(weights=weights)
        backbone.heads = nn.Identity()
        return FrozenEncoder(backbone, 768)
    if name == "wav2vec2":
        import torchaudio
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        if pretrained:
            backbone = bundle.get_model()  # downloads weights (Colab only)
        else:
            # Same architecture, random weights: offline tests never download.
            from torchaudio.models import wav2vec2_model
            backbone = wav2vec2_model(**bundle._params)

        def _audio_features(model: nn.Module, waveform: torch.Tensor) -> torch.Tensor:
            features, _ = model(waveform)
            return features.mean(dim=1)

        return FrozenEncoder(backbone, 768, _audio_features)
    raise ValueError(f"unknown encoder {name!r}; choose resnet50, vit_b_16, wav2vec2, video_resnet50")


def preprocess_image(source: Any, *, size: int = 224) -> torch.Tensor:
    """PIL image / path / bytes -> normalised (1, 3, size, size) tensor."""
    from PIL import Image
    from torchvision import transforms
    if isinstance(source, (bytes, bytearray)):
        image = Image.open(io.BytesIO(bytes(source)))
    elif isinstance(source, (str, os.PathLike)):
        image = Image.open(source)
    else:
        image = source
    pipeline = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return pipeline(image.convert("RGB")).unsqueeze(0)


def sample_video_frames(frames: torch.Tensor, *, max_frames: int = 8) -> torch.Tensor:
    """``(T, 3, H, W)`` frame stack -> evenly spaced subset (video = images)."""
    if frames.ndim != 4:
        raise ValueError("frames must be (T, 3, H, W)")
    count = min(max_frames, frames.shape[0])
    idx = torch.linspace(0, frames.shape[0] - 1, count).round().long()
    return frames[idx]


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------
class MultimodalIngestor(nn.Module):
    """Frozen encoder + trainable projectors = one modality's ingestion path.

    ``forward(raw)`` returns ``(key (B, 1024), value (B, 8192))`` where ONLY
    the projectors carry gradients. ``ingest`` writes a single sample to a
    persistent bank (gradient-free). ``batch_view`` builds the differentiable
    batch-local view used during training.
    """

    def __init__(self, encoder: FrozenEncoder | None = None, memory_dim: int = VALUE_DIM,
                 key_dim: int = KEY_DIM, *, encoder_dim: int | None = None) -> None:
        super().__init__()
        if encoder is None and encoder_dim is None:
            raise ValueError("provide an encoder or encoder_dim")
        self.encoder = encoder
        width = encoder.output_dim if encoder is not None else int(encoder_dim)
        self.value_projector = SkillProjector(width, memory_dim)
        self.key_projector = SkillKeyProjector(width, key_dim)
        self.memory_dim = int(memory_dim)
        self.key_dim = int(key_dim)

    def forward(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.encoder is None:
            raise ValueError("projector-only ingestor: use project_features()")
        features = self.encoder(raw)
        return self.key_projector(features), self.value_projector(features)

    def project_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project pre-computed encoder features (datasets may cache them)."""
        return self.key_projector(features), self.value_projector(features)

    def encode_video(self, frames: torch.Tensor, *, max_frames: int = 8
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """Video = sampled frames through the image encoder, mean-pooled.

        Returns a single ``(1, key_dim)`` / ``(1, memory_dim)`` pair: the core
        model reads one 8192-dim matrix per video, agnostic to frame count.
        """
        subset = sample_video_frames(frames, max_frames=max_frames)
        keys, values = self.forward(subset)
        return keys.mean(dim=0, keepdim=True), values.mean(dim=0, keepdim=True)

    @torch.no_grad()
    def ingest(self, raw: torch.Tensor, bank: ProceduralSkillBank, *,
               modality: str = "image", metadata: dict[str, Any] | None = None) -> int:
        key, value = self.forward(raw)
        return bank.write(key[0], value[0], modality=modality, metadata=metadata)

    def batch_view(self, keys: torch.Tensor, values: torch.Tensor) -> SkillBankView:
        """``(B, D)`` keys / ``(B, M)`` values -> batch-local one-slot view."""
        if keys.ndim != 2 or values.ndim != 2:
            raise ValueError("batch_view expects (B, D) keys and (B, M) values")
        batch = keys.shape[0]
        return SkillBankView(keys.unsqueeze(1), values.unsqueeze(1),
                             torch.ones(batch, 1, dtype=torch.bool, device=keys.device))


def pooled_text_embedding(model: nn.Module, ids: torch.Tensor,
                          ignore_index: int) -> torch.Tensor:
    """Mean of core token embeddings over non-pad positions -> (B, d_model).

    Keeps the autograd graph: the TextSkillProjector is trained through it.
    """
    embeddings = model.embed(ids)
    mask = (ids != ignore_index).unsqueeze(-1).to(embeddings.dtype)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return (embeddings * mask).sum(dim=1) / counts
