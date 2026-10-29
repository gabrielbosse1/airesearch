"""Raw-byte embedding + positional + sinusoidal timestep helpers."""

import math
import torch
import torch.nn as nn


class ByteEmbedding(nn.Module):
    """Embed raw bytes (0-255) into d_model. No tokenizer involved."""

    def __init__(self, vocab_size: int = 256, d_model: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) long byte ids
        return self.emb(x)  # (B, L, d)


class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal timestep embedding -> d_model vector."""

    def __init__(self, d_model: int, max_period: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]
        device = t.device
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=device).float()
            / half
        )  # (half,)
        args = t[:, None].float() * freqs[None, :]  # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, d)
        if emb.shape[-1] < self.d_model:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class PositionalEncoding(nn.Module):
    """Learned positional embeddings."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l = x.shape[:2]
        pos = torch.arange(l, device=x.device).unsqueeze(0).expand(b, l)
        return self.emb(pos)
