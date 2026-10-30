"""Style-conditioned autoregressive raw-byte CHAT decoder.

Diversity is learned through differentiable predicted distributions, not
cosine similarity between discrete sampled byte IDs. Styles and sentence
patterns are synthetic; this is not an open-domain language model.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class DiverseByteDecoder(nn.Module):
    EOS = 256
    START = 257
    IGNORE = -100

    def __init__(self, context_dim: int = 64, *, num_styles: int = 32, hidden_dim: int = 96,
                 byte_dim: int = 48, style_dim: int = 32) -> None:
        super().__init__()
        if min(context_dim, num_styles, hidden_dim, byte_dim, style_dim) < 1:
            raise ValueError("decoder dimensions must be positive")
        self.context_dim, self.num_styles, self.hidden_dim = context_dim, num_styles, hidden_dim
        self.byte_dim, self.style_dim = byte_dim, style_dim
        self.byte_embedding = nn.Embedding(258, byte_dim)
        self.style_embedding = nn.Embedding(num_styles, style_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.initial = nn.Linear(hidden_dim + style_dim, hidden_dim)
        self.recurrent = nn.GRU(byte_dim + style_dim + hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 257)
        self.register_buffer("response_trained", torch.zeros((), dtype=torch.bool))

    def forward(self, context: torch.Tensor, style: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("context must have shape (batch, context_dim)")
        if (style.dtype != torch.long or style.shape != (context.shape[0],) or style.device != context.device
                or bool(((style < 0) | (style >= self.num_styles)).any())):
            raise ValueError("style IDs must match batch and decoder range")
        if (previous.dtype != torch.long or previous.ndim != 2 or previous.shape[0] != context.shape[0]
                or previous.device != context.device or bool(((previous < 0) | (previous > self.START)).any())):
            raise ValueError("previous must contain byte/START IDs on the context device")
        projected = torch.tanh(self.context_projection(context))
        style_vector = self.style_embedding(style)
        initial = torch.tanh(self.initial(torch.cat((projected, style_vector), dim=-1))).unsqueeze(0)
        steps = previous.shape[1]
        recurrent_input = torch.cat((self.byte_embedding(previous), style_vector.unsqueeze(1).expand(-1, steps, -1),
                                     projected.unsqueeze(1).expand(-1, steps, -1)), dim=-1)
        output, _ = self.recurrent(recurrent_input, initial)
        return self.output(output)

    @torch.no_grad()
    def generate(self, context: torch.Tensor, *, style: int | None = None, max_bytes: int = 80,
                 temperature: float = 0.15, sample: bool = True,
                 rng: torch.Generator | None = None) -> tuple[bytes, bool, int]:
        if context.shape != (1, self.context_dim) or not 1 <= max_bytes <= 512:
            raise ValueError("generation requires one context and a bounded byte budget")
        if not 0 < temperature <= 10:
            raise ValueError("temperature must be in (0, 10]")
        if style is None:
            style = int(torch.randint(self.num_styles, (1,), generator=rng, device=context.device).item())
        if not isinstance(style, int) or not 0 <= style < self.num_styles:
            raise ValueError("invalid style")
        style_tensor = torch.tensor([style], dtype=torch.long, device=context.device)
        projected = torch.tanh(self.context_projection(context))
        style_vector = self.style_embedding(style_tensor)
        state = torch.tanh(self.initial(torch.cat((projected, style_vector), dim=-1))).unsqueeze(0)
        previous = torch.tensor([[self.START]], dtype=torch.long, device=context.device)
        result = bytearray()
        for _ in range(max_bytes):
            inp = torch.cat((self.byte_embedding(previous), style_vector.unsqueeze(1), projected.unsqueeze(1)), dim=-1)
            hidden, state = self.recurrent(inp, state)
            logits = self.output(hidden[:, -1, :]).squeeze(0)
            token = (int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1, generator=rng).item())
                     if sample else int(logits.argmax().item()))
            if token == self.EOS:
                return bytes(result), True, style
            result.append(token)
            previous = torch.tensor([[token]], dtype=torch.long, device=context.device)
        return bytes(result), False, style


def distribution_diversity_loss(logits: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Mean pairwise cosine similarity of *predicted* byte distributions.

    ``logits`` has shape (prompts, styles, positions, 257). Minimize this
    differentiable term alongside byte NLL; no argmax/sample is in the graph.
    It rewards surface variety, not truthfulness or grammaticality.
    """
    if logits.ndim != 4 or logits.shape[-1] != 257 or logits.shape[1] < 2:
        raise ValueError("logits must have shape (prompts, at least 2 styles, positions, 257)")
    b, styles, length, _ = logits.shape
    if valid is None:
        valid = torch.ones((b, styles, length), dtype=torch.bool, device=logits.device)
    if valid.shape != (b, styles, length) or valid.dtype != torch.bool or valid.device != logits.device:
        raise ValueError("valid mask must match logits batch/style/position axes")
    probs = F.normalize(torch.softmax(logits, dim=-1), dim=-1, eps=1e-8)
    similarities = torch.einsum("bstv,brtv->bsrt", probs, probs)
    pairs = valid.unsqueeze(2) & valid.unsqueeze(1)
    pairs = pairs & ~torch.eye(styles, dtype=torch.bool, device=logits.device).view(1, styles, styles, 1)
    if not bool(pairs.any()):
        return logits.sum() * 0
    return similarities.masked_select(pairs).mean()


def load_diverse_decoder(path: str | Path, *, generator_path: str | Path) -> DiverseByteDecoder:
    state = torch.load(path, map_location="cpu", weights_only=True)
    actual = hashlib.sha256(Path(generator_path).read_bytes()).hexdigest()
    if not isinstance(state, dict) or state.get("schema_version") != 1 or state.get("generator_sha256") != actual:
        raise ValueError("diverse response checkpoint is incompatible with this generator")
    decoder = DiverseByteDecoder(state["context_dim"], num_styles=state["num_styles"], hidden_dim=state["hidden_dim"],
                                 byte_dim=state["byte_dim"], style_dim=state["style_dim"])
    decoder.load_state_dict(state["state_dict"], strict=True)
    if not bool(decoder.response_trained):
        raise ValueError("diverse response checkpoint is not trained")
    return decoder.eval()
