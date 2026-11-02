"""Small decoder-only UTF-8 byte model; not a Phase 18 checkpoint architecture."""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from notebook.preprocess import BOS, EOS, PAD


@dataclass
class Config:
    # Byte ids 0..255, then BOS/EOS/PAD. Higher ids are reserved so checkpoints
    # can add task markers without changing the base byte rows.
    vocab_size: int = 268
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 1024
    dropout: float = 0.0
    gradient_checkpointing: bool = True

    def __post_init__(self):
        if not isinstance(self.vocab_size, int) or self.vocab_size <= PAD:
            raise ValueError(f'vocab_size must be at least {PAD + 1}')
        for name in ('d_model', 'n_layers', 'n_heads', 'd_ff', 'max_seq_len'):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.max_seq_len < 2:
            raise ValueError('max_seq_len must be at least 2 for BOS/EOS')
        if self.d_model % self.n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        if not isinstance(self.dropout, (int, float)) or not 0.0 <= self.dropout < 1.0:
            raise ValueError('dropout must be in [0, 1)')


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.heads = cfg.n_heads
        self.dim = d // cfg.n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, d))
        self.dropout = cfg.dropout

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = [z.view(b, t, self.heads, self.dim).transpose(1, 2) for z in (q, k, v)]
        if torch.onnx.is_in_onnx_export():
            mask = torch.ones((t, t), device=x.device, dtype=torch.bool).tril()
            scores = (q @ k.transpose(-2, -1)) * self.dim ** -0.5
            a = torch.softmax(scores.masked_fill(~mask, float('-inf')), dim=-1) @ v
        else:
            # PyTorch chooses a compatible fused or math SDPA kernel. FlashAttention-2
            # generally requires Ampere or newer; the T4 falls back automatically.
            a = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        x = x + self.proj(a.transpose(1, 2).contiguous().view(b, t, d))
        return x + self.ff(self.ln2(x))


class PhoneLM(nn.Module):
    def __init__(self, cfg: Config | None = None):
        super().__init__()
        self.cfg = cfg or Config()
        self.token = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.pos = nn.Embedding(self.cfg.max_seq_len, self.cfg.d_model)
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layers)])
        self.norm = nn.LayerNorm(self.cfg.d_model)
        self.head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        self.head.weight = self.token.weight
        # GPT-style initialization keeps tied-embedding logits and loss well scaled.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2:
            raise ValueError('input_ids must be a rank-2 integer tensor [batch, sequence]')
        if ids.shape[1] == 0:
            raise ValueError('input_ids must contain at least one token')
        if ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError(f'input length {ids.shape[1]} exceeds max_seq_len {self.cfg.max_seq_len}')
        if ids.dtype != torch.long:
            raise TypeError('input_ids must have dtype torch.long')
        if torch.any(ids < 0) or torch.any(ids >= self.cfg.vocab_size):
            raise ValueError('input_ids contains a token outside the configured vocabulary')
        t = ids.shape[1]
        x = self.token(ids) + self.pos(torch.arange(t, device=ids.device))
        for block in self.blocks:
            x = (checkpoint(block, x, use_reentrant=False)
                 if self.training and self.cfg.gradient_checkpointing else block(x))
        return self.head(self.norm(x))

    def loss(self, ids):
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[1] < 2:
            raise ValueError('loss requires a rank-2 batch with at least BOS and EOS')
        logits = self(ids[:, :-1])
        target = ids[:, 1:]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=PAD)

    def freeze(self, count):
        if not isinstance(count, int) or not 0 <= count <= len(self.blocks):
            raise ValueError(f'freeze count must be between 0 and {len(self.blocks)}')
        for parameter in self.parameters():
            parameter.requires_grad = True
        for block in self.blocks[:count]:
            for parameter in block.parameters():
                parameter.requires_grad = False


@torch.no_grad()
def generate(model, prompt, max_new=96, sample=False, temperature=1.0):
    """Generate decoded UTF-8 bytes while preserving the caller's train/eval mode."""
    if not isinstance(prompt, str):
        raise TypeError('prompt must be a str')
    if not isinstance(max_new, int) or max_new < 0:
        raise ValueError('max_new must be a non-negative integer')
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ValueError('temperature must be positive')
    device = next(model.parameters()).device
    # Keep BOS even for a long prompt; trimming from the left is preferable to
    # silently changing the sequence format by dropping it.
    prompt_bytes = list(prompt.encode('utf8'))[-(model.cfg.max_seq_len - 1):]
    ids = [BOS] + prompt_bytes
    output = []
    was_training = model.training
    model.eval()
    try:
        for _ in range(min(max_new, model.cfg.max_seq_len - len(ids))):
            logits = model(torch.tensor([ids], dtype=torch.long, device=device))[0, -1].float()
            logits = logits / float(temperature)
            logits[PAD:] = -float('inf')  # PAD and reserved ids; EOS remains available.
            nxt = (int(torch.multinomial(torch.softmax(logits, -1), 1))
                   if sample else int(logits.argmax()))
            if nxt == EOS:
                break
            ids.append(nxt)
            output.append(nxt)
    finally:
        model.train(was_training)
    return bytes(output).decode('utf8', errors='replace')


def export_onnx(model, path):
    # ONNX uses the explicit mathematical causal attention path; no external weights
    # at this model size. Export CPU FP32 for compatibility; FP16 .pt is smaller.
    # ``dynamo=False`` selects the TorchScript exporter, which honours opset 17 and
    # dynamic_axes; the newer dynamo path emits ops that older onnxruntime rejects.
    model = model.to('cpu').float().eval()
    torch.onnx.export(
        model,
        (torch.tensor([[BOS, 79, 108, 195, 161]], dtype=torch.long),),
        path,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'},
                      'logits': {0: 'batch', 1: 'sequence'}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
