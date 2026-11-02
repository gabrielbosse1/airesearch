"""Small decoder-only UTF-8 byte model; not a Phase 18 checkpoint architecture."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class Config:
    vocab_size: int = 268
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 1024
    dropout: float = 0.0
    gradient_checkpointing: bool = True

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__(); d = cfg.d_model
        self.heads = cfg.n_heads; self.dim = d // cfg.n_heads
        self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3*d); self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, d))
        self.dropout = cfg.dropout
    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = [z.view(b, t, self.heads, self.dim).transpose(1, 2) for z in (q,k,v)]
        if torch.onnx.is_in_onnx_export():
            mask = torch.ones((t, t), device=x.device, dtype=torch.bool).tril()
            scores = (q @ k.transpose(-2, -1)) * self.dim ** -0.5
            a = torch.softmax(scores.masked_fill(~mask, float('-inf')), dim=-1) @ v
        else:
            # PyTorch chooses a compatible fused or math SDPA kernel. FlashAttention-2
            # generally requires Ampere or newer; the T4 falls back automatically.
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).contiguous().view(b,t,d))
        return x + self.ff(self.ln2(x))

class PhoneLM(nn.Module):
    def __init__(self, cfg=Config()):
        super().__init__(); self.cfg = cfg
        assert cfg.d_model % cfg.n_heads == 0
        self.token = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model); self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.token.weight
        # GPT-style initialization keeps tied-embedding logits and loss well scaled.
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None: nn.init.zeros_(module.bias)
    def forward(self, ids):
        t = ids.shape[1]
        x = self.token(ids) + self.pos(torch.arange(t, device=ids.device))
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training and self.cfg.gradient_checkpointing else block(x)
        return self.head(self.norm(x))
    def loss(self, ids):
        logits = self(ids[:, :-1]); target = ids[:, 1:]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=258)
    def freeze(self, count):
        for p in self.parameters(): p.requires_grad = True
        for block in self.blocks[:count]:
            for p in block.parameters(): p.requires_grad = False

@torch.no_grad()
def generate(model, prompt, max_new=96, sample=False, temperature=1.0):
    model.eval(); device = next(model.parameters()).device
    ids = [256] + list(prompt.encode('utf8'))
    ids = ids[-(model.cfg.max_seq_len-1):]
    output = []
    for _ in range(min(max_new, model.cfg.max_seq_len-len(ids))):
        logits = model(torch.tensor([ids], device=device))[0, -1].float() / temperature
        logits[258:] = -float('inf') # forbid PAD and reserved ids; EOS remains available
        nxt = int(torch.multinomial(torch.softmax(logits, -1), 1)) if sample else int(logits.argmax())
        if nxt == 257: break
        ids.append(nxt); output.append(nxt)
    return bytes(output).decode('utf8', errors='replace')

def export_onnx(model, path):
    # ONNX uses the explicit mathematical causal attention path; no external weights
    # at this model size. Export CPU FP32 for compatibility; FP16 .pt is smaller.
    # ``dynamo=False`` selects the TorchScript exporter, which honours opset 17 and
    # dynamic_axes; the newer dynamo path emits ops that older onnxruntime rejects.
    model = model.to('cpu').float().eval()
    torch.onnx.export(model, (torch.tensor([[256, 79, 108, 195, 161]], dtype=torch.long),), path,
                      input_names=['input_ids'], output_names=['logits'],
                      dynamic_axes={'input_ids': {0:'batch', 1:'sequence'}, 'logits': {0:'batch', 1:'sequence'}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
