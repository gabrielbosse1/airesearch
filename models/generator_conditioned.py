"""GeneratorConditioned: a small byte-level generator conditioned on memory
context.

It turns (user_input + retrieved_memory_context) into output bytes. The
output can be:
  * plain text / code
  * a special FUNCTION_CALL token sequence that the host intercepts and
    executes (enables tool use, e.g. "write code", "play minecraft action")

It is conditioned on a memory context vector from MemoryModel, so its
generation is grounded in long-term memory (people, past sessions, facts)
rather than only the current context window.

Shared d_model space with MemoryModel so the two can be co-trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorConditioned(nn.Module):
    """Byte-level generator conditioned on long-term memory context.

    Vocabulary is raw bytes (0-255) plus a few reserved control tokens that
    live *above* 255 and are handled by the host (never embedded as bytes):
        FUNC_CALL = 256  -> begin a function call
        FUNC_ARG  = 257  -> function argument follows
        FUNC_END  = 258  -> end function call
    These are embedded in a separate control embedding table.
    """

    VOCAB_BYTES = 256
    FUNC_CALL = 256
    FUNC_ARG = 257
    FUNC_END = 258
    VOCAB_SIZE = 259  # 256 bytes + 3 control tokens

    def __init__(
        self,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 256,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Byte embedding (0-255) + control-token embedding (256-258)
        self.byte_emb = nn.Embedding(self.VOCAB_BYTES, d_model)
        self.control_emb = nn.Embedding(3, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Memory context injection: memory vector is added to every position
        # via a gated projection.
        self.memory_gate = nn.Linear(d_model, d_model)
        self.memory_proj = nn.Linear(d_model, d_model)

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.VOCAB_SIZE)
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_context: torch.Tensor | None = None,
        return_logits: bool = True,
    ):
        """Generate next-byte logits conditioned on memory context.

        Parameters
        ----------
        input_ids : (B, L) long. Values in [0, VOCAB_SIZE). Control tokens
            (>=256) use the control embedding.
        memory_context : (B, d_model) or (d_model,) retrieved memory vector,
            or None.
        return_logits : bool

        Returns
        -------
        logits : (B, L, VOCAB_SIZE)
        """
        b, l = input_ids.shape
        device = input_ids.device

        # Build embeddings: bytes vs control tokens.
        is_control = input_ids >= self.VOCAB_BYTES
        byte_ids = input_ids.clamp(max=self.VOCAB_BYTES - 1)
        ctrl_ids = (input_ids - self.VOCAB_BYTES).clamp(min=0, max=2)

        h = self.byte_emb(byte_ids)  # (B, L, d)
        h = h + self.control_emb(ctrl_ids) * is_control.unsqueeze(-1).float()

        # Inject memory context (gated) at every position.
        if memory_context is not None:
            if memory_context.dim() == 1:
                memory_context = memory_context.unsqueeze(0).expand(b, -1)
            mem_gate = torch.sigmoid(self.memory_gate(memory_context))  # (B, d)
            mem_proj = self.memory_proj(memory_context)  # (B, d)
            h = h + mem_gate.unsqueeze(1) * mem_proj.unsqueeze(1)

        pos = torch.arange(l, device=device).unsqueeze(0).expand(b, l)
        h = h + self.pos_emb(pos)  # positional over sequence index
        h = self.dropout(h)

        # Causal mask so generation is autoregressive.
        causal_mask = torch.triu(
            torch.ones(l, l, device=device, dtype=torch.bool), diagonal=1
        )
        for block in self.blocks:
            h = block(h, src_mask=causal_mask, is_causal=True)
        h = self.norm(h)

        if return_logits:
            return self.head(h)  # (B, L, VOCAB_SIZE)
        return h

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        memory_context: torch.Tensor | None = None,
        max_new: int = 40,
        temperature: float = 0.8,
        func_handler=None,
    ):
        """Autoregressively generate bytes. If a FUNC_CALL is emitted and
        ``func_handler`` is provided, the handler is invoked and its result
        injected back as context.

        Returns (generated_ids, func_calls_made).
        """
        self.eval()
        x = prompt_ids.clone().unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids
        func_calls = []
        for _ in range(max_new):
            if x.shape[1] >= self.max_len:
                break
            logits = self.forward(x, memory_context)  # (1, L, V)
            next_logits = logits[0, -1] / temperature
            probs = F.softmax(next_logits, dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1).item())
            x = torch.cat([x, torch.tensor([[nxt]], device=x.device)], dim=1)

            if nxt == self.FUNC_CALL and func_handler is not None:
                # Read the argument tokens until FUNC_END.
                arg_tokens = []
                while True:
                    logits = self.forward(x, memory_context)
                    nl = logits[0, -1] / temperature
                    pr = F.softmax(nl, dim=-1)
                    nt = int(torch.multinomial(pr, num_samples=1).item())
                    x = torch.cat([x, torch.tensor([[nt]], device=x.device)], dim=1)
                    if nt == self.FUNC_END:
                        break
                    if nt < self.VOCAB_BYTES:
                        arg_tokens.append(nt)
                func_name = bytes(arg_tokens).decode("utf-8", errors="replace")
                func_calls.append(func_name)
                result = func_handler(func_name)
                # Inject result as a pseudo-prompt: "RESULT:<result>"
                result_bytes = list(f"RESULT:{result}".encode("utf-8"))
                for rb in result_bytes:
                    x = torch.cat([x, torch.tensor([[rb]], device=x.device)], dim=1)
        return x[0], func_calls
