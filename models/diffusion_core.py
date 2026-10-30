"""Diffusion backbone: denoise a noisy *byte embedding* given timestep + concept.

Works in continuous embedding space (true Gaussian diffusion) but stays
byte-level at the boundaries: inputs are raw byte ids (embedded internally for
the noising target) and outputs are decoded back to the nearest byte id by
embedding distance. No external tokenizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .byte_embed import ByteEmbedding, PositionalEncoding, SinusoidalTimeEmb
from .epistemic import ConfidenceHead
from .alignment import soft_dtw_alignment
from .masking import binary_edit_mask, requested_bytes


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d)  (full self-attention, causal not required for diffusion)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DiffusionCore(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 64,
                 n_layers: int = 3, n_heads: int = 4, d_ff: int = 256,
                 max_len: int = 130, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len

        # Shared byte embedding: used to (a) embed clean bytes as the
        # diffusion target and (b) decode predictions back to bytes via
        # nearest-neighbor in embedding space.
        self.byte_emb = ByteEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # Learned condition-token base: timestep + concept are added to it.
        self.cond_tok = nn.Parameter(torch.zeros(1, d_model))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout)
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_model)  # predicts clean *embedding*
        # Mask and explicitly requested replacement bytes are separate, optional
        # conditioning. Zero initialization leaves legacy denoising unchanged.
        self.edit_mask_embedding = nn.Parameter(torch.zeros(d_model))
        self.edit_value_proj = nn.Linear(d_model, d_model, bias=False)
        self.confidence_head = ConfidenceHead(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)
        nn.init.normal_(self.cond_tok, std=0.02)
        nn.init.zeros_(self.edit_value_proj.weight)

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

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ) -> None:
        # Called both when loading DiffusionCore directly and recursively from
        # ByteCoreDiffusion. Only entirely absent legacy heads are initialized;
        # partially missing/corrupt current heads still fail strict loading.
        head_prefix = prefix + "confidence_head."
        if not any(key.startswith(head_prefix) for key in state_dict):
            for key, value in self.confidence_head.state_dict().items():
                state_dict[head_prefix + key] = value.detach().clone()
        # Both Phase 9 tensors are absent from old checkpoints. Initialize the
        # entire absent feature to zero, but don't hide partially missing new
        # checkpoints under strict loading.
        edit_keys = (prefix + "edit_mask_embedding", prefix + "edit_value_proj.weight")
        if not any(key in state_dict for key in edit_keys):
            state_dict[edit_keys[0]] = torch.zeros_like(self.edit_mask_embedding)
            state_dict[edit_keys[1]] = torch.zeros_like(self.edit_value_proj.weight)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )

    def alignment_loss(self, prediction: torch.Tensor, target: torch.Tensor, *,
                       prediction_mask: torch.Tensor | None = None, target_mask: torch.Tensor | None = None,
                       gamma: float = 0.02) -> torch.Tensor:
        """Differentiable soft-DTW on real-byte embeddings; no new parameters."""
        return soft_dtw_alignment(prediction, target, prediction_mask=prediction_mask,
                                  target_mask=target_mask, gamma=gamma)

    # ------------------------------------------------------------------ #
    # Decode: continuous embedding -> nearest byte id
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def embed_to_bytes(self, emb: torch.Tensor) -> torch.Tensor:
        """Map (B, L, d) embeddings to (B, L) byte ids via nearest embedding."""
        b, l, d = emb.shape
        flat = emb.reshape(-1, d)  # (B*L, d)
        # Squared distances via matrix multiplication, without materializing a
        # prohibitively large (B*L, 256, d) tensor during confidence training.
        w = self.byte_emb.emb.weight  # (256, d)
        dist = flat.square().sum(-1, keepdim=True) + w.square().sum(-1).unsqueeze(0) - 2 * (flat @ w.T)
        byte_ids = dist.argmin(-1).reshape(b, l)
        return byte_ids

    # ------------------------------------------------------------------ #
    # Forward: denoise
    # ------------------------------------------------------------------ #
    def forward(self, xt_emb: torch.Tensor, t: torch.Tensor,
                concept: torch.Tensor | None = None, *,
                return_confidence: bool = False,
                edit_mask: torch.Tensor | None = None,
                requested_byte_ids: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict clean byte embedding from noised embedding xt_emb.

        Parameters
        ----------
        xt_emb : (B, L, d) continuous noised byte embedding.
        t : (B,) timestep in [0, 1] (0 = clean, 1 = pure noise).
        concept : (B, d_model) concept vector, or None.

        Returns
        -------
        pred_emb : (B, L, d) predicted clean byte embedding.
        """
        b, l, d = xt_emb.shape
        device = xt_emb.device
        mask: torch.Tensor | None = None
        request_embedding: torch.Tensor | None = None
        if edit_mask is not None:
            mask = binary_edit_mask(edit_mask, (b, l), device)
            if requested_byte_ids is not None:
                ids = requested_bytes(requested_byte_ids, (b, l), device)
                # Values outside the mask cannot condition an edit.
                ids = torch.where(mask, ids, torch.zeros_like(ids))
                request_embedding = self.byte_emb(ids)
        elif requested_byte_ids is not None:
            raise ValueError("requested_byte_ids requires an edit_mask")

        h = self.dropout(xt_emb)
        if mask is not None:
            h = h + mask.unsqueeze(-1).to(h.dtype) * self.edit_mask_embedding

        # Condition token: learned base + timestep + concept.
        t_emb = self.time_mlp(t)  # (B, d)
        if concept is None:
            concept = torch.zeros(b, self.d_model, device=device)
        cond = self.cond_tok + t_emb + concept  # (B, d)
        h = torch.cat([cond.unsqueeze(1), h], dim=1)  # (B, L+1, d)
        h = h + self.pos_emb(h)

        for block in self.blocks:
            h = block(h)
        h = self.norm(h)

        seq_out = h[:, 1:, :]  # drop cond token -> (B, L, d)
        pred_emb = self.head(seq_out)  # (B, L, d)
        if mask is not None and request_embedding is not None:
            # This is a learned projection, not a hard overwrite. Only the
            # caller-declared replacement bytes are supplied as conditioning.
            pred_emb = pred_emb + mask.unsqueeze(-1).to(pred_emb.dtype) * self.edit_value_proj(request_embedding)
        if return_confidence:
            logits = self.confidence_head(torch.cat([seq_out, pred_emb], dim=-1))
            return pred_emb, logits
        return pred_emb
