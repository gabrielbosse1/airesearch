"""ByteCore-Diffusion: byte-level diffusion model with a concept bank.

Puts together:
  * DiffusionCore  -- the denoising backbone (continuous embedding space).
  * ConceptBank    -- write-only memory for concepts (learn w/o retraining).
  * Noise schedule -- Gaussian diffusion over byte embeddings.

Everything is raw bytes (0-255). Modalities (text / image / audio) are
encoded to raw byte arrays by thin wrappers; the model itself only sees bytes.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .concept_bank import ConceptBank
from .diffusion_core import DiffusionCore


class ByteCoreDiffusion(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 64,
                 n_layers: int = 3, n_heads: int = 4, d_ff: int = 256,
                 max_len: int = 130, diffusion_timesteps: int = 50,
                 concept_capacity: int = 256, concept_dim: int = 64,
                 dropout: float = 0.1, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.diffusion_timesteps = diffusion_timesteps
        self.d_model = d_model
        self.max_len = max_len

        self.core = DiffusionCore(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=dropout,
        )
        self.bank = ConceptBank(capacity=concept_capacity, d=concept_dim)

        # ----- Gaussian noise schedule (linear betas) -----
        T = diffusion_timesteps
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        # --- Fixed, well-separated byte embeddings ---------------------
        # Freeze the byte embedding so it cannot collapse (which would make
        # every byte indistinguishable). Use normalized random vectors so
        # each byte id maps to a distinct point on the unit sphere; only the
        # denoising core is trained.
        self._init_fixed_byte_embeddings()

        self.to(device)

    def _init_fixed_byte_embeddings(self) -> None:
        emb = self.core.byte_emb.emb.weight  # (vocab, d_model)
        nn.init.kaiming_uniform_(emb, a=math.sqrt(5))
        with torch.no_grad():
            emb.div_(emb.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        emb.requires_grad = False

    # ---------------------------------------------------------------- #
    # Noise schedule helpers
    # ---------------------------------------------------------------- #
    def q_sample(self, x0_emb: torch.Tensor, t_int: torch.Tensor) -> torch.Tensor:
        """q(x_t | x_0) = sqrt(ab_t) x_0 + sqrt(1 - ab_t) eps.

        t_int: integer timestep indices in {0, ..., T-1}.
        """
        b = x0_emb.shape[0]
        t_int = t_int.clamp(0, self.diffusion_timesteps - 1).long()
        ab = self.alpha_bars[t_int].view(b, 1, 1)  # (B,1,1)
        eps = torch.randn_like(x0_emb)
        return ab.sqrt() * x0_emb + (1.0 - ab).sqrt() * eps

    def p_mean_variance(self, xt_emb: torch.Tensor, t_idx: torch.Tensor,
                        concept: torch.Tensor | None,
                        guidance_scale: float = 1.0):
        """DDPM posterior mean/variance for x_{t-1} given x_t, predicting x0.

        t_idx: integer timestep indices in {0, ..., T-1}.
        """
        b = xt_emb.shape[0]
        device = xt_emb.device
        t_idx = t_idx.clamp(0, self.diffusion_timesteps - 1).long()

        # Classifier-free guidance: cond vs uncond prediction.
        if guidance_scale != 1.0 and concept is not None:
            pred_c = self.core(xt_emb, t_idx / self.diffusion_timesteps, concept)
            pred_u = self.core(xt_emb, t_idx / self.diffusion_timesteps, None)
            pred_emb = pred_u + guidance_scale * (pred_c - pred_u)
        else:
            pred_emb = self.core(xt_emb, t_idx / self.diffusion_timesteps, concept)

        # Posterior q(x_{t-1} | x_t, x_0) with x_0 = pred_emb.
        ab_t = self.alpha_bars[t_idx].view(b, 1, 1)
        # alpha_bar_{t-1}, with alpha_bar_{-1} := 1 (clean start).
        ab_prev_vals = self.alpha_bars[t_idx - 1].clone()
        ab_prev_vals[t_idx == 0] = 1.0
        ab_prev = ab_prev_vals.view(b, 1, 1)
        beta_t = self.betas[t_idx].view(b, 1, 1)
        alpha_t = self.alphas[t_idx].view(b, 1, 1)

        coef1 = (ab_prev.sqrt() * beta_t) / (1.0 - ab_t)
        coef2 = (alpha_t.sqrt() * (1.0 - ab_prev)) / (1.0 - ab_t)
        mean = coef1 * pred_emb + coef2 * xt_emb
        var = beta_t  # simplified fixed variance
        return mean, var, pred_emb

    @torch.no_grad()
    def p_sample_loop(self, shape, concept: torch.Tensor | None = None,
                      guidance_scale: float = 2.0) -> torch.Tensor:
        """Ancestral sampling from pure noise -> clean byte embedding.

        shape: (B, L, d). Returns (B, L, d) clean embedding.
        """
        device = self.device
        b = shape[0]
        xt = torch.randn(shape, device=device)  # x_T ~ N(0, I)
        for t in range(self.diffusion_timesteps - 1, -1, -1):
            tb = torch.full((b,), t, device=device, dtype=torch.long)
            mean, var, _ = self.p_mean_variance(xt, tb, concept,
                                                guidance_scale)
            if t > 0:
                z = torch.randn_like(xt)
                xt = mean + var.sqrt() * z
            else:
                xt = mean
        return xt

    # ---------------------------------------------------------------- #
    # Training loss
    # ---------------------------------------------------------------- #
    def compute_loss(self, x0_emb: torch.Tensor,
                     concept: torch.Tensor | None = None) -> torch.Tensor:
        """Denoising score-matching loss (MSE between pred and clean emb)."""
        b = x0_emb.shape[0]
        device = x0_emb.device
        t_int = torch.randint(0, self.diffusion_timesteps, (b,), device=device)
        t_float = t_int.float() / self.diffusion_timesteps
        xt_emb = self.q_sample(x0_emb, t_int)
        pred_emb = self.core(xt_emb, t_float, concept)
        loss = F.mse_loss(pred_emb, x0_emb)
        return loss

    # ---------------------------------------------------------------- #
    # Bytes <-> embeddings
    # ---------------------------------------------------------------- #
    def bytes_to_emb(self, byte_ids: torch.Tensor) -> torch.Tensor:
        return self.core.byte_emb(byte_ids)  # (B, L, d)

    @torch.no_grad()
    def generate(self, n_samples: int, seq_len: int,
                 concept: torch.Tensor | None = None,
                 guidance_scale: float = 2.0) -> torch.Tensor:
        """Generate clean byte ids: (n_samples, seq_len)."""
        shape = (n_samples, seq_len, self.d_model)
        emb = self.p_sample_loop(shape, concept, guidance_scale)
        byte_ids = self.core.embed_to_bytes(emb)
        return byte_ids  # (B, L)

    # ---------------------------------------------------------------- #
    # Concept bank interface (learn without retraining)
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def encode_concept(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Derive a concept vector (d_model,) from example bytes: mean emb."""
        emb = self.core.byte_emb(byte_ids)  # (B, L, d) or (L, d)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return emb.mean(dim=0)  # (d,)

    @torch.no_grad()
    def write_concept(self, name: str, example_bytes):
        """Store a new concept from example byte ids (list or tensor). No
        retraining -- direct memory write."""
        if isinstance(example_bytes, list):
            example_bytes = torch.tensor(example_bytes, dtype=torch.long,
                                         device=self.device)
        key = self.encode_concept(example_bytes)
        self.bank.write(name, key, key)

    @torch.no_grad()
    def read_concept(self, name_or_query) -> torch.Tensor | None:
        """Recall a concept vector by name, or by query vector (d,)."""
        if isinstance(name_or_query, str):
            if name_or_query not in self.bank.name_to_idx:
                return None
            idx = self.bank.name_to_idx[name_or_query]
            return self.bank.vals[idx].clone()
        else:
            q = torch.tensor(name_or_query, dtype=torch.float,
                             device=self.device) if not \
                torch.is_tensor(name_or_query) else name_or_query
            return self.bank.read(q)

    def save(self, dirpath: str):
        os.makedirs(dirpath, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(dirpath, "model.pt"))
        self.bank.save(os.path.join(dirpath, "concept_bank.pt"))

    def load(self, dirpath: str, map_location="cpu"):
        self.load_state_dict(torch.load(os.path.join(dirpath, "model.pt"),
                                        map_location=map_location))
        self.bank.load(os.path.join(dirpath, "concept_bank.pt"))
        self.to(self.device)
        return self
