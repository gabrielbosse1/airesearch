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
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .concept_bank import ConceptBank
from .adaptive import configured_threshold
from .holdout import HoldoutViolation
from .diffusion_core import DiffusionCore
from .masking import binary_edit_mask, requested_bytes


@dataclass(frozen=True)
class SamplingStats:
    """Per-sample computation and confidence from a diffusion sampling run."""

    steps: torch.Tensor
    confidence: torch.Tensor
    halted_early: torch.Tensor


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
        # A zero-initialized, migrated legacy head is not a trained editor.
        self.register_buffer("mask_edit_trained", torch.zeros((), dtype=torch.bool))

        # --- Fixed, well-separated byte embeddings ---------------------
        # Freeze the byte embedding so it cannot collapse (which would make
        # every byte indistinguishable). Use normalized random vectors so
        # each byte id maps to a distinct point on the unit sphere; only the
        # denoising core is trained.
        self._init_fixed_byte_embeddings()

        self.to(device)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if self.bank.heldout_guard is not None and not getattr(self, "_loading_audited_model", False):
            raise HoldoutViolation("direct state loading bypasses concept provenance; use ByteCoreDiffusion.load")
        edit_keys = ("core.edit_mask_embedding", "core.edit_value_proj.weight")
        has_edit = any(key in state_dict for key in edit_keys)
        if "mask_edit_trained" not in state_dict and not has_edit:
            migrated = OrderedDict(state_dict)
            if hasattr(state_dict, "_metadata"):
                migrated._metadata = state_dict._metadata
            migrated["mask_edit_trained"] = torch.zeros_like(self.mask_edit_trained)
            state_dict = migrated
        elif "mask_edit_trained" in state_dict and not has_edit:
            raise ValueError("checkpoint contains mask metadata but has no mask editor weights")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def mark_mask_edit_trained(self) -> None:
        """Record completed optimizer updates; this is not an accuracy guarantee."""
        self.mask_edit_trained.fill_(True)

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

    def p_mean_variance(
        self,
        xt_emb: torch.Tensor,
        t_idx: torch.Tensor,
        concept: torch.Tensor | None,
        guidance_scale: float = 1.0,
        *,
        return_confidence: bool = False,
        edit_mask: torch.Tensor | None = None,
        requested_byte_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """DDPM posterior and optional per-byte confidence in predicted x0.

        For classifier-free guidance, confidence is conservatively the lower
        of the conditional and unconditional estimates. It still requires
        validation for that guidance scale; it is not an error guarantee.
        """
        b = xt_emb.shape[0]
        t_idx = t_idx.clamp(0, self.diffusion_timesteps - 1).long()
        confidence: torch.Tensor | None = None
        core_kwargs = {}
        if edit_mask is not None:
            core_kwargs = {"edit_mask": edit_mask, "requested_byte_ids": requested_byte_ids}
        elif requested_byte_ids is not None:
            raise ValueError("requested_byte_ids requires an edit_mask")
        if guidance_scale != 1.0 and concept is not None:
            if return_confidence:
                pred_c, conf_c = self.core(xt_emb, t_idx / self.diffusion_timesteps, concept, return_confidence=True, **core_kwargs)
                pred_u, conf_u = self.core(xt_emb, t_idx / self.diffusion_timesteps, None, return_confidence=True, **core_kwargs)
                confidence = torch.minimum(self.core.confidence_head.probability(conf_c), self.core.confidence_head.probability(conf_u))
            else:
                pred_c = self.core(xt_emb, t_idx / self.diffusion_timesteps, concept, **core_kwargs)
                pred_u = self.core(xt_emb, t_idx / self.diffusion_timesteps, None, **core_kwargs)
            pred_emb = pred_u + guidance_scale * (pred_c - pred_u)
        elif return_confidence:
            pred_emb, conf_logits = self.core(xt_emb, t_idx / self.diffusion_timesteps, concept, return_confidence=True, **core_kwargs)
            confidence = self.core.confidence_head.probability(conf_logits)
        else:
            pred_emb = self.core(xt_emb, t_idx / self.diffusion_timesteps, concept, **core_kwargs)
        assert isinstance(pred_emb, torch.Tensor)

        ab_t = self.alpha_bars[t_idx].view(b, 1, 1)
        ab_prev_vals = self.alpha_bars[t_idx - 1].clone()
        ab_prev_vals[t_idx == 0] = 1.0
        ab_prev = ab_prev_vals.view(b, 1, 1)
        beta_t = self.betas[t_idx].view(b, 1, 1)
        alpha_t = self.alphas[t_idx].view(b, 1, 1)
        coef1 = (ab_prev.sqrt() * beta_t) / (1.0 - ab_t)
        coef2 = (alpha_t.sqrt() * (1.0 - ab_prev)) / (1.0 - ab_t)
        mean = coef1 * pred_emb + coef2 * xt_emb
        var = beta_t
        if return_confidence:
            assert confidence is not None
            return mean, var, pred_emb, confidence
        return mean, var, pred_emb

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: tuple[int, int, int],
        concept: torch.Tensor | None = None,
        guidance_scale: float = 2.0,
        *,
        confidence_threshold: float | None = None,
        min_steps: int = 1,
        confidence_reduction: str = "min",
        return_stats: bool = False,
        initial_xt: torch.Tensor | None = None,
        adaptive_config: Mapping[str, Any] | None = None,
        edit_mask: torch.Tensor | None = None,
        original_emb: torch.Tensor | None = None,
        requested_byte_ids: torch.Tensor | None = None,
        allow_untrained_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SamplingStats]:
        """Ancestral sampling with opt-in, per-sample adaptive computation.

        Early-stopped samples receive the predicted clean embedding, not an
        intermediate noisy state. Remaining active samples continue denoising.
        Default behavior always uses all diffusion timesteps. An optional
        initial noisy embedding allows paired reconstruction experiments; it
        must correspond to the last timestep. This is not automatic inversion.
        With edit_mask, original_emb is required and unmasked embeddings are
        clamped to it at every step. requested_byte_ids supplies desired bytes
        separately from the mask; the network predicts the masked outputs.
        """
        if adaptive_config is not None:
            if confidence_threshold is not None:
                raise ValueError("pass either adaptive_config or an explicit threshold, not both")
            confidence_threshold = configured_threshold(adaptive_config, self.core.confidence_head, "diffusion_sampling")
            if confidence_threshold is not None:
                min_steps = adaptive_config.get("min_steps", min_steps)
                confidence_reduction = adaptive_config.get("confidence_reduction", confidence_reduction)
        if len(shape) != 3 or shape[0] < 1 or shape[1] < 1 or shape[2] != self.d_model:
            raise ValueError("shape must be (positive batch, positive length, d_model)")
        if shape[1] + 1 > self.core.max_len:
            raise ValueError("sequence length exceeds the diffusion core's positional limit")
        if not isinstance(min_steps, int) or isinstance(min_steps, bool) or not 1 <= min_steps <= self.diffusion_timesteps:
            raise ValueError("min_steps must lie within the diffusion schedule")
        if confidence_reduction not in ("min", "mean"):
            raise ValueError("confidence_reduction must be 'min' or 'mean'")
        if not math.isfinite(guidance_scale):
            raise ValueError("guidance_scale must be finite")
        if confidence_threshold is not None:
            if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
                raise ValueError("confidence_threshold must be in [0, 1]")
            self.core.confidence_head.warn_if_uncalibrated("diffusion_sampling")
        device = self.betas.device
        b = shape[0]
        mask: torch.Tensor | None = None
        source: torch.Tensor | None = None
        request: torch.Tensor | None = None
        if edit_mask is not None:
            mask = binary_edit_mask(edit_mask, (b, shape[1]), device)
            if not torch.is_tensor(original_emb) or tuple(original_emb.shape) != tuple(shape) or not torch.is_floating_point(original_emb) or not bool(torch.isfinite(original_emb).all()):
                raise ValueError("original_emb must be finite floating point and match shape")
            source = original_emb.detach().to(device=device, dtype=self.core.byte_emb.emb.weight.dtype).clone()
            if requested_byte_ids is not None:
                request = requested_bytes(requested_byte_ids, (b, shape[1]), device)
            if bool(mask.any()) and not bool(self.mask_edit_trained) and not allow_untrained_mask:
                raise RuntimeError("mask-conditioned editor has not been trained; train it or explicitly allow an untrained experiment")
            if confidence_threshold is not None:
                raise ValueError("adaptive confidence thresholds are not calibrated for masked edits")
        elif original_emb is not None or requested_byte_ids is not None:
            raise ValueError("original_emb and requested_byte_ids require an edit_mask")
        if concept is not None:
            concept = concept.to(device)
            if concept.ndim not in (1, 2) or concept.shape[-1] != self.d_model:
                raise ValueError("concept must have shape (d_model,), (1,d_model) or (B,d_model)")
            if concept.ndim == 2 and concept.shape[0] not in (1, b):
                raise ValueError("concept batch must match sampling batch")
        if initial_xt is None:
            xt = torch.randn(shape, device=device, dtype=self.core.byte_emb.emb.weight.dtype)
        else:
            if tuple(initial_xt.shape) != tuple(shape) or not torch.is_floating_point(initial_xt) or not bool(torch.isfinite(initial_xt).all()):
                raise ValueError("initial_xt must be finite floating point and match shape")
            xt = initial_xt.detach().to(device=device, dtype=self.core.byte_emb.emb.weight.dtype).clone()
        if mask is not None:
            assert source is not None
            xt = torch.where(mask.unsqueeze(-1), xt, source)
            active = mask.any(dim=1)
        else:
            active = torch.ones(b, dtype=torch.bool, device=device)
        steps = torch.zeros(b, dtype=torch.long, device=device)
        last_confidence = torch.full((b,), float("nan"), device=device)
        halted_early = torch.zeros(b, dtype=torch.bool, device=device)
        for t in range(self.diffusion_timesteps - 1, -1, -1):
            active_idx = active.nonzero(as_tuple=False).flatten()
            if active_idx.numel() == 0:
                break
            tb = torch.full((active_idx.numel(),), t, device=device, dtype=torch.long)
            active_concept = concept
            if concept is not None and concept.ndim == 2 and concept.shape[0] == b:
                active_concept = concept.index_select(0, active_idx)
            mask_kwargs = {}
            active_mask = None
            if mask is not None:
                active_mask = mask.index_select(0, active_idx)
                mask_kwargs = {"edit_mask": active_mask,
                               "requested_byte_ids": None if request is None else request.index_select(0, active_idx)}
            need_confidence = confidence_threshold is not None or return_stats
            if need_confidence:
                mean, var, pred, per_byte = self.p_mean_variance(
                    xt.index_select(0, active_idx), tb, active_concept,
                    guidance_scale, return_confidence=True, **mask_kwargs,
                )
                if active_mask is None:
                    confidence = per_byte.amin(dim=1) if confidence_reduction == "min" else per_byte.mean(dim=1)
                elif confidence_reduction == "min":
                    confidence = per_byte.masked_fill(~active_mask, float("inf")).amin(dim=1)
                else:
                    confidence = (per_byte * active_mask).sum(dim=1) / active_mask.sum(dim=1)
                last_confidence[active_idx] = confidence
            else:
                mean, var, pred = self.p_mean_variance(
                    xt.index_select(0, active_idx), tb, active_concept, guidance_scale, **mask_kwargs,
                )
                confidence = None
            steps[active_idx] += 1
            early = torch.zeros(active_idx.numel(), dtype=torch.bool, device=device)
            if confidence_threshold is not None and t > 0:
                assert confidence is not None
                early = torch.isfinite(confidence) & (confidence >= confidence_threshold) & (steps[active_idx] >= min_steps)
            if t > 0:
                proposal = mean + var.sqrt() * torch.randn_like(mean)
                proposal[early] = pred[early]
            else:
                proposal = mean
            if active_mask is not None:
                assert source is not None
                proposal = torch.where(active_mask.unsqueeze(-1), proposal, source.index_select(0, active_idx))
            xt[active_idx] = proposal
            if bool(early.any()):
                stopped = active_idx[early]
                active[stopped] = False
                halted_early[stopped] = True
            if t == 0:
                active[active_idx] = False
        if return_stats:
            return xt, SamplingStats(steps, last_confidence, halted_early)
        return xt

    @torch.no_grad()
    def p_sample_loop_with_mask(
        self, shape: tuple[int, int, int], edit_mask: torch.Tensor, *,
        original_emb: torch.Tensor, requested_byte_ids: torch.Tensor | None = None,
        concept: torch.Tensor | None = None, guidance_scale: float = 1.0,
        **sampling_options: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, SamplingStats]:
        """Sample only masked positions while retaining original embeddings.

        A mask alone is inpainting, not a request for a particular value. Pass
        requested_byte_ids to specify replacement values; only entries inside
        the mask are visible to the model. Output positions remain predictions.
        """
        return self.p_sample_loop(
            shape, concept, guidance_scale, edit_mask=edit_mask,
            original_emb=original_emb, requested_byte_ids=requested_byte_ids,
            **sampling_options,
        )

    @torch.no_grad()
    def masked_edit(
        self, input_ids: torch.Tensor, edit_mask: torch.Tensor,
        requested_byte_ids: torch.Tensor | None = None, *,
        concept: torch.Tensor | None = None, guidance_scale: float = 1.0,
        return_stats: bool = False, **sampling_options: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, SamplingStats]:
        """Edit a batch of equal-length byte rows, never copying requested bytes into the output."""
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (B, L)")
        shape = tuple(input_ids.shape)
        ids = requested_bytes(input_ids, shape, self.betas.device, name="input_ids")
        mask = binary_edit_mask(edit_mask, shape, self.betas.device)
        original = self.bytes_to_emb(ids)
        sampled = self.p_sample_loop_with_mask(
            tuple(original.shape), mask, original_emb=original,
            requested_byte_ids=requested_byte_ids, concept=concept,
            guidance_scale=guidance_scale, return_stats=return_stats, **sampling_options,
        )
        if return_stats:
            emb, stats = sampled
        else:
            emb = sampled
        decoded = self.core.embed_to_bytes(emb)
        # Guard byte identity even if an embedding table contains a tie. This
        # copies only unmasked originals; masked bytes come from the model.
        decoded = torch.where(mask, decoded, ids)
        return (decoded, stats) if return_stats else decoded

    # ---------------------------------------------------------------- #
    # Training loss
    # ---------------------------------------------------------------- #
    def compute_loss(
        self,
        x0_emb: torch.Tensor,
        concept: torch.Tensor | None = None,
        *,
        confidence_weight: float = 0.1,
        valid_mask: torch.Tensor | None = None,
        corrupted_emb: torch.Tensor | None = None,
        alignment_loss_weight: float = 0.0,
        alignment_gamma: float = 0.02,
        timestep_limit: int | None = None,
        return_details: bool = False,
        edit_mask: torch.Tensor | None = None,
        random_edit_mask_probability: float = 0.0,
        requested_byte_ids: torch.Tensor | None = None,
        source_emb: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """MSE + optional soft-DTW alignment + supervised confidence BCE.

        ``corrupted_emb`` supplies an edit-corrupted input while ``x0_emb`` is
        the clean target. MSE compares original positions; alignment matches
        real-byte embeddings monotonically. The default weight zero preserves
        the legacy objective. Soft-DTW cannot change output length by itself.
        Confidence targets are detached exact nearest-byte correctness. Masks
        exclude padding; absent a mask all positions remain valid. Optional
        edit masks retain the original context and supervise masked bytes only.
        Random masks are sampled only on valid bytes (with at least one per
        nonempty row). requested_byte_ids must be an actual edit request, not
        an unannounced source of test targets; it is visible only under the mask.
        """
        if x0_emb.ndim != 3 or x0_emb.shape[0] < 1 or x0_emb.shape[1] < 1 or x0_emb.shape[-1] != self.d_model:
            raise ValueError("x0_emb must have shape (positive B, positive L, d_model)")
        if not math.isfinite(confidence_weight) or confidence_weight < 0:
            raise ValueError("confidence_weight must be finite and non-negative")
        if not math.isfinite(alignment_loss_weight) or alignment_loss_weight < 0:
            raise ValueError("alignment_loss_weight must be finite and non-negative")
        if not math.isfinite(alignment_gamma) or alignment_gamma <= 0:
            raise ValueError("alignment_gamma must be finite and positive")
        if timestep_limit is None:
            timestep_limit = self.diffusion_timesteps
        if not isinstance(timestep_limit, int) or isinstance(timestep_limit, bool) or not 1 <= timestep_limit <= self.diffusion_timesteps:
            raise ValueError("timestep_limit must lie within the diffusion schedule")
        if corrupted_emb is not None and (corrupted_emb.shape != x0_emb.shape or corrupted_emb.device != x0_emb.device or corrupted_emb.dtype != x0_emb.dtype):
            raise ValueError("corrupted_emb must match the target shape, device and dtype")
        if source_emb is not None and (not torch.is_tensor(source_emb) or source_emb.shape != x0_emb.shape or source_emb.device != x0_emb.device or source_emb.dtype != x0_emb.dtype or not bool(torch.isfinite(source_emb).all())):
            raise ValueError("source_emb must be finite and match the target shape, device and dtype")
        if not math.isfinite(random_edit_mask_probability) or not 0 <= random_edit_mask_probability <= 1:
            raise ValueError("random_edit_mask_probability must be in [0, 1]")
        if edit_mask is not None and random_edit_mask_probability:
            raise ValueError("pass either edit_mask or random_edit_mask_probability, not both")
        b = x0_emb.shape[0]
        device = x0_emb.device
        if valid_mask is None:
            mask = torch.ones(x0_emb.shape[:2], device=device, dtype=torch.bool)
        else:
            mask = valid_mask.to(device=device, dtype=torch.bool)
            if mask.shape != x0_emb.shape[:2] or not bool(mask.any()):
                raise ValueError("valid_mask must match (B, L) and contain a valid byte")
        training_edit_mask: torch.Tensor | None = None
        if edit_mask is not None:
            training_edit_mask = binary_edit_mask(edit_mask, tuple(x0_emb.shape[:2]), device)
            if bool((training_edit_mask & ~mask).any()):
                raise ValueError("edit_mask cannot include invalid/padded positions")
        elif random_edit_mask_probability:
            training_edit_mask = (torch.rand(x0_emb.shape[:2], device=device) < random_edit_mask_probability) & mask
            # Avoid a silently empty objective on short sequences.
            for row in range(b):
                valid = mask[row].nonzero(as_tuple=False).flatten()
                if valid.numel() and not bool(training_edit_mask[row].any()):
                    choice = torch.randint(valid.numel(), (), device=device)
                    training_edit_mask[row, valid[choice]] = True
        if training_edit_mask is not None:
            if not bool(training_edit_mask.any()):
                raise ValueError("masked training requires at least one valid edit position")
            if alignment_loss_weight:
                raise ValueError("soft-DTW alignment is not defined for isolated masked edits")
            mask = training_edit_mask
        elif requested_byte_ids is not None or source_emb is not None:
            raise ValueError("requested_byte_ids and source_emb require masked training")
        t_int = torch.randint(0, timestep_limit, (b,), device=device)
        t_float = t_int.float() / self.diffusion_timesteps
        xt_emb = self.q_sample(x0_emb if corrupted_emb is None else corrupted_emb, t_int)
        core_kwargs = {}
        if training_edit_mask is not None:
            source = x0_emb if source_emb is None else source_emb
            xt_emb = torch.where(training_edit_mask.unsqueeze(-1), xt_emb, source)
            core_kwargs = {"edit_mask": training_edit_mask, "requested_byte_ids": requested_byte_ids}
        if confidence_weight > 0 or return_details:
            pred_emb, confidence_logits = self.core(xt_emb, t_float, concept, return_confidence=True, **core_kwargs)
        else:
            pred_emb = self.core(xt_emb, t_float, concept, **core_kwargs)
            confidence_logits = None
        assert isinstance(pred_emb, torch.Tensor)
        per_byte_mse = (pred_emb - x0_emb).square().mean(dim=-1)
        mse = per_byte_mse[mask].mean()
        confidence_loss = mse.new_zeros(())
        if confidence_logits is not None:
            with torch.no_grad():
                predicted_bytes = self.core.embed_to_bytes(pred_emb.detach())
                true_bytes = self.core.embed_to_bytes(x0_emb.detach())
                correct = (predicted_bytes == true_bytes).to(confidence_logits.dtype)
            confidence_loss = F.binary_cross_entropy_with_logits(confidence_logits[mask], correct[mask])
        alignment = mse.new_zeros(())
        if alignment_loss_weight:
            alignment = self.core.alignment_loss(pred_emb, x0_emb, prediction_mask=mask, target_mask=mask,
                                                 gamma=alignment_gamma)
        loss = mse + alignment_loss_weight * alignment + confidence_weight * confidence_loss
        if return_details:
            details = {"denoising_mse": mse, "alignment_loss": alignment, "confidence_bce": confidence_loss}
            if training_edit_mask is not None:
                details["masked_positions"] = training_edit_mask.sum().detach()
            return loss, details
        return loss

    # ---------------------------------------------------------------- #
    # Bytes <-> embeddings
    # ---------------------------------------------------------------- #
    def bytes_to_emb(self, byte_ids: torch.Tensor) -> torch.Tensor:
        return self.core.byte_emb(byte_ids)  # (B, L, d)

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        seq_len: int,
        concept: torch.Tensor | None = None,
        guidance_scale: float = 2.0,
        *,
        confidence_threshold: float | None = None,
        min_steps: int = 1,
        confidence_reduction: str = "min",
        return_stats: bool = False,
        adaptive_config: Mapping[str, Any] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, SamplingStats]:
        """Generate raw byte IDs, optionally reporting adaptive sampling steps."""
        shape = (n_samples, seq_len, self.d_model)
        result = self.p_sample_loop(
            shape, concept, guidance_scale, confidence_threshold=confidence_threshold,
            min_steps=min_steps, confidence_reduction=confidence_reduction,
            return_stats=return_stats, adaptive_config=adaptive_config,
        )
        if return_stats:
            emb, stats = result
            return self.core.embed_to_bytes(emb), stats
        assert isinstance(result, torch.Tensor)
        return self.core.embed_to_bytes(result)

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
        if isinstance(example_bytes, (bytes, bytearray, list, tuple)):
            try:
                raw = bytes(example_bytes)
            except (ValueError, TypeError) as exc:
                raise ValueError("concept examples must contain raw bytes") from exc
            ids = torch.tensor(list(raw), dtype=torch.long, device=self.betas.device)
        elif torch.is_tensor(example_bytes) and example_bytes.ndim == 1 and example_bytes.dtype in (torch.uint8, torch.int32, torch.int64):
            ids = example_bytes.to(device=self.betas.device, dtype=torch.long)
            if bool(((ids < 0) | (ids > 255)).any()):
                raise ValueError("concept examples must contain raw bytes")
            raw = bytes(ids.cpu().tolist())
        else:
            raise TypeError("concept example must be one-dimensional raw bytes")
        if not raw:
            raise ValueError("concept example must not be empty")
        # Validate before deriving or writing any vector.
        if self.bank.heldout_guard is not None:
            self.bank.heldout_guard.validate(raw, metadata={"name": name})
        key = self.encode_concept(ids)
        self.bank.write(name, key, key, raw_input=raw)

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
        if self.bank.heldout_guard is not None:
            self.bank.audit_holdouts()
        os.makedirs(dirpath, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(dirpath, "model.pt"))
        self.bank.save(os.path.join(dirpath, "concept_bank.pt"))

    def load(self, dirpath: str, map_location="cpu"):
        # Validate separately stored concept metadata before mutating weights.
        self.bank.preflight_load(os.path.join(dirpath, "concept_bank.pt"))
        self._loading_audited_model = True
        try:
            self.load_state_dict(torch.load(os.path.join(dirpath, "model.pt"),
                                            map_location=map_location, weights_only=True))
        finally:
            self._loading_audited_model = False
        self.bank.load(os.path.join(dirpath, "concept_bank.pt"))
        self.to(self.device)
        return self
