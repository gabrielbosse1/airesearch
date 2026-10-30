"""Latent thought and trainable, explicitly calibrated confidence utilities.

Confidence is a learned estimate, not a guarantee.  Temperature fitting must use
held-out labels; merely training a head does not make it calibrated.  No token
is emitted by either component.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceHead(nn.Module):
    """Predict the log-odds that a prediction is correct.

    ``forward`` returns raw logits for binary cross entropy. ``probability``
    applies a fitted scalar temperature when available.  Temperature fitting is
    optional and does not certify calibration under distribution shift.
    """

    SCOPES = {"generic": 0, "next_byte": 1, "thought": 2,
              "diffusion_denoising": 3, "diffusion_sampling": 4}

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("temperature", torch.ones(()))
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))
        self.register_buffer("calibration_scope", torch.zeros((), dtype=torch.long))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    def probability(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits / self.temperature.clamp_min(0.05))

    def train(self, mode: bool = True) -> ConfidenceHead:
        # Any subsequent optimization can invalidate a previous calibration.
        if mode:
            self.calibrated.fill_(False)
            self.calibration_scope.zero_()
        return super().train(mode)

    def warn_if_uncalibrated(self, expected_scope: str | None = None) -> None:
        if expected_scope is not None and expected_scope not in self.SCOPES:
            raise ValueError("unknown confidence calibration scope")
        if not bool(self.calibrated.item()):
            warnings.warn(
                "Adaptive halting is using an uncalibrated confidence head; "
                "validate and calibrate on held-out data before relying on it.",
                RuntimeWarning,
                stacklevel=3,
            )
        elif expected_scope is not None and int(self.calibration_scope.item()) != self.SCOPES[expected_scope]:
            warnings.warn(
                f"Confidence was calibrated for a different task, not {expected_scope}; "
                "validate and recalibrate for this halting use.", RuntimeWarning, stacklevel=3,
            )

    def calibrate(
        self, logits: torch.Tensor, targets: torch.Tensor, *, max_iter: int = 50,
        scope: str = "generic",
    ) -> float:
        """Fit a temperature by held-out binary negative log likelihood.

        Inputs are detached: this adjusts only the temperature buffer, not the
        model. Both outcomes must occur in the validation set. Call again after
        training and whenever the deployment distribution changes.
        """
        logits = logits.detach().reshape(-1).to(device=self.temperature.device, dtype=torch.float64)
        targets = targets.detach().reshape(-1).to(device=self.temperature.device, dtype=torch.float64)
        if logits.numel() != targets.numel() or logits.numel() < 2:
            raise ValueError("calibration requires equally sized, non-empty held-out logits and targets")
        if not bool(torch.isfinite(logits).all() and torch.isfinite(targets).all()):
            raise ValueError("calibration inputs must be finite")
        if bool(((targets < 0) | (targets > 1)).any()):
            raise ValueError("confidence targets must be in [0, 1]")
        if not bool((targets == 0).any() and (targets == 1).any()):
            raise ValueError("calibration requires both correct and incorrect examples")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        if scope not in self.SCOPES:
            raise ValueError(f"scope must be one of {tuple(self.SCOPES)}")

        with torch.enable_grad():
            log_temp = torch.zeros((), device=logits.device, dtype=logits.dtype, requires_grad=True)
            optimizer = torch.optim.LBFGS([log_temp], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe")

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                temp = log_temp.clamp(-3.0, 3.0).exp()
                loss = F.binary_cross_entropy_with_logits(logits / temp, targets)
                loss.backward()
                return loss

            optimizer.step(closure)
            result = float(log_temp.detach().clamp(-3.0, 3.0).exp().item())
        self.temperature.fill_(result)
        self.calibration_scope.fill_(self.SCOPES[scope])
        self.calibrated.fill_(True)
        return result


class ThoughtStreamLayer(nn.Module):
    """A per-position recurrent thought stream coupled to a causal byte stream.

    No position reads a future position here. The main transformer supplies
    causal contextualization; this stream maintains a separate latent state
    across transformer layers and feeds it back through a zero-initialized gate.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_norm = nn.LayerNorm(d_model)
        self.input_proj = nn.Linear(d_model, d_model)
        self.cell = nn.GRUCell(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.output_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self, hidden: torch.Tensor, thought: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.shape != thought.shape or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden and thought must have matching (..., d_model) shapes")
        shape = hidden.shape
        inp = F.silu(self.input_proj(self.input_norm(hidden))).reshape(-1, self.d_model)
        next_thought = self.cell(inp, thought.reshape(-1, self.d_model)).reshape(shape)
        hidden = hidden + torch.tanh(self.output_gate) * self.output_proj(next_thought)
        return hidden, next_thought
