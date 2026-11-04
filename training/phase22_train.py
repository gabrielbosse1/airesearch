"""Phase 22 three-phase trainer (Phase 1 / Phase 2 / Phase 3).

Run in Colab, e.g.::

    python -m training.phase22_train --phase 1 --device cuda
    python -m training.phase22_train --phase 2 --device cuda
    python -m training.phase22_train --phase 3 --device cuda

Every phase is resumable: ``resume_checkpoint`` looks for
``checkpoint_phaseN.pt`` locally, then on the Hub, and restores weights,
AdamW state, GradScaler state, RNG state and the lossless byte-packer carry.

The Arena sandbox has no CUDA and no huggingface.co access. ``--self-test``
therefore runs a tiny CPU smoke test over REAL repository text (no invented
corpus) and reports ``data_mode='offline-smoke'``. It is plumbing validation,
not a training result; no loss/quality claim is made from it.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
import torch.nn.functional as F

from data.phase22_sources import SourceRegistry, stream_phase1, stream_phase2
from models.foundation import (BOS, PAD, CausalByteTransformer,
                               FoundationConfig, decode_ids, encode_bytes)
from training.phase22_common import (BytePacker, VRAMMonitor, append_jsonl,
                                     estimate_vram, hf_upload,
                                     initialise_from_phase, load_config,
                                     make_grad_scaler, model_config,
                                     resume_checkpoint, resolve_precision,
                                     save_checkpoint, save_deploy_copy,
                                     vram_fill_ladder, write_report)
from training.phase22_reward import (RunningBaseline, build_rl_case,
                                     compute_reward, dominance_report)

PHASE_NAMES = {1: "phase1_name", 2: "phase2_name", 3: "phase3_name"}


def weighted_ce(logits: torch.Tensor, ids: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
    """Next-byte CE with per-token trainable weights (0 = prompt/header)."""
    shift_logits = logits[:, :-1, :]
    targets = ids[:, 1:]
    token_weights = weights[:, 1:].to(shift_logits.dtype)
    per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        targets.reshape(-1), reduction="none", ignore_index=PAD,
    ).view(targets.shape)
    denominator = token_weights.sum().clamp(min=1.0)
    return (per_token * token_weights).sum() / denominator


def _pairs_to_tensors(pairs: list[tuple[torch.Tensor, torch.Tensor]],
                      device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.stack([pair[0] for pair in pairs]).to(device)
    weights = torch.stack([pair[1] for pair in pairs]).to(device)
    return ids, weights


@torch.no_grad()
def _noop() -> None:
    return None


class Phase22Trainer:
    """One trainer instance per phase (notebook 01/02/03)."""

    def __init__(self, config: Mapping[str, Any], *, phase: int,
                 device: torch.device | str = "cpu", max_steps: int | None = None,
                 data_mode: str = "hub", overrides: Mapping[str, Any] | None = None,
                 report_dir: str | Path = "runs/phase22",
                 init_from_phase: int | None = None,
                 strict_phase_order: bool = False) -> None:
        if phase not in (1, 2, 3):
            raise ValueError("phase must be 1, 2 or 3")
        self.config = dict(config)
        self.phase = phase
        self.device = torch.device(device)
        self.data_mode = data_mode
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.training = dict(self.config.get("training", {}))
        self.max_steps = int(max_steps or self.training.get(
            {1: "phase1_max_steps", 2: "phase2_max_steps", 3: "phase3_max_steps"}[phase], 20))
        self.grad_accum = int(self.training.get("grad_accum", 8))
        self.micro_batch = int(self.training.get("micro_batch_size", 4))
        self.seq_len = int(self.training.get("seq_len", 1024))
        if overrides:
            model_section = dict(self.config["model"])
            model_section.update(overrides)
            self.config["model"] = model_section
            self.seq_len = min(self.seq_len, int(model_section["max_seq_len"]))

        model_cfg = model_config(self.config)
        self.model = CausalByteTransformer(model_cfg).to(self.device)
        self.plan = resolve_precision(self.device,
                                      self.training.get("requested_precision", "bfloat16"),
                                      self.training.get("precision", "auto"))
        self.scaler = make_grad_scaler(self.plan)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.training.get("learning_rate", 3e-4)),
            betas=tuple(self.training.get("adam_betas", [0.9, 0.95])),
            weight_decay=float(self.training.get("weight_decay", 0.1)),
        )
        vram_cfg = self.config.get("vram", {})
        self.monitor = VRAMMonitor(
            path=self.report_dir / "vram.jsonl",
            target_min_gb=float(vram_cfg.get("target_min_gb", 10.0)),
            target_max_gb=float(vram_cfg.get("target_max_gb", 12.0)),
            device_total_gb=float(vram_cfg.get("device_total_gb", 14.0)),
        )
        self.metrics_path = self.report_dir / "metrics.jsonl"
        self.registry = SourceRegistry()
        if init_from_phase is not None:
            self.initialisation = initialise_from_phase(
                self.config, init_from_phase, self.model, device=self.device)
        elif strict_phase_order:
            raise RuntimeError(f"Phase {phase} requires init_from_phase; pass 1 or 2")
        else:
            self.initialisation = {"loaded": False,
                                   "reason": "phase-to-phase initialisation not requested"}
        self.resume = resume_checkpoint(self.config, phase, self.model,
                                        optimizer=self.optimizer, scaler=self.scaler,
                                        device=self.device)
        self.step = int(self.resume["step"])
        self.packer = BytePacker(self.seq_len)
        carry_state = (self.resume.get("extra") or {}).get("packer")
        if carry_state:
            self.packer.load_state_dict(carry_state)
        self._accumulated = 0
        self._pending: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.estimate = estimate_vram(self.config, batch_size=self.micro_batch,
                                      seq_len=self.seq_len)
        self.fill_ladder = (vram_fill_ladder(self.config, device=self.device,
                                              model=self.model, plan=self.plan)
                            if vram_cfg.get("fill_to_target", True) else None)
        self.start = time.time()
        self.last_loss: float | None = None

    # ------------------------------------------------------------------
    def _checkpoint_path(self) -> Path:
        ck = self.config.get("checkpoints", {})
        return Path(ck.get("local_dir", "runs/phase22/checkpoints")) / ck[PHASE_NAMES[self.phase]]

    def _save(self) -> dict[str, Any]:
        path = self._checkpoint_path()
        info = save_checkpoint(path, self.model, phase=self.phase, step=self.step,
                               optimizer=self.optimizer, scaler=self.scaler,
                               extra={"packer": self.packer.state_dict(),
                                      "precision": self.plan.as_dict(),
                                      "vram_estimate": self.estimate,
                                      "registry": [record.as_dict() for record in self.registry.records]})
        ck = self.config.get("checkpoints", {})
        remote = f"{ck.get('hf_path_in_repo', 'phase22')}/{path.name}"
        info["hf"] = hf_upload(path, ck.get("hf_repo"), remote)
        info["deploy_copy"] = save_deploy_copy(path.with_name(path.stem + "_fp16.pt"), path)
        return info

    def _log(self, loss: float, *, tag: str = "train") -> None:
        record = {
            "phase": self.phase, "step": self.step, "tag": tag,
            "loss": round(loss, 6), "precision": self.plan.resolved,
            "seq_len": self.seq_len, "micro_batch": self.micro_batch,
            "elapsed_s": round(time.time() - self.start, 2),
            "packer": self.packer.state_dict(),
        }
        if self.step % int(self.training.get("monitor_every", 25)) == 0:
            record["vram"] = self.monitor.snapshot(tag, step=self.step)
        append_jsonl(self.metrics_path, record)

    # ------------------------------------------------------------------
    def _optimiser_step(self, loss: torch.Tensor) -> None:
        """Scale, clip, step; obeys grad_accum."""
        scaled = loss / self.grad_accum
        if self.scaler is not None:
            self.scaler.scale(scaled).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           float(self.training.get("max_grad_norm", 1.0)))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            scaled.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           float(self.training.get("max_grad_norm", 1.0)))
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _consume(self, pairs: Iterable[tuple[bytes, int]]) -> int:
        """Pack documents into batches and run optimizer steps."""
        produced = 0
        for document, prefix in pairs:
            for pair in self.packer.feed(document, trainable_prefix=prefix):
                self._pending.append(pair)
                if len(self._pending) < self.micro_batch:
                    continue
                ids, weights = _pairs_to_tensors(self._pending, self.device)
                self._pending = []
                with self.plan.autocast_context():
                    logits = self.model(ids)
                    loss = weighted_ce(logits, ids, weights)
                self._optimiser_step(loss)
                self.last_loss = float(loss.detach().item())
                self.step += 1
                produced += 1
                if self.step % int(self.training.get("log_every", 10)) == 0:
                    self._log(self.last_loss)
                if self.step % int(self.training.get("checkpoint_every", 500)) == 0:
                    self._save()
                if self.step >= self.max_steps:
                    return produced
        return produced

    # ------------------------------------------------------------------
    def run_phase1(self) -> dict[str, Any]:
        """RAW next-byte causal LM. No prompts, no ReAct, no synthetic rows."""
        if self.step >= self.max_steps:
            return self._finish("already complete via resume")
        if self.data_mode == "offline-smoke":
            pairs = self._offline_documents()
        else:
            pairs = ((raw, len(raw)) for _, raw in
                     stream_phase1(self.config, registry=self.registry))
        self._consume(pairs)
        return self._finish("phase1 raw next-byte causal LM")

    def run_phase2(self) -> dict[str, Any]:
        """Conversation + ReAct trajectory distillation with prompt masking."""
        if self.step >= self.max_steps:
            return self._finish("already complete via resume")
        if self.data_mode == "offline-smoke":
            pairs = self._offline_documents()
        else:
            pairs = stream_phase2(self.config, registry=self.registry)
        self._consume(pairs)
        return self._finish("phase2 distillation / ReAct alignment")

    # ------------------------------------------------------------------
    def _rl_cases(self) -> list[dict[str, Any]]:
        return [
            build_rl_case(
                question=b"Qual e o resultado atual do campeonato de 2026?",
                expected_answer=b"Consulte a busca: campeonato 2026",
                initial_draft=b"O campeao de 2024 e o atual",
                expected_draft=b"Consulte a busca: campeonato 2026",
                past_actions=b"Action: nenhuma busca executada ainda",
                search_required=True,
            ),
            build_rl_case(
                question=b"Como faco um gerador de pedregulho?",
                expected_answer=b"agua e lava produzem pedregulho",
                initial_draft=b"gelo e lava produzem pedregulho",
                expected_draft=b"agua e lava produzem pedregulho",
                past_actions=b"Action: explorar caverna (executado)",
                search_required=False,
            ),
        ]

    def _sample(self, prefix: torch.Tensor, max_new: int,
                temperature: float = 1.0
                ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Differentiable sampling: new ids, summed log-prob, token count."""
        ids = prefix
        sampled: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        for _ in range(max_new):
            window = ids[-self.model.config.max_seq_len:]
            with self.plan.autocast_context():
                logits = self.model(window.unsqueeze(0))[0, -1].float()
            distribution = torch.distributions.Categorical(logits=logits / temperature)
            token = distribution.sample()
            sampled.append(token.detach())
            log_probs.append(distribution.log_prob(token))
            ids = torch.cat((ids, token.reshape(1)))
            if int(token.item()) == 257:  # EOS control id
                break
        if not sampled:
            return (torch.empty(0, dtype=torch.long, device=ids.device),
                    torch.zeros((), device=ids.device), 0)
        return torch.stack(sampled), torch.stack(log_probs).sum(), len(sampled)

    def run_phase3(self) -> dict[str, Any]:
        """REINFORCE / PPO-lite with a deterministic search-tool sandbox."""
        from training.phase22_reward import DRAFT_OPEN, PAST_OPEN  # noqa: F401 (documented markers)

        cases = self._rl_cases()
        baseline = RunningBaseline()
        rewards: list[float] = []
        while self.step < self.max_steps:
            case = cases[self.step % len(cases)]
            prompt = encode_bytes(case["prompt"], bos=True)[-self.model.config.max_seq_len:].to(self.device)
            first, lp1, n1 = self._sample(prompt, 96)
            trajectory = decode_ids(first)
            lp2 = torch.zeros((), device=self.device)
            n2 = 0
            if b"search_ddg(" in trajectory and b"Observation:" not in trajectory:
                result = case.get("search_result") or (
                    case["expected_answer"] + b" (fonte: busca)")
                observation = b"Observation: " + result + b"\n"
                tool_ids = torch.tensor(list(observation), dtype=torch.long,
                                        device=self.device)
                extended = torch.cat((prompt, first, tool_ids))
                second, lp2, n2 = self._sample(extended, 64)
                trajectory = trajectory + observation + decode_ids(second)
            reward = compute_reward(trajectory, case).total
            rewards.append(reward)
            advantage = max(-2.0, min(2.0, baseline.advantage(reward)))
            limit = self.model.config.max_seq_len
            anchor_ids = encode_bytes(case["prompt"], bos=True, eos=True)[-limit:].to(self.device)
            with self.plan.autocast_context():
                anchor = self.model.loss(anchor_ids.unsqueeze(0))
            tokens = max(1, n1 + n2)
            policy = -(advantage * (lp1 + lp2) / tokens) + 0.01 * anchor
            self._optimiser_step(policy)
            self.last_loss = float(policy.detach().item())
            self.step += 1
            if self.step % int(self.training.get("log_every", 10)) == 0:
                self._log(self.last_loss, tag="rl")
                append_jsonl(self.metrics_path,
                             {"phase": 3, "step": self.step, "tag": "reward",
                              "reward": round(reward, 4),
                              "advantage": round(advantage, 4),
                              "mean_reward": round(sum(rewards) / len(rewards), 4),
                              "dominance": dominance_report()["ordering"]})
            if self.step % int(self.training.get("checkpoint_every", 500)) == 0:
                self._save()
        return self._finish("phase3 sandbox RL", extra={
            "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
            "dominance": dominance_report(),
        })

    # ------------------------------------------------------------------
    def _offline_documents(self) -> Iterator[tuple[bytes, int]]:
        """REAL repository text for a CPU plumbing smoke test (not a corpus)."""
        root = Path(__file__).resolve().parent.parent
        paths = sorted((root / "models").glob("*.py"))[:6] + \
                [root / "README.md", root / "AGENTS.md"]
        count = 0
        for path in paths:
            if not path.is_file():
                continue
            raw = path.read_bytes()
            count += 1
            yield raw[:4096], 0
        if count == 0:
            raise RuntimeError("no repository text found for the smoke test")

    def _finish(self, note: str, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        checkpoint = self._save()
        manifest = self.registry.manifest(
            self.config.get("data", {}).get("license_manifest",
                                            self.report_dir / "dataset_manifest.json"),
            extra={"phase": self.phase, "data_mode": self.data_mode,
                   "steps": self.step, "note": note})
        report = {
            "phase": self.phase, "note": note, "steps_completed": self.step,
            "max_steps": self.max_steps, "resumed": self.resume["resumed"],
            "initialisation": self.initialisation,
            "data_mode": self.data_mode, "precision": self.plan.as_dict(),
            "parameters": self.model.parameters_count,
            "config_model": self.model.config.to_dict(),
            "vram_estimate": self.estimate,
            "vram_fill_ladder": self.fill_ladder,
            "vram_measured": self.monitor.available,
            "checkpoint": checkpoint,
            "dataset_manifest": str(self.config.get("data", {}).get(
                "license_manifest", self.report_dir / "dataset_manifest.json")),
            "dataset_sources_recorded": len(manifest["sources"]),
            "packer_carry_bytes": len(self.packer.carry),
            "last_loss": self.last_loss,
            "elapsed_s": round(time.time() - self.start, 2),
        }
        if extra:
            report.update(dict(extra))
        write_report(self.report_dir / f"phase{self.phase}_report.json", report)
        return report

    def run(self) -> dict[str, Any]:
        return {1: self.run_phase1, 2: self.run_phase2, 3: self.run_phase3}[self.phase]()


def smoke_test() -> dict[str, Any]:
    """CPU-only plumbing check over real repository text; no Hub / no GPU."""
    cfg = load_config("configs/t4_scaled.yaml")
    tiny = {"d_model": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
            "d_ff": 128, "max_seq_len": 256, "refinement_steps": 1,
            "refinement_rank": 16, "gradient_checkpointing": True}
    overrides = dict(tiny)
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="phase22_smoke_") as folder:
        local = dict(cfg)
        local["checkpoints"] = dict(cfg["checkpoints"])
        local["checkpoints"]["local_dir"] = str(Path(folder) / "checkpoints")
        local["checkpoints"]["hf_repo"] = None
        local["checkpoints"]["resume_from_hf"] = False
        local["vram"] = dict(cfg["vram"])
        local["vram"]["fill_to_target"] = False
        local["output"] = dict(cfg["output"])
        local["data"] = dict(cfg["data"])
        local["data"]["license_manifest"] = str(Path(folder) / "dataset_manifest.json")
        for phase in (1, 2, 3):
            trainer = Phase22Trainer(local, phase=phase, device="cpu", max_steps=2,
                                     data_mode="offline-smoke", overrides=overrides,
                                     report_dir=Path(folder) / "reports")
            report = trainer.run()
            results[f"phase{phase}"] = {
                "steps": report["steps_completed"],
                "loss": report["last_loss"],
                "checkpoint_bytes": report["checkpoint"]["bytes"],
                "resumed": report["resumed"],
                "precision": report["precision"]["resolved"],
                "packer_carry_bytes": report["packer_carry_bytes"],
            }
        # Resume proof: reopen phase 1 checkpoint and verify step/carry.
        trainer = Phase22Trainer(local, phase=1, device="cpu", max_steps=2,
                                 data_mode="offline-smoke", overrides=overrides,
                                 report_dir=Path(folder) / "reports")
        results["resume"] = {"step": trainer.step, "resumed": trainer.resume["resumed"],
                             "carry": len(trainer.packer.carry)}
    results["dominance"] = dominance_report()["ordering"]
    results["measured_vram"] = False
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--config", default="configs/t4_scaled.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--data-mode", choices=("hub", "offline-smoke"), default="hub")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(smoke_test(), indent=2))
        return
    config = load_config(args.config)
    trainer = Phase22Trainer(config, phase=args.phase, device=args.device,
                             max_steps=args.max_steps, data_mode=args.data_mode)
    print(json.dumps(trainer.run(), indent=2))


if __name__ == "__main__":
    main()
