"""Train style-conditioned raw-byte CHAT replies with differentiable diversity.

The response corpus is deliberately small and synthetic. Selection uses
held-out prompt validation NLL/teacher-forced exactness, never the final prompt.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from data.diverse_dialogue import response_bytes, training_prompts
from data.phase12_eval_sets import DIVERSITY_TEST_PROMPT, DIVERSITY_VALIDATION_PROMPTS
from models.diverse_response import DiverseByteDecoder, distribution_diversity_loss
from models.holdout import HoldoutGuard, HoldoutViolation
from training.train_confidence import load_generator
from training.train_conversation import sha256
from training.train_replay_intents import phase12_guard


@torch.no_grad()
def _features(generator, prompts):
    framed = [b"U:" + prompt for prompt in prompts]
    lengths = torch.tensor([len(prompt) for prompt in framed], dtype=torch.long)
    ids = torch.zeros((len(framed), int(lengths.max())), dtype=torch.long)
    for i, prompt in enumerate(framed):
        ids[i, :len(prompt)] = torch.tensor(list(prompt), dtype=torch.long)
    return generator.conversation_features(ids, lengths, focus_start=torch.full_like(lengths, 2)).detach()


def _target_tensors(responses):
    max_len = max(len(response) + 1 for response in responses)
    previous = torch.full((len(responses), max_len), DiverseByteDecoder.START, dtype=torch.long)
    targets = torch.full((len(responses), max_len), DiverseByteDecoder.IGNORE, dtype=torch.long)
    for i, response in enumerate(responses):
        ids = torch.tensor(list(response), dtype=torch.long)
        targets[i, :len(response)] = ids; targets[i, len(response)] = DiverseByteDecoder.EOS
        previous[i, 1:len(response) + 1] = ids
    return previous, targets


@torch.no_grad()
def _validate(decoder, features, previous, targets):
    styles = torch.arange(decoder.num_styles).repeat(len(features))
    context = features.repeat_interleave(decoder.num_styles, dim=0)
    p = previous.repeat(len(features), 1)
    t = targets.repeat(len(features), 1)
    logits = decoder(context, styles, p)
    nll = F.cross_entropy(logits.reshape(-1, 257), t.reshape(-1), ignore_index=decoder.IGNORE)
    valid = t != decoder.IGNORE
    correct = ((logits.argmax(-1) == t) | ~valid).all(dim=1)
    return {"nll": float(nll), "teacher_sequence_exact": int(correct.sum()), "sequences": int(correct.numel()),
            "teacher_sequence_accuracy": float(correct.float().mean())}


def train(args):
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); rng = random.Random(args.seed)
    if min(args.epochs, args.steps_per_epoch, args.prompts_per_batch, args.styles_per_prompt) < 1:
        raise ValueError("invalid training configuration")
    responses = response_bytes(); prompts = training_prompts()
    if not 2 <= args.styles_per_prompt <= len(responses):
        raise ValueError("styles per prompt must be at least two and no more than the response count")
    base_guard = phase12_guard(args)
    validation_guard = HoldoutGuard.from_records([{"input": p} for p in DIVERSITY_VALIDATION_PROMPTS] +
                                                  [{"input": DIVERSITY_TEST_PROMPT}])
    guard = HoldoutGuard(base_guard.contents | validation_guard.contents,
                         base_guard.held_out_hashes | validation_guard.held_out_hashes)
    accepted, blocked = [], 0
    # One refusal aborts rather than silently dropping a style or leaking it.
    for prompt in prompts:
        safe = True
        for response in responses:
            try:
                guard.validate(prompt, response, metadata={"split": "train"})
            except HoldoutViolation:
                blocked += 1; safe = False; break
        if safe:
            accepted.append(prompt)
    if len(accepted) < 4:
        raise RuntimeError("insufficient guarded response prompts")
    generator = load_generator(args.generator, "cpu", args.generator_heads).eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    train_features = _features(generator, accepted)
    validation_features = _features(generator, DIVERSITY_VALIDATION_PROMPTS)
    previous, targets = _target_tensors(responses)
    decoder = DiverseByteDecoder(generator.d_model, num_styles=len(responses), hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    best, rank, selected_epoch, history = None, None, 0, []
    print(f"preflight: {len(accepted)} prompts x {len(responses)} styles, {blocked} blocked; protected={len(guard.contents)}", flush=True)
    for epoch in range(args.epochs):
        decoder.train(); total_nll = total_div = 0.0
        for _ in range(args.steps_per_epoch):
            prompt_indices = [rng.randrange(len(accepted)) for _ in range(args.prompts_per_batch)]
            style_ids = [rng.sample(range(len(responses)), args.styles_per_prompt) for _ in prompt_indices]
            style_tensor = torch.tensor(style_ids, dtype=torch.long)
            contexts = train_features[prompt_indices].unsqueeze(1).expand(-1, args.styles_per_prompt, -1)
            flattened_styles = style_tensor.reshape(-1)
            inputs, output = previous[flattened_styles], targets[flattened_styles]
            logits = decoder(contexts.reshape(-1, generator.d_model), flattened_styles, inputs)
            nll = F.cross_entropy(logits.reshape(-1, 257), output.reshape(-1), ignore_index=decoder.IGNORE)
            shaped = logits.reshape(args.prompts_per_batch, args.styles_per_prompt, logits.shape[1], 257)
            valid = (output != decoder.IGNORE).reshape(args.prompts_per_batch, args.styles_per_prompt, -1)
            diversity = distribution_diversity_loss(shaped, valid)
            loss = nll + args.diversity_weight * diversity
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite response loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step(); total_nll += float(nll.detach()); total_div += float(diversity.detach())
        decoder.eval(); val = _validate(decoder, validation_features, previous, targets)
        current = (val["teacher_sequence_accuracy"], -val["nll"])
        if rank is None or current > rank:
            rank, selected_epoch = current, epoch + 1
            best = {key: value.detach().clone() for key, value in decoder.state_dict().items()}
        row = {"epoch": epoch + 1, "train_nll": total_nll / args.steps_per_epoch,
               "diversity_loss": total_div / args.steps_per_epoch, "validation": val}
        history.append(row)
        if epoch == 0 or (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"epoch {epoch+1}/{args.epochs}: val-nll={val['nll']:.4f} "
                  f"teacher-exact={val['teacher_sequence_exact']}/{val['sequences']} "
                  f"div={row['diversity_loss']:.4f}", flush=True)
    assert best is not None
    decoder.load_state_dict(best); decoder.response_trained.fill_(True); decoder.eval()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "diverse_response.pt"
    torch.save({"schema_version": 1, "generator_sha256": sha256(args.generator), "context_dim": generator.d_model,
                "num_styles": decoder.num_styles, "hidden_dim": decoder.hidden_dim,
                "byte_dim": decoder.byte_dim, "style_dim": decoder.style_dim, "state_dict": decoder.state_dict()}, path)
    report = {"configuration": vars(args), "generator_sha256": sha256(args.generator), "decoder_sha256": sha256(path),
              "guard_fingerprint": guard.fingerprint, "preflight": {"training_prompts": len(accepted),
              "styles": len(responses), "blocked_prompts": blocked, "protected_hashes": len(guard.contents)},
              "selected_epoch": selected_epoch, "selection_rule": "validation teacher-forced sequence accuracy, then validation NLL",
              "validation": _validate(decoder, validation_features, previous, targets),
              "decoder_parameters": sum(p.numel() for p in decoder.parameters()), "epochs": history,
              "limitations": "Synthetic fixed style corpus: surface variety is not evidence of semantic quality. Loss uses soft predicted distributions, not discrete sampled token IDs."}
    (out / "diverse_response_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generator", default="runs/phase12/generator.pt")
    p.add_argument("--generator-heads", type=int, default=4)
    p.add_argument("--pairs", default="data/real_edit_pairs.jsonl")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--output-dir", default="runs/phase12")
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--steps-per-epoch", type=int, default=25)
    p.add_argument("--prompts-per-batch", type=int, default=8)
    p.add_argument("--styles-per-prompt", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--diversity-weight", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.0015)
    p.add_argument("--seed", type=int, default=121230)
    p.add_argument("--threads", type=int, default=1)
    train(p.parse_args())


if __name__ == "__main__":
    main()
