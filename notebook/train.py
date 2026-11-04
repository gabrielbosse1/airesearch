"""Wall-clock bounded four-stage training with recoverable HF snapshots."""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from notebook.hf_upload import upload_checkpoint
from notebook.preprocess import batch, sample_mix
from notebook.tools import reward

# Stage 2 (index 1) is the Phase 19 change: 40% code, 40% Mineflayer player
# trajectories, 20% pt-br. Four 47-minute stages fit a free Colab session better
# than four 60-minute stages; the mix is independent of the wall-clock budget.
MIXES = [
    dict(pt=50, en=30, code=20),
    dict(code=40, mineflayer=40, pt=20),
    dict(react=40, search=30, code=30),
    dict(edit=50, pt=20, code=15, mineflayer=15),
]
FREEZE = [0, 6, 8, 8]
RL_CASES = [
    ('code', 'Request: Write a Python function to sort a list\nCode: ', None, None),
    ('mineflayer', 'Request: Mine the diamond ore at (50, 30, 75).\nThought: I need diamonds.\nAction: ', None, None),
    ('pt', 'Request: Olá, como você está?\nAnswer: ', None, None),
    ('edit', 'Request: Corrija: voce\nDraft: voce\n', 'voce', 'você'),
]
SNAPSHOT_VERSION = 2


def _stages(stages) -> tuple[int, ...]:
    selected = tuple(range(len(MIXES))) if stages is None else tuple(stages)
    if not selected or any(not isinstance(stage, int) or stage not in range(len(MIXES))
                           for stage in selected):
        raise ValueError('stages must be a non-empty subset of 0..3')
    if len(set(selected)) != len(selected):
        raise ValueError('stages must not contain duplicates')
    if tuple(sorted(selected)) != selected:
        raise ValueError('stages must be in ascending order')
    return selected


# Phase 21 recovery fix. A group with only a handful of unique rows must not be
# allowed to occupy 30-50% of a stage's gradient updates: at 32 sequences per
# optimizer step, four unique ``edit`` rows are replayed thousands of times per
# stage while the tied byte embedding/output matrix stays fully trainable.
MIN_DIVERSE_ROWS = 64
LANGUAGE_GROUPS = ('pt', 'en', 'code')
# Stages 3 and 4 (indices 2, 3) reserve this share of their updates for a
# rehearsal of the Stage 1 bilingual mixture. Without it, both stages train
# almost exclusively on code / tool templates and rewrite the shared byte
# embedding, which is what produced the reported language collapse.
REPLAY_FRACTION = 0.4


def balanced_weights(groups, weights, min_unique=MIN_DIVERSE_ROWS):
    """Scale each mixture weight by its unique-row diversity, then renormalise.

    ``diversity = min(1, unique_rows / min_unique)``. Stage 1 (large pt/en/code
    groups) and Stage 2 (521 Mineflayer rows) are essentially unchanged; a
    4-row ``edit`` group at 50% drops to ~6% (and ``stage_mixture`` reduces it
    further in Stage 4) instead of being replayed thousands of times per stage. Redistribution is proportional to ``w * diversity`` over
    ALL groups, so no single group can absorb the removed mass.
    """
    if not isinstance(weights, dict) or not weights:
        raise ValueError('weights must be a non-empty mapping')
    if not isinstance(min_unique, int) or min_unique <= 0:
        raise ValueError('min_unique must be a positive integer')
    required = [kind for kind, weight in weights.items()
                if isinstance(weight, (int, float)) and weight > 0 and groups.get(kind)]
    if not required:
        raise ValueError('at least one positively weighted non-empty group is required')
    diversity = {kind: min(1.0, len(set(groups[kind])) / min_unique) for kind in required}
    scaled = {kind: weights[kind] * diversity[kind] for kind in required}
    total = sum(scaled.values())
    if total <= 0:
        raise ValueError('all groups have zero diversity')
    original = sum(weights[kind] for kind in required)
    return {kind: (original * scaled.get(kind, 0.0) / total if kind in scaled else 0.0)
            for kind in weights}


def stage_mixture(groups, weights, stage=None):
    """The Phase 21 recovery fix: diversity-capped weights + language rehearsal.

    ``stage`` is 0-based; stages 3 and 4 keep 40% of their gradient updates on
    the Stage 1 language groups so the tied byte embedding is not rewritten by a
    handful of tool/edit templates. This is ONE curriculum fix, applied by
    ``draw``; it needs no architecture change and no extra GPU time.
    """
    capped = balanced_weights(groups, weights)
    if stage is None or stage < 2:
        return capped
    available = [kind for kind in LANGUAGE_GROUPS if groups.get(kind)]
    if not available:
        return capped
    total = sum(capped.values())
    result = {kind: value * (1.0 - REPLAY_FRACTION) for kind, value in capped.items()}
    reserve = total * REPLAY_FRACTION / len(available)
    for kind in available:
        result[kind] = result.get(kind, 0.0) + reserve
    return result


def draw(groups, weights, count, balanced=True, stage=None):
    """Sample a checked mixture. ``balanced=False`` reproduces the Phase 19 mix.

    Delegates validation to the shared helper so neither path can silently
    sample a missing or negatively weighted group.
    """
    active = stage_mixture(groups, weights, stage) if balanced else weights
    return sample_mix(groups, active, count, rng=random)

def check_groups(groups, stages=None):
    """Fail before training when a selected mixture has no usable rows."""
    selected = _stages(stages)
    missing = sorted({kind for stage in selected for kind in MIXES[stage] if not groups.get(kind)})
    if missing:
        hint = (' Generate Mineflayer rows with notebook.mineflayer_training_data.'
                if 'mineflayer' in missing else '')
        raise ValueError('Missing training groups: ' + ', '.join(missing) + '.' + hint +
                         ' Automatic dataset search found no usable rows; see data_cache/manifest.json '
                         'for every source attempted. Rerun the discovery cell or check Hub network '
                         'access. Synthetic data is not substituted for language or code rows.')


def snapshot(model, opt, scaler, stage, step, elapsed, cfg):
    """Capture the exact state needed to resume the current stage."""
    return {
        'version': SNAPSHOT_VERSION,
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'stage': stage,
        'step': step,
        'elapsed': elapsed,
        'config': asdict(cfg),
        'random': random.getstate(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _validate_resume(resume, cfg):
    """Reject incomplete/wrong snapshots before mutating model or RNG state."""
    if not isinstance(resume, dict):
        raise ValueError('Checkpoint must be a dictionary produced by notebook.train.snapshot')
    required = {'model', 'optimizer', 'scaler', 'stage', 'step', 'elapsed', 'config', 'random', 'torch_rng', 'cuda_rng'}
    missing = sorted(required - set(resume))
    if missing:
        raise ValueError('Checkpoint is incomplete; missing: ' + ', '.join(missing))
    if resume['config'] != asdict(cfg):
        raise ValueError('Checkpoint architecture mismatch')
    if not isinstance(resume['stage'], int) or resume['stage'] not in range(len(MIXES)):
        raise ValueError('Checkpoint has an invalid stage')
    if not isinstance(resume['step'], int) or resume['step'] < 0:
        raise ValueError('Checkpoint has an invalid step')
    if not isinstance(resume['elapsed'], (int, float)) or resume['elapsed'] < 0:
        raise ValueError('Checkpoint has an invalid elapsed time')


def policy_loss(model, device, max_new=48):
    """REINFORCE: sample token IDs, then replay them to preserve policy gradients."""
    kind, prompt, draft, target = random.choice(RL_CASES)
    prefix = [256] + list(prompt.encode('utf8'))
    # Prompts are currently much shorter than the context window, but preserving
    # BOS here prevents future edits from producing an overlong forward pass.
    prefix = [256] + prefix[1:][-(model.cfg.max_seq_len - 1):]
    ids = prefix[:]
    model.eval()
    with torch.no_grad():
        for _ in range(min(max_new, model.cfg.max_seq_len - len(ids))):
            logits = model(torch.tensor([ids], dtype=torch.long, device=device))[0, -1].float()
            logits[258:] = -float('inf')
            token = int(torch.multinomial(torch.softmax(logits, -1), 1))
            ids.append(token)
            if token == 257:
                break
    output = bytes(token for token in ids[len(prefix):] if token < 256).decode('utf8', 'replace')
    score = reward(kind, output, draft, target)
    model.train()
    if score == 0 or len(ids) == len(prefix):
        return None, score
    inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    logits = model(inp)[0, len(prefix) - 1:].float()
    logits = torch.cat([logits[:, :258], torch.full_like(logits[:, 258:], -float('inf'))], dim=-1)
    labels = torch.tensor(ids[len(prefix):], dtype=torch.long, device=device)
    return -score * F.log_softmax(logits, dim=-1).gather(1, labels[:, None]).mean(), score


def train(model, groups, token, repo_id, session_deadline, stage_seconds=47 * 60,
          upload_seconds=15 * 60, micro_batch=1, accumulation=32, resume=None, lr=3e-4,
          stages=None, legacy_mix=False, metrics_path=None):
    """Train selected stages (default all four), resuming from an HF snapshot.

    ``stages=[k]`` lets the notebook keep one cell per stage while sharing the
    in-memory model. A resume snapshot restores weights for any selected stage;
    optimizer, elapsed time and RNG are restored only if the snapshot is for the
    same stage. The notebook clears its consumed resume handle after a successful
    stage, preventing later cells from accidentally rewinding completed work.

    ``legacy_mix=True`` restores the unbalanced Phase 19 mixtures (forensics
    only). ``metrics_path`` appends one JSON line per optimizer step: the Colab
    run kept loss only inside a transient tqdm bar, which made the Phase 21 log
    autopsy impossible after the runtime ended.
    """
    selected = _stages(stages)
    check_groups(groups, selected)
    if not isinstance(stage_seconds, (int, float)) or stage_seconds <= 0:
        raise ValueError('stage_seconds must be positive')
    if not isinstance(upload_seconds, (int, float)) or upload_seconds <= 0:
        raise ValueError('upload_seconds must be positive')
    if not isinstance(micro_batch, int) or micro_batch <= 0:
        raise ValueError('micro_batch must be a positive integer')
    if not isinstance(accumulation, int) or accumulation <= 0:
        raise ValueError('accumulation must be a positive integer')
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError('lr must be positive')
    if not isinstance(session_deadline, (int, float)):
        raise ValueError('session_deadline must be a Unix timestamp')
    if not isinstance(legacy_mix, bool):
        raise ValueError('legacy_mix must be a bool')
    metrics_file = None
    if metrics_path is not None:
        metrics_file = Path(metrics_path)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        if not metrics_file.exists():
            metrics_file.write_text('', encoding='utf8')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if resume is not None:
        _validate_resume(resume, model.cfg)
        model.load_state_dict(resume['model'])
        random.setstate(resume['random'])
        torch.set_rng_state(resume['torch_rng'])
        if torch.cuda.is_available() and resume['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
        if resume['stage'] in selected:
            start_stage = resume['stage']
        else:
            start_stage = min(selected)
            print(f"Checkpoint is at stage {resume['stage']}; starting stage {start_stage} instead.")
    else:
        start_stage = min(selected)

    step = resume['step'] if resume is not None else 0
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    for stage in selected:
        if stage < start_stage:
            continue
        model.freeze(min(FREEZE[stage], len(model.blocks)))
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError(f'Stage {stage} has no trainable parameters after freezing')
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=.01)
        elapsed = resume['elapsed'] if resume is not None and stage == start_stage == resume['stage'] else 0.0
        if resume is not None and stage == start_stage == resume['stage']:
            optimizer.load_state_dict(resume['optimizer'])
            scaler.load_state_dict(resume['scaler'])

        last_upload = time.monotonic()
        bar = tqdm(total=stage_seconds, initial=min(elapsed, stage_seconds),
                   desc=f'Stage {stage + 1}', unit='s')
        tick = time.monotonic()
        model.train()
        while elapsed < stage_seconds and time.time() < session_deadline:
            optimizer.zero_grad(set_to_none=True)
            fraction = min(1.0, elapsed / stage_seconds)
            for parameter_group in optimizer.param_groups:
                parameter_group['lr'] = lr * 0.5 * (1 + math.cos(math.pi * fraction))
            losses = []
            for _ in range(accumulation):
                texts = draw(groups, MIXES[stage], micro_batch,
                            balanced=not legacy_mix, stage=stage)
                ids = batch(texts, model.cfg.max_seq_len).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == 'cuda'):
                    loss = model.loss(ids) / accumulation
                scaler.scale(loss).backward()
                losses.append(float(loss.detach()) * accumulation)
            if stage == 3 and step % 8 == 0:
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == 'cuda'):
                    rl_loss, _score = policy_loss(model, device)
                if rl_loss is not None:
                    scaler.scale(rl_loss * 0.1).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if metrics_file is not None:
                mean_loss = sum(losses) / len(losses)
                record = {
                    'source': 'notebook.train', 'stage': stage + 1,
                    'step': step, 'loss': mean_loss,
                    'lr': float(optimizer.param_groups[0]['lr']),
                    'grad_norm': grad_norm,
                    'nan': not math.isfinite(mean_loss) or not math.isfinite(grad_norm),
                    'mixture': 'legacy' if legacy_mix else 'balanced',
                    'time': time.time(),
                }
                with metrics_file.open('a', encoding='utf8') as handle:
                    handle.write(json.dumps(record) + '\n')

            now = time.monotonic()
            delta = now - tick
            tick = now
            elapsed += delta
            bar.update(min(delta, max(0, stage_seconds - bar.n)))
            bar.set_postfix(loss=f'{sum(losses) / len(losses):.3f}', step=step)
            if now - last_upload >= upload_seconds:
                upload_checkpoint(snapshot(model, optimizer, scaler, stage, step, elapsed, model.cfg),
                                  stage, step, token, repo_id)
                # Include upload latency in the wall-clock stage budget.
                last_upload = time.monotonic()

        bar.close()
        # Upload the current stage even on a wall-clock stop. At resume, an
        # exhausted stage immediately advances after restoring its optimizer.
        upload_checkpoint(snapshot(model, optimizer, scaler, stage, step, elapsed, model.cfg),
                          stage, step, token, repo_id)
        resume = None
        if time.time() >= session_deadline:
            print('Stopped at session deadline; resume from the latest uploaded checkpoint.')
            return False
    return True
