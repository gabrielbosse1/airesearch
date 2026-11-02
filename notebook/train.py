"""Wall-clock bounded four-stage training with recoverable HF snapshots."""
from __future__ import annotations

import math
import random
import time
from dataclasses import asdict

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


def draw(groups, weights, count):
    """Sample a checked mixture. Delegates validation to the shared helper."""
    return sample_mix(groups, weights, count, rng=random)


def check_groups(groups, stages=None):
    """Fail before training when a selected mixture has no usable rows."""
    selected = _stages(stages)
    missing = sorted({kind for stage in selected for kind in MIXES[stage] if not groups.get(kind)})
    if missing:
        hint = (' Generate Mineflayer rows with notebook.mineflayer_training_data.'
                if 'mineflayer' in missing else '')
        raise ValueError('Missing training groups: ' + ', '.join(missing) + '.' + hint +
                         ' Supply licensed UTF-8 text files; do not silently substitute synthetic data.')


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
          stages=None):
    """Train selected stages (default all four), resuming from an HF snapshot.

    ``stages=[k]`` lets the notebook keep one cell per stage while sharing the
    in-memory model. A resume snapshot restores weights for any selected stage;
    optimizer, elapsed time and RNG are restored only if the snapshot is for the
    same stage. The notebook clears its consumed resume handle after a successful
    stage, preventing later cells from accidentally rewinding completed work.
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
                ids = batch(draw(groups, MIXES[stage], micro_batch), model.cfg.max_seq_len).to(device)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1

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
