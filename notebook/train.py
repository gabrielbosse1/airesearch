"""Wall-clock bounded four-stage training with recoverable HF snapshots."""
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from notebook.preprocess import batch
from notebook.tools import reward
from notebook.hf_upload import upload_checkpoint

# Stage 2 (index 1) is the Phase 19 change: 40% code, 40% Mineflayer player
# trajectories, 20% pt-br. Four 47-minute stages fit a free Colab session better
# than four 60-minute stages; the mix is independent of the wall-clock budget.
MIXES = [dict(pt=50, en=30, code=20), dict(code=40, mineflayer=40, pt=20),
         dict(react=40, search=30, code=30), dict(edit=50, pt=20, code=15, mineflayer=15)]
FREEZE = [0, 6, 8, 8]
RL_CASES = [('code', 'Request: Write a Python function to sort a list\nCode: ', None, None),
            ('mineflayer', 'Request: Mine the diamond ore at (50, 30, 75).\nThought: I need diamonds.\nAction: ', None, None),
            ('pt', 'Request: Olá, como você está?\nAnswer: ', None, None),
            ('edit', 'Request: Corrija: voce\nDraft: voce\n', 'voce', 'você')]

def draw(groups, weights, count):
    keys = list(weights)
    return [random.choice(groups[k]) for k in random.choices(keys, weights=list(weights.values()), k=count)]

def check_groups(groups, stages=None):
    stages = range(len(MIXES)) if stages is None else stages
    missing = sorted({k for stage in stages for k in MIXES[stage] if not groups.get(k)})
    if missing:
        hint = ' Generate Mineflayer rows with notebook.mineflayer_training_data.' if 'mineflayer' in missing else ''
        raise ValueError('Missing training groups: ' + ', '.join(missing) + '.' + hint +
                         ' Supply licensed UTF-8 text files; do not silently substitute synthetic data.')

def snapshot(model, opt, scaler, stage, step, elapsed, cfg):
    return dict(version=1, model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                stage=stage, step=step, elapsed=elapsed, config=asdict(cfg), random=random.getstate(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def policy_loss(model, device, max_new=48):
    """REINFORCE: sample token IDs, then replay them to preserve policy gradients."""
    kind, prompt, draft, target = random.choice(RL_CASES)
    prefix = [256] + list(prompt.encode('utf8')); ids = prefix[:]
    model.eval()
    with torch.no_grad():
        for _ in range(min(max_new, model.cfg.max_seq_len-len(ids))):
            logits = model(torch.tensor([ids], device=device))[0, -1].float()
            logits[258:] = -float('inf')
            token = int(torch.multinomial(torch.softmax(logits, -1), 1)); ids.append(token)
            if token == 257: break
    output = bytes(x for x in ids[len(prefix):] if x < 256).decode('utf8', 'replace')
    score = reward(kind, output, draft, target)
    model.train()
    if score == 0 or len(ids) == len(prefix): return None, score
    inp = torch.tensor([ids[:-1]], device=device)
    logits = model(inp)[0, len(prefix)-1:].float()
    logits = torch.cat([logits[:, :258], torch.full_like(logits[:, 258:], -float('inf'))], dim=-1)
    labels = torch.tensor(ids[len(prefix):], device=device)
    return -score * F.log_softmax(logits, dim=-1).gather(1, labels[:,None]).mean(), score

def train(model, groups, token, repo_id, session_deadline, stage_seconds=47*60,
          upload_seconds=15*60, micro_batch=1, accumulation=32, resume=None, lr=3e-4,
          stages=None):
    """Train selected stages (default all four), resuming from an HF snapshot.

    ``stages=[k]`` lets the notebook keep one cell per stage while sharing the
    in-memory model. ``resume`` always restores weights; the optimizer, elapsed
    time and RNG are restored only for the checkpoint's own stage.
    """
    stages = tuple(range(len(MIXES))) if stages is None else tuple(stages)
    if not stages or any(stage not in range(len(MIXES)) for stage in stages):
        raise ValueError('stages must be a non-empty subset of 0..3')
    check_groups(groups, stages)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if resume:
        if resume['config'] != asdict(model.cfg): raise ValueError('Checkpoint architecture mismatch')
        model.load_state_dict(resume['model']); random.setstate(resume['random']); torch.set_rng_state(resume['torch_rng'])
        if torch.cuda.is_available() and resume['cuda_rng'] is not None: torch.cuda.set_rng_state_all(resume['cuda_rng'])
        if resume['stage'] in stages:
            start_stage = resume['stage']
        else:
            start_stage = min(stages)
            print(f"Checkpoint is at stage {resume['stage']}; starting stage {start_stage} instead.")
    else:
        start_stage = min(stages)
    step = resume['step'] if resume else 0
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    for stage in stages:
        if stage < start_stage:
            continue
        model.freeze(min(FREEZE[stage], len(model.blocks)))
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=.01)
        elapsed = resume['elapsed'] if resume and stage == start_stage == resume['stage'] else 0.0
        if resume and stage == start_stage == resume['stage']:
            opt.load_state_dict(resume['optimizer']); scaler.load_state_dict(resume['scaler'])
        last_upload = time.monotonic()
        bar = tqdm(total=stage_seconds, initial=min(elapsed, stage_seconds), desc=f'Stage {stage+1}', unit='s')
        tick = time.monotonic(); model.train()
        while elapsed < stage_seconds and time.time() < session_deadline:
            opt.zero_grad(set_to_none=True)
            frac = min(1., elapsed/stage_seconds)
            for param in opt.param_groups: param['lr'] = lr * .5 * (1 + math.cos(math.pi * frac))
            losses = []
            for _ in range(accumulation):
                ids = batch(draw(groups, MIXES[stage], micro_batch), model.cfg.max_seq_len).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    loss = model.loss(ids)/accumulation
                scaler.scale(loss).backward(); losses.append(float(loss.detach())*accumulation)
            if stage == 3 and step % 8 == 0:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    rl_loss, score = policy_loss(model, device)
                if rl_loss is not None: scaler.scale(rl_loss * .1).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); step += 1
            now = time.monotonic(); delta = now-tick; tick = now; elapsed += delta
            bar.update(min(delta, max(0, stage_seconds-bar.n))); bar.set_postfix(loss=f'{sum(losses)/len(losses):.3f}', step=step)
            if now-last_upload >= upload_seconds:
                upload_checkpoint(snapshot(model, opt, scaler, stage, step, elapsed, model.cfg), stage, step, token, repo_id)
                last_upload = time.monotonic()  # upload time counts against the budget
        bar.close()
        # Upload current stage even on wall-clock stop. At resume, an exhausted
        # stage immediately advances after restoring its optimizer.
        upload_checkpoint(snapshot(model, opt, scaler, stage, step, elapsed, model.cfg), stage, step, token, repo_id)
        resume = None
        if time.time() >= session_deadline:
            print('Stopped at session deadline; resume from the latest uploaded checkpoint.')
            return False
    return True
