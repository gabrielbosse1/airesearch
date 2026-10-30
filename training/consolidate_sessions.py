"""Experimental Phase 12 session intent rehearsal, preserving the frozen backbone.

Synthetic training only. Validate against newly declared prompts, protect old
held-outs, and never alter the Phase 12 checkpoint. Not deployment calibration.
"""
import argparse
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from data.consolidation_examples import greeting_train, greeting_validation, greeting_test, session_test, TRAIN_HISTORY
from models.conversation import pack_conversation
from models.holdout import HoldoutGuard, HoldoutViolation
from training.finalize_replay import final_guard
from training.replay import ReplayBuffer
from training.train_confidence import load_generator
from training.train_conversation import sha256


def cache(model, rows):
    payloads = [pack_conversation(msg, hist[-2:], min(model.max_len, len(msg) + 202)) for msg, _, hist in rows]
    lengths = torch.tensor([len(x) for x in payloads])
    ids = torch.zeros((len(rows), max(lengths)), dtype=torch.long)
    for i, payload in enumerate(payloads): ids[i, :len(payload)] = torch.tensor(list(payload))
    focus = torch.tensor([len(payload)-len(row[0]) for payload, row in zip(payloads, rows)])
    labels = torch.tensor([int(row[1]) for row in rows])
    with torch.no_grad():
        base = torch.cat([model.conversation_backbone_features(ids[ix, :int(lengths[ix].max())], lengths[ix]) for ix in torch.arange(len(rows)).split(32)])
    return base, ids, lengths, focus, labels


def features(model, c, ix):
    base, ids, lengths, focus, _ = c
    return model.conversation_features(ids[ix, :int(lengths[ix].max())], lengths[ix], focus_start=focus[ix], backbone_features=base[ix])


def train(args):
    torch.set_num_threads(1); torch.manual_seed(args.seed); rng = random.Random(args.seed)
    model = load_generator(args.generator, 'cpu', 4).eval()
    original = {k:v.clone() for k,v in model.state_dict().items()}
    # New evaluation prompts and historic held-outs excluded from new training.
    guard0 = final_guard(args)
    new_guard = HoldoutGuard.from_records([{'input': x[0]} for x in greeting_validation()+greeting_test()+session_test()])
    guard = HoldoutGuard(guard0.contents | new_guard.contents, guard0.held_out_hashes | new_guard.held_out_hashes)
    safe = []
    for msg, label, hist in greeting_train():
        try: guard.validate(msg, extra=tuple(x for t in hist for x in (t.user, t.assistant)))
        except HoldoutViolation: continue
        safe.append((msg, label, hist))
    replay = ReplayBuffer.load('runs/phase12/replay_multitask_phase12.pt', guard=guard0)
    old = []
    for i, row in enumerate(replay.buffer):
        if row.task != 'intent': continue
        try: guard.validate(row.message)
        except HoldoutViolation: continue
        # Include both no-history and contextual variants for historical rehearsal.
        old.append((row.message, row.label, TRAIN_HISTORY[i % len(TRAIN_HISTORY)]))
        old.append((row.message, row.label, ()))
    if len(safe) < 50 or len(old) < 100: raise RuntimeError(f'insufficient guarded rows: {len(safe)}, {len(old)}')
    tc, oc, vc = cache(model, safe), cache(model, old), cache(model, greeting_validation())
    for p in model.parameters(): p.requires_grad_(False)
    params = list(model.intent_encoder.parameters()) + list(model.intent_classifier.parameters())
    for p in params: p.requires_grad_(True)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    best = None; best_rank = None; curves = []
    for epoch in range(args.epochs):
        losses = []
        for _ in range(args.steps):
            a = torch.tensor(rng.choices(range(len(safe)), k=32)); b = torch.tensor(rng.choices(range(len(old)), k=96))
            logits = torch.cat((model.intent_classifier(features(model, tc, a)), model.intent_classifier(features(model, oc, b))))
            labels = torch.cat((tc[4][a], oc[4][b]))
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 2, error_if_nonfinite=True); optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            val = float((model.intent_classifier(features(model, vc, torch.arange(len(vc[4])))).argmax(-1) == vc[4]).float().mean())
            # Historical replay train metric is a sanity constraint, not held-out evaluation.
            acc = float((model.intent_classifier(features(model, oc, torch.arange(len(oc[4])))).argmax(-1) == oc[4]).float().mean())
        row = {'epoch':epoch+1,'train_loss':sum(losses)/len(losses),'greeting_validation':val,'replay_train_accuracy':acc}; curves.append(row)
        rank = (val + acc, acc)
        if best_rank is None or rank > best_rank:
            best_rank = rank; best = {k:v.detach().clone() for k,v in model.state_dict().items()}; selected = epoch+1
        print(json.dumps(row), flush=True)
    model.load_state_dict(best)
    for name, value in model.state_dict().items():
        if not name.startswith(('intent_encoder.', 'intent_classifier.')) and not torch.equal(value, original[name]):
            raise RuntimeError('frozen tensor changed: '+name)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(), out/'generator.pt')
    report={'configuration':vars(args),'source_sha256':sha256(args.generator),'generator_sha256':sha256(out/'generator.pt'),
            'train_greetings':len(safe),'replay_context_and_no_context':len(old),'selected_epoch':selected,'epochs':curves,
            'selection':'greeting validation accuracy plus replay training accuracy; no test selection',
            'limitations':'Synthetic and replay training metrics are not a safety or retention guarantee.'}
    (out/'intent_training.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--generator',default='runs/phase12/generator.pt'); p.add_argument('--output-dir',default='runs/phase12_consolidation'); p.add_argument('--pairs',default='data/real_edit_pairs.jsonl'); p.add_argument('--processed-dir',default='data/processed'); p.add_argument('--epochs',type=int,default=12); p.add_argument('--steps',type=int,default=15); p.add_argument('--lr',type=float,default=0.0002); p.add_argument('--seed',type=int,default=12501); train(p.parse_args())
if __name__=='__main__': main()
