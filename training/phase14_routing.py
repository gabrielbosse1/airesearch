"""Ten-epoch Phase 14 routing probe: balanced replay, teacher KL, diagonal EWC."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from data.phase14_data import routing_examples
from data.conversation_examples import conversation_examples
from evaluation.conversational_eval_v2 import classification_metrics
from models.conversation import pack_conversation
from models.holdout import HoldoutGuard
from training.finalize_replay import final_guard
from training.replay import ReplayBuffer
from training.consolidate_sessions import cache, features
from training.phase13_feasibility import grad_norm
from training.train_confidence import load_generator
from training.train_conversation import sha256


def balanced_indices(labels, count, rng):
    """Round-robin class stratification; no repeated index inside a batch."""
    groups=defaultdict(list)
    for i,label in enumerate(labels): groups[int(label)].append(i)
    classes=list(groups)
    for values in groups.values(): rng.shuffle(values)
    rng.shuffle(classes); result=[]; pos={k:0 for k in classes}
    while len(result)<count:
        progressed=False
        for key in classes:
            if pos[key] < len(groups[key]) and len(result)<count:
                result.append(groups[key][pos[key]]); pos[key]+=1; progressed=True
        if not progressed: raise ValueError('not enough distinct replay items')
    rng.shuffle(result)
    return torch.tensor(result)


def empirical_fisher(model, cached, indices, params):
    """Mean of per-example squared supervised log-likelihood gradients."""
    fisher=[torch.zeros_like(p) for p in params]
    for i in indices:
        ix=torch.tensor([int(i)])
        loss=F.cross_entropy(model.intent_classifier(features(model,cached,ix)),cached[4][ix])
        grads=torch.autograd.grad(loss,params)
        for f,g in zip(fisher,grads): f.add_(g.detach().square())
    return [x/len(indices) for x in fisher]


def ewc_penalty(params, anchor, fisher):
    return sum((f*(p-old).square()).sum() for p,old,f in zip(params,anchor,fisher)) / 2


def train(args):
    torch.set_num_threads(1); torch.manual_seed(args.seed); rng=random.Random(args.seed)
    model=load_generator(args.generator,'cpu',4).eval()
    for p in model.parameters(): p.requires_grad_(False)
    params=list(model.intent_encoder.parameters())+list(model.intent_classifier.parameters())
    for p in params:p.requires_grad_(True)
    anchor=[p.detach().clone() for p in params]
    original={k:v.clone() for k,v in model.state_dict().items()}
    train_rows=routing_examples('train'); val_rows=routing_examples('validation')
    # Validation pairs intentionally share prompts with train; protect complete
    # history+prompt frames instead of forbidding the shared prompt alone.
    base=final_guard(args)
    valguard=HoldoutGuard.from_records([{'input':pack_conversation(m,h,model.max_len)} for m,_,h in val_rows])
    guard=HoldoutGuard(base.contents|valguard.contents,base.held_out_hashes|valguard.held_out_hashes)
    for m,_,h in train_rows:
        guard.validate(pack_conversation(m,h,model.max_len), extra=tuple(x for t in h for x in (t.user,t.assistant)))
    replay=ReplayBuffer.load('runs/phase12/replay_multitask_phase12.pt',guard=base)
    old=[(r.message,r.label,()) for r in replay.buffer if r.task=='intent']
    for m,_,h in old: guard.validate(m)
    tc=cache(model,train_rows); vc=cache(model,val_rows); oc=cache(model,old)
    with torch.no_grad():
        teacher=torch.cat([model.intent_classifier(features(model,oc,ix)) for ix in torch.arange(len(old)).split(64)]).detach()
    chosen=balanced_indices(oc[4],70,rng); fisher=empirical_fisher(model,oc,chosen,params)
    opt=torch.optim.Adam(params,lr=args.lr)
    oldtest=[(r.message,r.intent) for r in conversation_examples('test')]
    def measure(c):
        with torch.no_grad():
            logits=torch.cat([model.intent_classifier(features(model,c,ix)) for ix in torch.arange(len(c[4])).split(64)])
            return {'loss':float(F.cross_entropy(logits,c[4])), 'accuracy':float((logits.argmax(-1)==c[4]).float().mean())}
    initial={'train':measure(tc),'validation':measure(vc),'phase10':classification_metrics(model,oldtest,threshold=.55)['correct']}
    history=[]
    for epoch in range(10):
        order=list(range(len(train_rows))); rng.shuffle(order); losses=[]; penalties=[]; grads=[]
        for start in range(0,len(order),32):
            ni=torch.tensor(order[start:start+32]); oi=balanced_indices(oc[4],len(ni),rng)
            new_logits=model.intent_classifier(features(model,tc,ni)); old_logits=model.intent_classifier(features(model,oc,oi))
            ce=(F.cross_entropy(new_logits,tc[4][ni])+F.cross_entropy(old_logits,oc[4][oi]))/2
            kl=F.kl_div(F.log_softmax(old_logits,dim=-1),F.softmax(teacher[oi],dim=-1),reduction='batchmean')
            penalty=ewc_penalty(params,anchor,fisher)
            objective=ce+args.ewc_lambda*penalty+args.distill*kl
            opt.zero_grad(); objective.backward(); grads.append(grad_norm(params)); torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True); opt.step()
            losses.append(float(ce.detach())); penalties.append(float(penalty.detach()))
        train_m=measure(tc); val_m=measure(vc)
        phase10=classification_metrics(model,oldtest,threshold=.55)['correct']
        row={'epoch':epoch+1,'batch_ce':sum(losses)/len(losses),'train':train_m,'validation':val_m,
             'phase10_correct':phase10,'gradient_min':min(grads),'gradient_max':max(grads),'gradient_mean':sum(grads)/len(grads),
             'ewc_penalty':sum(penalties)/len(penalties)}
        history.append(row); print(json.dumps(row),flush=True)
    for k,v in model.state_dict().items():
        if not k.startswith(('intent_encoder.','intent_classifier.')) and not torch.equal(v,original[k]): raise RuntimeError('frozen tensor changed')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out/'routing_generator.pt')
    report={'configuration':vars(args),'source_sha256':sha256(args.generator),'checkpoint_sha256':sha256(out/'routing_generator.pt'),
            'train_examples':len(train_rows),'validation_examples':len(val_rows),'old_replay':len(old),'mix':'50% old / 50% new per batch',
            'replay_sampling':'class-stratified without replacement within each batch','fisher_examples':len(chosen),
            'fisher_mean':sum(float(x.sum()) for x in fisher)/sum(x.numel() for x in fisher),
            'initial':initial,'epochs':history,'selection':'fixed final epoch; no test-based checkpoint selection',
            'limits':'Synthetic labels and held-out topic entities; prompts and templates shared between train and validation.'}
    (out/'routing_metrics.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--generator',default='runs/phase12/generator.pt');p.add_argument('--output-dir',default='runs/phase14');p.add_argument('--pairs',default='data/real_edit_pairs.jsonl');p.add_argument('--processed-dir',default='data/processed');p.add_argument('--lr',type=float,default=.00012);p.add_argument('--ewc-lambda',type=float,default=20.0);p.add_argument('--distill',type=float,default=.5);p.add_argument('--seed',type=int,default=14001);train(p.parse_args())
if __name__=='__main__':main()
