"""Five-epoch isolated feasibility probes. Never writes modified model checkpoints.

The chunk probe trains an autoregressive byte decoder on chunk-framed data, but
THINK is host metadata, not a learned cognitive action. Held-out prompts, target
code and histories are synthetic; conclusions are restricted to this experiment.
"""
import argparse
import ast
import json
import math
from pathlib import Path
from argparse import Namespace
from models.holdout import HoldoutGuard
from training.finalize_replay import final_guard
import torch
import torch.nn.functional as F
from data.phase13_feasibility import routing_examples, chunk_examples
from models.chunked_decoder import generate_diverse_in_chunks
from models.diverse_response import load_diverse_decoder
from training.consolidate_sessions import cache, features
from training.train_confidence import load_generator
from training.train_diverse_responses import _features, _target_tensors
from evaluation.phase13_eval import distance


def grad_norm(params):
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in params if p.grad is not None))


def train_routing(generator):
    for p in generator.parameters(): p.requires_grad_(False)
    params=list(generator.intent_encoder.parameters())+list(generator.intent_classifier.parameters())
    for p in params: p.requires_grad_(True)
    train=cache(generator,routing_examples('train')); val=cache(generator,routing_examples('validation'))
    opt=torch.optim.Adam(params,lr=.001)
    def metric(c):
        with torch.no_grad():
            logits=generator.intent_classifier(features(generator,c,torch.arange(len(c[4]))))
            return {'loss':float(F.cross_entropy(logits,c[4])), 'accuracy':float((logits.argmax(-1)==c[4]).float().mean())}
    initial={'train':metric(train),'validation':metric(val)}; history=[]
    rng=torch.Generator().manual_seed(1301)
    for epoch in range(5):
        norms=[]
        for ix in torch.randperm(len(train[4]),generator=rng).split(10):
            logits=generator.intent_classifier(features(generator,train,ix)); loss=F.cross_entropy(logits,train[4][ix])
            opt.zero_grad(); loss.backward(); norms.append(grad_norm(params)); torch.nn.utils.clip_grad_norm_(params, 5, error_if_nonfinite=True); opt.step()
        history.append({'epoch':epoch+1,'train':metric(train),'validation':metric(val),'gradient_min':min(norms),'gradient_max':max(norms),'gradient_mean':sum(norms)/len(norms)})
    return {'train_examples':len(train[4]),'validation_examples':len(val[4]),'initial':initial,'epochs':history}


def train_chunks(model, decoder):
    train=chunk_examples('train'); val=chunk_examples('validation')
    for p in model.parameters(): p.requires_grad_(False)
    for p in decoder.parameters(): p.requires_grad_(True)
    tf=_features(model,[x[0] for x in train]); vf=_features(model,[x[0] for x in val])
    tp,tt=_target_tensors([x[1] for x in train]); vp,vt=_target_tensors([x[1] for x in val])
    # Fixed style; no variation or target bytes supplied at generation time.
    style=torch.zeros(len(train),dtype=torch.long); vs=torch.zeros(len(val),dtype=torch.long)
    params=list(decoder.parameters()); opt=torch.optim.Adam(params,lr=.003)
    def loss(features_, styles, previous, targets):
        logits=decoder(features_,styles,previous)
        return F.cross_entropy(logits.reshape(-1,257),targets.reshape(-1),ignore_index=decoder.IGNORE)
    def nll():
        with torch.no_grad(): return {'train':float(loss(tf,style,tp,tt)), 'validation':float(loss(vf,vs,vp,vt))}
    initial=nll(); history=[]; rng=torch.Generator().manual_seed(1302)
    for epoch in range(5):
        norms=[]
        for ix in torch.randperm(len(train),generator=rng).split(5):
            cost=loss(tf[ix],style[ix],tp[ix],tt[ix]); opt.zero_grad(); cost.backward(); norms.append(grad_norm(params)); torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True); opt.step()
        history.append({'epoch':epoch+1,**nll(),'gradient_min':min(norms),'gradient_max':max(norms),'gradient_mean':sum(norms)/len(norms)})
    def score(rows, feats):
        results=[]
        for (prompt,target,_), feature in zip(rows,feats):
            result=generate_diverse_in_chunks(decoder,feature.unsqueeze(0),lambda p,c: True,style=0,target_length=150,chunk_size=30,max_chunks=5)
            try:
                text=result.output.decode('utf-8')
                printable=all(c.isprintable() or c in '\n\r\t' for c in text)
            except UnicodeDecodeError:
                text = None
                printable = False
            try:
                tree = ast.parse(text) if text is not None else None
                syntax = bool(tree is not None and len(result.output) >= 100 and
                              any(isinstance(n, ast.FunctionDef) for n in tree.body))
            except (SyntaxError, ValueError):
                syntax = False
            edit=distance(result.output,target)
            results.append({'prompt':prompt.decode(),'target':target.decode(),'output':result.output.decode('utf-8','replace'),'output_hex':result.output.hex(),
                            'halt_reason':result.halt_reason,'edit_distance':edit,'byte_similarity':1-edit/max(len(target),len(result.output)),
                            'printable_utf8':printable,'structural_proxy':syntax})
        return {'count':len(results),'mean_byte_similarity':sum(x['byte_similarity'] for x in results)/len(results),
                'structural_count':sum(x['structural_proxy'] for x in results),'printable_count':sum(x['printable_utf8'] for x in results),'examples':results}
    return {'train_examples':len(train),'validation_examples':len(val),'initial':initial,'epochs':history,
            'train_generation':score(train,tf),'validation_generation':score(val,vf),
            'caveat':'Targets contain four 30-byte chunks; teacher forcing trains bytes/EOS, not THINK or verifier decisions. AST/printability are weak structural proxies.'}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--generator',default='runs/phase12_consolidation/generator.pt'); p.add_argument('--decoder',default='runs/phase12_consolidation/diverse_response.pt'); p.add_argument('--output',default='runs/phase13_feasibility.json'); a=p.parse_args()
    torch.set_num_threads(1); torch.manual_seed(1300)
    prior = final_guard(Namespace(pairs='data/real_edit_pairs.jsonl', processed_dir='data/processed'))
    valid = HoldoutGuard.from_records([{'input': row[0]} for row in routing_examples('validation')] +
                                     [{'input': row[0], 'output': row[1]} for row in chunk_examples('validation')])
    guard = HoldoutGuard(prior.contents | valid.contents, prior.held_out_hashes | valid.held_out_hashes)
    for msg, _, history in routing_examples('train'):
        guard.validate(msg, extra=tuple(raw for turn in history for raw in (turn.user, turn.assistant)))
    for prompt, target, _ in chunk_examples('train'):
        guard.validate(prompt, target)
    # Independent copies prevent the routing experiment from changing chunk features.
    g1=load_generator(a.generator,'cpu',4).eval(); g2=load_generator(a.generator,'cpu',4).eval()
    d=load_diverse_decoder(a.decoder,generator_path=a.generator).eval()
    report={'configuration':vars(a),'seed':1300,'routing':train_routing(g1),'chunk':train_chunks(g2,d)}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'routing_initial':report['routing']['initial'],'routing_final':report['routing']['epochs'][-1], 'chunk_initial':report['chunk']['initial'],'chunk_final':report['chunk']['epochs'][-1], 'chunk_validation':{k:v for k,v in report['chunk']['validation_generation'].items() if k != 'examples'}},indent=2))
if __name__=='__main__': main()
