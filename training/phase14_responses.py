"""Train experimental seven-intent, twelve-style byte responses with diversity loss."""
import argparse
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from models.conversation_intents import ConversationIntent as I
from models.diverse_response import distribution_diversity_loss
from models.phase14 import IntentResponseDecoder
from data.phase14_data import RESPONSE_VARIATIONS
from training.train_confidence import load_generator
from training.train_diverse_responses import _features, _target_tensors
from training.phase13_feasibility import grad_norm
from models.conversation import _visible_text
from models.holdout import HoldoutGuard
from training.finalize_replay import final_guard
from training.train_conversation import sha256


def train(args):
    torch.set_num_threads(1); torch.manual_seed(args.seed); rng=random.Random(args.seed)
    g=load_generator(args.generator,'cpu',4).eval()
    train_prompts=[f'Please respond about topic number {i} in this session.'.encode() for i in range(8)]
    val_prompts=[b'Please respond to the current conversation.', b'Please reply to this turn.']
    prior = final_guard(args)
    validation = HoldoutGuard.from_records([{'input': x} for x in val_prompts])
    guard = HoldoutGuard(prior.contents | validation.contents, prior.held_out_hashes | validation.held_out_hashes)
    for prompt in train_prompts:
        for variants in RESPONSE_VARIATIONS.values():
            for text in variants:
                guard.validate(prompt, text.encode())
    train_features=_features(g,train_prompts); val_features=_features(g,val_prompts)
    responses=[s.encode() for label in I for s in RESPONSE_VARIATIONS[label]]
    previous,targets=_target_tensors(responses)
    d=IntentResponseDecoder(g.d_model,styles=12); params=list(d.parameters()); opt=torch.optim.Adam(params,lr=.0015)
    def batch():
        intent=torch.tensor([rng.randrange(7) for _ in range(8)])
        contexts=train_features[torch.tensor([rng.randrange(len(train_prompts)) for _ in range(8)])]
        styles=torch.tensor([rng.sample(range(12),6) for _ in range(8)])
        ids=(intent[:,None]*12+styles).reshape(-1)
        logits=d(contexts.repeat_interleave(6,0),intent.repeat_interleave(6),styles.reshape(-1),previous[ids])
        output=targets[ids]
        nll=F.cross_entropy(logits.reshape(-1,257),output.reshape(-1),ignore_index=-100)
        div=distribution_diversity_loss(logits.reshape(8,6,logits.shape[1],257),(output!=-100).reshape(8,6,-1))
        return nll,div
    def validation():
        with torch.no_grad():
            intents=torch.arange(7).repeat_interleave(12); styles=torch.arange(12).repeat(7)
            contexts=val_features[0].repeat(84,1); logits=d(contexts,intents,styles,previous)
            nll=F.cross_entropy(logits.reshape(-1,257),targets.reshape(-1),ignore_index=-100)
            valid=targets!=-100; exact=(((logits.argmax(-1)==targets)|~valid).all(-1)).sum()
            return {'nll':float(nll),'teacher_exact':int(exact),'count':84}
    initial=validation(); history=[]
    for epoch in range(10):
        losses=[]; divs=[]; grads=[]
        for _ in range(args.steps):
            nll,div=batch(); loss=nll+args.diversity_weight*div
            opt.zero_grad();loss.backward();grads.append(grad_norm(params));torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True);opt.step()
            losses.append(float(nll.detach()));divs.append(float(div.detach()))
        row={'epoch':epoch+1,'train_nll':sum(losses)/len(losses),'diversity_loss':sum(divs)/len(divs),
             'validation':validation(),'gradient_mean':sum(grads)/len(grads),'gradient_min':min(grads),'gradient_max':max(grads)}
        history.append(row);print(json.dumps(row),flush=True)
    d.eval(); results={}
    # Fixed seed, sampling on held-out prompt for each intent; do not select samples.
    for label in I:
        torchrng=torch.Generator().manual_seed(args.seed+int(label))
        raw=[d.generate(val_features[1:2],torch.tensor([int(label)]),temperature=.3,sample=True,rng=torchrng,max_bytes=80) for _ in range(20)]
        results[label.name.lower()]={'unique':len({x[0] for x in raw}),
          'complete_valid':sum(complete and _visible_text(data) for data,complete,_ in raw),
          'known_variation_matches':sum(data.decode('utf-8','replace') in RESPONSE_VARIATIONS[label] for data,_,_ in raw),
          'outputs':[{'text':data.decode('utf-8','replace'),'hex':data.hex(),'complete':complete,'style':style} for data,complete,style in raw]}
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    torch.save({'generator_sha256':sha256(args.generator),'state_dict':d.state_dict(),'styles':12},out/'intent_response.pt')
    report={'config':vars(args),'train_prompt_count':len(train_prompts),'validation_prompt_count':len(val_prompts),
            'guard_fingerprint':guard.fingerprint,'responses_per_intent':12,'trained_intents':list(results),'initial':initial,'epochs':history,'evaluation':results,
            'limitations':'Synthetic intent-conditioned generic responses; shared templates across train/validation; no factual grounding or side-effect authorization. Direct decoder test, not session.'}
    (out/'response_metrics.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--generator',default='runs/phase12/generator.pt');p.add_argument('--output-dir',default='runs/phase14');p.add_argument('--pairs',default='data/real_edit_pairs.jsonl');p.add_argument('--processed-dir',default='data/processed');p.add_argument('--steps',type=int,default=45);p.add_argument('--diversity-weight',type=float,default=.1);p.add_argument('--seed',type=int,default=14002);train(p.parse_args())
if __name__=='__main__':main()
