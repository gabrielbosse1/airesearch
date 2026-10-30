"""Ten-epoch byte-generation and learned chunk-verifier feasibility experiment."""
import argparse
import ast
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from data.phase14_data import code_examples
from models.diverse_response import load_diverse_decoder
from models.phase14 import ChunkVerificationHead, VerifiedChunkGenerator, verifier_batch
from models.chunked_decoder import generate_diverse_in_chunks
from models.holdout import HoldoutGuard
from training.finalize_replay import final_guard
from training.train_confidence import load_generator
from training.train_diverse_responses import _features, _target_tensors
from training.phase13_feasibility import grad_norm
from training.train_conversation import sha256
from evaluation.phase13_eval import distance


def verification_rows(examples,seed):
    rng=random.Random(seed); rows=[]
    for _, target in examples:
        for start in range(0,len(target),30):
            prefix, good=target[:start],target[start:start+30]
            if not good: continue
            # Same length/characters as positive, so NUL/noise shortcuts are avoided.
            shuffled=list(good);rng.shuffle(shuffled);bad=bytes(shuffled)
            if bad==good: bad=good[::-1]
            if bad==good: continue
            rows.extend(((prefix,good,1),(prefix,bad,0)))
    return rows


def verifier_cache(rows):
    ids,seg,lengths=verifier_batch([(x[0],x[1]) for x in rows])
    return ids,seg,lengths,torch.tensor([x[2] for x in rows],dtype=torch.float)


def train(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=random.Random(args.seed)
    gen=load_generator(args.generator,'cpu',4).eval();decoder=load_diverse_decoder(args.decoder,generator_path=args.generator).eval()
    train=code_examples('train');val=code_examples('validation')
    base=final_guard(args)
    protection=HoldoutGuard.from_records([{'input':p,'output':t} for p,t in val])
    guard=HoldoutGuard(base.contents|protection.contents,base.held_out_hashes|protection.held_out_hashes)
    for p,t in train: guard.validate(p,t)
    tf=_features(gen,[x[0] for x in train]);vf=_features(gen,[x[0] for x in val]);tp,tt=_target_tensors([x[1] for x in train]);vp,vt=_target_tensors([x[1] for x in val])
    vr=verification_rows(train,args.seed+1); vv=verification_rows(val,args.seed+2)
    for prefix, chunk, _ in vr:
        guard.validate(prefix + chunk)
    a=verifier_cache(vr);b=verifier_cache(vv)
    verifier=ChunkVerificationHead();dparams=list(decoder.parameters());vparams=list(verifier.parameters())
    dop=torch.optim.Adam(dparams,lr=.0015);vop=torch.optim.Adam(vparams,lr=.001)
    def byte_loss(feats,previous,target):
        logits=decoder(feats,torch.zeros(len(feats),dtype=torch.long),previous)
        return F.cross_entropy(logits.reshape(-1,257),target.reshape(-1),ignore_index=-100)
    def verifier_logits(c,ix):
        return verifier(c[0][ix],c[1][ix],c[2][ix])
    def metric():
        with torch.no_grad():
            logits=torch.cat([verifier_logits(b,ix) for ix in torch.arange(len(b[3])).split(64)])
            pred=logits>=0; truth=b[3].bool()
            return {'train_nll':float(byte_loss(tf,tp,tt)),'validation_nll':float(byte_loss(vf,vp,vt)),
                    'verifier_validation_bce':float(F.binary_cross_entropy_with_logits(logits,b[3])),
                    'verifier_validation_accuracy':float((pred==truth).float().mean()),
                    'true_positive':int((pred&truth).sum()),'true_negative':int((~pred&~truth).sum()),
                    'positive':int(truth.sum()),'negative':int((~truth).sum())}
    initial=metric(); history=[]
    for epoch in range(10):
        dn=[];vn=[];losses=[];verifier_losses=[]
        order=list(range(len(train)));rng.shuffle(order)
        for start in range(0,len(order),8):
            ix=torch.tensor(order[start:start+8]);cost=byte_loss(tf[ix],tp[ix],tt[ix])
            dop.zero_grad();cost.backward();dn.append(grad_norm(dparams));torch.nn.utils.clip_grad_norm_(dparams,5,error_if_nonfinite=True);dop.step();losses.append(float(cost.detach()))
        order=list(range(len(vr)));rng.shuffle(order)
        for start in range(0,len(order),64):
            ix=torch.tensor(order[start:start+64]);cost=F.binary_cross_entropy_with_logits(verifier_logits(a,ix),a[3][ix])
            vop.zero_grad();cost.backward();vn.append(grad_norm(vparams));torch.nn.utils.clip_grad_norm_(vparams,5,error_if_nonfinite=True);vop.step();verifier_losses.append(float(cost.detach()))
        row={'epoch':epoch+1,'decoder_train_batch_nll':sum(losses)/len(losses),'verifier_train_bce':sum(verifier_losses)/len(verifier_losses),
             'decoder_gradient_min':min(dn),'decoder_gradient_max':max(dn),'decoder_gradient_mean':sum(dn)/len(dn),
             'verifier_gradient_min':min(vn),'verifier_gradient_max':max(vn),'verifier_gradient_mean':sum(vn)/len(vn),**metric()}
        history.append(row);print(json.dumps(row),flush=True)
    decoder.eval();verifier.eval();wrapper=VerifiedChunkGenerator(decoder,verifier).eval()
    results=[]
    for (prompt,target),feature in zip(val[:10],vf[:10]):
        gated,scores=wrapper.generate(feature.unsqueeze(0),style=0,target_length=150,chunk_size=30,max_chunks=5,threshold=.5)
        plain=generate_diverse_in_chunks(decoder,feature.unsqueeze(0),lambda p,c:True,style=0,target_length=150,chunk_size=30,max_chunks=5)
        def evaluate(x):
            try:
                text=x.output.decode('utf-8'); tree=ast.parse(text)
                structural=len(x.output)>=100 and any(isinstance(n,ast.FunctionDef) for n in tree.body)
            except (UnicodeDecodeError,SyntaxError,ValueError): structural=False
            edit=distance(x.output,target)
            return {'output':x.output.decode('utf-8','replace'),'hex':x.output.hex(),'bytes':len(x.output),'halt_reason':x.halt_reason,
                    'structural':structural,'edit_distance':edit,'byte_similarity':1-edit/max(len(target),len(x.output))}
        results.append({'prompt':prompt.decode(),'reference':target.decode(),'verified':evaluate(gated),'ungated':evaluate(plain),'verifier_scores':scores})
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    torch.save({'generator_sha256':sha256(args.generator),'decoder':decoder.state_dict(),'verifier':verifier.state_dict()},out/'verified_chunks.pt')
    report={'configuration':vars(args),'code_train':len(train),'code_validation':len(val),'verifier_train':len(vr),'verifier_validation':len(vv),
            'initial':initial,'epochs':history,'generation':results,
            'limitations':'Code templates recur across splits; verifier learns shuffled-byte vs original chunks, not semantics. No prompt is supplied to the verifier. Syntax is not functional correctness.'}
    (out/'chunk_metrics.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--generator',default='runs/phase12/generator.pt');p.add_argument('--decoder',default='runs/phase12/diverse_response.pt');p.add_argument('--output-dir',default='runs/phase14');p.add_argument('--pairs',default='data/real_edit_pairs.jsonl');p.add_argument('--processed-dir',default='data/processed');p.add_argument('--seed',type=int,default=14003);train(p.parse_args())
if __name__=='__main__':main()
