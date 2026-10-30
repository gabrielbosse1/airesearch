"""Experimental consolidation evaluation on actual isolated session routes."""
import argparse
import json
from pathlib import Path
import torch
from data.consolidation_examples import greeting_test, session_test
from data.conversation_examples import conversation_examples
from data.phase12_eval_sets import DIVERSITY_TEST_PROMPT
from evaluation.conversational_eval_v2 import classification_metrics
from evaluation.memory_eval import _load, _session
from models.conversation_intents import INTENT_NAMES
from models.conversation import _visible_text
from training.train_diverse_responses import _features


def evaluate(args):
    torch.set_num_threads(1); components = _load(args); model, _, _, _, decoder = components
    def cases(data):
        rows=[]
        for msg, intent, history in data:
            s, _ = _session(components, b'consolidation-eval'); s.max_history_turns=2; s.history=list(history)
            r=s.respond(msg); expected=INTENT_NAMES[int(intent)]
            rows.append({'prompt':msg.decode(), 'history':[{'user':t.user.decode(),'assistant':t.assistant.decode()} for t in history],
                         'expected':expected,'predicted':r.intent,'action':r.action,'correct':r.intent==expected})
        return {'count':len(rows),'correct':sum(x['correct'] for x in rows),'rows':rows}
    # Same held-out prompt repeated with changing real session history and RNG.
    s, _ = _session(components, b'diversity-session'); s.max_history_turns=2; s.rng.manual_seed(12555)
    routed=[s.respond(DIVERSITY_TEST_PROMPT) for _ in range(20)]
    features=_features(model,[DIVERSITY_TEST_PROMPT]); rng=torch.Generator().manual_seed(12555)
    direct=[decoder.generate(features,temperature=0.15,sample=True,rng=rng) for _ in range(20)]
    output=[{'text':r.response.decode('utf-8','replace'),'intent':r.intent,'source':r.response_source,'action':r.action} for r in routed]
    historical=classification_metrics(model,[(x.message,x.intent) for x in conversation_examples('test')],threshold=.55)
    return {'greetings':cases(greeting_test()),'session_intents':cases(session_test()),
            'diversity':{'routed_unique':len({x.response for x in routed}),
                'routed_diverse_decoder':sum(x.response_source=='diverse_byte_decoder' for x in routed),
                'direct_unique':len({x[0] for x in direct}), 'direct_complete_valid':sum(complete and _visible_text(data) for data,complete,_ in direct),
                'seed':12555,'routed':output, 'direct':[{'text':x[0].decode('utf-8','replace'),'complete':x[1],'style':x[2]} for x in direct]},
            'phase10_development':historical,
            'limitations':'Synthetic English, fixed seed, previously inspected Phase 12 and Phase 10 sets are development tests; distinct bytes do not measure appropriate responses.'}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--generator',default='runs/phase12_consolidation/generator.pt'); p.add_argument('--response',default='runs/phase12_consolidation/response.pt'); p.add_argument('--vector-reader',default='runs/phase12_consolidation/vector_reader.pt'); p.add_argument('--diverse-response',default='runs/phase12_consolidation/diverse_response.pt'); p.add_argument('--heads',type=int,default=4); p.add_argument('--output',default='runs/phase12_consolidation/benchmark.json'); a=p.parse_args()
    result=evaluate(a); Path(a.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'greetings':f"{result['greetings']['correct']}/{result['greetings']['count']}", 'session':f"{result['session_intents']['correct']}/{result['session_intents']['count']}", 'routed_unique':result['diversity']['routed_unique'],'direct_unique':result['diversity']['direct_unique'],'phase10':f"{result['phase10_development']['correct']}/{result['phase10_development']['count']}"},indent=2))
if __name__=='__main__': main()
