"""Coverage for Phase 14 datasets, EWC, replay and learned verifier wiring."""
import unittest
import random
import torch
from data.phase14_data import routing_examples, code_examples, RESPONSE_VARIATIONS
from models.conversation_intents import ConversationIntent
from models.diverse_response import DiverseByteDecoder
from models.phase14 import ChunkVerificationHead, IntentResponseDecoder, VerifiedChunkGenerator, verifier_batch
from training.phase14_routing import balanced_indices, ewc_penalty
from training.phase14_chunks import verification_rows


class Phase14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_data_splits_and_pairs(self):
        tr, va = routing_examples('train'), routing_examples('validation')
        self.assertEqual((len(tr),len(va)),(800,200))
        self.assertFalse({(m,h) for m,_,h in tr} & {(m,h) for m,_,h in va})
        for i in range(0,len(tr),2):
            self.assertEqual(tr[i][0],tr[i+1][0]);self.assertNotEqual(tr[i][1],tr[i+1][1]);self.assertNotEqual(tr[i][2],tr[i+1][2])
        self.assertEqual((len(code_examples('train')),len(code_examples('validation'))),(120,30))
        self.assertFalse(set(code_examples('train')) & set(code_examples('validation')))
        self.assertEqual(set(RESPONSE_VARIATIONS),set(ConversationIntent))
        self.assertTrue(all(len(x)>=10 for x in RESPONSE_VARIATIONS.values()))

    def test_fisher_penalty_and_balanced_replay(self):
        p=torch.nn.Parameter(torch.tensor([1.,3.])); anchor=[torch.tensor([1.,2.])];fisher=[torch.tensor([4.,2.])]
        loss=ewc_penalty([p],anchor,fisher);self.assertAlmostEqual(loss.item(),1.0)
        loss.backward();self.assertTrue(torch.equal(p.grad,torch.tensor([0.,2.])))
        idx=balanced_indices([0]*10+[1]*10+[2]*10,9,random.Random(1)).tolist()
        self.assertEqual(len(set(idx)),9);self.assertEqual([sum(i//10==j for i in idx) for j in range(3)],[3,3,3])

    def test_verifier_padding_and_gradient(self):
        torch.manual_seed(1);head=ChunkVerificationHead().eval()
        x=verifier_batch([(b'prefix',b'abc')]); y=verifier_batch([(b'prefix',b'abc'),(b'a'*45,b'bc')])
        self.assertTrue(torch.allclose(head(*x)[0],head(*y)[0],atol=1e-6))
        head(*y).sum().backward();self.assertGreater(float(head.output.weight.grad.norm()),0)
        with self.assertRaises(ValueError): verifier_batch([(b'prefix',b'')])

    def test_generated_chunk_is_rejected_by_verifier(self):
        d=DiverseByteDecoder(context_dim=8,num_styles=2,hidden_dim=8,byte_dim=4,style_dim=4).eval()
        with torch.no_grad():
            d.output.weight.zero_();d.output.bias.fill_(-100);d.output.bias[65]=100
        v=ChunkVerificationHead().eval()
        with torch.no_grad(): v.output.weight.zero_();v.output.bias.fill_(-10)
        result,scores=VerifiedChunkGenerator(d,v).generate(torch.zeros(1,8),style=0,target_length=10,chunk_size=4,max_chunks=3)
        self.assertEqual(result.halt_reason,'rejected');self.assertEqual(result.output,b'');self.assertEqual(len(scores),1)
        self.assertLess(scores[0],.5)

    def test_intent_conditioning_and_verifier_labels(self):
        decoder=IntentResponseDecoder(8,styles=12)
        context=torch.zeros(2,8);intent=torch.tensor([0,1]);style=torch.tensor([0,0]);previous=torch.full((2,2),257,dtype=torch.long)
        self.assertEqual(decoder(context,intent,style,previous).shape,(2,2,257))
        rows=verification_rows(code_examples('train')[:2],10)
        for good,bad in zip(rows[::2],rows[1::2]):
            self.assertEqual(good[0],bad[0]);self.assertEqual((good[2],bad[2]),(1,0));self.assertNotEqual(good[1],bad[1]);self.assertEqual(sorted(good[1]),sorted(bad[1]))

if __name__=='__main__': unittest.main()
