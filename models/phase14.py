"""Experimental intent-aware decoder and learned byte-chunk verifier.

A verifier score is not a safety guarantee and never authorizes side effects.
"""
import torch
from torch import nn
from .diverse_response import DiverseByteDecoder
from .chunked_decoder import generate_diverse_in_chunks


class IntentResponseDecoder(nn.Module):
    def __init__(self, context_dim, styles=12):
        super().__init__()
        self.intent = nn.Embedding(7, context_dim)
        self.decoder = DiverseByteDecoder(context_dim, num_styles=styles, hidden_dim=96)

    def conditioned(self, context, intent):
        return context + self.intent(intent)

    def forward(self, context, intent, style, previous):
        return self.decoder(self.conditioned(context,intent), style, previous)

    def generate(self, context, intent, **kwargs):
        return self.decoder.generate(self.conditioned(context,intent), **kwargs)


class ChunkVerificationHead(nn.Module):
    """Binary scorer over a bounded suffix and candidate, with explicit segment IDs."""
    def __init__(self, width=32, max_bytes=128):
        super().__init__()
        self.max_bytes=max_bytes
        self.byte=nn.Embedding(256,width)
        self.segment=nn.Embedding(2,width)
        self.conv=nn.Sequential(nn.Conv1d(width,width,5,padding=2),nn.SiLU(),nn.Conv1d(width,width,3,padding=1),nn.SiLU())
        self.output=nn.Linear(width*2,1)

    def forward(self, ids, segments, lengths):
        if ids.ndim != 2 or segments.shape != ids.shape or lengths.shape != (ids.shape[0],) or ids.shape[1] > self.max_bytes:
            raise ValueError('invalid verifier shapes')
        if ids.dtype != torch.long or segments.dtype != torch.long or lengths.dtype != torch.long:
            raise TypeError('verifier inputs must be long tensors')
        if bool(((ids < 0)|(ids > 255)).any()) or bool(((segments < 0)|(segments > 1)).any()) or bool(((lengths < 1)|(lengths > ids.shape[1])).any()):
            raise ValueError('invalid raw bytes, segments, or lengths')
        mask=torch.arange(ids.shape[1],device=ids.device).unsqueeze(0)<lengths.unsqueeze(1)
        x=(self.byte(ids)+self.segment(segments))*mask.unsqueeze(-1)
        # Mask between convolutions so padding cannot create a boundary signal.
        x=self.conv[1](self.conv[0](x.transpose(1,2)))*mask.unsqueeze(1)
        x=self.conv[3](self.conv[2](x))*mask.unsqueeze(1)
        candidate=mask & (segments==1)
        if bool((candidate.sum(1)==0).any()): raise ValueError('candidate must not be empty')
        mean=(x*candidate.unsqueeze(1)).sum(-1)/candidate.sum(-1,keepdim=True)
        maximum=x.masked_fill(~candidate.unsqueeze(1),float('-inf')).amax(-1)
        return self.output(torch.cat((mean,maximum),-1)).squeeze(-1)


def verifier_batch(pairs, max_bytes=128, device='cpu'):
    prepared=[]
    for prefix,chunk in pairs:
        if not isinstance(prefix,bytes) or not isinstance(chunk,bytes) or not chunk or len(chunk)>max_bytes:
            raise ValueError('verifier expects nonempty bounded candidate bytes')
        prefix=prefix[-(max_bytes-len(chunk)):] if len(chunk)<max_bytes else b''
        prepared.append((prefix+chunk, len(prefix)))
    width=max(len(x[0]) for x in prepared)
    ids=torch.zeros((len(pairs),width),dtype=torch.long,device=device); segments=torch.zeros_like(ids)
    lengths=torch.tensor([len(x[0]) for x in prepared],dtype=torch.long,device=device)
    for i,(raw,start) in enumerate(prepared):
        ids[i,:len(raw)]=torch.tensor(list(raw),device=device); segments[i,start:len(raw)]=1
    return ids,segments,lengths


class VerifiedChunkGenerator(nn.Module):
    """Wrap a decoder with a learned per-chunk stop/reject head."""
    def __init__(self, decoder, verifier):
        super().__init__(); self.decoder=decoder; self.verifier=verifier

    @torch.no_grad()
    def generate(self, context, *, threshold=.5, **kwargs):
        if not 0 <= threshold <= 1: raise ValueError('invalid threshold')
        scores=[]
        def verify(prefix, chunk):
            if not chunk: return True  # EOS/no-progress handled by the loop.
            ids,segments,lengths=verifier_batch([(prefix,chunk)],self.verifier.max_bytes,device=context.device)
            score=float(torch.sigmoid(self.verifier(ids,segments,lengths))[0]); scores.append(score)
            return score >= threshold
        result=generate_diverse_in_chunks(self.decoder,context,verify,**kwargs)
        return result,scores
