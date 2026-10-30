"""Adapter for the existing diverse byte decoder, without resetting its prefix."""
import torch
from .chunked_generation import generate_in_chunks


@torch.no_grad()
def generate_diverse_in_chunks(decoder, context, verify, *, style=0, target_length=200,
                               chunk_size=40, max_chunks=10):
    if context.shape != (1, decoder.context_dim) or not 0 <= style < decoder.num_styles:
        raise ValueError('invalid context or style')
    style_tensor = torch.tensor([style], dtype=torch.long, device=context.device)

    def next_chunk(prefix, budget):
        chunk = bytearray()
        for _ in range(budget):
            previous = torch.tensor([[decoder.START, *prefix, *chunk]], dtype=torch.long, device=context.device)
            token = int(decoder(context, style_tensor, previous)[0, -1].argmax())
            if token == decoder.EOS:
                return bytes(chunk), True
            chunk.append(token)
        return bytes(chunk), False

    return generate_in_chunks(next_chunk, verify, target_length=target_length,
                              chunk_size=chunk_size, max_chunks=max_chunks)
