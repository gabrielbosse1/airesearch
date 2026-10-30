"""Bounded incremental generation with an explicit per-chunk verifier.

Control events are metadata, not bytes. A verifier is host-supplied; it is not
assumed to be a trained cognitive model, nor to establish semantic correctness.
"""
from dataclasses import dataclass
from typing import Callable, Literal


@dataclass(frozen=True)
class ChunkResult:
    output: bytes
    halt_reason: Literal['done', 'target_length', 'max_chunks', 'rejected', 'no_progress']
    events: tuple[tuple[str, int], ...]


def generate_in_chunks(next_chunk: Callable[[bytes, int], tuple[bytes, bool]],
                       verify: Callable[[bytes, bytes], bool], *, target_length: int,
                       chunk_size: int = 40, max_chunks: int = 10) -> ChunkResult:
    """Generate output bytes only; next_chunk sees the entire accepted prefix.

    ``done`` comes from the generator's actual EOS, not literal text. A failed
    verification discards the candidate chunk. Target length counts output only.
    """
    for value in (target_length, chunk_size, max_chunks):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError('length and chunk limits must be positive integers')
    output = b''
    events = []
    for _ in range(max_chunks):
        budget = min(chunk_size, target_length - len(output))
        chunk, done = next_chunk(output, budget)
        if not isinstance(chunk, bytes) or not isinstance(done, bool) or len(chunk) > budget:
            raise ValueError('generator returned invalid chunk or EOS')
        events.append(('GENERATE_CHUNK', len(chunk)))
        accepted = verify(output, chunk)
        if not isinstance(accepted, bool):
            raise TypeError('verifier must return bool')
        events.append(('THINK', len(output) + len(chunk)))
        if not accepted:
            return ChunkResult(output, 'rejected', tuple(events))
        output += chunk
        if done:
            events.append(('DONE', len(output)))
            return ChunkResult(output, 'done', tuple(events))
        if len(output) == target_length:
            return ChunkResult(output, 'target_length', tuple(events))
        if not chunk:
            return ChunkResult(output, 'no_progress', tuple(events))
    return ChunkResult(output, 'max_chunks', tuple(events))
