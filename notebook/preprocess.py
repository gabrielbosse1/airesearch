"""Byte-level examples, bounded mixtures, deterministic synthetic data."""
import random
import re

import torch

BOS, EOS, PAD = 256, 257, 258


def _utf8_prefix(text: str, byte_limit: int) -> bytes:
    """Return a complete-codepoint UTF-8 prefix of ``text`` within ``byte_limit``.

    The model works on bytes, but its source corpora are declared to be UTF-8.
    Cutting the final byte sequence in the middle of a codepoint teaches malformed
    fragments at every sequence boundary, so drop only that incomplete character.
    """
    if not isinstance(text, str):
        raise TypeError('text must be a str')
    raw = text.encode('utf8')
    if len(raw) <= byte_limit:
        return raw
    raw = raw[:byte_limit]
    # ``text`` was a Python str, hence valid UTF-8 before truncation. Any decode
    # error can only be the final partial codepoint.
    while raw:
        try:
            raw.decode('utf8')
            return raw
        except UnicodeDecodeError as exc:
            raw = raw[:exc.start]
    return raw


def encode(text, max_seq_len=1024):
    """UTF-8 encode one record with BOS/EOS without splitting a codepoint."""
    if not isinstance(max_seq_len, int) or max_seq_len < 2:
        raise ValueError('max_seq_len must be an integer of at least 2')
    return [BOS] + list(_utf8_prefix(text, max_seq_len - 2)) + [EOS]


def batch(texts, max_seq_len=1024):
    """Pad a non-empty batch of UTF-8 records to its local maximum length."""
    if not texts:
        raise ValueError('texts must contain at least one record')
    rows = [encode(t, max_seq_len) for t in texts]
    result = torch.full((len(rows), max(map(len, rows))), PAD, dtype=torch.long)
    for i, row in enumerate(rows):
        result[i, :len(row)] = torch.tensor(row, dtype=torch.long)
    return result


def synthetic():
    """Small, labelled task fixtures; never used as a language-data fallback."""
    minecraft, search, react, edit = [], [], [], []
    for x in range(-10, 11):
        minecraft.append(f'Request: Build a house at {x} 64 0\nAction: minecraft.place_block("stone", {x}, 64, 0)\nObservation: simulated placement only\nAnswer: Primeiro bloco posicionado.')
    for query in ['clima em São Paulo', 'Python documentation', 'notícias do Brasil', 'weather in São Paulo']:
        search.append(f'Request: {query}\nThought: Não tenho certeza; vou pesquisar.\nAction: search_ddg("{query}")\nObservation: [exemplo fictício; consulte fontes atuais]\nAnswer: Confira os resultados atuais antes de concluir.')
        react.append(f'Request: {query}\nThought: Preciso consultar uma fonte.\nAction: search_ddg("{query}")\nObservation: busca não executada neste exemplo\nAnswer: Não posso verificar sem resultados reais.')
    for wrong, right in [('ola', 'olá'), ('voce', 'você'), ('bom diaa', 'bom dia'), ('casaa', 'casa')]:
        # Edit offsets and replacement bytes use UTF-8 byte indices, not character indices.
        old, new = wrong.encode(), right.encode()
        start = 0
        while start < min(len(old), len(new)) and old[start] == new[start]:
            start += 1
        # General edits support replacement of a byte range with arbitrary UTF-8 bytes.
        edit.append(f'Request: Corrija: {wrong}\nDraft: {wrong}\n[EDIT pos={start} delete={len(old)-start} bytes={new[start:].hex()}]\nAnswer: {right}')
    return {'minecraft': minecraft, 'search': search, 'react': react, 'edit': edit}


def apply_edits(draft, commands):
    """Strictly apply one or more byte-range replacements; reject malformed UTF-8."""
    pattern = re.compile(r'\[EDIT pos=(\d+) delete=(\d+) bytes=([0-9a-f]*)\]')
    data = draft.encode('utf8')
    matches = list(pattern.finditer(commands))
    if not matches or ''.join(m.group(0) for m in matches) != commands.strip():
        raise ValueError('invalid edit syntax')
    for match in matches:
        pos, delete = int(match[1]), int(match[2])
        raw = bytes.fromhex(match[3])
        if pos > len(data) or pos + delete > len(data):
            raise ValueError('out of range')
        data = data[:pos] + raw + data[pos + delete:]
    return data.decode('utf8')


def sample_mix(groups, weights, count, rng=random):
    """Sample only non-empty, positively weighted groups."""
    if not isinstance(count, int) or count < 0:
        raise ValueError('count must be a non-negative integer')
    if not weights:
        raise ValueError('weights must not be empty')
    invalid = [k for k, weight in weights.items()
               if not isinstance(weight, (int, float)) or weight < 0]
    if invalid:
        raise ValueError('weights must be non-negative numbers: ' + ', '.join(map(str, invalid)))
    required = [k for k, weight in weights.items() if weight > 0]
    if not required:
        raise ValueError('at least one mixture weight must be positive')
    missing = [k for k in required if not groups.get(k)]
    if missing:
        raise ValueError('Missing required data groups: ' + ', '.join(missing))
    if count == 0:
        return []
    return [rng.choice(groups[k]) for k in rng.choices(required,
            weights=[weights[k] for k in required], k=count)]
