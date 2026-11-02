"""Byte-level examples, bounded mixtures, deterministic synthetic data."""
import random
import re
import torch

BOS, EOS, PAD = 256, 257, 258

def encode(text, max_seq_len=1024):
    # Reserve EOS and BOS; no hidden tokenizer or Unicode codepoint conversion.
    return [BOS] + list(text.encode('utf8')[:max_seq_len - 2]) + [EOS]

def batch(texts, max_seq_len=1024):
    rows = [encode(t, max_seq_len) for t in texts]
    result = torch.full((len(rows), max(map(len, rows))), PAD, dtype=torch.long)
    for i, row in enumerate(rows): result[i, :len(row)] = torch.tensor(row)
    return result

def synthetic():
    minecraft, search, react, edit = [], [], [], []
    for x in range(-10, 11):
        minecraft.append(f'Request: Build a house at {x} 64 0\nAction: minecraft.place_block("stone", {x}, 64, 0)\nObservation: simulated placement only\nAnswer: Primeiro bloco posicionado.')
    for query in ['clima em São Paulo', 'Python documentation', 'notícias do Brasil', 'weather in São Paulo']:
        search.append(f'Request: {query}\nThought: Não tenho certeza; vou pesquisar.\nAction: search_ddg("{query}")\nObservation: [exemplo fictício; consulte fontes atuais]\nAnswer: Confira os resultados atuais antes de concluir.')
        react.append(f'Request: {query}\nThought: Preciso consultar uma fonte.\nAction: search_ddg("{query}")\nObservation: busca não executada neste exemplo\nAnswer: Não posso verificar sem resultados reais.')
    for wrong, right in [('ola','olá'), ('voce','você'), ('bom diaa','bom dia'), ('casaa','casa')]:
        # Edit offsets and replacement bytes use UTF-8 byte indices, not character indices.
        old, new = wrong.encode(), right.encode()
        start = 0
        while start < min(len(old), len(new)) and old[start] == new[start]: start += 1
        # General edits support replacement of a byte range with arbitrary UTF-8 bytes.
        edit.append(f'Request: Corrija: {wrong}\nDraft: {wrong}\n[EDIT pos={start} delete={len(old)-start} bytes={new[start:].hex()}]\nAnswer: {right}')
    return {'minecraft':minecraft, 'search':search, 'react':react, 'edit':edit}

def apply_edits(draft, commands):
    """Strictly apply one or more byte-range replacements; reject malformed UTF-8."""
    pattern = re.compile(r'\[EDIT pos=(\d+) delete=(\d+) bytes=([0-9a-f]*)\]')
    data = draft.encode('utf8')
    matches = list(pattern.finditer(commands))
    if not matches or ''.join(m.group(0) for m in matches) != commands.strip(): raise ValueError('invalid edit syntax')
    for match in matches:
        pos, delete = int(match[1]), int(match[2]); raw = bytes.fromhex(match[3])
        if pos > len(data) or pos + delete > len(data): raise ValueError('out of range')
        data = data[:pos] + raw + data[pos+delete:]
    return data.decode('utf8')

def sample_mix(groups, weights, count, rng=random):
    missing = [k for k, w in weights.items() if w and not groups.get(k)]
    if missing: raise ValueError('Missing required data groups: ' + ', '.join(missing))
    keys = list(weights); probs = list(weights.values())
    return [rng.choice(groups[k]) for k in rng.choices(keys, weights=probs, k=count)]
