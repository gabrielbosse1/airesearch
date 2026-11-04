"""Shared helpers for the Phase 21 collapse investigation.

Design rule: this module must import with the standard library alone. PyTorch is
imported lazily inside the functions that need a real checkpoint, so the log,
data and encoding forensics still run on a machine without a GPU or torch.

Everything written here is a *diagnostic*, not a language-quality benchmark.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / 'runs' / 'phase21'
CHECKPOINTS = REPORTS / 'checkpoints'
REPLAY = REPORTS / 'replay'

# Colab artifacts are deliberately gitignored. These are the paths the Phase 19
# notebook writes inside /content/airesearch.
COLAB_ARTIFACTS = (
    'data_cache/samples.json',
    'data_cache/manifest.json',
    'data_cache/mineflayer_trajectories.jsonl',
    'checkpoints',
    'final_model_fp16.pt',
    'final_model/config.json',
    'runs/train.log',
)

PT_WORDS = {
    'olá', 'ola', 'como', 'você', 'voce', 'estou', 'bem', 'obrigado', 'obrigada',
    'bom', 'dia', 'não', 'nao', 'está', 'esta', 'são', 'sao', 'para', 'com',
    'mais', 'mas', 'isso', 'meu', 'minha', 'uma', 'dos', 'das', 'português',
    'brasil', 'aqui', 'fazer', 'pode', 'tem', 'ele', 'ela', 'nós', 'também',
    'sobre', 'quando', 'quem', 'qual', 'onde', 'ser', 'ter', 'sendo', 'entre',
    'tudo', 'hoje', 'está', 'vai', 'vamos', 'posso', 'gosto', 'animal',
    'favorito', 'capivara', 'resposta', 'certo', 'sim', 'então', 'porque',
    'estão', 'me', 'seu', 'sua', 'nos', 'um', 'ou', 'de', 'da', 'do', 'em',
}

MOJIBAKE = re.compile(r'(?:Ã|Â|Ð|Ñ)[\x80-\xbf]|Ã¢|Ã£|Ã¡|Ã©|Ãª|Ã­|Ã³|Ãµ|Âº|Âª|\ufffd')


def report_dir() -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    return REPORTS


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf8'))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return default


def write_json(path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf8')
    return path


def write_text(path, text) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if isinstance(text, str) else str(text), encoding='utf8')
    return path


def artifact_status() -> dict:
    """Which Phase 19 Colab artifacts are actually available in this checkout?"""
    found = {}
    for relative in COLAB_ARTIFACTS:
        path = ROOT / relative
        found[relative] = {'exists': path.exists(),
                           'size_bytes': path.stat().st_size if path.is_file() else None}
    return {
        'artifacts': found,
        'any_colab_artifact': any(row['exists'] for row in found.values()),
        'note': ('Phase 19 Colab artifacts are gitignored (.gitignore: data_cache/, '
                 'checkpoints/, final_model/, final_model_fp16.pt). They live only in the '
                 'ephemeral Colab runtime and are NOT in this checkout. Any conclusion '
                 'about the 4-hour run must be derived from source forensics plus a '
                 'deterministic local replay, or from artifacts the user re-uploads.'),
    }


def _entropy(values) -> float:
    counts = Counter(values)
    n = len(values) or 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def coherence(byte_ids) -> dict:
    """Score a raw byte continuation from a Portuguese prompt.

    Returns individual, inspectable signals plus a scalar ``score`` in [0, 1].
    A random byte decoder produces invalid UTF-8, many U+FFFD replacements, a
    high share of bytes >= 0x80 and near-maximal byte entropy. A coherent
    Portuguese continuation does the opposite. This is a heuristic: it rewards
    plausible Portuguese byte statistics, not semantics.
    """
    ids = [int(b) for b in byte_ids]
    oob = [b for b in ids if b < 0 or b > 255]
    raw = bytes(b for b in ids if 0 <= b <= 255)
    try:
        text = raw.decode('utf8')
        valid_utf8 = True
    except UnicodeDecodeError:
        text = raw.decode('utf8', errors='replace')
        valid_utf8 = False

    n_chars = max(1, len(text))
    replacement_ratio = text.count('\ufffd') / n_chars
    printable = sum(1 for ch in text
                    if ch.isprintable() or ch in '\n\t\r') / n_chars
    words = re.findall(r"[A-Za-zÀ-ÿ']{2,}", text.lower())
    hits = sum(1 for word in words if word in PT_WORDS)
    pt_hit = min(1.0, hits / 2.0)
    high_bytes = sum(1 for b in raw if b >= 128) / max(1, len(raw))
    control_bytes = sum(1 for b in raw if b < 32 and b not in (9, 10, 13)) / max(1, len(raw))
    entropy = _entropy(raw) / 8.0 if raw else 0.0
    trigrams = [tuple(raw[i:i + 3]) for i in range(max(0, len(raw) - 2))]
    distinct_trigrams = len(set(trigrams)) / max(1, len(trigrams))
    runs = [len(m.group()) for m in re.finditer(r'(.)\1{3,}', text)]
    max_run = max(runs) if runs else 0

    score = (0.35 * float(valid_utf8)
             + 0.25 * printable
             + 0.25 * pt_hit
             + 0.15 * (1.0 - min(1.0, entropy / 0.75)))
    score *= 0.5 + 0.5 * distinct_trigrams
    score = round(max(0.0, min(1.0, score)), 4)
    gibberish = (not valid_utf8) or printable < 0.65 or (pt_hit == 0 and len(text) > 12)
    return {
        'n_bytes': len(raw),
        'text': text[:400],
        'valid_utf8': valid_utf8,
        'replacement_ratio': round(replacement_ratio, 4),
        'printable_ratio': round(printable, 4),
        'pt_word_hits': hits,
        'pt_word_score': round(pt_hit, 4),
        'high_byte_ratio': round(high_bytes, 4),
        'control_byte_ratio': round(control_bytes, 4),
        'byte_entropy_bits_per_byte': round(entropy * 8, 4),
        'distinct_trigram_ratio': round(distinct_trigrams, 4),
        'max_repeat_run': max_run,
        'out_of_range_ids': len(oob),
        'score': score,
        'gibberish': bool(gibberish),
    }


def generate_raw(model, prompt, max_new=100, sample=False, temperature=1.0,
                 mode='faithful'):
    """Deterministic byte-level generation with full diagnostics.

    ``mode='faithful'`` reproduces notebook.phone_model.generate: mask ids >= PAD,
    stop on EOS, but ALSO records never-trained control ids instead of crashing
    in ``bytes()``. ``mode='byte_only'`` restricts decoding to real bytes 0..255.
    ``mode='raw'`` applies no masking at all, exposing what the logits actually
    prefer (including BOS=256, which is never a training target).
    """
    import torch
    from notebook.preprocess import BOS, EOS, PAD

    if mode not in ('faithful', 'byte_only', 'raw'):
        raise ValueError("mode must be 'faithful', 'byte_only' or 'raw'")
    cfg = model.cfg
    prompt_bytes = list(prompt.encode('utf8'))[-(cfg.max_seq_len - 1):]
    ids = [BOS] + prompt_bytes
    emitted = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(min(max_new, cfg.max_seq_len - len(ids))):
                logits = model(torch.tensor([ids], dtype=torch.long,
                                            device=next(model.parameters()).device))[0, -1].float()
                if temperature != 1.0:
                    logits = logits / float(temperature)
                if mode == 'faithful':
                    logits[PAD:] = -float('inf')
                elif mode == 'byte_only':
                    logits[256:] = -float('inf')
                nxt = (int(torch.multinomial(torch.softmax(logits, -1), 1))
                       if sample else int(logits.argmax()))
                if mode == 'faithful' and nxt == EOS:
                    break
                ids.append(nxt)
                emitted.append(nxt)
    finally:
        model.train(was_training)

    control_ids = [i for i in emitted if i > 255]
    byte_ids = [i for i in emitted if 0 <= i <= 255]
    result = {
        'prompt': prompt, 'mode': mode, 'sample': bool(sample),
        'n_emitted': len(emitted), 'n_bytes': len(byte_ids),
        'eos_at': emitted.index(EOS) if EOS in emitted else None,
        'bos_count': emitted.count(BOS),
        'control_ids': control_ids[:20], 'control_id_count': len(control_ids),
        'byte_ids': byte_ids,
    }
    result.update(coherence(byte_ids))
    return result


def original_generate(model, prompt, max_new=100):
    """Call the shipped generator verbatim and capture any crash as evidence."""
    from notebook.phone_model import generate
    try:
        return {'ok': True, 'text': generate(model, prompt, max_new=max_new)}
    except Exception as exc:  # bytes([256]) raises ValueError; record it
        return {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}


def model_from_checkpoint(path, device='cpu'):
    """Load a final state dict or a full resume snapshot into a real PhoneLM.

    Only load checkpoints you trust: torch.load can execute arbitrary code.
    Strict loading is attempted first; a shape mismatch is reported loudly
    instead of being silently filled with random initialisation.
    """
    import torch
    from dataclasses import asdict
    from notebook.phone_model import Config, PhoneLM

    payload = torch.load(str(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f'{path}: unsupported checkpoint payload')
    state = payload.get('model', payload)
    config_data = payload.get('config')
    if isinstance(config_data, dict):
        allowed = set(asdict(Config()))
        config_data = {k: v for k, v in config_data.items() if k in allowed}
        cfg = Config(**config_data)
    else:
        cfg = Config()
    model = PhoneLM(cfg)
    load_warning = None
    try:
        model.load_state_dict(state, strict=True)
        missing, unexpected = [], []
    except RuntimeError as exc:
        load_warning = f'strict load failed ({str(exc)[:200]}); fell back to non-strict'
        missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return model, {'missing_keys': list(missing), 'unexpected_keys': list(unexpected),
                   'load_warning': load_warning,
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'config': asdict(cfg)}

def find_checkpoint(explicit=None, variant='baseline', stage=3):
    """Pick a checkpoint to analyse, preferring an explicit user-supplied path.

    ``runs/model.pt`` is a Phase 5 checkpoint with a DIFFERENT architecture; it
    is deliberately excluded so a forensic run can never silently analyse an
    unrelated model with randomly initialised weights.
    """
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    candidates = [
        ROOT / 'final_model_fp16.pt',
        CHECKPOINTS / f'{variant}_stage{stage + 1}.pt',
    ]
    for path in candidates:
        if path.exists():
            return path
    replay_files = sorted(CHECKPOINTS.glob(f'{variant}_stage*.pt'))
    if replay_files:
        return replay_files[-1]
    others = sorted(CHECKPOINTS.glob('*.pt'))
    return others[-1] if others else None

def md_table(headers, rows) -> str:
    out = ['| ' + ' | '.join(str(h) for h in headers) + ' |',
           '| ' + ' | '.join('---' for _ in headers) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(str(v) for v in row) + ' |')
    return '\n'.join(out)
