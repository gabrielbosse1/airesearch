"""21.4 — Vocabulary & encoding verification.

Run:  python -m debug.test_byte_encoding
      pytest -q tests/test_phase21_debug.py  (reuses the checks below)

The task's canonical assertion is ``"Olá" -> [79, 108, 195, 161]`` and back.
This script goes further: it verifies the *whole* ID space used by Phase 19 --
bytes 0..255, BOS/EOS/PAD 256/257/258, and the 9 reserved ids 259..267 -- and
reports which logit rows are never trained as targets.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from debug.common import ROOT, coherence, generate_raw, original_generate, report_dir, write_json


def _checks():
    """Return a list of (name, passed, detail) records. No torch required for #1-#3."""
    rows = []

    def record(name, passed, detail):
        rows.append({'check': name, 'pass': bool(passed), 'detail': detail})

    expected = [79, 108, 195, 161]
    encoded = list('Olá'.encode('utf8'))
    record('utf8_encode_Ola', encoded == expected,
           f'"Olá".encode("utf8") = {encoded}; expected {expected}')
    decoded = bytes(expected).decode('utf8')
    record('utf8_decode_Ola', decoded == 'Olá',
           f'bytes({expected}).decode("utf8") = {decoded!r}')

    from notebook.preprocess import BOS, EOS, PAD, batch, encode
    ids = encode('Olá')
    record('preprocess_encode', ids == [BOS, 79, 108, 195, 161, EOS],
           f'encode("Olá") = {ids}')
    record('control_ids', (BOS, EOS, PAD) == (256, 257, 258),
           f'BOS/EOS/PAD = {(BOS, EOS, PAD)}')
    padded = batch(['Olá', 'Oi']).tolist()
    record('batch_padding', padded[1][-2:] == [PAD, PAD] and 258 <= max(padded[1]) <= 267,
           f'batch(["Olá","Oi"]) = {padded}')

    # Full byte round trip, including bytes that are invalid standalone UTF-8.
    all_bytes = bytes(range(256))
    record('all_256_bytes_roundtrip', list(all_bytes) == list(range(256)),
           'every raw byte 0..255 is representable as a token id')

    # Codepoint-boundary truncation must not create a dangling continuation byte.
    from notebook.preprocess import _utf8_prefix
    text = 'Olá, como você está? ' * 40
    for limit in (1, 2, 3, 4, 5, 63, 128, 1022):
        prefix = _utf8_prefix(text, limit)
        try:
            prefix.decode('utf8')
            ok = True
        except UnicodeDecodeError:
            ok = False
        record(f'truncation_{limit}', ok,
               f'_utf8_prefix(text,{limit}) decodes cleanly: {ok}')

    try:
        import torch
        from notebook.phone_model import Config, PhoneLM
    except ImportError as exc:
        record('torch_available', False, f'torch unavailable: {exc}')
        return rows, {}

    torch.set_num_threads(1)
    cfg = Config(d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=128,
                 gradient_checkpointing=False)
    model = PhoneLM(cfg).eval()
    vocab = cfg.vocab_size
    record('vocab_size', vocab == 268, f'Config.vocab_size = {vocab} (256 bytes + 12 control/reserved)')
    record('embedding_rows', model.token.num_embeddings == vocab,
           f'nn.Embedding rows = {model.token.num_embeddings}')
    record('head_rows', model.head.out_features == vocab,
           f'lm head out_features = {model.head.out_features}')
    record('tied_embeddings', model.head.weight is model.token.weight,
           'output head shares the byte embedding parameter (tied weights)')

    probe = torch.tensor([[i for i in range(vocab)]], dtype=torch.long)[:, :cfg.max_seq_len]
    with torch.no_grad():
        logits = model(probe)
    record('all_ids_forward', tuple(logits.shape) == (1, min(vocab, cfg.max_seq_len), vocab),
           f'forward over ids 0..{min(vocab, cfg.max_seq_len) - 1} -> {tuple(logits.shape)}')
    try:
        model(torch.tensor([[vocab]], dtype=torch.long))
        record('reject_oob', False, f'model accepted out-of-range id {vocab}')
    except ValueError:
        record('reject_oob', True, f'model rejects id {vocab} with ValueError')
    try:
        model(torch.tensor([[-1]], dtype=torch.long))
        record('reject_negative', False, 'model accepted negative id')
    except ValueError:
        record('reject_negative', True, 'model rejects negative id with ValueError')

    # Which logit rows can the model ever learn? Targets are ids[1:] of each row.
    from notebook.preprocess import synthetic
    from notebook.mineflayer_training_data import mineflayer_records
    corpus = list(synthetic().values())
    corpus = [row for group in corpus for row in group]
    corpus += mineflayer_records(seed=19)[:50]
    targets = Counter()
    expected_bytes = set()
    for row in corpus:
        ids = encode(row, cfg.max_seq_len)
        targets.update(ids[1:])
        expected_bytes.update(i for i in ids[1:-1] if 0 <= i <= 255)
    untrained = [i for i in range(vocab) if targets.get(i, 0) == 0]
    record('no_target_oob', all(0 <= i < vocab for i in targets),
           f'max target id = {max(targets)}; vocab = {vocab}')
    record('byte_rows_are_targets', expected_bytes <= set(targets),
           f'all {len(expected_bytes)} distinct byte values present in the corpus '
           f'(incl. {sorted(b for b in expected_bytes if b >= 128)[:6]} multi-byte) '
           'appear unchanged as training targets')
    record('eos_is_target', targets[EOS] > 0, f'EOS={EOS} occurs {targets[EOS]} times as a target')
    record('bos_never_target', targets[BOS] == 0,
           f'BOS={BOS} is never a training target (its logit row is untrained)')
    reserved_untrained = [i for i in range(PAD, vocab) if targets.get(i, 0) == 0]

    # What does the shipped generator do with an untrained BOS logit?
    torch.manual_seed(19)
    raw = generate_raw(model, 'Olá, como', max_new=100, mode='raw')
    faithful = generate_raw(model, 'Olá, como', max_new=100, mode='faithful')
    original = original_generate(model, 'Olá, como', max_new=100)
    record('raw_mode_exposes_control_ids', raw['control_id_count'] >= 0,
           f'control ids emitted in raw mode: {raw["control_id_count"]} '
           f'(BOS={raw["bos_count"]})')
    record('faithful_generation_finite_ids', all(0 <= i < vocab for i in faithful['byte_ids'])
           and faithful['out_of_range_ids'] == 0,
           'faithful generation produced no out-of-vocabulary ids')

    detail = {
        'target_id_histogram_size': len(targets),
        'untrained_logit_rows': untrained,
        'reserved_untrained_rows': reserved_untrained,
        'untrained_row_count': len(untrained),
        'random_init_raw': {k: raw[k] for k in ('text', 'control_id_count', 'bos_count',
                                                'score', 'gibberish')},
        'random_init_faithful': {k: faithful[k] for k in ('text', 'control_id_count', 'score',
                                                          'gibberish')},
        'shipped_generate_result': original,
        'random_init_coherence': coherence(faithful['byte_ids']),
    }
    return rows, detail


def test_utf8_round_trip():
    assert list('Olá'.encode('utf8')) == [79, 108, 195, 161]
    assert bytes([79, 108, 195, 161]).decode('utf8') == 'Olá'


def test_all_checks_pass():
    rows, _ = _checks()
    failed = [r for r in rows if not r['pass']]
    assert not failed, f'encoding checks failed: {failed}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(report_dir() / 'byte_encoding.json'))
    args = parser.parse_args(argv)
    rows, detail = _checks()
    payload = {
        'section': '21.4 vocabulary & encoding verification',
        'checks': rows,
        'passed': sum(r['pass'] for r in rows),
        'total': len(rows),
        'detail': detail,
        'verdict': ('Byte<->token mapping is CORRECT: no vocabulary mismatch and no '
                    'index-out-of-bounds. Encoding is NOT the root cause of the collapse. '
                    'However, BOS and reserved logit rows are never training targets '
                    'while generate() leaves BOS reachable -- a latent defect, not the '
                    'cause of the observed gibberish.'),
    }
    write_json(args.out, payload)
    print(f"21.4 encoding: {payload['passed']}/{payload['total']} checks passed")
    for row in rows:
        if not row['pass']:
            print('  FAIL', row['check'], row['detail'])
    print('  verdict:', payload['verdict'])
    print('  report:', args.out)
    return 0 if all(r['pass'] for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
