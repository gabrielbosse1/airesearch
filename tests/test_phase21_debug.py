"""Phase 21 forensic-tooling regressions.

Run: .venv/bin/python -m pytest tests/test_phase21_debug.py -q
"""
from __future__ import annotations

import pytest

from debug.analyze_training_logs import (check_lr_schedule, detect_collapse,
                                         detect_nan_inf, detect_spikes,
                                         parse_text)
from debug.common import coherence, md_table
from debug.test_byte_encoding import test_utf8_round_trip  # noqa: F401 (re-exported check)
from notebook.train import MIXES, balanced_weights, draw, stage_mixture


def _groups():
    return {'big': [f'row-{i}' for i in range(200)],
            'small': ['a', 'b', 'c', 'd'],
            'pt': [f'pt-{i}' for i in range(100)],
            'en': [f'en-{i}' for i in range(100)]}


def test_utf8_ola_roundtrip():
    assert list('Olá'.encode('utf8')) == [79, 108, 195, 161]
    assert bytes([79, 108, 195, 161]).decode('utf8') == 'Olá'


def test_balanced_weights_caps_tiny_group():
    weights = {'big': 50, 'small': 50}
    capped = balanced_weights(_groups(), weights)
    total = sum(capped.values())
    assert abs(total - 100) < 1e-6
    assert capped['small'] / total < 0.10, capped
    assert capped['big'] > capped['small'] * 5


def test_balanced_weights_leaves_diverse_mix_untouched():
    groups = {'big': _groups()['big'], 'pt': _groups()['pt']}
    capped = balanced_weights(groups, {'big': 50, 'pt': 50})
    assert abs(capped['big'] - 50) < 1e-6
    assert abs(capped['pt'] - 50) < 1e-6


def test_stage_mixture_replays_language_in_stages_3_and_4():
    groups = _groups()
    for stage, weights in enumerate(MIXES[:2]):
        assert stage_mixture(groups, weights, stage) == balanced_weights(groups, weights)
    for stage in (2, 3):
        fixed = stage_mixture(groups, {'small': 50, 'big': 50}, stage)
        language = sum(fixed.get(k, 0) for k in ('pt', 'en'))
        assert abs(language / sum(fixed.values()) - 0.4) < 1e-6, fixed


def test_draw_balanced_actually_samples_diverse_groups():
    groups = {'big': [f'row-{i}' for i in range(200)], 'small': ['a', 'b', 'c', 'd']}
    weights = {'big': 50, 'small': 50}
    legacy = draw(groups, weights, 200, balanced=False)
    fixed = draw(groups, weights, 200, balanced=True)
    assert legacy.count('a') + legacy.count('b') + legacy.count('c') + legacy.count('d') > 60
    assert fixed.count('a') + fixed.count('b') + fixed.count('c') + fixed.count('d') < 40


def test_coherence_separates_portuguese_from_random_bytes():
    good = coherence('estou bem, obrigado, como você está hoje'.encode('utf8'))
    bad = coherence(bytes([0xC3, 0x28, 0xFF, 0x00, 0x9A, 0x80, 0x13] * 12))
    assert good['valid_utf8'] and not good['gibberish']
    assert not bad['valid_utf8'] and bad['gibberish']
    assert good['score'] > bad['score'] + 0.3


def test_log_parser_and_detectors():
    text = '\n'.join(f'Stage 1: loss={2.0 + i * 0.001:.4f} step={i} lr={0.003 - i * 1e-6:.6f}'
                     for i in range(30))
    text += '\nStage 1: loss=nan step=31 lr=0.0029\n'
    records = parse_text(text)
    assert len(records) == 31
    assert detect_nan_inf(records)['count'] == 1
    assert detect_spikes(records)['count'] == 0
    assert check_lr_schedule(records)['schedule_failed'] is False
    assert detect_collapse(records)['suspiciously_early_collapse'] is False


def test_spike_detector_flags_explosion():
    text = '\n'.join(
        f'Stage 2: loss={1.0:.4f} step={i} lr=0.001' for i in range(20))
    text += '\nStage 2: loss=9.0 step=20 lr=0.001\n'
    spikes = detect_spikes(parse_text(text))
    assert spikes['count'] >= 1
    assert spikes['gradient_explosion_in_loss']


def test_training_data_pipeline_is_clean():
    from debug.verify_training_data import _pipeline_roundtrip
    report = _pipeline_roundtrip({'pt': ['Olá, como você está?', 'Não tenho certeza.'],
                                  'code': ['def f(x):\n    return x\n']}, limit=5)
    assert report['problem_count'] == 0
    assert report['padded_batch_ok']
    assert report['max_token_id'] <= 258


def test_mixture_exposure_identifies_stages_3_and_4():
    from debug.verify_training_data import _mixture_exposure
    groups = {'pt': [f'p{i}' for i in range(1000)], 'en': [f'e{i}' for i in range(1000)],
              'code': [f'c{i}' for i in range(1000)], 'react': [f'r{i}' for i in range(1000)],
              'search': ['s1', 's2', 's3', 's4'], 'edit': ['x1', 'x2', 'x3', 'x4'],
              'mineflayer': [f'm{i}' for i in range(521)]}
    stages = _mixture_exposure(groups)
    assert stages[2]['max_over_min_exposure_ratio'] > 100
    assert stages[3]['max_over_min_exposure_ratio'] > 100


def test_md_table_shape():
    table = md_table(['a', 'b'], [[1, 2], [3, 4]])
    assert len(table.splitlines()) == 4
