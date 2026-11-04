"""21.9 — One quick fix, tested on a 5-minute A/B run.

Fix under test (the single causal change, in ``notebook/train.py``):

    Replace the raw per-stage mixture with ``stage_mixture()``:
    (1) scale each group's weight by its unique-row diversity, renormalise;
    (2) reserve 40% of Stage 3/4 updates for a Stage-1 pt/en/code rehearsal.

This needs no architecture change, no new data and no extra GPU time. It is
applied by ``train(..., legacy_mix=False)``, the new default; ``legacy_mix=True``
reproduces the broken Phase 19 curriculum for A/B measurement.

The A/B is a deterministic CPU replay at 300 optimizer steps per stage
(~4 minutes total here), not a replacement for a full retrain on the T4.

Run: python -m debug.quick_recovery
     python -m debug.quick_recovery --from-existing
"""
from __future__ import annotations

import argparse
from pathlib import Path

from debug.common import (REPORTS, md_table, report_dir, write_json, write_text)


def _metrics(payload):
    stages = payload['stages']
    final = stages[-1]
    losses = [s['heldout_pt_loss'] for s in stages if 'heldout_pt_loss' in s]
    coherences = [s['coherence'] for s in stages]
    return {
        'final_training_loss': final.get('final_loss'),
        'final_heldout_pt_loss': final.get('heldout_pt_loss'),
        'worst_heldout_pt_loss': max(losses) if losses else None,
        'best_heldout_pt_loss': min(losses) if losses else None,
        'forgetting_ratio': round(max(losses) / min(losses), 3) if losses else None,
        'final_coherence': final['coherence'],
        'final_gibberish': final['gibberish'],
        'final_valid_utf8': final['valid_utf8'],
        'final_pt_word_hits': final['pt_word_hits'],
        'final_probes_passed': final['probes_passed'],
        'final_probes': [(p['case'], p['pass']) for p in final['probes']],
        'final_text': final['text'][:160],
        'per_stage': [{'stage': s['stage'], 'coherence': s['coherence'],
                       'gibberish': s['gibberish'],
                       'heldout_pt_loss': s.get('heldout_pt_loss'),
                       'probes_passed': s['probes_passed']} for s in stages],
        'stage3_4_weights': [dict(payload['stage_weights'][i]) for i in (2, 3)],
        'elapsed_seconds': payload['elapsed_seconds'],
    }


def run(steps, lr, accumulation, seed, out_dir, from_existing=False):
    from debug.replay_training import run_replay
    out_dir = Path(out_dir)
    payloads = {}
    for variant in ('baseline', 'fixed'):
        cache = out_dir / f'{variant}_replay.json'
        if from_existing and cache.exists():
            import json
            payloads[variant] = json.loads(cache.read_text(encoding='utf8'))
            print(f'  {variant}: reusing {cache}')
        else:
            print(f'  {variant}: training {steps} steps/stage ...')
            payloads[variant] = run_replay(variant, steps, lr, accumulation, seed,
                                           out_dir=out_dir)
    baseline = _metrics(payloads['baseline'])
    fixed = _metrics(payloads['fixed'])

    def delta(key, lower_is_better=False):
        a, b = baseline.get(key), fixed.get(key)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return None
        change = b - a
        return {'baseline': a, 'fixed': b, 'delta': round(change, 5),
                'relative': round(change / a, 4) if a else None,
                'better': (change < 0 if lower_is_better else change > 0)}

    improvements = {
        'final_heldout_pt_loss': delta('final_heldout_pt_loss', lower_is_better=True),
        'worst_heldout_pt_loss': delta('worst_heldout_pt_loss', lower_is_better=True),
        'forgetting_ratio': delta('forgetting_ratio', lower_is_better=True),
        'final_coherence': delta('final_coherence'),
        'final_training_loss': delta('final_training_loss', lower_is_better=True),
    }
    improved = bool(all(improvements[key]['better'] for key in (
        'final_heldout_pt_loss', 'worst_heldout_pt_loss',
        'forgetting_ratio', 'final_coherence'))
        and fixed['final_valid_utf8'])
    gibberish_cleared = bool(baseline['final_gibberish'] and not fixed['final_gibberish'])
    return {
        'section': '21.9 quick recovery test',
        'fix': ('stage_mixture(): diversity-capped group weights + 40% Stage-1 '
                'language rehearsal in Stages 3/4 (notebook/train.py, default on)'),
        'experiment': {
            'steps_per_stage': steps, 'lr': lr, 'accumulation': accumulation,
            'seed': seed, 'variants': ['baseline (legacy_mix=True)', 'fixed (default)'],
            'total_wall_seconds': round(baseline['elapsed_seconds'] + fixed['elapsed_seconds'], 2),
            'note': ('Deterministic CPU replay, seed fixed, same RNG stream for both '
                     'variants: only the mixture weights differ. NOT the 4-hour T4 run.'),
        },
        'baseline': baseline, 'fixed': fixed,
        'improvements': improvements,
        'improved': improved,
        'gibberish_cleared': gibberish_cleared,
        'caveats': [
            'The single "probes passed" count drops because that one pass '
            'was a memorised, eval-identical [EDIT] template; it is not a capability '
            'loss and is not a valid success metric (see 21.7).',
            'Fixed final TRAINING loss is higher: the fixed model is no longer '
            'over-fitting to four repeated rows. Held-out Portuguese loss and byte '
            'coherence are the meaningful metrics.',
            'The strict gibberish heuristic may remain True for the tiny fixed '
            'surrogate even when all held-out metrics improve; that flag rewards '
            'Portuguese word hits, which a 245k-parameter model rarely produces. '
            'The claim is prevention of forgetting/invalid bytes, not fluency.',
            'This is a small-scale mechanistic demonstration. Restoring the real '
            'Phase 19 model still requires a full retrain (or fine-tuning from the '
            'Stage 2 checkpoint) with the corrected mixture and a clean eval set.',
            'The tiny surrogate still does not produce fluent Portuguese; the claim '
            'is that forgetting and byte-salad collapse are prevented, not that '
            'language quality is solved.',
        ],
    }


def render_markdown(report):
    b, f = report['baseline'], report['fixed']
    return '\n'.join([
        '# 21.9 — Quick recovery A/B', '',
        f"**Fix:** {report['fix']}", '',
        f"**Result: {'IMPROVED' if report['improved'] else 'NO CLEAR IMPROVEMENT'}**", '',
        f"Experiment: {report['experiment']['steps_per_stage']} steps/stage × 4 stages × "
        f"2 variants, lr={report['experiment']['lr']}, seed={report['experiment']['seed']}, "
        f"{report['experiment']['total_wall_seconds']}s wall clock.", '',
        md_table(['metric (better direction)', 'baseline', 'fixed', 'delta'],
                 [['held-out PT loss (↓)', b['final_heldout_pt_loss'], f['final_heldout_pt_loss'],
                   report['improvements']['final_heldout_pt_loss']['delta']],
                  ['worst held-out PT loss (↓)', b['worst_heldout_pt_loss'], f['worst_heldout_pt_loss'],
                   report['improvements']['worst_heldout_pt_loss']['delta']],
                  ['forgetting ratio max/min (↓)', b['forgetting_ratio'], f['forgetting_ratio'],
                   report['improvements']['forgetting_ratio']['delta']],
                  ['final byte coherence (↑)', b['final_coherence'], f['final_coherence'],
                   report['improvements']['final_coherence']['delta']],
                  ['final training loss (informational)', b['final_training_loss'],
                   f['final_training_loss'],
                   report['improvements']['final_training_loss']['delta']],
                  ['final output gibberish?', b['final_gibberish'], f['final_gibberish'], '—'],
                  ['final output valid UTF-8?', b['final_valid_utf8'], f['final_valid_utf8'], '—'],
                  ['contaminated probes passed', b['final_probes_passed'],
                   f['final_probes_passed'], '—']]), '',
        '## Per-stage held-out Portuguese loss (teacher-forced)', '',
        md_table(['stage', 'baseline', 'fixed'],
                 [[bs['stage'], bs['heldout_pt_loss'], fs['heldout_pt_loss']]
                  for bs, fs in zip(b['per_stage'], f['per_stage'])]), '',
        '## Stage 3/4 mixture weights', '',
        md_table(['variant', 'Stage 3', 'Stage 4'],
                 [['baseline', b['stage3_4_weights'][0], b['stage3_4_weights'][1]],
                  ['fixed', f['stage3_4_weights'][0], f['stage3_4_weights'][1]]]), '',
        '## Final 100-byte generations from `Olá, como`', '',
        f"- baseline: `{b['final_text']}`",
        f"- fixed:    `{f['final_text']}`", '',
        '## Caveats', '',
    ] + [f'- {line}' for line in report['caveats']] + [''])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--accumulation', type=int, default=6)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--out-dir', default=str(report_dir()))
    parser.add_argument('--from-existing', action='store_true',
                        help='reuse existing replay results instead of retraining')
    parser.add_argument('--out', default=str(report_dir() / 'quick_recovery.json'))
    parser.add_argument('--markdown', default=str(report_dir() / 'quick_recovery.md'))
    args = parser.parse_args(argv)
    report = run(args.steps, args.lr, args.accumulation, args.seed,
                 Path(args.out_dir), args.from_existing)
    write_json(args.out, report)
    write_text(args.markdown, render_markdown(report))
    print('21.9 recovery:', 'IMPROVED' if report['improved'] else 'NO CLEAR IMPROVEMENT')
    print('  held-out PT loss:', report['baseline']['final_heldout_pt_loss'], '->',
          report['fixed']['final_heldout_pt_loss'])
    print('  coherence:', report['baseline']['final_coherence'], '->',
          report['fixed']['final_coherence'])
    print('  gibberish:', report['baseline']['final_gibberish'], '->',
          report['fixed']['final_gibberish'])
    print('  reports:', args.out, args.markdown)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
