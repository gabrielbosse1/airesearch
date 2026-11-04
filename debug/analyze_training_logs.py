"""21.1 — Training log autopsy.

The Phase 19 Colab run printed loss ONLY into a transient tqdm bar; no metrics
file was persisted, and ``checkpoints/`` and ``data_cache/`` are gitignored. This
script therefore does two things:

  1. If you supply real logs (``--logs``: .jsonl, .json, .log/.txt, or an
     executed .ipynb with cell outputs), it parses them, plots per-stage loss
     curves and runs the spike / collapse / NaN / LR / forgetting detectors.
  2. Otherwise it analyses the deterministic local replay written by
     ``debug/replay_training.py`` and labels the source explicitly as a replay,
     never as the Colab run.

Run: python -m debug.analyze_training_logs
     python -m debug.analyze_training_logs --logs /path/to/colab.log
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from debug.common import (REPLAY, ROOT, artifact_status, md_table, report_dir,
                          write_json, write_text)

NUMBER = r'(?:nan|inf|-?inf|[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
LOSS_RE = re.compile(rf'loss[=:\s]+({NUMBER})', re.I)
STEP_RE = re.compile(r'step[=:\s]+(\d+)', re.I)
STAGE_RE = re.compile(r'stage[=:\s]*(\d+)', re.I)
LR_RE = re.compile(rf'(?:^|[\s,])lr[=:\s]+({NUMBER})', re.I)
GRAD_RE = re.compile(rf'grad(?:ient)?[_ -]?norm[=:\s]+({NUMBER})', re.I)


def _to_float(text):
    try:
        return float(text)
    except (ValueError, TypeError):
        return float('nan')


def _normalise(record, default_stage=None, seq=0):
    stage = record.get('stage', default_stage)
    if isinstance(stage, str):
        match = STAGE_RE.search(stage)
        stage = int(match.group(1)) if match else None
    loss = _to_float(record.get('loss'))
    lr = record.get('lr')
    grad = record.get('grad_norm', record.get('gradient_norm'))
    return {
        'stage': int(stage) if isinstance(stage, (int, float)) else None,
        'step': int(record.get('step', seq)) if str(record.get('step', seq)).isdigit() else seq,
        'loss': loss,
        'lr': _to_float(lr) if lr is not None else None,
        'grad_norm': _to_float(grad) if grad is not None else None,
        'nan': bool(record.get('nan', False)) or not math.isfinite(loss),
        'kind': record.get('kind', 'step'),
        'coherence': record.get('coherence'),
        'heldout_pt_loss': record.get('heldout_pt_loss'),
        'heldout_code_loss': record.get('heldout_code_loss'),
        'gibberish': record.get('gibberish'),
        'probes_passed': record.get('probes_passed'),
        'variant': record.get('variant'),
    }


def parse_text(text):
    """Parse tqdm / print style log text into step records."""
    records = []
    current_stage = None
    seq = 0
    for line in text.splitlines():
        stage_match = STAGE_RE.search(line)
        if stage_match:
            current_stage = int(stage_match.group(1))
        loss_match = LOSS_RE.search(line)
        if not loss_match:
            continue
        seq += 1
        step_match = STEP_RE.search(line)
        lr_match = LR_RE.search(line)
        grad_match = GRAD_RE.search(line)
        records.append(_normalise({
            'stage': current_stage, 'step': int(step_match.group(1)) if step_match else seq,
            'loss': loss_match.group(1),
            'lr': lr_match.group(1) if lr_match else None,
            'grad_norm': grad_match.group(1) if grad_match else None,
        }, current_stage, seq))
    return records


def parse_path(path):
    path = Path(path)
    if not path.exists():
        return [], f'missing: {path}'
    if path.suffix == '.ipynb':
        notebook = json.loads(path.read_text(encoding='utf8'))
        chunks = []
        for cell in notebook.get('cells', []):
            for output in cell.get('outputs', []):
                chunks.extend(output.get('text', []) if isinstance(output.get('text'), list)
                              else [output.get('text', '')])
                data = output.get('data', {})
                if isinstance(data.get('text/plain'), list):
                    chunks.extend(data['text/plain'])
                elif isinstance(data.get('text/plain'), str):
                    chunks.append(data['text/plain'])
        return parse_text('\n'.join(str(c) for c in chunks)), f'notebook outputs: {path}'
    if path.suffix in ('.jsonl', '.ndjson'):
        rows = []
        for seq, line in enumerate(path.read_text(encoding='utf8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(_normalise(payload, seq=seq))
        return rows, f'jsonl: {path}'
    if path.suffix == '.json':
        payload = json.loads(path.read_text(encoding='utf8'))
        if isinstance(payload, dict):
            payload = payload.get('records', payload.get('metrics', [payload]))
        rows = [_normalise(r, seq=i) for i, r in enumerate(payload, 1)
                if isinstance(r, dict)]
        return rows, f'json: {path}'
    return parse_text(path.read_text(encoding='utf8', errors='replace')), f'text: {path}'


def _steps(records):
    return [r for r in records if r['kind'] == 'step' and math.isfinite(r['loss'])]


def detect_nan_inf(records):
    bad = [r for r in records if not math.isfinite(r['loss']) or r['nan']]
    return {'count': len(bad), 'examples': bad[:5]}


def detect_spikes(records, factor=3.0):
    """Loss > factor * rolling median of the same stage."""
    per_stage = {}
    for row in _steps(records):
        per_stage.setdefault(row['stage'], []).append(row)
    spikes = []
    for stage, rows in sorted(per_stage.items(), key=lambda kv: (kv[0] is None, kv[0])):
        losses = [r['loss'] for r in rows]
        for i in range(8, len(losses)):
            window = losses[max(0, i - 20):i]
            baseline = statistics.median(window)
            if baseline > 0 and losses[i] > factor * baseline:
                spikes.append({'stage': stage, 'step': rows[i]['step'],
                               'loss': round(losses[i], 5),
                               'baseline': round(baseline, 5),
                               'ratio': round(losses[i] / baseline, 2)})
    return {'factor': factor, 'count': len(spikes), 'spikes': spikes[:20],
            'gradient_explosion_in_loss': len(spikes) > 0}


def detect_collapse(records, threshold=0.05):
    """Loss collapsing to near zero suspiciously early in a stage."""
    per_stage = {}
    for row in _steps(records):
        per_stage.setdefault(row['stage'], []).append(row)
    results = []
    for stage, rows in sorted(per_stage.items(), key=lambda kv: (kv[0] is None, kv[0])):
        losses = [r['loss'] for r in rows]
        first = next((i for i, loss in enumerate(losses) if loss < threshold), None)
        results.append({'stage': stage, 'steps': len(losses),
                        'initial_loss': round(losses[0], 5) if losses else None,
                        'final_loss': round(losses[-1], 5) if losses else None,
                        'min_loss': round(min(losses), 5) if losses else None,
                        'first_below_threshold': first,
                        'fraction_of_stage': round(first / len(losses), 4) if first is not None and losses else None})
    early = [r for r in results if r['first_below_threshold'] is not None
             and r['fraction_of_stage'] is not None and r['fraction_of_stage'] < 0.1]
    return {'threshold': threshold, 'stages': results,
            'suspiciously_early_collapse': bool(early)}


def check_lr_schedule(records):
    per_stage = {}
    for row in _steps(records):
        if row['lr'] is not None and math.isfinite(row['lr']):
            per_stage.setdefault(row['stage'], []).append(row['lr'])
    results = []
    for stage, values in sorted(per_stage.items(), key=lambda kv: (kv[0] is None, kv[0])):
        increases = sum(1 for a, b in zip(values, values[1:]) if b > a + 1e-12)
        results.append({'stage': stage, 'n': len(values),
                        'start': values[0], 'end': values[-1],
                        'max': max(values), 'min': min(values),
                        'increases_within_stage': increases,
                        'decayed_to_near_zero': values[-1] <= 0.05 * max(values) + 1e-12,
                        'constant': max(values) - min(values) < 1e-12})
    return {'stages': results,
            'schedule_failed': any(r['increases_within_stage'] > 0 or r['constant']
                                   for r in results) if results else None}


def detect_gradient_explosion(records, factor=10.0, absolute=100.0):
    values = [r['grad_norm'] for r in _steps(records)
              if r['grad_norm'] is not None and math.isfinite(r['grad_norm'])]
    if not values:
        return {'available': False, 'note': 'no grad_norm records; Phase 19 did not log them'}
    baseline = statistics.median(values) or 1e-9
    outliers = [v for v in values if v > factor * baseline or v > absolute]
    fraction = len(outliers) / len(values)
    return {'available': True, 'median': round(baseline, 5),
            'max': round(max(values), 5), 'outliers': len(outliers),
            'outlier_fraction': round(fraction, 5),
            'explosion': max(values) > 1000.0 or fraction > 0.05,
            'values_are_pre_clip': True,
            'note': ('These are the norms returned by clip_grad_norm_ BEFORE clipping. '
                     'notebook.train clips the global norm to 1.0 before every AdamW '
                     'step, so isolated large norms never reach the weights. A few '
                     'early spikes are normal initialisation, not explosion.')}

def detect_forgetting(records):
    """Prefer teacher-forced held-out loss; fall back to generation coherence.

    Catastrophic forgetting can be TRANSIENT: Stage 3 can destroy Portuguese and
    Stage 4 can partially recover the loss while still being useless. We flag
    the worst stage, not just the final one.
    """
    evals = [r for r in records if r['kind'] == 'eval']
    losses = [(r['stage'], r['heldout_pt_loss']) for r in evals
              if r.get('heldout_pt_loss') is not None]
    coherences = [(r['stage'], r['coherence']) for r in evals
                  if r.get('coherence') is not None]
    if not losses and not coherences:
        return {'available': False, 'note': 'no per-stage evaluations in the log'}
    result = {'available': True}
    if losses:
        first = losses[0][1]
        best_stage, best = min(losses, key=lambda x: x[1])
        worst_stage, worst = max(losses, key=lambda x: x[1])
        final_stage, final = losses[-1]
        result['heldout_pt_loss_per_stage'] = [
            {'stage': stage, 'loss': value} for stage, value in losses]
        result['best_stage'] = best_stage
        result['worst_stage'] = worst_stage
        result['first_loss'] = first
        result['best_loss'] = best
        result['worst_loss'] = worst
        result['final_loss'] = final
        result['worst_over_first_ratio'] = round(worst / max(1e-9, first), 3)
        result['forgetting'] = worst > 1.20 * first or final > 1.20 * best
        result['collapse_stage'] = worst_stage if worst > 1.20 * first else None
    if coherences:
        best_stage, best = max(coherences, key=lambda x: x[1])
        result['coherence_per_stage'] = [
            {'stage': stage, 'coherence': value} for stage, value in coherences]
        result['best_stage_by_coherence'] = best_stage
        result['final_coherence'] = coherences[-1][1]
        result['coherence_degradation'] = round(best - coherences[-1][1], 4)
        if 'forgetting' not in result:
            result['forgetting'] = coherences[-1][1] < best - 0.1
    return result

def detect_encoding_corruption(records):
    return {'available': False,
            'note': ('Token/byte encoding corruption cannot be detected from loss curves; '
                     'see debug/test_byte_encoding.py and debug/verify_training_data.py. '
                     'A clean loss curve is fully compatible with a broken decode path.')}


def analyse(records, source, variant=None):
    per_stage = {}
    for row in _steps(records):
        per_stage.setdefault(row['stage'], []).append(row['loss'])
    summary = [{'stage': stage, 'steps': len(losses),
                'mean': round(statistics.fmean(losses), 5),
                'first': round(losses[0], 5), 'last': round(losses[-1], 5),
                'min': round(min(losses), 5), 'max': round(max(losses), 5)}
               for stage, losses in sorted(per_stage.items(),
                                           key=lambda kv: (kv[0] is None, kv[0]))]
    return {
        'section': '21.1 training log autopsy',
        'source': source,
        'artifact_status': artifact_status(),
        'n_records': len(records),
        'variant': variant,
        'variants': sorted({r['variant'] for r in records if r.get('variant')}),
        'source_is_colab_log': 'colab' in source.lower() and 'replay' not in source.lower(),
        'per_stage_loss': summary,
        'nan_inf': detect_nan_inf(records),
        'loss_spikes': detect_spikes(records),
        'loss_collapse': detect_collapse(records),
        'lr_schedule': check_lr_schedule(records),
        'gradient_explosion': detect_gradient_explosion(records),
        'catastrophic_forgetting': detect_forgetting(records),
        'encoding_corruption': detect_encoding_corruption(records),
        'checks': {
            'loss_spike_gradient_explosion': detect_spikes(records)['gradient_explosion_in_loss'],
            'loss_collapsed_to_near_zero_too_fast': detect_collapse(records)['suspiciously_early_collapse'],
            'nan_or_inf_values': detect_nan_inf(records)['count'] > 0,
            'lr_scheduling_failed': check_lr_schedule(records)['schedule_failed'],
            'signs_of_catastrophic_forgetting': detect_forgetting(records).get('forgetting', None),
        },
    }


def plot(records, out_path, source='', variant=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    per_stage = {}
    for row in _steps(records):
        per_stage.setdefault(row['stage'], []).append(row)
    stages = sorted(per_stage, key=lambda s: (s is None, s))
    if not stages:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
    for ax, stage in zip(axes.ravel(), stages[:4]):
        rows = per_stage[stage]
        steps = [r['step'] for r in rows]
        losses = [r['loss'] for r in rows]
        ax.plot(steps, losses, lw=0.8, color='#1f77b4', label='loss')
        if len(losses) >= 10:
            window = max(3, len(losses) // 25)
            smooth = [statistics.fmean(losses[max(0, i - window):i + 1])
                      for i in range(len(losses))]
            ax.plot(steps, smooth, lw=2, color='#d62728', label=f'rolling({window})')
        ax.set_title(f'Stage {stage} loss')
        ax.set_xlabel('optimizer step')
        ax.set_ylabel('cross-entropy (nats/byte)')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        lrs = [r['lr'] for r in rows if r['lr'] is not None]
        if lrs and len(lrs) == len(rows):
            twin = ax.twinx()
            twin.plot(steps, lrs, lw=0.9, ls='--', color='#2ca02c', alpha=.7)
            twin.set_ylabel('lr', color='#2ca02c')
    for ax in axes.ravel()[len(stages[:4]):]:
        ax.axis('off')
    label = ('LOCAL REPLAY (not Colab)' if 'replay' in source.lower()
            else 'supplied logs')
    if variant:
        label += f' — {variant}'
    fig.suptitle('Phase 21 loss curves — ' + label, fontsize=11)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return str(out_path)


def render_markdown(report, plot_path):
    out = ['# 21.1 — Training log autopsy', '',
           f"**Log source:** `{report['source']}`", '',
           report['artifact_status']['note'], '',
           '## Per-stage loss', '',
           md_table(['stage', 'steps', 'first', 'last', 'mean', 'min', 'max'],
                    [[r['stage'], r['steps'], r['first'], r['last'], r['mean'], r['min'], r['max']]
                     for r in report['per_stage_loss']]), '',
           '## Detector results', '',
           md_table(['question', 'result', 'evidence'], [
               ['Loss spike (gradient explosion)?', report['checks']['loss_spike_gradient_explosion'],
                f"{report['loss_spikes']['count']} spikes > 3x rolling median"],
               ['Loss collapsed to near zero too fast?',
                report['checks']['loss_collapsed_to_near_zero_too_fast'],
                f"{len([s for s in report['loss_collapse']['stages'] if s['first_below_threshold'] is not None])} stages reached < 0.05"],
               ['NaN / Inf values?', report['checks']['nan_or_inf_values'],
                f"{report['nan_inf']['count']} records"],
               ['LR scheduling failed?', report['checks']['lr_scheduling_failed'],
                f"{len(report['lr_schedule']['stages'])} stages with lr data"],
               ['Catastrophic forgetting?', report['checks']['signs_of_catastrophic_forgetting'],
                str(report['catastrophic_forgetting'].get('per_stage', 'no eval records'))],
               ['Gradient norms logged?', report['gradient_explosion'].get('available'),
                report['gradient_explosion'].get('note', '')],
               ['Encoding corruption visible?', 'not testable from loss',
                report['encoding_corruption']['note']],
           ]), '']
    if plot_path:
        out += [f'![loss curves]({Path(plot_path).name})', '']
    return '\n'.join(out)


def default_records():
    """Prefer real Colab logs if present; otherwise use the labelled local replay."""
    candidates = [ROOT / 'runs' / 'train.log', ROOT / 'colab_training.log',
                  ROOT / 'data_cache' / 'training_metrics.jsonl']
    for path in candidates:
        if path.exists():
            records, source = parse_path(path)
            if records:
                return records, source
    replay_files = sorted(REPLAY.glob('*_metrics.jsonl'))
    records = []
    sources = []
    for path in replay_files:
        rows, _ = parse_path(path)
        records.extend(rows)
        sources.append(path.name)
    if records:
        return records, ('LOCAL REPLAY — deterministic CPU re-run of the Phase 19 '
                         f'pipeline, NOT the Colab logs: {", ".join(sources)}')
    return [], 'NO LOGS AVAILABLE'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', action='append', default=[],
                        help='repeatable: real Colab log/jsonl/json/ipynb path')
    parser.add_argument('--out', default=str(report_dir() / 'training_logs.json'))
    parser.add_argument('--plot', default=str(report_dir() / 'loss_curves.png'))
    parser.add_argument('--markdown', default=str(report_dir() / 'training_logs.md'))
    args = parser.parse_args(argv)

    records, sources = [], []
    for path in args.logs:
        rows, label = parse_path(path)
        records.extend(rows)
        sources.append(label)
    if not records:
        records, label = default_records()
        sources = [label]
    source = '; '.join(sources) if sources else 'NO LOGS AVAILABLE'

    by_variant = {}
    for variant in sorted({r.get('variant') for r in records}):
        subset = [r for r in records if r.get('variant') == variant]
        by_variant[variant or 'all'] = subset
    if len(by_variant) <= 1:
        by_variant = {'all': records}

    reports = {}
    plots = {}
    for variant, subset in by_variant.items():
        suffix = '' if variant == 'all' else f'_{variant}'
        out_path = str(Path(args.out).with_name(Path(args.out).stem + suffix + '.json'))
        plot_path = str(Path(args.plot).with_name(Path(args.plot).stem + suffix + '.png'))
        report = analyse(subset, source, None if variant == 'all' else variant)
        report['plot'] = plot(subset, plot_path, source,
                              None if variant == 'all' else variant)
        write_json(out_path, report)
        reports[variant] = report
        plots[variant] = report['plot']
        print(f'21.1 [{variant}]: {report["n_records"]} records, '
              f'checks={report["checks"]}, '
              f'collapse_stage={report["catastrophic_forgetting"].get("collapse_stage")}')

    primary = reports.get('baseline', reports.get('all'))
    combined = {
        'section': '21.1 training log autopsy', 'source': source,
        'artifact_status': artifact_status(), 'n_records': len(records),
        'source_is_colab_log': 'colab' in source.lower() and 'replay' not in source.lower(),
        'variants': {name: {'report_path': str(Path(args.out).with_name(
                                Path(args.out).stem + ('' if name == 'all' else f'_{name}') + '.json')),
                           'plot': plots[name],
                           'checks': rep['checks'],
                           'per_stage_loss': rep['per_stage_loss'],
                           'catastrophic_forgetting': rep['catastrophic_forgetting'],
                           'nan_inf': rep['nan_inf'],
                           'lr_schedule': rep['lr_schedule'],
                           'gradient_explosion': rep['gradient_explosion'],
                           'loss_spikes': rep['loss_spikes'],
                           'loss_collapse': rep['loss_collapse']}
                     for name, rep in reports.items()},
        'primary_checks': primary['checks'],
    }
    write_json(args.out, combined)

    md = [render_markdown(primary, primary.get('plot')), '',
          '## Per-variant comparison', '',
          md_table(['variant', 'spikes', 'NaN/Inf', 'LR failed', 'forgetting',
                    'worst held-out PT stage', 'worst/first ratio'],
                   [[name, rep['checks']['loss_spike_gradient_explosion'],
                     rep['checks']['nan_or_inf_values'],
                     rep['checks']['lr_scheduling_failed'],
                     rep['checks']['signs_of_catastrophic_forgetting'],
                     rep['catastrophic_forgetting'].get('worst_stage'),
                     rep['catastrophic_forgetting'].get('worst_over_first_ratio')]
                    for name, rep in reports.items()]), '']
    for name, path in plots.items():
        if path:
            md.append(f'### {name} loss curves\n\n![loss curves]({Path(path).name})\n')
    write_text(args.markdown, '\n'.join(md))
    print('21.1 logs:', source)
    print('  reports:', args.out, args.markdown, ', '.join(str(v) for v in plots.values() if v))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
