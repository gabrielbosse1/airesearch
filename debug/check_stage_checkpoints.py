"""21.2 — Checkpoint degradation analysis.

For each available stage checkpoint, generate 100 bytes from ``"Olá, como"``
with the shipped greedy decoding path, score Portuguese byte coherence, and
locate the stage where the output becomes gibberish.

Checkpoint sources, in priority order:

  * ``--checkpoints-dir DIR``  — local ``checkpoints/`` from the Colab runtime
    (path pattern ``checkpoints/stage_<k>_step_<n>/checkpoint.pt``);
  * ``--repo-id user/repo``   — downloads stage snapshots from a private HF
    model repo using ``HF_TOKEN`` (read token; never printed or stored);
  * default                    — the deterministic local replay checkpoints in
    ``runs/phase21/checkpoints/`` (clearly labelled as a replay, not Colab).

Run: python -m debug.check_stage_checkpoints
     python -m debug.check_stage_checkpoints --checkpoints-dir /path/to/checkpoints
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from debug.common import (CHECKPOINTS, ROOT, artifact_status, coherence,
                          generate_raw, md_table, model_from_checkpoint,
                          original_generate, report_dir, write_json, write_text)

PROMPT = 'Olá, como'
STAGE_RE = re.compile(r'stage[_-]?(\d+)', re.I)


def _stage_of(path, payload=None):
    match = STAGE_RE.search(path.name)
    if match:
        value = int(match.group(1))
        # HF snapshots use 0-based stage ids; replay files use 1-based names.
        if payload is not None and isinstance(payload.get('stage'), int):
            return payload['stage'] + 1
        return value
    return None


def discover(args):
    """Return [(stage, path, origin)] for all checkpoints we can actually load."""
    found = []
    if args.checkpoints_dir:
        root = Path(args.checkpoints_dir)
        for path in sorted(root.rglob('*.pt')):
            found.append((_stage_of(path), path, 'local Colab checkpoints dir'))
        return found, 'local Colab checkpoints dir'
    if args.repo_id:
        token = os.environ.get('HF_TOKEN')
        if not token:
            return [], 'HF repo requested but HF_TOKEN is not set'
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError:
            return [], 'huggingface_hub is not installed'
        try:
            api = HfApi(token=token)
            names = [n for n in api.list_repo_files(repo_id=args.repo_id,
                                                    repo_type='model', token=token)
                     if re.fullmatch(r'checkpoints/stage_\d+_step_\d+/checkpoint\.pt', n)]
            # Keep the highest step per stage, i.e. the END of each stage.
            latest = {}
            for name in names:
                stage, step = map(int, re.findall(r'\d+', name)[:2])
                if stage not in latest or step > latest[stage][0]:
                    latest[stage] = (step, name)
            out = []
            for stage, (step, name) in sorted(latest.items()):
                path = Path(hf_hub_download(repo_id=args.repo_id, filename=name,
                                            repo_type='model', token=token))
                out.append((stage + 1, path, f'HF {args.repo_id} step {step}'))
            return out, f'HF repo {args.repo_id}'
        except Exception as exc:
            return [], f'HF download failed: {type(exc).__name__}: {exc}'

    variants = args.variants or ['baseline', 'fixed']
    for variant in variants:
        for stage in range(1, 5):
            path = CHECKPOINTS / f'{variant}_stage{stage}.pt'
            if path.exists():
                found.append((stage, path, f'LOCAL REPLAY ({variant})'))
    return found, 'LOCAL REPLAY — not the Colab checkpoints'


def test_checkpoint(path, stage, origin):
    import torch
    torch.set_num_threads(1)
    payload = torch.load(str(path), map_location='cpu', weights_only=False)
    model, meta = model_from_checkpoint(path)
    probe = generate_raw(model, PROMPT, max_new=100, mode='faithful')
    shipped = original_generate(model, PROMPT, max_new=100)
    byte_only = generate_raw(model, PROMPT, max_new=100, mode='byte_only')
    saved_metrics = payload.get('metrics', {}) if isinstance(payload, dict) else {}
    return {
        'stage': stage, 'origin': origin, 'path': str(path),
        'parameters': meta['parameters'], 'config': meta['config'],
        'load_missing_keys': meta['missing_keys'][:5],
        'prompt': PROMPT,
        'n_bytes': probe['n_bytes'],
        'text': probe['text'][:200],
        'coherence': probe['score'],
        'gibberish': probe['gibberish'],
        'valid_utf8': probe['valid_utf8'],
        'replacement_ratio': probe['replacement_ratio'],
        'pt_word_hits': probe['pt_word_hits'],
        'byte_entropy_bits_per_byte': probe['byte_entropy_bits_per_byte'],
        'control_ids_emitted': probe['control_ids'],
        'byte_only_text': byte_only['text'][:120],
        'shipped_generate': shipped,
        'heldout_pt_loss': saved_metrics.get('heldout_pt_loss'),
        'heldout_code_loss': saved_metrics.get('heldout_code_loss'),
        'probes_passed': saved_metrics.get('probes_passed'),
        'training_final_loss': saved_metrics.get('final_loss'),
    }


def locate_collapse(rows):
    """First stage whose held-out Portuguese loss jumps >20% over the BEST
    EARLIER stage, or whose generated bytes turn gibberish after a clean stage."""
    ordered = sorted(rows, key=lambda r: r['stage'])
    loss_signal = [r for r in ordered if r.get('heldout_pt_loss') is not None]
    best_so_far = None
    for row in loss_signal:
        if best_so_far is not None and row['heldout_pt_loss'] > best_so_far['heldout_pt_loss'] * 1.20:
            return {'stage': row['stage'], 'signal': 'heldout_pt_loss',
                    'best_prior_stage': best_so_far['stage'],
                    'best_prior_loss': best_so_far['heldout_pt_loss'],
                    'collapse_loss': row['heldout_pt_loss'],
                    'ratio': round(row['heldout_pt_loss'] / best_so_far['heldout_pt_loss'], 3)}
        if best_so_far is None or row['heldout_pt_loss'] < best_so_far['heldout_pt_loss']:
            best_so_far = row
    for previous, current in zip(ordered, ordered[1:]):
        if current['gibberish'] and not previous['gibberish']:
            return {'stage': current['stage'], 'signal': 'gibberish_transition',
                    'previous_stage': previous['stage']}
    return {'stage': None, 'signal': 'no clear transition'}

def render_markdown(report):
    out = ['# 21.2 — Checkpoint degradation', '',
           f"**Checkpoint source:** `{report['source']}`", '',
           report['artifact_status']['note'], '',
           f"Prompt: `{PROMPT}` (greedy, 100 bytes, faithful to `phone_model.generate`)", '',
           md_table(['stage', 'origin', 'coherence', 'gibberish', 'valid UTF-8',
                     'PT word hits', 'held-out PT loss', 'text (first 60 chars)'],
                    [[r['stage'], r['origin'], r['coherence'], r['gibberish'],
                      r['valid_utf8'], r['pt_word_hits'], r.get('heldout_pt_loss'),
                      repr(r['text'][:60])] for r in report['checkpoints']]), '',
           f"**Collapse located at stage {report['collapse']['stage']}** "
           f"(signal: {report['collapse']['signal']})", '']
    if report.get('variants'):
        out += ['## Baseline vs fixed (local replay)', '',
                md_table(['variant', 'final coherence', 'final gibberish',
                          'final held-out PT loss', 'probes passed'],
                         [[name, v['final_coherence'], v['final_gibberish'],
                           v['final_heldout_pt_loss'], v['final_probes_passed']]
                          for name, v in report['variants'].items()]), '']
    return '\n'.join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints-dir', default=None)
    parser.add_argument('--repo-id', default=None)
    parser.add_argument('--variants', nargs='*', default=None)
    parser.add_argument('--out', default=str(report_dir() / 'stage_checkpoints.json'))
    parser.add_argument('--markdown', default=str(report_dir() / 'stage_checkpoints.md'))
    args = parser.parse_args(argv)

    discovered, source = discover(args)
    rows = []
    for stage, path, origin in discovered:
        try:
            rows.append(test_checkpoint(path, stage, origin))
        except Exception as exc:
            rows.append({'stage': stage, 'path': str(path), 'origin': origin,
                         'error': f'{type(exc).__name__}: {exc}'})
    valid = [r for r in rows if 'coherence' in r]
    by_variant = {}
    for variant in ('baseline', 'fixed'):
        subset = sorted([r for r in valid if variant in r['origin']],
                        key=lambda r: r['stage'])
        if subset:
            last = subset[-1]
            by_variant[variant] = {
                'final_coherence': last['coherence'],
                'final_gibberish': last['gibberish'],
                'final_heldout_pt_loss': last.get('heldout_pt_loss'),
                'final_probes_passed': last.get('probes_passed'),
                'per_stage': [{'stage': r['stage'], 'coherence': r['coherence'],
                               'gibberish': r['gibberish'],
                               'heldout_pt_loss': r.get('heldout_pt_loss')}
                              for r in subset],
            }
    baseline_rows = [r for r in valid if 'baseline' in r['origin']] or valid
    report = {
        'section': '21.2 checkpoint degradation analysis',
        'source': source, 'artifact_status': artifact_status(),
        'checkpoints': rows,
        'collapse': locate_collapse(baseline_rows),
        'variants': by_variant,
        'note': ('Local replay checkpoints are surrogate evidence. To analyse the '
                 'real Colab run, re-run with --checkpoints-dir or --repo-id.'),
    }
    write_json(args.out, report)
    write_text(args.markdown, render_markdown(report))
    print('21.2 checkpoints:', source, '| tested', len(valid))
    for row in valid:
        print(f"  stage {row['stage']} [{row['origin']}]: coherence {row['coherence']:.3f} "
              f"gibberish={row['gibberish']} "
              f"heldout_pt={row.get('heldout_pt_loss')} text={row['text'][:50]!r}")
    print('  collapse:', report['collapse'])
    print('  reports:', args.out, args.markdown)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
