"""Run the whole Phase 21 forensic workflow in dependency order.

    python -m debug.run_all                 # full run (~4 min on CPU)
    python -m debug.run_all --from-existing  # reuse the replay checkpoints/logs

Individual steps can also be run separately; see debug/README.md.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from debug.common import read_json, report_dir, write_json

STEPS = [
    ('21.4 encoding', 'debug.test_byte_encoding', []),
    ('21.3 data', 'debug.verify_training_data', []),
]
LATE_STEPS = [
    ('21.1 logs', 'debug.analyze_training_logs', []),
    ('21.2 checkpoints', 'debug.check_stage_checkpoints', []),
    ('21.5 weights', 'debug.analyze_model_weights', []),
    ('21.6 activations', 'debug.probe_activations', []),
    ('21.7 edit mystery', 'debug.investigate_edit_success', []),
]


def _run(module, args):
    command = [sys.executable, '-m', module] + args
    print('\n>>>', ' '.join(command), flush=True)
    completed = subprocess.run(command, cwd=str(Path(__file__).resolve().parent.parent))
    return {'module': module, 'returncode': completed.returncode}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-existing', action='store_true',
                        help='reuse existing replay results (no retraining)')
    parser.add_argument('--steps', type=int, default=300)
    args = parser.parse_args(argv)
    out = report_dir()
    results = []

    for label, module, extra in STEPS:
        results.append({**_run(module, extra), 'label': label})

    recovery = ['debug.quick_recovery', '--steps', str(args.steps)]
    if args.from_existing:
        recovery.append('--from-existing')
    results.append({**_run(recovery[0], recovery[1:]), 'label': '21.9 recovery A/B'})

    for label, module, extra in LATE_STEPS:
        results.append({**_run(module, extra), 'label': label})

    encoding = read_json(out / 'byte_encoding.json', {}) or {}
    data = read_json(out / 'training_data.json', {}) or {}
    logs = read_json(out / 'training_logs.json', {}) or {}
    checkpoints = read_json(out / 'stage_checkpoints.json', {}) or {}
    weights = read_json(out / 'model_weights.json', {}) or {}
    activations = read_json(out / 'activations.json', {}) or {}
    edit = read_json(out / 'edit_mystery.json', {}) or {}
    recovery_report = read_json(out / 'quick_recovery.json', {}) or {}

    summary = {
        'section': 'Phase 21 consolidated forensic report',
        'steps': results,
        'all_steps_succeeded': all(r['returncode'] == 0 for r in results),
        'colab_artifacts_available': (data.get('artifact_status', {})
                                      .get('any_colab_artifact', False)),
        'encoding_checks_passed': f"{encoding.get('passed', '?')}/{encoding.get('total', '?')}",
        'data_corruption_found': (data.get('verdict', {}).get('data_corruption_found')),
        'eval_contaminated_cases': (data.get('eval_leakage', {}).get('contaminated_cases')),
        'collapse_stage': checkpoints.get('collapse', {}).get('stage'),
        'collapse_signal': checkpoints.get('collapse', {}).get('signal'),
        'baseline_forgetting': (logs.get('variants', {}).get('baseline', {})
                                .get('checks', {}).get('signs_of_catastrophic_forgetting')),
        'numerical_pathology': any((weights.get('verdict') or {}).values()),
        'activations_healthy': (bool(activations.get('flags')) and
                                    activations['flags'].get('early_layers_produce_reasonable_vectors', False)
                                    and activations['flags'].get('middle_layers_transform_activations', False)
                                    and activations['flags'].get('final_distribution_valid', False)
                                    and not activations['flags'].get('garbage_propagating', True)),
        'edit_pass_is_memorisation': edit.get('static_evidence', {}).get(
            'eval_prompt_equals_rl_prompt'),
        'quick_fix_improved': recovery_report.get('improved'),
        'root_cause_report': 'debug/root_cause_analysis.md',
    }
    write_json(out / 'forensic_report.json', summary)
    print('\n=== PHASE 21 SUMMARY ===')
    for key, value in summary.items():
        if key not in ('steps',):
            print(f'  {key}: {value}')
    print('  consolidated report:', out / 'forensic_report.json')
    return 0 if summary['all_steps_succeeded'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
