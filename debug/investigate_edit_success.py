"""21.7 — The "edit" success mystery.

Why did exactly one probe pass? This script reconstructs the edit case from
``notebook/evaluate.py``, compares it byte-for-byte with the Stage 4 supervised
rows and the Stage 4 REINFORCE cases, measures the longest common prefix between
the model's output and those training rows, and finally tests a HELD-OUT edit
typo the model has never seen.

Run: python -m debug.investigate_edit_success
"""
from __future__ import annotations

import argparse

from debug.common import (find_checkpoint, generate_raw, md_table,
                          model_from_checkpoint, report_dir, write_json, write_text)

HELD_OUT = ('Request: Corrija: obrigadoo\nDraft: obrigadoo\n', 'obrigadoo', 'obrigado')


def _lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def analyse(checkpoint=None, variant='baseline', stage=4):
    from notebook.evaluate import CASES
    from notebook.preprocess import synthetic
    from notebook.tools import reward
    from notebook.train import FREEZE, MIXES, RL_CASES

    eval_prompt = dict(CASES)['edit']
    edit_rows = synthetic()['edit']
    rl = [row for row in RL_CASES if row[0] == 'edit']
    rl_prompt = rl[0][1] if rl else None
    rl_reward_args = rl[0][2:] if rl else ()

    static = {
        'eval_prompt': eval_prompt,
        'stage4_mix': MIXES[3],
        'stage4_freeze_blocks': FREEZE[3],
        'edit_rows': edit_rows,
        'edit_unique_rows': len(set(edit_rows)),
        'edit_rows_total_bytes': sum(len(r.encode('utf8')) for r in edit_rows),
        'rl_edit_cases': [{'kind': r[0], 'prompt': r[1], 'draft': r[2], 'target': r[3]}
                          for r in rl],
        'eval_prompt_equals_rl_prompt': eval_prompt == rl_prompt,
        'eval_prompt_is_prefix_of_training_row': [
            {'row_index': i, 'lcp_chars': _lcp(eval_prompt, row),
             'is_prefix': row.startswith(eval_prompt)}
            for i, row in enumerate(edit_rows)],
        'eval_and_rl_use_same_reward': (
            rl and rl[0][0] == 'edit' and rl[0][2] == 'voce' and rl[0][3] == 'você'),
        'stage4_edit_fraction': MIXES[3]['edit'] / sum(MIXES[3].values()),
        'reward_definition': ('tools.reward("edit") returns 0.5 only if an [EDIT ...] '
                             'command transforms draft="voce" into target="você"'),
    }
    static['hypothesis_overtrained_on_edit'] = (
        static['stage4_edit_fraction'] >= 0.5 and static['edit_unique_rows'] <= 4)
    static['hypothesis_edit_skill_learned'] = False  # tested below, not assumed

    result = {'section': '21.7 edit success investigation', 'static_evidence': static}
    path = find_checkpoint(checkpoint, variant, stage - 1)
    if path is None:
        result['generation'] = {'available': False, 'note': 'no checkpoint available'}
        result['verdict'] = ('Static evidence is conclusive: the passing probe is a '
                             'memorised 4-row template, reinforced by REINFORCE on the '
                             'identical prompt. No checkpoint was available to confirm '
                             'behaviourally.')
        return result

    import torch
    torch.set_num_threads(1)
    model, meta = model_from_checkpoint(path)
    train_output = generate_raw(model, eval_prompt, max_new=96, mode='faithful')
    held_output = generate_raw(model, HELD_OUT[0], max_new=96, mode='faithful')
    # The model only generates the CONTINUATION after the prompt, so compare
    # against the training row's suffix (the memorised [EDIT ...] command).
    lcp = [{'row_index': i, 'lcp_chars': _lcp(train_output['text'], row[len(eval_prompt):]),
            'training_suffix': row[len(eval_prompt):],
            'suffix_length': len(row[len(eval_prompt):])}
           for i, row in enumerate(edit_rows) if row.startswith(eval_prompt)]
    exact_suffix_match = any(x['lcp_chars'] >= x['suffix_length'] for x in lcp)
    train_pass = bool(reward('edit', train_output['text'], 'voce', 'você'))
    held_pass = bool(reward('edit', held_output['text'], HELD_OUT[1], HELD_OUT[2]))
    result['generation'] = {
        'available': True, 'checkpoint': str(path), 'model': meta,
        'training_prompt_output': train_output['text'][:200],
        'training_prompt_pass': train_pass,
        'longest_common_prefix_with_training_suffix': max((x['lcp_chars'] for x in lcp), default=0),
        'exact_training_suffix_reproduced': exact_suffix_match,
        'lcp_details': lcp,
        'held_out_prompt': HELD_OUT[0],
        'held_out_output': held_output['text'][:200],
        'held_out_pass': held_pass,
        'held_out_coherence': held_output['score'],
    }
    result['hypothesis_edit_skill_learned'] = held_pass and train_pass
    result['verdict'] = (
        'The edit probe passed because Stage 4 spent 50% of its supervised updates '
        'on FOUR unique [EDIT] templates and the Stage 4 REINFORCE case uses the '
        'byte-identical evaluation prompt and reward. '
        + (f"The model reproduces the memorised [EDIT] continuation "
           f"({'verbatim' if exact_suffix_match else 'partially'}), yet fails a held-out typo "
           f"({'PASS' if held_pass else 'FAIL'}). This is memorisation of one test "
           "item, not an edit capability."
           if not held_pass else
           'A held-out typo also passed; a broader edit benchmark would be needed.'))
    return result


def render_markdown(report):
    s = report['static_evidence']
    g = report['generation']
    return '\n'.join([
        '# 21.7 — Why only `edit` passed', '',
        f"Eval prompt: `{s['eval_prompt']!r}`", '',
        md_table(['fact', 'value'], [
            ['Stage 4 edit weight', f"{s['stage4_edit_fraction']:.0%}"],
            ['Unique edit training rows', s['edit_unique_rows']],
            ['Total edit training bytes', s['edit_rows_total_bytes']],
            ['Stage 4 frozen blocks', f"{s['stage4_freeze_blocks']}/12"],
            ['Eval prompt == REINFORCE prompt', s['eval_prompt_equals_rl_prompt']],
            ['Eval and RL share the same reward', s['eval_and_rl_use_same_reward']],
            ['Eval prompt is a training-row prefix',
             [x['is_prefix'] for x in s['eval_prompt_is_prefix_of_training_row']]],
            ['Over-trained on edit?', s['hypothesis_overtrained_on_edit']],
        ]), '',
        '## Behavioural test', '',
        md_table(['probe', 'pass', 'output (first 100 chars)'], [
            ['training/eval prompt', g.get('training_prompt_pass'),
             repr(g.get('training_prompt_output', '')[:100])],
            ['held-out typo `obrigadoo`', g.get('held_out_pass'),
             repr(g.get('held_out_output', '')[:100])],
        ]), '',
        (f"Longest common prefix with a training row: "
         f"**{g.get('longest_common_prefix_with_training_suffix')} characters** "
         f"(exact suffix reproduced: {g.get('exact_training_suffix_reproduced')})."
         if g.get('available') else ''), '',
        f"**Verdict:** {report['verdict']}", '',
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--variant', default='baseline')
    parser.add_argument('--stage', type=int, default=4)
    parser.add_argument('--out', default=str(report_dir() / 'edit_mystery.json'))
    parser.add_argument('--markdown', default=str(report_dir() / 'edit_mystery.md'))
    args = parser.parse_args(argv)
    report = analyse(args.checkpoint, args.variant, args.stage)
    write_json(args.out, report)
    write_text(args.markdown, render_markdown(report))
    s = report['static_evidence']
    print('21.7 edit: eval prompt == RL prompt:', s['eval_prompt_equals_rl_prompt'])
    print('  stage 4 mix:', s['stage4_mix'], '| unique edit rows:', s['edit_unique_rows'])
    print('  training-prompt pass:', report['generation'].get('training_prompt_pass'),
          '| held-out pass:', report['generation'].get('held_out_pass'))
    print('  verdict:', report['verdict'][:220])
    print('  reports:', args.out, args.markdown)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
