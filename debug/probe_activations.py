"""21.6 — Layer-by-layer activation analysis.

Feeds ``"Olá, como você"`` through the checkpoint, captures activations at the
embedding, every transformer block, the final LayerNorm and the output head,
and checks whether the final softmax is a valid probability distribution over
raw bytes (as opposed to mass leaking into never-trained control ids).

Run: python -m debug.probe_activations
     python -m debug.probe_activations --checkpoint final_model_fp16.pt
"""
from __future__ import annotations

import argparse
import math

from debug.common import (find_checkpoint, md_table, model_from_checkpoint,
                          report_dir, write_json, write_text)

TEXT = 'Olá, como você'


def analyse(path, text=TEXT):
    import torch
    torch.set_num_threads(1)
    from notebook.preprocess import BOS

    model, meta = model_from_checkpoint(path)
    captured = {}
    handles = []

    def make_hook(name):
        def inner(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            value = output.detach().float()
            finite = torch.isfinite(value)
            safe = torch.where(finite, value, torch.zeros_like(value))
            captured[name] = {
                'shape': list(value.shape),
                'mean': round(float(safe.mean()), 6),
                'std': round(float(safe.std()), 6),
                'max_abs': round(float(safe.abs().max()), 6),
                'zero_fraction': round(float((safe.abs() < 1e-6).float().mean()), 6),
                'saturated_fraction': round(float((safe.abs() > 10).float().mean()), 6),
                'nonfinite': int((~finite).sum()),
                'vector': value[0],  # kept in memory only, for layer-to-layer deltas
            }
        return inner

    handles.append(model.token.register_forward_hook(make_hook('embedding')))
    handles.append(model.pos.register_forward_hook(make_hook('positional')))
    for index, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(make_hook(f'block_{index + 1:02d}')))
    handles.append(model.norm.register_forward_hook(make_hook('final_norm')))
    handles.append(model.head.register_forward_hook(make_hook('logits')))

    ids = [BOS] + list(text.encode('utf8'))
    with torch.no_grad():
        logits = model(torch.tensor([ids], dtype=torch.long))[0, -1].float()
    for handle in handles:
        handle.remove()

    probs = torch.softmax(logits, dim=-1)
    total = float(probs.sum())
    finite = bool(torch.isfinite(probs).all())
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
    top = torch.topk(probs, 10)
    top_rows = []
    for value, index in zip(top.values.tolist(), top.indices.tolist()):
        label = (repr(chr(index)) if 0 <= index < 256 and chr(index).isprintable()
                 else f'id={index} (control)' if index >= 256 else f'byte 0x{index:02x}')
        top_rows.append({'id': index, 'label': label, 'probability': round(value, 6)})

    mass = {
        'raw_bytes_0_255': round(float(probs[:256].sum()), 6),
        'BOS_256_untrained_target': round(float(probs[256]), 6),
        'EOS_257': round(float(probs[257]), 6),
        'PAD_and_reserved_258_267': round(float(probs[258:].sum()), 6),
        'ascii_letters': round(float(probs[65:91].sum() + probs[97:123].sum()), 6),
        'space': round(float(probs[32]), 6),
        'utf8_continuation_0x80_0xbf': round(float(probs[128:192].sum()), 6),
        'ascii_control_0_31': round(float(probs[:32].sum()), 6),
    }
    distribution = {
        'softmax_finite': finite,
        'softmax_sum': round(total, 6),
        'valid_distribution': finite and abs(total - 1.0) < 1e-4,
        'entropy_nats': round(entropy, 4),
        'max_entropy_nats': round(math.log(len(probs)), 4),
        'uniformity': round(entropy / math.log(len(probs)), 4),
        'argmax_id': int(logits.argmax()),
        'argmax_label': top_rows[0]['label'],
        'top10': top_rows,
        'probability_mass': mass,
    }

    layers = []
    for name in sorted(captured):
        row = {k: v for k, v in captured[name].items() if k != 'vector'}
        layers.append({'layer': name, **row})
    deltas = []
    names = [f'block_{i + 1:02d}' for i in range(len(model.blocks))]
    names = [n for n in names if n in captured]
    for previous, current in zip(names, names[1:]):
        a = captured[previous]['vector'][-1]
        b = captured[current]['vector'][-1]
        delta = (b - a).norm().item() / max(1e-9, a.norm().item())
        cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        deltas.append({'layers': f'{previous}->{current}',
                       'relative_change': round(delta, 5),
                       'cosine': round(cosine, 5)})
    for row in layers:
        row.pop('vector', None)

    early_ok = all(captured[n]['nonfinite'] == 0 and captured[n]['std'] > 1e-5
                   for n in ('embedding', 'positional') if n in captured)
    middle_ok = bool(deltas) and all(d['relative_change'] > 1e-4 for d in deltas)
    flags = {
        'early_layers_produce_reasonable_vectors': early_ok,
        'middle_layers_transform_activations': middle_ok,
        'final_distribution_valid': distribution['valid_distribution'],
        'garbage_propagating': any(l['nonfinite'] > 0 for l in layers),
        'dead_layers': sum(1 for l in layers if l.get('zero_fraction', 0) > 0.9),
    }
    return {
        'section': '21.6 layer-by-layer activation analysis',
        'checkpoint': str(path), 'model': meta, 'input': text,
        'input_ids': ids,
        'layers': layers,
        'block_to_block_deltas': deltas,
        'final_distribution': distribution,
        'flags': flags,
        'interpretation': ('Activations are finite and transform across layers: the '
                          'network is computationally healthy. The failure is in WHAT '
                          'the output distribution encodes, not in garbage propagating '
                          'through dead or exploding layers.' if early_ok and middle_ok
                          and distribution['valid_distribution']
                          and not flags['garbage_propagating'] else
                          'A structural activation problem was detected; see flags.'),
    }


def render_markdown(report):
    d = report['final_distribution']
    out = [
        '# 21.6 — Activation probe', '',
        f"Checkpoint: `{report['checkpoint']}` · input `{report['input']}`", '',
        md_table(['flag', 'result'], [[k, v] for k, v in report['flags'].items()]), '',
        '## Layer statistics', '',
        md_table(['layer', 'std', 'mean', 'max |x|', 'zero frac', 'nonfinite'],
                 [[l['layer'], l['std'], l['mean'], l['max_abs'],
                   l['zero_fraction'], l['nonfinite']]
                  for l in report['layers'][:6]] +
                 ([['…', '…', '…', '…', '…', '…']] if len(report['layers']) > 12 else []) +
                 [[l['layer'], l['std'], l['mean'], l['max_abs'],
                   l['zero_fraction'], l['nonfinite']]
                  for l in report['layers'][-3:]]), '',
        '## Final softmax', '',
        md_table(['metric', 'value'], [
            ['valid distribution', d['valid_distribution']],
            ['softmax sum', d['softmax_sum']],
            ['entropy / max entropy', f"{d['entropy_nats']} / {d['max_entropy_nats']}"],
            ['uniformity', d['uniformity']],
            ['argmax', f"{d['argmax_id']} {d['argmax_label']}"],
        ]), '',
        md_table(['probability mass', 'value'],
                 [[k, v] for k, v in d['probability_mass'].items()]), '',
        'Top 10 ids: ' + ', '.join(f"{r['id']}:{r['label']}={r['probability']:.4f}"
                                   for r in d['top10']), '',
        f"**Interpretation:** {report['interpretation']}", '',
    ]
    return '\n'.join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--variant', default='baseline')
    parser.add_argument('--stage', type=int, default=4)
    parser.add_argument('--text', default=TEXT)
    parser.add_argument('--out', default=str(report_dir() / 'activations.json'))
    parser.add_argument('--markdown', default=str(report_dir() / 'activations.md'))
    args = parser.parse_args(argv)
    path = find_checkpoint(args.checkpoint, args.variant, args.stage - 1)
    if path is None:
        print('21.6: no checkpoint available')
        return 1
    report = analyse(path, args.text)
    write_json(args.out, report)
    write_text(args.markdown, render_markdown(report))
    print('21.6 activations:', path.name, '|', len(report['layers']), 'layers probed')
    print('  flags:', report['flags'])
    print('  final distribution:', report['final_distribution']['valid_distribution'],
          'entropy', report['final_distribution']['entropy_nats'],
          'argmax', report['final_distribution']['argmax_id'])
    print('  ', report['interpretation'])
    print('  reports:', args.out, args.markdown)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
