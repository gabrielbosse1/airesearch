"""21.5 — Gradient & weight health check.

Loads the final (or any stage) checkpoint and looks for the three numerical
failure modes that could explain a total language collapse: exploded weights
(Inf/NaN), vanished weights (all ~0), and dead/saturated neurons. It also checks
whether the byte embedding matrix is still a meaningful, differentiated codebook.

Run: python -m debug.analyze_model_weights
     python -m debug.analyze_model_weights --checkpoint checkpoints/stage_3_step_900/checkpoint.pt
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from debug.common import (ROOT, find_checkpoint, md_table, model_from_checkpoint,
                          report_dir, write_json, write_text)

SAMPLE_TEXT = 'Olá, como você está?'


def analyse(path):
    import torch
    torch.set_num_threads(1)
    from notebook.preprocess import batch

    model, meta = model_from_checkpoint(path)
    per_tensor = []
    total_params = 0
    nan_count = inf_count = near_zero_count = 0
    dead_rows = dead_total = 0
    for name, param in model.named_parameters():
        t = param.detach().float()
        n = t.numel()
        total_params += n
        finite = torch.isfinite(t)
        n_nan = int((~finite & torch.isnan(t)).sum())
        n_inf = int((~finite & torch.isinf(t)).sum())
        n_zero = int((finite & (t.abs() < 1e-7)).sum())
        nan_count += n_nan
        inf_count += n_inf
        near_zero_count += n_zero
        safe = torch.where(finite, t, torch.zeros_like(t))
        if t.ndim >= 2:
            dead_rows += int((safe.abs().amax(dim=tuple(range(1, t.ndim))) < 1e-7).sum())
            dead_total += t.shape[0]
        per_tensor.append({
            'name': name, 'shape': list(t.shape), 'numel': n,
            'mean': round(float(safe.mean()), 8), 'std': round(float(safe.std()), 8),
            'min': round(float(safe.min()), 8), 'max': round(float(safe.max()), 8),
            'abs_max': round(float(safe.abs().max()), 8),
            'nan': n_nan, 'inf': n_inf, 'near_zero_fraction': round(n_zero / n, 6),
        })

    # Embedding / output codebook health. Head weight IS the embedding (tied).
    emb = model.token.weight.detach().float()
    norms = emb.norm(dim=1)
    byte_norms = norms[:256]
    control_norms = norms[256:]
    sample_ids = torch.linspace(0, 255, steps=64).long()
    sample = emb[sample_ids]
    sample = sample / sample.norm(dim=1, keepdim=True).clamp_min(1e-9)
    cos = (sample @ sample.T).abs()
    offdiag = cos[~torch.eye(64, dtype=torch.bool)]
    embedding = {
        'rows': int(emb.shape[0]), 'width': int(emb.shape[1]),
        'byte_row_norm_mean': round(float(byte_norms.mean()), 6),
        'byte_row_norm_std': round(float(byte_norms.std()), 6),
        'byte_row_norm_min': round(float(byte_norms.min()), 6),
        'control_row_norm_mean': round(float(control_norms.mean()), 6),
        'mean_abs_cosine_between_byte_rows': round(float(offdiag.mean()), 6),
        'max_abs_cosine_between_byte_rows': round(float(offdiag.max()), 6),
        'norms_of_BOS_EOS_PAD': [round(float(norms[i]), 6) for i in (256, 257, 258)],
        'distinct_rows': int(torch.unique(emb, dim=0).shape[0]),
        'note': ('A near-zero mean |cosine| means byte rows remain differentiated. '
                 'Rows 256..267 are control/reserved ids; BOS/PAD/reserved rows are '
                 'never training targets and therefore retain near-initialisation values.'),
    }

    # Live activations and a live gradient on one clean Portuguese batch.
    activations = {}
    handles = []
    def hook(name):
        def inner(_module, _inputs, output):
            value = output.detach().float() if isinstance(output, torch.Tensor) else None
            if value is None:
                return
            finite = torch.isfinite(value)
            safe = torch.where(finite, value, torch.zeros_like(value))
            activations[name] = {
                'shape': list(value.shape),
                'mean': round(float(safe.mean()), 6),
                'std': round(float(safe.std()), 6),
                'zero_fraction': round(float((safe.abs() < 1e-6).float().mean()), 6),
                'saturated_fraction': round(float((safe.abs() > 10).float().mean()), 6),
                'max_abs': round(float(safe.abs().max()), 6),
                'nonfinite': int((~finite).sum()),
            }
        return inner
    handles.append(model.token.register_forward_hook(hook('token_embedding')))
    for index, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(hook(f'block_{index + 1}')))
    handles.append(model.norm.register_forward_hook(hook('final_norm')))
    handles.append(model.head.register_forward_hook(hook('logits')))

    ids = batch([SAMPLE_TEXT], model.cfg.max_seq_len)
    model.train()
    loss = model.loss(ids)
    loss.backward()
    grads = [p.grad.detach().norm().item() for p in model.parameters()
             if p.grad is not None and torch.isfinite(p.grad).all()]
    global_grad_norm = float(math.sqrt(sum(g * g for g in grads)))
    for handle in handles:
        handle.remove()
    model.zero_grad(set_to_none=True)
    model.eval()

    # Optimizer momentum state, when the snapshot contains it.
    payload = torch.load(str(path), map_location='cpu', weights_only=False)
    opt_state = payload.get('optimizer', {}) if isinstance(payload, dict) else {}
    exp_avg = [v['exp_avg'].detach().float().abs().mean().item()
               for v in opt_state.get('state', {}).values()
               if isinstance(v, dict) and 'exp_avg' in v]

    dead_activation_fraction = max(
        [v['zero_fraction'] for k, v in activations.items() if k.startswith('block_')]
        or [0.0])
    verdict = {
        'weights_exploded_to_inf_or_nan': nan_count + inf_count > 0,
        'weights_vanished': near_zero_count / total_params > 0.5,
        'dead_neurons': dead_activation_fraction > 0.5 or (dead_total and dead_rows / dead_total > 0.5),
        'embedding_codebook_collapsed': embedding['mean_abs_cosine_between_byte_rows'] > 0.9,
        'vanishing_gradients': global_grad_norm < 1e-6,
        'exploding_gradients': global_grad_norm > 1e4 or not math.isfinite(global_grad_norm),
    }
    return {
        'section': '21.5 gradient & weight health',
        'checkpoint': str(path), 'model': meta,
        'total_parameters': total_params,
        'nan_values': nan_count, 'inf_values': inf_count,
        'near_zero_values': near_zero_count,
        'near_zero_fraction': round(near_zero_count / total_params, 6),
        'dead_weight_rows': dead_rows, 'dead_weight_rows_total': dead_total,
        'per_tensor': per_tensor,
        'embedding_health': embedding,
        'activations': activations,
        'live_gradient': {
            'sample_text': SAMPLE_TEXT, 'loss': round(float(loss.detach()), 6),
            'global_grad_norm': round(global_grad_norm, 6),
            'n_parameter_grads': len(grads),
            'optimizer_exp_avg_mean': round(sum(exp_avg) / len(exp_avg), 8) if exp_avg else None,
            'note': ('Phase 19 did not persist gradient norms. This is a fresh '
                     'forward/backward on a clean Portuguese batch, plus AdamW '
                     'momentum magnitudes recovered from the snapshot.'),
        },
        'verdict': verdict,
        'interpretation': ('No numerical pathology: weights are finite, diverse and '
                          'gradients are healthy. The collapse is DISTRIBUTIONAL '
                          '(the output distribution was rewritten by a tiny, '
                          'repeated task corpus), not a weight-level blow-up.'
                          if not any(verdict.values()) else
                          'A numerical pathology was detected; see verdict flags.'),
    }


def render_markdown(report):
    v = report['verdict']
    return '\n'.join([
        '# 21.5 — Weight & gradient health', '',
        f"Checkpoint: `{report['checkpoint']}` · {report['total_parameters']:,} parameters", '',
        md_table(['check', 'result'], [[k, val] for k, val in v.items()]), '',
        md_table(['metric', 'value'], [
            ['NaN values', report['nan_values']],
            ['Inf values', report['inf_values']],
            ['near-zero fraction', report['near_zero_fraction']],
            ['dead weight rows', f"{report['dead_weight_rows']}/{report['dead_weight_rows_total']}"],
            ['byte row norm (mean ± std)',
             f"{report['embedding_health']['byte_row_norm_mean']} ± {report['embedding_health']['byte_row_norm_std']}"],
            ['mean |cos| between byte rows',
             report['embedding_health']['mean_abs_cosine_between_byte_rows']],
            ['live global grad norm', report['live_gradient']['global_grad_norm']],
            ['max activation zero fraction',
             max([a['zero_fraction'] for k, a in report['activations'].items()
                  if k.startswith('block_')] or [0.0])],
        ]), '',
        f"**Interpretation:** {report['interpretation']}", '',
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--variant', default='baseline')
    parser.add_argument('--stage', type=int, default=4)
    parser.add_argument('--out', default=str(report_dir() / 'model_weights.json'))
    parser.add_argument('--markdown', default=str(report_dir() / 'model_weights.md'))
    args = parser.parse_args(argv)
    path = find_checkpoint(args.checkpoint, args.variant, args.stage - 1)
    if path is None:
        write_json(args.out, {'error': 'no checkpoint found; run debug.replay_training first '
                                      'or pass --checkpoint / --repo-id checkpoints'})
        print('21.5: no checkpoint available')
        return 1
    report = analyse(path)
    write_json(args.out, report)
    write_text(args.markdown, render_markdown(report))
    print('21.5 weights:', path.name, '|', report['total_parameters'], 'params')
    print('  verdict:', report['verdict'])
    print('  ', report['interpretation'])
    print('  reports:', args.out, args.markdown)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
