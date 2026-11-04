"""Deterministic CPU replay of the Phase 19 four-stage pipeline.

The real Colab run's logs and checkpoints are gitignored and were never
uploaded to this checkout, so there is nothing to autopsy directly. This module
re-runs the *same* code paths at reduced scale on CPU:

  * ``notebook.train.MIXES`` / ``FREEZE`` / ``draw`` (the real sampler),
  * ``notebook.preprocess.batch`` (real byte encoding),
  * the real ``Cell 25`` probe prompts and ``notebook.tools.reward``,
  * AdamW + per-stage cosine LR + global grad clip 1.0 + a fresh optimizer per
    stage, matching ``notebook.train.train``.

It writes per-step metrics, a checkpoint at the END of each stage, and a
per-stage "Olá, como" 100-byte generation result. Absolute quality numbers do
NOT transfer to the 4-hour T4 run; the mechanism (which stage destroys the byte
distribution) does.

Run: python -m debug.replay_training --variant baseline --steps 120
     python -m debug.replay_training --variant fixed --steps 120
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import asdict
from pathlib import Path

from debug.common import (CHECKPOINTS, REPLAY, ROOT, coherence, generate_raw,
                          report_dir, write_json)

# A small, clean, authored PT-BR probe corpus. It is a CONTROLLED SUBSTRATE, not
# a claim that these exact sentences were in the Colab corpus. The real run got
# Portuguese from Hugging Face; we cannot replay that offline. All sentences are
# valid UTF-8 and include accented multi-byte characters.
PT_SENTENCES = [
    'Olá, como vai você hoje? Estou bem, obrigado.',
    'Olá, como está o tempo em São Paulo? Não tenho certeza.',
    'Olá, como posso ajudar você com o código?',
    'Bom dia! Você está pronto para começar o trabalho?',
    'Estou bem e a resposta está correta, obrigada.',
    'Não sei a resposta, mas posso procurar a informação.',
    'Meu animal favorito é a capivara, um bicho muito tranquilo.',
    'Você pode me dizer qual é o seu nome?',
    'Obrigado pela ajuda, tudo está ótimo agora.',
    'Como você está? Estou bem, e você?',
    'Para fazer isso, precisamos de mais tempo e de dados.',
    'Aqui no Brasil o português tem muitos acentos: á, ã, ç, ê.',
    'Quando o modelo erra, é preciso verificar a codificação.',
    'Não é possível concluir sem observar os resultados reais.',
    'Vamos escrever uma função simples em Python para ordenar a lista.',
    'A resposta certa depende da pergunta e do contexto.',
    'Ele não tem dados suficientes para responder agora.',
    'Sobre a memória, o valor guardado é a capivara.',
    'Também é importante separar treino e avaliação.',
    'Entre as opções, a melhor é perguntar antes de agir.',
    'Olá, como você está? Estou muito bem, obrigado!',
    'Bom dia, como posso ajudar? Quero entender o problema.',
    'Não há erro de codificação nesta frase portuguesa.',
    'Você está certo: a sequência de bytes precisa ser válida.',
    'Minha resposta é simples e está escrita em português.',
    'Qual é o animal favorito? A capivara é o meu favorito.',
    'Estou aqui para ajudar com a tarefa de hoje.',
    'O código deve ser claro, curto e fácil de testar.',
    'Não podemos confiar em uma avaliação contaminada.',
    'Obrigada, a correção está correta e o texto ficou bom.',
    'Olá, como vai? Tudo bem comigo, e com você?',
    'São Paulo tem clima instável; consulte fontes atuais.',
    'A função ordena a lista e devolve o resultado.',
    'Não é seguro executar código gerado sem um sandbox.',
    'Está tudo pronto para a próxima etapa do treino.',
    'Olá, como foi o seu dia? Espero que tenha sido muito bom.',
    'Você sabia que a capivara é o maior roedor do mundo?',
    'Não tenho dados suficientes, por isso vou perguntar de novo.',
    'Estou verificando a codificação antes de continuar o treino.',
    'Obrigada por explicar, agora a resposta ficou bem mais clara.',
    'Bom dia! Qual é a pergunta que você quer fazer hoje?',
    'A lista precisa ser ordenada e o resultado deve ser conferido.',
    'Quando houver dúvida, é melhor não inventar uma resposta.',
    'Meu nome não importa; o importante é o conteúdo da resposta.',
    'Sobre o clima, consulte sempre uma fonte atualizada e confiável.',
    'Também precisamos separar os dados de treino dos dados de teste.',
    'Entre todas as opções, prefiro a que evita o esquecimento.',
    'Está tudo certo com a sequência de bytes deste texto?',
    'Você está certa: a mistura de dados precisa ser equilibrada.',
    'Não é possível confiar em um único exemplo repetido mil vezes.',
    'Olá, como posso ajudar com a correção deste pequeno texto?',
    'O modelo aprendeu quatro frases e esqueceu todo o resto.',
    'Para medir a qualidade, usamos uma frase que nunca foi treinada.',
    'A capivara aparece de novo nesta frase de verificação.',
    'Estou bem, obrigado, e desejo um ótimo dia para você.',
    'Qual é o resultado esperado? Ainda não sei dizer.',
    'Bom dia, vamos revisar o código e depois testar tudo.',
    'Não há motivo para trocar a codificação neste momento.',
    'O português brasileiro usa acentos em muitas palavras comuns.',
    'Você pode repetir a pergunta de uma forma mais simples?',
    'Minha resposta está pronta, mas ainda precisa de revisão.',
    'São Paulo, Joinville e Curitiba aparecem neste texto.',
    'Também é preciso registrar as métricas de cada etapa.',
    'Estou escrevendo mais uma frase para ampliar o corpus.',
    'Obrigado, a explicação anterior estava correta e completa.',
    'Como você está hoje? Estou muito bem, e agradeço a pergunta.',
    'Não podemos usar o mesmo texto para treinar e avaliar.',
    'A função deve devolver a lista ordenada sem modificar a original.',
    'Você tem alguma dúvida sobre a etapa de ferramentas?',
    'Bom dia, o treino começou cedo e terminou depois do almoço.',
    'Estou comparando dois modelos com a mesma semente aleatória.',
    'Olá, como vai o projeto? Está avançando bem, obrigado.',
    'A memória do modelo não substitui uma fonte confiável.',
    'Não invente resultados quando a busca não trouxer nada.',
    'Por favor, verifique se a saída é um texto UTF-8 válido.',
    'Qual é o animal favorito? Continuo escolhendo a capivara.',
    'Todo cuidado é pouco ao interpretar uma métrica isolada.',
    'Você está certo, a repetição prejudicou a qualidade final.',
    'Estou satisfeito com o teste, mas ainda falta um retreino completo.',
    'Olá, como você está? A resposta deve ser em português.',
]


def _truncate(text, limit):
    from notebook.preprocess import _utf8_prefix
    return _utf8_prefix(str(text), limit).decode('utf8')


def build_replay_groups(seq_len=96):
    """Real in-repo rows where available, plus the controlled PT substrate."""
    from notebook.mineflayer_training_data import mineflayer_records
    from notebook.preprocess import synthetic

    limit = seq_len - 2
    groups = {
        'pt': [_truncate(row, limit) for row in PT_SENTENCES],
        'en': [], 'code': [], 'react': [],
    }
    raw = ROOT / 'data' / 'github_raw'
    stories = raw / 'github_stories.jsonl'
    if stories.exists():
        for line in stories.read_text(encoding='utf8').splitlines()[:60]:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(payload.get('body') or payload.get('description') or '')
            if len(text) > 40:
                groups['en'].append(_truncate(text, limit))
    react = raw / 'github_react.jsonl'
    if react.exists():
        for line in react.read_text(encoding='utf8').splitlines()[:60]:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload.get('text'), str):
                groups['react'].append(_truncate(payload['text'], limit))
    # Pin the source directories: rglob over the whole checkout would silently
    # change the replay corpus every time a debug/ file is added, which would
    # invalidate A/B comparability. Only pre-Phase-21 source dirs are used.
    code_paths = []
    for directory in ('models', 'evaluation', 'notebook', 'data', 'training'):
        code_paths.extend(sorted((ROOT / directory).rglob('*.py')))
    for path in sorted(code_paths, key=lambda q: str(q.relative_to(ROOT))):
        if len(groups['code']) >= 60:
            break
        try:
            text = path.read_text(encoding='utf8')
        except (OSError, UnicodeDecodeError):
            continue
        if len(text) > 80:
            groups['code'].append(_truncate(text, limit))
    # The notebook EXTENDS groups with the small labelled fixtures; it does not
    # overwrite the real rows already found. Preserve that behaviour.
    for kind, rows in synthetic().items():
        groups.setdefault(kind, [])
        groups[kind] = list(dict.fromkeys(
            groups[kind] + [_truncate(row, limit) for row in rows]))
    groups['mineflayer'] = [_truncate(row, limit)
                            for row in mineflayer_records(seed=19)[:100]]
    groups = {k: v for k, v in groups.items() if v}
    return groups


# Held-out teacher-forced probes. These never appear in any training group and
# are NOT the same strings as the Cell 25 prompts. Cross-entropy on them is a
# much more sensitive forgetting signal than greedy free-running generation from
# a 243k-parameter model.
PT_HELDOUT = [
    'Olá, tudo bem? A resposta está pronta para você.',
    'Não consigo confirmar sem consultar uma fonte confiável.',
    'Obrigado, o texto em português ficou correto agora.',
    'Minha capivara está no jardim e gosta de água.',
    'Qual é a melhor maneira de verificar esta codificação?',
    'Estou escrevendo uma frase nova para medir o esquecimento.',
    'A memória guarda o valor e depois devolve a resposta.',
    'Você pode explicar de novo, por favor? Não entendi.',
    'Bom dia, o código precisa ser testado antes do envio.',
    'São Paulo e Joinville aparecem nesta frase acentuada.',
]
CODE_HELDOUT = [
    'def soma(a, b):\n    return a + b\n',
    'def ordena(lista):\n    return sorted(lista)\n',
    'class Item:\n    def __init__(self, nome):\n        self.nome = nome\n',
]


def heldout_loss(model, texts, seq_len):
    import torch
    from notebook.preprocess import batch
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return round(sum(float(model.loss(batch([t], seq_len))) for t in texts) / len(texts), 5)
    finally:
        model.train(was_training)


def _probe_pass(kind, text):
    from notebook.tools import reward
    if kind == 'pt-br':
        return bool(re.search(r'\b(estou|bem|olá|obrigad[oa]|você)\b', text.lower()))
    if kind == 'code':
        return bool(reward('code', text))
    if kind == 'search':
        return 'search_ddg(' in text
    if kind == 'edit':
        return bool(reward('edit', text, 'voce', 'você'))
    return 'capivara' in text.lower()


def evaluate_stage(model, stage, seq_len=96):
    from notebook.evaluate import CASES
    main_probe = generate_raw(model, 'Olá, como', max_new=100, mode='faithful')
    probes = []
    for kind, prompt in CASES:
        output = generate_raw(model, prompt, max_new=48, mode='faithful')
        probes.append({'case': kind, 'pass': _probe_pass(kind, output['text']),
                       'score': output['score'], 'gibberish': output['gibberish'],
                       'text': output['text'][:120]})
    return {
        'stage': stage,
        'heldout_pt_loss': heldout_loss(model, PT_HELDOUT, seq_len),
        'heldout_code_loss': heldout_loss(model, CODE_HELDOUT, seq_len),
        'coherence': main_probe['score'],
        'gibberish': main_probe['gibberish'],
        'valid_utf8': main_probe['valid_utf8'],
        'replacement_ratio': main_probe['replacement_ratio'],
        'pt_word_hits': main_probe['pt_word_hits'],
        'text': main_probe['text'][:160],
        'probes': probes,
        'probes_passed': sum(p['pass'] for p in probes),
    }


def run_replay(variant='baseline', steps=120, lr=2e-3, accumulation=6, seed=19,
               seq_len=96, out_dir=None, verbose=True):
    """Run all four stages and return a summary. ``variant`` is baseline|fixed."""
    import torch
    from notebook import train as training
    from notebook.phone_model import Config, PhoneLM
    from notebook.preprocess import BOS, PAD, batch

    if variant not in ('baseline', 'fixed'):
        raise ValueError('variant must be baseline or fixed')
    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)
    groups = build_replay_groups(seq_len)
    cfg = Config(d_model=48, n_layers=12, n_heads=4, d_ff=96,
                 max_seq_len=seq_len, dropout=0.0, gradient_checkpointing=False)
    model = PhoneLM(cfg)
    out_dir = Path(out_dir) if out_dir else REPORTS_DIR()
    replay_dir = out_dir / 'replay'
    ckpt_dir = out_dir / 'checkpoints'
    replay_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = replay_dir / f'{variant}_metrics.jsonl'
    metrics_path.write_text('', encoding='utf8')

    balanced = variant == 'fixed'
    stage_summaries = []
    global_step = 0
    started = time.monotonic()
    for stage, mix in enumerate(training.MIXES):
        model.freeze(min(training.FREEZE[stage], len(model.blocks)))
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=.01)
        model.train()
        losses = []
        for local_step in range(steps):
            fraction = local_step / max(1, steps - 1)
            current_lr = lr * 0.5 * (1 + math.cos(math.pi * fraction))
            for group in optimizer.param_groups:
                group['lr'] = current_lr
            optimizer.zero_grad(set_to_none=True)
            step_losses = []
            for _ in range(accumulation):
                texts = training.draw(groups, mix, 1, balanced=balanced, stage=stage)
                ids = batch(texts, cfg.max_seq_len)
                loss = model.loss(ids) / accumulation
                loss.backward()
                step_losses.append(float(loss.detach()) * accumulation)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            global_step += 1
            mean_loss = sum(step_losses) / len(step_losses)
            losses.append(mean_loss)
            with metrics_path.open('a', encoding='utf8') as handle:
                handle.write(json.dumps({
                    'kind': 'step', 'source': 'debug.replay_training',
                    'variant': variant, 'stage': stage + 1, 'step': global_step,
                    'loss': round(mean_loss, 6), 'lr': current_lr,
                    'grad_norm': round(grad_norm, 6),
                    'nan': not math.isfinite(mean_loss) or not math.isfinite(grad_norm),
                    'mixture': 'balanced' if balanced else 'legacy',
                }) + '\n')

        summary = evaluate_stage(model, stage + 1, cfg.max_seq_len)
        summary['mean_loss'] = round(sum(losses) / len(losses), 5)
        summary['final_loss'] = round(losses[-1], 5)
        stage_summaries.append(summary)
        with metrics_path.open('a', encoding='utf8') as handle:
            handle.write(json.dumps({
                'kind': 'eval', 'source': 'debug.replay_training',
                'variant': variant, 'stage': stage + 1, 'step': global_step,
                'loss': summary['final_loss'], 'lr': 0.0, 'grad_norm': None,
                'nan': False, 'coherence': summary['coherence'],
                'heldout_pt_loss': summary['heldout_pt_loss'],
                'heldout_code_loss': summary['heldout_code_loss'],
                'gibberish': summary['gibberish'],
                'probes_passed': summary['probes_passed'],
            }) + '\n')
        checkpoint = ckpt_dir / f'{variant}_stage{stage + 1}.pt'
        torch.save({
            'version': 'phase21-replay', 'variant': variant,
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'stage': stage, 'step': global_step, 'config': asdict(cfg),
            'metrics': summary,
        }, checkpoint)
        if verbose:
            print(f'  stage {stage + 1}: loss {summary["final_loss"]:.4f} '
                  f'coherence {summary["coherence"]:.3f} '
          f'heldout-pt {summary["heldout_pt_loss"]:.3f} '
                  f'gibberish={summary["gibberish"]} '
                  f'probes {summary["probes_passed"]}/5')

    payload = {
        'variant': variant, 'mixture': 'balanced' if balanced else 'legacy',
        'steps_per_stage': steps, 'accumulation': accumulation, 'lr': lr, 'seed': seed,
        'config': asdict(cfg),
        'group_counts': {k: len(v) for k, v in groups.items()},
        'stage_weights': [training.stage_mixture(groups, mix, i) if balanced else mix
                          for i, mix in enumerate(training.MIXES)],
        'stages': stage_summaries,
        'elapsed_seconds': round(time.monotonic() - started, 2),
        'metrics_path': str(metrics_path),
    }
    write_json(out_dir / f'{variant}_replay.json', payload)
    return payload


def REPORTS_DIR():
    return report_dir()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('baseline', 'fixed'), default='baseline')
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--accumulation', type=int, default=6)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--out-dir', default=str(report_dir()))
    args = parser.parse_args(argv)
    print(f'Phase 21 replay: variant={args.variant} steps={args.steps}/stage '
          f'lr={args.lr} accum={args.accumulation} seed={args.seed}')
    payload = run_replay(args.variant, args.steps, args.lr, args.accumulation,
                         args.seed, out_dir=Path(args.out_dir))
    print(f'  finished in {payload["elapsed_seconds"]}s; '
          f'metrics: {payload["metrics_path"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
