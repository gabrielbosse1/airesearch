import json
import random
import re
import subprocess
import time
from pathlib import Path
import pytest
import torch

from notebook.preprocess import encode, batch, apply_edits, sample_mix, synthetic
from notebook.phone_model import Config, PhoneLM, export_onnx, generate
from notebook.tools import reward, run_search_action, Memory
from notebook.hf_upload import save_checkpoint, load_checkpoint, upload_checkpoint, latest_checkpoint
from notebook.evaluate import evaluate
from notebook.mineflayer_training_data import mineflayer_records
from notebook import train as training

torch.set_num_threads(1)

def tiny():
    return PhoneLM(Config(d_model=16, n_layers=2, n_heads=2, d_ff=32, max_seq_len=256, gradient_checkpointing=True))

def groups():
    data = {'pt': ['Olá! Estou bem.'], 'en': ['Hello, how are you?'],
            'code': ['def f(x): return x']}
    data.update(synthetic())
    data['mineflayer'] = mineflayer_records()[:8]
    return data

def test_utf8_batch_and_edit():
    assert encode('Olá', 8) == [256, *'Olá'.encode(), 257]
    assert encode('abcdef', 4) == [256, 97, 98, 257]
    x = batch(['Olá', 'x'], 10)
    assert x[1, -1] == 258
    for line in synthetic()['edit']:
        wrong = line.split('Draft: ')[1].split('\n')[0]
        edit = line.split('\n')[2]
        assert apply_edits(wrong, edit) == line.split('Answer: ')[1]
    with pytest.raises(ValueError): apply_edits('a', '[EDIT pos=4 delete=1 bytes=62]')
    with pytest.raises(ValueError): sample_mix({'pt': []}, {'pt': 1}, 1)
    # Sequence limits never leave a partial UTF-8 character before EOS.
    assert bytes(encode('aé', 4)[1:-1]).decode('utf8') == 'a'
    with pytest.raises(ValueError): encode('x', 1)
    with pytest.raises(ValueError): batch([], 8)
    with pytest.raises(ValueError): sample_mix({'pt': ['x']}, {'pt': 0}, 1)

def test_model_configuration_and_generation_mode():
    with pytest.raises(ValueError, match='divisible'):
        Config(d_model=15, n_heads=2)
    with pytest.raises(ValueError, match='vocab_size'):
        Config(vocab_size=258)
    model = tiny().train()
    assert generate(model, 'Olá', max_new=0) == ''
    assert model.training  # generation must not surprise a training caller.
    with pytest.raises(ValueError, match='temperature'):
        generate(model, 'Olá', temperature=0)

def test_tools_and_real_inputs():
    assert reward('code', 'def sort_list(xs):\n    return sorted(xs)') == 1
    assert reward('code', 'def broken(') == 0
    assert reward('minecraft', 'minecraft.place_block("stone", 0, 64, 0)') == 1
    assert reward('mineflayer', 'await bot.pathfinder.goto(new GoalNear(100, 64, 200, 1));') == 1
    assert reward('mineflayer', 'print("not javascript")') == 0
    assert reward('mineflayer', "await bot.dig(bot.blockAt(new Vec3(0, 64, 0)));") == 1
    assert reward('edit', '[EDIT pos=3 delete=1 bytes=c3aa]', 'voce', 'você') == .5
    assert reward('pt', 'Olá, estou bem!') == .5
    assert run_search_action('Action: search_ddg("clima em São Paulo")', lambda q: q.encode()) == 'clima em São Paulo'.encode()
    assert run_search_action('Action: search_ddg(__import__("os"))', lambda q: b'bad') is None
    memory = Memory(); memory.remember('animal', 'capivara'); assert memory.recall('animal') == 'capivara'

def test_training_and_model_eval():
    model = tiny(); model.train()
    before = model.token.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    data = groups()
    x = batch([data['pt'][0] + '\nAnswer: Estou bem.', 'Question: sort a list\nCode: def sort_list(xs): return sorted(xs)',
               data['mineflayer'][0]], 128)
    model.loss(x).backward(); opt.step()
    assert not torch.equal(before, model.token.weight)
    rows = evaluate(model, max_new=2)
    assert [r['case'] for r in rows] == ['pt-br', 'code', 'mineflayer-navigate', 'mineflayer-mine_diamond',
                                         'mineflayer-craft_pickaxe', 'mineflayer-attack_zombie',
                                         'search', 'edit', 'memory']
    assert all(isinstance(r['pass'], bool) and isinstance(r['output'], str) for r in rows)
    model.freeze(1)
    assert all(not p.requires_grad for p in model.blocks[0].parameters())
    assert all(p.requires_grad for p in model.blocks[1].parameters())

def test_stage_two_mix_and_freeze_contract():
    assert training.MIXES[1] == dict(code=40, mineflayer=40, pt=20)
    assert training.FREEZE[1] == 6
    assert training.MIXES[3]['mineflayer'] == 15
    with pytest.raises(ValueError, match='mineflayer'):
        training.check_groups({'pt': ['a'], 'en': ['b'], 'code': ['c'], 'react': ['d'],
                               'search': ['e'], 'edit': ['f']})
    training.check_groups({'pt': ['a'], 'code': ['b'], 'mineflayer': ['c']}, stages=[1])

def test_snapshots_and_upload(tmp_path, monkeypatch):
    class FakeAPI:
        def __init__(self): self.files = []; self.uploaded = None
        def create_repo(self, **kw): self.created = kw
        def upload_file(self, **kw): self.files.append(kw['path_in_repo']); self.uploaded = kw
        def list_repo_files(self, **kw): return self.files + ['checkpoints/stage_1_step_2/checkpoint.pt', 'bad']
    api = FakeAPI()
    state = {'value': torch.tensor([42])}
    file = upload_checkpoint(state, 2, 11, 'redacted', 'me/private', root=tmp_path, api=api)
    assert load_checkpoint(file)['value'].item() == 42
    assert api.created['private'] and api.uploaded['path_in_repo'] == 'checkpoints/stage_2_step_11/checkpoint.pt'
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda **kw: kw['filename'])
    assert latest_checkpoint('me/private', 'redacted', api=api) == 'checkpoints/stage_2_step_11/checkpoint.pt'

def test_real_training_and_resume(tmp_path, monkeypatch):
    saved = []
    def mock_upload(state, stage, step, token, repo):
        file = save_checkpoint(state, stage, step, root=tmp_path)
        saved.append(file)
    monkeypatch.setattr(training, 'upload_checkpoint', mock_upload)
    data = groups()
    model = tiny()
    # A short first session reaches one checkpoint then pauses at its deadline.
    training.train(model, data, 'token', 'me/repo', time.time() + .01, stage_seconds=.2,
                   upload_seconds=.001, micro_batch=1, accumulation=1)
    assert saved
    state = load_checkpoint(saved[-1]); assert {'optimizer', 'scaler', 'random', 'torch_rng', 'elapsed'} <= state.keys()
    model2 = tiny()
    done = training.train(model2, data, 'token', 'me/repo', time.time() + 20, stage_seconds=.001,
                          upload_seconds=60, micro_batch=1, accumulation=1, resume=state)
    assert done and load_checkpoint(saved[-1])['stage'] == 3

def test_stage_selection_and_resume_mismatch(tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr(training, 'upload_checkpoint',
                        lambda state, stage, step, token, repo: saved.append(save_checkpoint(state, stage, step, root=tmp_path)))
    data = groups()
    model = tiny()
    assert training.train(model, data, 't', 'me/repo', time.time() + 20, stage_seconds=.001,
                          upload_seconds=60, micro_batch=1, accumulation=1, stages=[1])
    assert load_checkpoint(saved[-1])['stage'] == 1
    with pytest.raises(ValueError, match='stages'):
        training.train(model, data, 't', 'me/repo', time.time() + 20, stages=[4])

def test_onnx_dynamic(tmp_path):
    ort = pytest.importorskip('onnxruntime')
    model = tiny().eval(); path = tmp_path/'model.onnx'; export_onnx(model, str(path))
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    for length in (3, 9):
        ids = torch.randint(0, 256, (1, length))
        actual = session.run(None, {'input_ids': ids.numpy()})[0]
        torch.testing.assert_close(torch.from_numpy(actual), model(ids).detach(), atol=2e-4, rtol=2e-4)

def test_notebook_clones_repo_and_imports_sources():
    import nbformat
    nb = nbformat.read('notebook/phase19_colab.ipynb', as_version=4)
    nbformat.validate(nb)
    sources = [cell.source for cell in nb.cells]
    code_cells = [cell.source for cell in nb.cells if cell.cell_type == 'code']
    clone = code_cells[0]
    assert 'git' in clone and 'clone' in clone and 'REPO_URL' in clone
    assert '%cd' in clone
    assert any('install' in cell and 'requirements.txt' in cell for cell in code_cells[:2])
    remote = subprocess.run(['git', 'config', '--get', 'remote.origin.url'],
                            capture_output=True, text=True, check=True).stdout.strip()
    assert remote.rstrip('/') in '\n'.join(sources)
    # The clone must precede every repository import.
    first_import = next(i for i, cell in enumerate(nb.cells)
                        if cell.cell_type == 'code' and re.search(r'from notebook[.\s]|import notebook\b', cell.source))
    assert first_import >= 2, (first_import, clone[:80])
    assert 'sys.path.insert' in clone and 'import notebook' in clone
    # No embedded module sources from the old self-contained notebook.
    assert not any("Path('phase19').mkdir" in cell for cell in sources)
    for path in Path('notebook').glob('*.py'):
        if path.name == 'build_notebook.py':
            continue
        assert path.read_text(encoding='utf8') not in '\n'.join(sources)
    for cell in code_cells:
        for module in re.findall(r'from notebook\.(\w+) import', cell):
            assert Path(f'notebook/{module}.py').exists(), module

def test_committed_notebook_matches_its_builder():
    import importlib
    from notebook import build_notebook
    importlib.reload(build_notebook)
    expected = json.dumps(build_notebook.notebook, ensure_ascii=False, indent=1) + '\n'
    assert Path('notebook/phase19_colab.ipynb').read_text(encoding='utf8') == expected


def test_notebook_has_mineflayer_pipeline():
    import nbformat
    nb = nbformat.read('notebook/phase19_colab.ipynb', as_version=4)
    text = '\n'.join(cell.source for cell in nb.cells)
    for needle in ('mineflayer_training_data', 'mineflayer_records', 'validate_trajectories',
                   'evaluate_mineflayer', 'stages=[0]', 'stages=[1]', 'stages=[2]', 'stages=[3]',
                   '40% Mineflayer', 'Freeze', 'HF_TOKEN'):
        assert needle in text, needle
    assert 'mineflayer-py' in text  # the notebook explains that the package does not exist
    assert 'duckduckgo-search' in text and 'ddgs' in text
    # A checkpoint is consumed once: later stage cells must not rewind the model.
    assert text.count('RESUME = None') >= 4
    assert 'def resume_for(stage):' in text and 'resume=resume_for(0)' in text
    # Data is found by the notebook, never requested from the user.
    assert 'files.upload()' not in text and 'files.upload' not in text
    assert 'fill_missing(groups' in text and 'reports=reports' in text
    assert 'no files were requested' in text
    assert 'REPO_BRANCH' in text

def test_discovery_license_and_stream_limits(tmp_path, monkeypatch):
    from notebook import dataset_discovery as discovery
    import huggingface_hub, datasets
    from types import SimpleNamespace
    class API:
        def dataset_info(self, name):
            return SimpleNamespace(card_data={'license': 'mit' if name == 'good' else 'unknown', 'language': ['pt']}, size_in_bytes=123)
        def list_datasets(self, **kwargs):
            return [SimpleNamespace(id='good', card_data={'license':'mit','language':['pt']}, size_in_bytes=123)]
    monkeypatch.setattr(huggingface_hub, 'HfApi', API)
    # Portuguese-looking text with several distinct stopwords; a single
    # repeated word must not be enough to label a corpus as Portuguese.
    pt_row = {'text': 'Olá, você está bem? Não, mas para com isso agora e então farei mais.'}
    monkeypatch.setattr(datasets, 'load_dataset', lambda *a, **kw: iter([dict(pt_row)] * 30))
    monkeypatch.setattr(discovery, 'CANDIDATES', {'pt': ['bad', 'good']})
    groups_ = discovery.collect(str(tmp_path), max_rows=3, max_bytes=1000, min_rows=10)
    assert len(groups_['pt']) >= 1
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    bad = next(row for row in manifest if row.get('name') == 'bad')
    good = next(row for row in manifest if row.get('name') == 'good' and row.get('kind') == 'pt')
    assert bad['status'].startswith('skipped') and 'license' in good
    assert good['license'] == 'mit'
    assert discovery.discover(limit=1)[0]['accepted_license']


def test_discovery_searches_instead_of_asking(tmp_path, monkeypatch, capsys):
    """A failed curated source is replaced by another *found* dataset, not an upload."""
    from notebook import dataset_discovery as discovery
    import huggingface_hub, datasets
    from types import SimpleNamespace
    class API:
        def dataset_info(self, name):
            if name == 'curated/dead':
                raise ConnectionError('source unavailable')
            return SimpleNamespace(card_data={'license': 'odc-by', 'language': ['pt']},
                                   size_in_bytes=10, tags=[])
        def list_datasets(self, **kwargs):
            return [SimpleNamespace(id='search/found-pt', card_data={'license': 'odc-by',
                    'language': ['pt']}, size_in_bytes=10, downloads=42, tags=[])]
    monkeypatch.setattr(huggingface_hub, 'HfApi', API)
    monkeypatch.setattr(datasets, 'load_dataset',
                        lambda *a, **kw: iter([{'text': 'Você não está bem, mas o Brasil é grande.'}] * 20))
    monkeypatch.setattr(discovery, 'CANDIDATES', {'pt': ['curated/dead'], 'en': [], 'code': [], 'react': []})
    reports = discovery.discover(limit=2, api=API())
    assert 'search/found-pt' in discovery.search_candidates(reports)['pt']
    groups_ = discovery.collect(str(tmp_path), reports=reports, repo_root=str(tmp_path),
                                max_rows=30, max_bytes=100_000, min_rows=5, max_candidates=2)
    assert groups_['pt'], 'the discovered dataset was never streamed'
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    assert manifest[-1]['upload_required'] is False
    assert any(row.get('name') == 'search/found-pt' and row.get('status') == 'streamed'
               for row in manifest)
    assert 'files.upload' not in capsys.readouterr().out


def test_real_datasets_streaming_end_to_end(tmp_path, monkeypatch):
    """A verified source streams through the real `datasets` library, not a stub."""
    from notebook import dataset_discovery as discovery
    import huggingface_hub
    from types import SimpleNamespace
    local = tmp_path / 'localds'
    local.mkdir()
    rows = [
        {'text': 'Olá, você está bem? Não, mas para com isso agora e então farei mais.'},
        {'text': 'O Brasil é grande e você pode encontrar muitas coisas aqui todos os dias.'},
        {'instruction': 'Write a function', 'output': 'def f(x):\n    return x + 1'},
    ]
    (local / 'data.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n',
                                      encoding='utf8')

    class API:
        def dataset_info(self, name):
            return SimpleNamespace(card_data={'license': 'mit', 'language': ['pt']},
                                   tags=[], size_in_bytes=1, gated=False, private=False)
        def list_datasets(self, **kwargs):
            return []
    monkeypatch.setattr(huggingface_hub, 'HfApi', API)
    monkeypatch.setattr(discovery, 'CANDIDATES',
                        {'pt': [str(local)], 'en': [], 'code': [], 'react': []})
    groups_ = discovery.collect(str(tmp_path / 'cache'), search=False,
                                repo_root=str(tmp_path), min_rows=5, max_rows=20)
    assert len(groups_['pt']) == 2, groups_['pt']
    manifest = json.loads((tmp_path / 'cache' / 'manifest.json').read_text())
    assert any(row.get('name') == str(local) and row.get('status') == 'streamed'
               for row in manifest)


def test_local_corpus_is_real_licensed_data(tmp_path, monkeypatch):
    """Offline Hub still yields real en/code/react rows from the checkout."""
    from notebook import dataset_discovery as discovery
    import huggingface_hub, datasets
    from types import SimpleNamespace
    class DeadAPI:
        def dataset_info(self, name): raise ConnectionError('offline')
        def list_datasets(self, **kwargs): raise ConnectionError('offline')
    monkeypatch.setattr(huggingface_hub, 'HfApi', DeadAPI)
    monkeypatch.setattr(datasets, 'load_dataset', lambda *a, **kw: iter([]))
    monkeypatch.setattr(discovery, 'CANDIDATES', {'pt': [], 'en': [], 'code': [], 'react': []})
    groups_ = discovery.collect(str(tmp_path), search=False, max_rows=200, max_bytes=500_000)
    assert groups_['en'] and groups_['code'] and groups_['react']
    assert not groups_['pt'], 'Portuguese must never be fabricated locally'
    with pytest.raises(RuntimeError, match='Automatic dataset search'):
        discovery.fill_missing(groups_, cache_dir=str(tmp_path), search_limit=1)


def test_discovery_manual_data_and_known_configs(tmp_path):
    from notebook import dataset_discovery as discovery
    assert discovery.dataset_config('HuggingFaceFW/fineweb-edu', 'en') == 'sample-10BT'
    assert discovery.dataset_config('HuggingFaceFW/fineweb-2', 'pt') == 'por_Latn'
    assert discovery.dataset_config('allenai/c4', 'pt') == 'pt'
    assert discovery.dataset_config('allenai/c4', 'en') == 'en'
    assert 'odc-by' in discovery.ALLOWED
    # Every curated source must have a reviewed, permissive declared license.
    # OPUS-100 (license:unknown) and HuggingFaceH4/CodeAlpaca_20K (license:cc)
    # were silently skipped in Colab and used to leave groups empty.
    for names in discovery.CANDIDATES.values():
        for name in names:
            assert discovery.CURATED_LICENSES.get(name) in discovery.ALLOWED, name
    assert 'Helsinki-NLP/opus-100' not in str(discovery.CANDIDATES)
    assert 'HuggingFaceH4/CodeAlpaca_20K' not in str(discovery.CANDIDATES)
    manual = tmp_path / 'manual_data'; manual.mkdir()
    (manual / 'pt.txt').write_text('Olá mundo\nOlá mundo\n', encoding='utf8')
    groups_ = {'pt': ['texto existente'], 'en': [], 'code': []}
    first = discovery.merge_manual_data(groups_, manual)
    second = discovery.merge_manual_data(groups_, manual)
    assert groups_['pt'] == ['texto existente', 'Olá mundo']
    assert first['pt']['rows_added'] == 1 and second['pt']['rows_added'] == 0
    (manual / 'en.txt').write_bytes(b'not valid \xff utf8')
    with pytest.raises(ValueError, match='valid UTF-8'):
        discovery.merge_manual_data(groups_, manual)
