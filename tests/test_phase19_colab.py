import json
import random
import time
from pathlib import Path
import pytest
import torch

from notebook.preprocess import encode, batch, apply_edits, sample_mix, synthetic
from notebook.phone_model import Config, PhoneLM, export_onnx
from notebook.tools import reward, run_search_action, Memory
from notebook.hf_upload import save_checkpoint, load_checkpoint, upload_checkpoint, latest_checkpoint
from notebook.evaluate import evaluate
from notebook import train as training

torch.set_num_threads(1)

def tiny():
    return PhoneLM(Config(d_model=16, n_layers=2, n_heads=2, d_ff=32, max_seq_len=256, gradient_checkpointing=True))

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

def test_tools_and_real_inputs():
    assert reward('code', 'def sort_list(xs):\n    return sorted(xs)') == 1
    assert reward('code', 'def broken(') == 0
    assert reward('minecraft', 'minecraft.place_block("stone", 0, 64, 0)') == 1
    assert reward('edit', '[EDIT pos=3 delete=1 bytes=c3aa]', 'voce', 'você') == .5
    assert reward('pt', 'Olá, estou bem!') == .5
    assert run_search_action('Action: search_ddg("clima em São Paulo")', lambda q: q.encode()) == 'clima em São Paulo'.encode()
    assert run_search_action('Action: search_ddg(__import__("os"))', lambda q: b'bad') is None
    memory = Memory(); memory.remember('animal', 'capivara'); assert memory.recall('animal') == 'capivara'

def test_training_and_model_eval():
    model = tiny(); model.train()
    before = model.token.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    x = batch(['Request: Olá, como você está?\nAnswer: Estou bem.', 'Question: sort a list\nCode: def sort_list(xs): return sorted(xs)'], 128)
    model.loss(x).backward(); opt.step()
    assert not torch.equal(before, model.token.weight)
    rows = evaluate(model, max_new=2)
    assert [r['case'] for r in rows] == ['pt-br', 'code', 'minecraft', 'search', 'edit', 'memory']
    assert all(isinstance(r['pass'], bool) and isinstance(r['output'], str) for r in rows)
    model.freeze(1)
    assert all(not p.requires_grad for p in model.blocks[0].parameters())
    assert all(p.requires_grad for p in model.blocks[1].parameters())

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
    groups = {'pt': ['Olá! Estou bem.'], 'en': ['Hello, how are you?'], 'code': ['def f(x): return x']}
    groups.update(synthetic())
    model = tiny()
    # A short first session reaches one checkpoint then pauses at its deadline.
    training.train(model, groups, 'token', 'me/repo', time.time() + .01, stage_seconds=.2, upload_seconds=.001, micro_batch=1, accumulation=1)
    assert saved
    state = load_checkpoint(saved[-1]); assert {'optimizer', 'scaler', 'random', 'torch_rng', 'elapsed'} <= state.keys()
    model2 = tiny()
    done = training.train(model2, groups, 'token', 'me/repo', time.time()+20, stage_seconds=.001, upload_seconds=60, micro_batch=1, accumulation=1, resume=state)
    assert done and load_checkpoint(saved[-1])['stage'] == 3

def test_onnx_dynamic(tmp_path):
    ort = pytest.importorskip('onnxruntime')
    model = tiny().eval(); path = tmp_path/'model.onnx'; export_onnx(model, str(path))
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    for length in (3, 9):
        ids = torch.randint(0, 256, (1, length))
        actual = session.run(None, {'input_ids': ids.numpy()})[0]
        torch.testing.assert_close(torch.from_numpy(actual), model(ids).detach(), atol=2e-4, rtol=2e-4)

def test_notebook_is_self_contained():
    import nbformat
    nb = nbformat.read('notebook/phase19_colab.ipynb', as_version=4)
    nbformat.validate(nb)
    content = '\n'.join(cell.source for cell in nb.cells).replace('phase19.', 'notebook.')
    for path in Path('notebook').glob('*.py'):
        if path.name == 'build_notebook.py': continue
        assert path.read_text() in content.replace('\\n', '\n').replace('\\"','"').replace("\\'", "'") or repr(path.read_text()) in content

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
    monkeypatch.setattr(datasets, 'load_dataset', lambda *a, **kw: iter([{'text': 'olá ' * 20}] * 30))
    monkeypatch.setattr(discovery, 'CANDIDATES', {'pt': ['bad', 'good']})
    groups = discovery.collect(str(tmp_path), max_rows=3, max_bytes=1000)
    assert len(groups['pt']) == 3
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    assert manifest[0]['status'].startswith('skipped') and manifest[1]['license'] == 'mit'
    assert discovery.discover(limit=1)[0]['accepted_license']

def test_notebook_embedded_unpack(tmp_path):
    import nbformat, subprocess, sys
    nb = nbformat.read('notebook/phase19_colab.ipynb', as_version=4)
    snippets = [c.source for c in nb.cells if c.cell_type == 'code' and c.source.startswith("Path('phase19').mkdir")]
    assert len(snippets) == 8
    subprocess.run([sys.executable, '-c', 'from pathlib import Path\n'+'\n'.join(snippets)+'\nfrom phase19.phone_model import PhoneLM\nprint(PhoneLM.__name__)'], cwd=tmp_path, check=True)
