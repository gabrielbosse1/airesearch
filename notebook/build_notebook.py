"""Build the Phase 19 Colab notebook: clone the public repo, then train.

The notebook no longer embeds the repository source. It clones this repository,
installs its requirements and imports ``notebook.*`` modules from the checkout.
Run ``python -m notebook.build_notebook`` after editing notebook sources.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE = subprocess.run(['git', '-C', str(ROOT), 'config', '--get', 'remote.origin.url'],
                        capture_output=True, text=True, check=False).stdout.strip()
REPO_URL = REMOTE if REMOTE.startswith('http') else 'https://github.com/gabrielbosse1/airesearch.git'
# A notebook built on a feature branch must clone that branch, or Colab would
# silently run the old main-branch code (including its old upload prompt).  This
# is a literal, not `git rev-parse HEAD`, so the generated notebook is
# reproducible on any checkout (the builder/notebook consistency test relies on
# that).  The clone cell falls back to the default branch if it ever disappears
# (e.g. after this branch is merged and deleted); set it to '' for main.
REPO_BRANCH = 'arena/e6c9092c-airesearch'

cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': text.splitlines(True)})


md(f'''# ByteCore Phase 19 — clone, train, and test a Mineflayer player

This notebook **clones the public repository** `{REPO_URL}` and imports its code;
it does not embed a copy of the source. Run all cells in a fresh Colab runtime
(Runtime → Change runtime type → **T4 GPU**).

**What is new in this phase**

1. **Clone-first setup.** Cell 1 clones the repo, installs `requirements.txt` and
   switches the working directory; every later import comes from the checkout.
2. **Mineflayer player training.** `notebook/mineflayer_training_data.py` builds
   **521 deterministic trajectories** over 8 skills (navigation, mining, crafting,
   combat, building, inventory, chat, farming) in the real Mineflayer JavaScript
   API. The model learns whole player turns (`Request/Thought/Action/Observation/
   Answer`), not a command DSL.
3. **Stage 2 is now 40% code / 40% Mineflayer / 20% pt-br**, with the first six
   blocks frozen.
4. **A sandbox harness** validates generated JavaScript three ways: a static
   checker, `node --check` (parse only) and a Node mock of the Mineflayer API
   (`notebook/mineflayer_mock.js`) that records calls against a mock world.
5. **No data uploads.** Cell 4 *searches* the Hugging Face Hub for permissively
   licensed corpora, verifies each license, streams bounded samples, and also
   mines the real corpora already checked into `data/github_raw/` plus this
   repository's own source files. Discovery results feed collection directly;
   a missing group triggers a deeper automatic search, never a file picker.

**Reality check (read before trusting numbers).** The model starts from **random
weights** and does not inherit the Phase 18 checkpoint. Four time-boxed stages on
a free T4 cannot guarantee fluency, coding skill, search routing or competent
Minecraft play. The harness is a **test double**, not Minecraft: a passing call
means the mock accepted the API sequence, not that a real server would. There is
no `mineflayer-py` package — Mineflayer is Node.js — so the notebook trains the
model to *write* Mineflayer code and tests that code in a mock, and only a live
server (out of scope here) could prove real play.

**Time budget.** Four 47-minute stages plus installs/upload fit a free Colab
session (3h35m training deadline). Four 60-minute stages do not fit; set
`STAGE_MINUTES = 60` only if you accept the session ending mid-run. Checkpoints
upload roughly every 15 minutes and at every stage boundary.

Never paste your HF token into a saved cell, and never commit `checkpoints/`.
''')


md('## 1. Clone the repository and install its requirements\n'
   'The clone is shallow (depth 1). If the directory already exists it is reused and fast-forwarded.\n')
code(f'''import os, sys, shutil, subprocess, time
from pathlib import Path

SESSION_STARTED = time.time()
REPO_URL = {REPO_URL!r}
REPO_BRANCH = {REPO_BRANCH!r}   # '' means the default branch
REPO_DIR = Path('/content/airesearch')

if (REPO_DIR / '.git').exists():
    subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', '--depth', '1', 'origin'], check=False)
    subprocess.run(['git', '-C', str(REPO_DIR), 'pull', '--ff-only'], check=False)
else:
    command = ['git', 'clone', '--depth', '1']
    if REPO_BRANCH:
        command += ['--branch', REPO_BRANCH]
    attempt = subprocess.run(command + [REPO_URL, str(REPO_DIR)], check=False)
    if attempt.returncode and REPO_BRANCH:
        print('Branch clone failed; retrying the default branch.')
        shutil.rmtree(REPO_DIR, ignore_errors=True)
        subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(REPO_DIR)], check=True)
    elif attempt.returncode:
        raise RuntimeError('git clone failed; check the repository URL and network access')

%cd $REPO_DIR
assert (REPO_DIR / 'notebook' / 'mineflayer_training_data.py').exists(), 'clone did not contain the Phase 19 sources'
print('Repository:', REPO_URL)
print('Commit:', subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip())

# Make the checkout importable before any other cell runs. Jupyter ships a
# package also called `notebook`, so detach it and prove the checkout won.
sys.path.insert(0, str(REPO_DIR))
for name in [m for m in list(sys.modules) if m == 'notebook' or m.startswith('notebook.')]:
    del sys.modules[name]
import notebook as repo_notebook
assert str(Path(repo_notebook.__file__).resolve()).startswith(str(REPO_DIR)), \
    'another notebook package shadowed the checkout; restart the runtime'
print('Imported repository package from', repo_notebook.__file__)
''')
code('''%pip -q install -r requirements.txt
''')


md('''## 2. Extra dependencies and Node.js

`requirements.txt` covers PyTorch and the training stack. This cell adds the
notebook-only packages. Two corrections to the original plan:

* **`mineflayer-py` does not exist on PyPI.** Mineflayer is a Node.js library;
  training data comes from the repo's generator and code is validated by Node.
* **`duckduckgo-search` is frozen and renamed to `ddgs`** (mid-2025). New code
  must use `ddgs`; `notebook/tools.py` will use that name.
''')
code('''%pip -q install "datasets>=3,<5" "huggingface_hub>=0.25,<2" "ddgs>=9,<10" "onnx>=1.16,<2" tqdm

import shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
# Jupyter ships a package also called `notebook`; detach it so the checkout wins.
for name in [m for m in list(sys.modules) if m == 'notebook' or m.startswith('notebook.')]:
    del sys.modules[name]
import notebook as repo_notebook
assert str(Path(repo_notebook.__file__).resolve()).startswith(str(Path.cwd())), \\
    'the installed notebook package shadowed the repository; restart the runtime and rerun cell 1'
print('Imported notebook package from', repo_notebook.__file__)
NODE = shutil.which('node')
print('Node.js:', NODE or 'missing — JavaScript syntax checks will be skipped (mock harness still needs node)')
''')


md('''## 3. Credentials, model and resume state

Enter a **write-scoped** Hugging Face token privately. `REPO_ID` is your own
private model repository (`username/repo`); it is created if needed. Checkpoints
are PyTorch pickles: resume only from a repository you control.

`STAGE_MINUTES = 47` is deliberate: 4 × 60 minutes plus installs and uploads
does not fit a free Colab session. The stage mix and freezing are independent of
this number, so set it to 60 only if you accept an interrupted session.
''')
code('''from getpass import getpass
import json, random, torch
from pathlib import Path

from notebook.phone_model import Config, PhoneLM
from notebook.hf_upload import latest_checkpoint, load_checkpoint

HF_TOKEN = getpass('Hugging Face write token: ')
REPO_ID = input('Private model repo (username/repo): ').strip()
if not HF_TOKEN or '/' not in REPO_ID:
    raise ValueError('A token and username/repo are required')

STAGE_MINUTES = 47
STAGE_SECONDS = STAGE_MINUTES * 60
SESSION_DEADLINE = SESSION_STARTED + 3 * 3600 + 35 * 60
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (T4 strongly recommended)')

random.seed(19); torch.manual_seed(19)
cfg = Config()
model = PhoneLM(cfg)
print('Parameters:', sum(p.numel() for p in model.parameters()),
      '; raw FP16 MB:', round(sum(p.numel() for p in model.parameters()) * 2 / 1e6, 1))

checkpoint_path = latest_checkpoint(REPO_ID, HF_TOKEN)
RESUME = load_checkpoint(checkpoint_path) if checkpoint_path else None
print('Resume:', checkpoint_path or 'starting from scratch')

def resume_for(stage):
    \"\"\"Return the pending checkpoint or stop an unsafe out-of-order run.\"\"\"
    if RESUME is None:
        return None
    checkpoint_stage = int(RESUME['stage'])
    if checkpoint_stage != stage:
        raise RuntimeError(
            f'Latest checkpoint is at Stage {checkpoint_stage + 1}. '
            f'Run that stage first; skipping stages or rerunning an earlier one can corrupt the curriculum.')
    return RESUME
''')


md('''## 4. Search for and cache real datasets (no uploads)

`discover()` searches the Hugging Face Hub for Portuguese, English, code and
tool-trajectory corpora; those hits are **fed into `collect()`**, which verifies
each declared license with `dataset_info`, skips gated/unknown/non-permissive
sources, tries the right config (C4 `pt`/`en`, `sample-10BT`, `por_Latn`, ...) and
streams a bounded sample. It also mines the real corpora already checked into
this repository: `data/github_raw/github_stories.jsonl` and `github_react.jsonl`
(per-record SPDX licenses, `NOASSERTION` excluded), code blocks from those
records, and this repository's own `*.py` sources for code bytes. Open Data
Commons Attribution (`odc-by`) sources are accepted with their attribution
retained in `data_cache/manifest.json`; review the terms before publishing a
model. Chat messages are never relabelled as ReAct trajectories, and no
language or code data is fabricated. **Nothing is uploaded to this notebook.**
''')
code('''from notebook.dataset_discovery import collect, discover, group_counts
from notebook.preprocess import synthetic

# Search results are the collector's candidate list, not just a printout.
reports = discover(limit=8)
groups = collect(max_rows=4000, max_bytes=8_000_000, min_rows=200,
                 reports=reports, search_limit=8)
# Small labelled task fixtures (search/edit/ReAct prompts) extend, never
# overwrite, the real rows found above.
for kind, rows in synthetic().items():
    groups.setdefault(kind, [])
    groups[kind] = list(dict.fromkeys(groups[kind] + rows))
print(group_counts(groups))
''')


md('''### Required language/code data preflight (automatic retry)

Stage 1 needs real `pt`, `en` and `code` rows. If any are still empty,
`fill_missing()` runs a **deeper Hub search plus another pass over the local
repository corpora** and merges the results. It never opens a file picker and
never asks you for data. Files that already exist under
`data_cache/manual_data/` are merged silently and idempotently, but none are
required. `data_cache/manifest.json` lists every source tried and exactly why it
was skipped, so a failure here is a diagnosable search/network problem, not a
request for an upload. No language or code fallback is fabricated; check the
printed counts before spending GPU time.
''')
code('''from notebook.dataset_discovery import fill_missing, group_counts
from notebook.train import check_groups

preflight = fill_missing(groups, cache_dir='data_cache')
print('Preflight:', json.dumps(preflight, ensure_ascii=False))
manifest = json.loads((Path('data_cache') / 'manifest.json').read_text(encoding='utf8'))
summary = manifest[-1] if manifest else {}
print('Search summary:', json.dumps({
    key: summary.get(key) for key in
    ('hub_attempts', 'group_counts', 'missing_groups', 'upload_required')}, ensure_ascii=False))
print('All groups:', group_counts(groups))
# Stop here, before Mineflayer generation or GPU training, with a useful error.
check_groups(groups, stages=[0])
print('Stage 1 data preflight passed; no files were requested.')
''')


md('''## 5. Preprocess: Mineflayer player trajectories

Each trajectory is a complete player turn in real Mineflayer JavaScript. The
generator is deterministic for a seed and writes `data_cache/mineflayer_trajectories.jsonl`
so the corpus is auditable. Coverage is 8 skills with at least 50 rows each.
''')
code('''from notebook.mineflayer_training_data import dataset_stats, mineflayer_records, write_jsonl

groups['mineflayer'] = mineflayer_records(seed=19)
stats = dataset_stats(seed=19)
print(json.dumps(stats, indent=2))
assert stats['total'] >= 500, 'Phase 19 requires 500+ trajectories'
Path('data_cache').mkdir(exist_ok=True)
write_jsonl(Path('data_cache') / 'mineflayer_trajectories.jsonl', seed=19)
check_groups(groups)
print('Required groups present for all four stages.')
''')


md('''## 5b. Mineflayer sandbox self-test (run before training)

Three layers, all weaker than a real server:

1. static checks (Python) — wrong language, host capability words, unbalanced code;
2. `node --check` — parses the JavaScript, executes nothing;
3. the Node mock (`notebook/mineflayer_mock.js`) — executes against a fake bot in
   `node:vm`, records API calls and checks the expected action.

The 521-row corpus check takes a few seconds. **Never `exec()` model output** in
Python: the mock is isolation for non-adversarial model text, not a security
boundary for untrusted code.
''')
code('''from notebook.test_mineflayer import self_test, validate_trajectories

if NODE:
    reference_rows = self_test()
    for row in reference_rows:
        print(f"{row['case']:<16} {'PASS' if row['pass'] else 'FAIL'}", row['issues'] or '')
    assert all(row['pass'] for row in reference_rows), 'a reference Mineflayer snippet failed validation'

    corpus = validate_trajectories()
    print(json.dumps({k: v for k, v in corpus.items() if not k.endswith('failures')}, indent=2))
    for row in (corpus['static_failures'] + corpus['runtime_failures'])[:5]:
        print('FAIL', row)
    assert corpus['sandbox'] and corpus['passed'] == corpus['total'], \
        'Mineflayer corpus preflight failed; fix the reported rows before training.'
else:
    raise RuntimeError('Node.js is required for the Mineflayer corpus preflight. Install it (for example apt-get install -y nodejs) and rerun this cell before training.')
''')


md('''## 6. Stage 1 — bilingual foundation (47 min)

Mix 50/30/20 pt/en/code. Freeze 0 blocks. Each stage cell trains only its own
stage, keeps the model in memory, and uploads a resumable checkpoint on a fixed
cadence and at the end. If you restart the runtime, rerun cells 1–3 and start
from the cell matching the checkpoint's stage.
''')
code('''from notebook.train import train

completed = train(model, groups, HF_TOKEN, REPO_ID, SESSION_DEADLINE,
                  stage_seconds=STAGE_SECONDS, upload_seconds=15 * 60,
                  micro_batch=1, accumulation=32,
                  resume=resume_for(0), stages=[0])
if not completed:
    raise RuntimeError('Stage 1 stopped at the session deadline. Restart the runtime and resume from the latest checkpoint; do not continue to Stage 2.')
# A consumed checkpoint must not be reloaded by later stage cells and rewind the
# in-memory model to an earlier stage.
RESUME = None
print('Stage 1 completed:', completed)
''')


md('''## 7. Stage 2 — code + Mineflayer player (47 min)

**Data: 40% code, 40% Mineflayer trajectories, 20% pt-br.** Objective: next-byte
prediction. **Freeze: the first 6 blocks** to preserve language fluency. The model
sees real API calls (`pathfinder.goto(new GoalNear(...))`, `dig`, `craft`,
`attack`, `placeBlock`, inventory, chat, farming) inside complete player turns.
''')
code('''completed = train(model, groups, HF_TOKEN, REPO_ID, SESSION_DEADLINE,
                  stage_seconds=STAGE_SECONDS, upload_seconds=15 * 60,
                  micro_batch=1, accumulation=32,
                  resume=resume_for(1), stages=[1])
if not completed:
    raise RuntimeError('Stage 2 stopped at the session deadline. Restart the runtime and resume from the latest checkpoint; do not continue to Stage 3.')
RESUME = None
print('Stage 2 completed:', completed)
''')


md('''## 8. Stage 3 — tool use and search (47 min)

Mix 40/30/30 ReAct/search/code. Search is checked for a generated call, never
executed inside training. DDG requires a live network connection.
''')
code('''completed = train(model, groups, HF_TOKEN, REPO_ID, SESSION_DEADLINE,
                  stage_seconds=STAGE_SECONDS, upload_seconds=15 * 60,
                  micro_batch=1, accumulation=32,
                  resume=resume_for(2), stages=[2])
if not completed:
    raise RuntimeError('Stage 3 stopped at the session deadline. Restart the runtime and resume from the latest checkpoint; do not continue to Stage 4.')
RESUME = None
print('Stage 3 completed:', completed)
''')


md('''## 9. Stage 4 — RL + self-edit (47 min)

Mix 50/20/15/15 edit/pt/code/mineflayer. Alternates supervised edits with sparse
sampled REINFORCE. The reward is a weak, non-executing proxy: an exact-target
edit check, a Python AST shape check, and a static Mineflayer check. Scores are
not proof of quality.
''')
code('''completed = train(model, groups, HF_TOKEN, REPO_ID, SESSION_DEADLINE,
                  stage_seconds=STAGE_SECONDS, upload_seconds=15 * 60,
                  micro_batch=1, accumulation=32,
                  resume=resume_for(3), stages=[3])
if not completed:
    raise RuntimeError('Stage 4 stopped at the session deadline. Restart the runtime and resume from the latest checkpoint before evaluating or uploading a final model.')
RESUME = None
print('Stage 4 completed:', completed)
''')


md('''## 10. Final evaluation

Free-running probes: pt-br, code, four Mineflayer player tasks, search, edit and
host-managed memory. Mineflayer rows pass only if the static check, parse and
mock expectation all succeed. These probes are tiny and can all fail after four
hours — report them as measured, not as capability.
''')
code('''from notebook.evaluate import evaluate
from notebook.test_mineflayer import evaluate_mineflayer

results = evaluate(model, max_new=128)
for row in results:
    print(f"{row['case']:<22} {'PASS' if row['pass'] else 'FAIL'}")
    print('  ', repr(row['output'][:200]))
print('Probes passed:', sum(row['pass'] for row in results), '/', len(results))

mineflayer_rows = evaluate_mineflayer(model, max_new=160)
for row in mineflayer_rows:
    detail = row['expectation'] or row['issues'] or row['error'] or ''
    print(f"{row['case']:<16} {'PASS' if row['pass'] else 'FAIL'} {detail}")
print('Mineflayer probes passed:', sum(row['pass'] for row in mineflayer_rows), '/', len(mineflayer_rows))
''')


md('''## 11. Upload the final model and an honest model card

Uploads FP16 weights, the model card, configuration, and the dataset-attempt
manifest. The manifest records the declared licenses/configurations of sampled
sources; it is needed for attribution review, especially for `odc-by` data. The
token is used only in this runtime.
''')
code('''from dataclasses import asdict
from huggingface_hub import HfApi

probe_summary = {row['case']: bool(row['pass']) for row in results + mineflayer_rows}
manifest_path = Path('data_cache') / 'manifest.json'
card = f"""---
license: apache-2.0
tags: [byte-level, mineflayer, minecraft, phase19]
---

# ByteCore Phase 19 — phone-size byte model

Trained from **random weights** for four time-boxed stages in a free Colab
session (not four full hours of compute). Not compatible with Phase 18 weights.

* Stage 1: 50/30/20 pt/en/code
* Stage 2: 40/40/20 code/Mineflayer/pt-br, first 6 blocks frozen
* Stage 3: 40/30/30 ReAct/search/code
* Stage 4: 50/20/15/15 edit/pt/code/Mineflayer

## Data and attribution

`provenance/dataset_manifest.json` records every automated source attempt,
declared license, config and count. Data was found by the notebook itself: a
Hub search plus the real corpora checked into this repository
(`data/github_raw/*.jsonl`, per-record SPDX licenses) and first-party `*.py`
sources for code bytes. No training files were uploaded by the user. Optional
files already present under `data_cache/manual_data/` are merged silently; the
owner is responsible for their permissions and attribution. Review all source
terms before making this model public. In particular, `odc-by` sources require
attribution, and first-party source files are labelled as such.

Mineflayer ability was measured with a **mock** Node harness, not a real server:
generated calls were parsed and replayed against a fake bot. A pass means the
mock accepted the call sequence, not that the model can play Minecraft.

Probe results in this run: {json.dumps(probe_summary)}

Limitations: small decoder, weak reward proxies, tiny synthetic tool/edit
fixtures, no held-out benchmark, and no evidence of generalisation. Use at your
own risk.
"""
api = HfApi(token=HF_TOKEN)
api.create_repo(REPO_ID, repo_type='model', private=True, exist_ok=True)
torch.save({key: value.detach().cpu().half() for key, value in model.state_dict().items()},
           'final_model_fp16.pt')
Path('final_model').mkdir(exist_ok=True)
(Path('final_model') / 'config.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf8')
(Path('final_model') / 'README.md').write_text(card, encoding='utf8')
api.upload_file(path_or_fileobj='final_model_fp16.pt', path_in_repo='final/phase19_fp16.pt',
                repo_id=REPO_ID, repo_type='model', token=HF_TOKEN)
api.upload_file(path_or_fileobj='final_model/README.md', path_in_repo='README.md',
                repo_id=REPO_ID, repo_type='model', token=HF_TOKEN)
api.upload_file(path_or_fileobj='final_model/config.json', path_in_repo='final/config.json',
                repo_id=REPO_ID, repo_type='model', token=HF_TOKEN)
if manifest_path.exists():
    api.upload_file(path_or_fileobj=str(manifest_path), path_in_repo='provenance/dataset_manifest.json',
                    repo_id=REPO_ID, repo_type='model', token=HF_TOKEN)
else:
    print('WARNING: data manifest was not found; provenance was not uploaded.')
print('Uploaded to https://huggingface.co/' + REPO_ID)
''')


md('''## 12. Export and download (optional)

FP16 weights are exported separately from large resumable optimizer checkpoints.
ONNX exports FP32 and may be ~2× larger; it is a runtime representation, not a
verified mobile app. Move the model to CPU for export.
''')
code('''import shutil, zipfile
from dataclasses import asdict
from notebook.phone_model import export_onnx

out = Path('phone_export'); out.mkdir(exist_ok=True)
torch.save({k: v.detach().cpu().half() for k, v in model.state_dict().items()}, out / 'phone_fp16.pt')
(out / 'config.json').write_text(json.dumps(asdict(cfg), indent=2))
shutil.copytree(Path('notebook'), out / 'notebook', dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__', 'build_notebook.py', '*.ipynb'))
try:
    export_onnx(model, str(out / 'phone_fp32.onnx'))
    print('ONNX export succeeded:', round((out / 'phone_fp32.onnx').stat().st_size / 1e6, 1), 'MB')
except Exception as exc:
    print('ONNX export FAILED:', type(exc).__name__, str(exc))

with zipfile.ZipFile(Path('phone_model_fp16.zip'), 'w', compression=zipfile.ZIP_STORED) as handle:
    for path in out.rglob('*'):
        if path.is_file() and path.suffix != '.onnx':
            handle.write(path, path.relative_to(out))
print('FP16 bundle MB:', round(Path('phone_model_fp16.zip').stat().st_size / 1e6, 1))
try:
    from google.colab import files
    files.download(str(Path('phone_model_fp16.zip')))
except ImportError:
    print('Not running in Colab; bundle left at', Path('phone_model_fp16.zip').resolve())
''')


for index, cell in enumerate(cells):
    cell['id'] = f'phase19-{index:03d}'

notebook = {
    'cells': cells,
    'metadata': {
        'colab': {'name': 'phase19_colab.ipynb', 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

if __name__ == '__main__':
    target = ROOT / 'notebook' / 'phase19_colab.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf8')
    print('wrote', target, 'with', len(cells), 'cells')
