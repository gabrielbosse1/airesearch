"""Resume snapshots: immutable paths and full optimizer/training state."""
import os
import re
from pathlib import Path
import torch

PATTERN = re.compile(r'^checkpoints/stage_(\d+)_step_(\d+)/checkpoint\.pt$')

def save_checkpoint(state, stage, step, root='checkpoints'):
    folder = Path(root) / f'stage_{stage}_step_{step}'
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / 'checkpoint.tmp'
    torch.save(state, tmp)
    os.replace(tmp, folder / 'checkpoint.pt')
    return folder / 'checkpoint.pt'

def upload_checkpoint(state, stage, step, hf_token, repo_id, root='checkpoints', api=None):
    from huggingface_hub import HfApi
    path = save_checkpoint(state, stage, step, root)
    api = api or HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type='model', private=True, exist_ok=True, token=hf_token)
    api.upload_file(path_or_fileobj=str(path), path_in_repo=f'checkpoints/stage_{stage}_step_{step}/checkpoint.pt',
                    repo_id=repo_id, repo_type='model', token=hf_token,
                    commit_message=f'Stage {stage}, step {step}')
    return path

def latest_checkpoint(repo_id, hf_token, api=None):
    from huggingface_hub import HfApi, hf_hub_download
    api = api or HfApi(token=hf_token)
    try:
        names = api.list_repo_files(repo_id=repo_id, repo_type='model', token=hf_token)
    except Exception as exc:
        from huggingface_hub.errors import RepositoryNotFoundError
        if isinstance(exc, RepositoryNotFoundError): return None
        raise
    found = [(int(m[1]), int(m[2]), n) for n in names if (m := PATTERN.fullmatch(n))]
    if not found: return None
    name = max(found)[2]
    return hf_hub_download(repo_id=repo_id, filename=name, repo_type='model', token=hf_token)

def load_checkpoint(path):
    # ONLY load checkpoints you trust: pickle can execute arbitrary code.
    return torch.load(path, map_location='cpu', weights_only=False)
