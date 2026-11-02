"""Resume snapshots: immutable paths and full optimizer/training state."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import torch

PATTERN = re.compile(r'^checkpoints/stage_(\d+)_step_(\d+)/checkpoint\.pt$')


def save_checkpoint(state, stage, step, root='checkpoints'):
    """Atomically write one local checkpoint before any network operation."""
    folder = Path(root) / f'stage_{stage}_step_{step}'
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / 'checkpoint.tmp'
    target = folder / 'checkpoint.pt'
    torch.save(state, temporary)
    os.replace(temporary, target)
    return target


def upload_checkpoint(state, stage, step, hf_token, repo_id, root='checkpoints', api=None,
                      retries=3, retry_seconds=5.0):
    """Save locally first, then upload with bounded retries for transient outages.

    The returned path is always the durable local snapshot. If all uploads fail,
    the exception names that path so a live Colab runtime can retry it rather than
    silently continuing without a resumable remote checkpoint.
    """
    if not isinstance(retries, int) or retries < 0:
        raise ValueError('retries must be a non-negative integer')
    if not isinstance(retry_seconds, (int, float)) or retry_seconds < 0:
        raise ValueError('retry_seconds must be non-negative')
    from huggingface_hub import HfApi

    path = save_checkpoint(state, stage, step, root)
    api = api or HfApi(token=hf_token)
    last_error = None
    for attempt in range(retries + 1):
        try:
            api.create_repo(repo_id=repo_id, repo_type='model', private=True, exist_ok=True, token=hf_token)
            api.upload_file(path_or_fileobj=str(path),
                            path_in_repo=f'checkpoints/stage_{stage}_step_{step}/checkpoint.pt',
                            repo_id=repo_id, repo_type='model', token=hf_token,
                            commit_message=f'Stage {stage}, step {step}')
            return path
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            wait = retry_seconds * (2 ** attempt)
            print(f'Checkpoint upload failed ({type(exc).__name__}); retrying in {wait:.1f}s '
                  f'({attempt + 1}/{retries})')
            time.sleep(wait)
    raise RuntimeError(f'Checkpoint was saved locally at {path}, but upload failed after '
                       f'{retries + 1} attempt(s): {type(last_error).__name__}: {last_error}') from last_error


def latest_checkpoint(repo_id, hf_token, api=None):
    """Download the highest stage/step checkpoint, or ``None`` for a new repo."""
    from huggingface_hub import HfApi, hf_hub_download

    api = api or HfApi(token=hf_token)
    try:
        names = api.list_repo_files(repo_id=repo_id, repo_type='model', token=hf_token)
    except Exception as exc:
        from huggingface_hub.errors import RepositoryNotFoundError
        if isinstance(exc, RepositoryNotFoundError):
            return None
        raise
    found = [(int(match[1]), int(match[2]), name) for name in names
             if (match := PATTERN.fullmatch(name))]
    if not found:
        return None
    name = max(found)[2]
    return hf_hub_download(repo_id=repo_id, filename=name, repo_type='model', token=hf_token)


def load_checkpoint(path):
    # ONLY load checkpoints you trust: pickle can execute arbitrary code.
    return torch.load(path, map_location='cpu', weights_only=False)
