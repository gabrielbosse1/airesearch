"""Hugging Face discovery and bounded streaming for the Phase 19 notebook.

Only sources with a declared, reviewed permissive license are sampled.  Language
text is deliberately *not* replaced with synthetic examples: a missing group is
reported with a concrete manual-data path before expensive training can begin.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

QUERIES = ['pt-br', 'portuguese', 'brazilian', 'code', 'programming', 'python',
           'javascript', 'minecraft', 'game commands', 'openassistant', 'tool', 'agent']

# These are candidates, not a license or availability guarantee. A candidate is
# still checked with ``dataset_info`` at collection time. The CulturaX/FineWeb
# entries are useful fallbacks when OPUS metadata or a source config is absent.
CANDIDATES = {
    'pt': ['maritaca-ai/en2ptbridge', 'Helsinki-NLP/opus-100', 'uonlp/CulturaX'],
    'en': ['Helsinki-NLP/opus-100', 'HuggingFaceFW/fineweb-edu'],
    'code': ['m-a-p/CodeFeedback-Filtered-Instruction', 'HuggingFaceH4/CodeAlpaca_20K'],
    'react': ['OpenAssistant/oasst1'],
}

# Config names are deliberately source-specific. Passing ``None`` to datasets
# for FineWeb-Edu can fail because it has multiple configurations; that was the
# reason a valid English fallback could be skipped before streaming began.
DATASET_CONFIGS = {
    ('Helsinki-NLP/opus-100', 'pt'): 'en-pt',
    ('Helsinki-NLP/opus-100', 'en'): 'en-pt',
    ('uonlp/CulturaX', 'pt'): 'pt',
    ('HuggingFaceFW/fineweb-edu', 'en'): 'sample-10BT',
}

# ODC-BY is an attribution-only Open Data Commons license. It is permitted here
# alongside the code/content licenses below, but its attribution requirement is
# retained in data_cache/manifest.json and uploaded with the model card.
ALLOWED = {
    'apache-2.0', 'mit', 'cc0-1.0', 'cc-by-4.0', 'cc-by-3.0',
    'bsd-3-clause', 'bsd-2-clause', 'isc', 'unlicense', 'odc-by', 'odc-by-1.0',
}
MANUAL_KINDS = ('pt', 'en', 'code')


def _card_value(card, name, default=None):
    """Read a DatasetCardData value without assuming it is a plain dict."""
    if card is None:
        return default
    get = getattr(card, 'get', None)
    if callable(get):
        value = get(name, default)
    else:
        value = getattr(card, name, default)
    return default if value is None else value


def _normalise_license(value) -> str:
    """Return one canonical SPDX-ish identifier, or ``''`` if ambiguous."""
    if isinstance(value, (list, tuple, set)):
        values = {_normalise_license(item) for item in value}
        values.discard('')
        return values.pop() if len(values) == 1 else ''
    value = str(value or '').strip().lower()
    # Some cards use a display form while the hub API uses the SPDX form.
    aliases = {
        'apache 2.0': 'apache-2.0', 'apache-2.0 license': 'apache-2.0',
        'mit license': 'mit', 'cc by 4.0': 'cc-by-4.0',
        'cc by 3.0': 'cc-by-3.0', 'odc-by 1.0': 'odc-by-1.0',
        'odc-by-1.0 license': 'odc-by-1.0',
    }
    return aliases.get(value, value)


def license_of(info) -> str:
    card = getattr(info, 'card_data', None) or {}
    return _normalise_license(_card_value(card, 'license', ''))


def language_of(info):
    return _card_value(getattr(info, 'card_data', None) or {}, 'language', 'unknown')


def dataset_config(name: str, kind: str) -> str | None:
    """The intentional streaming configuration for a known candidate."""
    return DATASET_CONFIGS.get((name, kind))


def discover(limit=5):
    """Print discovered datasets and whether their declared license is accepted."""
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError('limit must be a positive integer')
    from huggingface_hub import HfApi

    api = HfApi()
    reports = []
    print('Note: pierreguillou/bert-base-cased-pt-br is a model, not a corpus dataset.')
    for query in QUERIES:
        try:
            rows = list(api.list_datasets(search=query, limit=limit, full=True))
            for row in rows:
                lic = license_of(row)
                reports.append({
                    'query': query,
                    'name': row.id,
                    'license': lic or 'unknown',
                    'language': language_of(row),
                    'size_bytes': getattr(row, 'size_in_bytes', None),
                    'accepted_license': lic in ALLOWED,
                })
        except Exception as exc:
            print('Search failed:', query, type(exc).__name__, str(exc)[:200])
    for row in reports:
        print(json.dumps(row, ensure_ascii=False, default=str))
    return reports


def extract(row, kind):
    """Turn a supported dataset row into a bounded trainable text record."""
    if kind == 'code':
        question = row.get('instruction') or row.get('query') or row.get('prompt') or row.get('problem')
        answer = row.get('output') or row.get('response') or row.get('answer') or row.get('completion')
        if isinstance(answer, list):
            answer = '\n'.join(str(item.get('content', '')) for item in answer if isinstance(item, dict))
        if not isinstance(question, str) or not isinstance(answer, str):
            return None
        if not re.search(r'python|javascript|\bjs\b|\bdef\b|\bfunction\b', question + ' ' + answer, re.I):
            return None
        return 'Question: ' + question + '\nCode: ' + answer
    if kind == 'react':
        # OASST messages are not tool trajectories; do not mislabel them as such.
        return None
    for key in ('text', 'content', 'sentence', 'translation'):
        value = row.get(key)
        if isinstance(value, dict):
            value = (value.get('pt') or value.get('pt-br')) if kind == 'pt' else value.get('en')
        if isinstance(value, str) and len(value.strip()) >= 30:
            return value
    return None


def _translation_text(row, kind):
    translation = row.get('translation')
    if not isinstance(translation, dict):
        return None
    keys = ('pt', 'pt-br') if kind == 'pt' else ('en',)
    for key in keys:
        value = translation.get(key)
        if isinstance(value, str) and len(value.strip()) >= 30:
            return value
    return None


def _bounded_lines(path: Path) -> list[str]:
    """Read newline-delimited UTF-8 records, rejecting broken input explicitly."""
    try:
        with path.open('r', encoding='utf8', errors='strict') as handle:
            return [line.strip()[:4096] for line in handle if line.strip()]
    except UnicodeDecodeError as exc:
        raise ValueError(f'{path} is not valid UTF-8: {exc}') from exc


def merge_manual_data(groups: dict, manual_dir='data_cache/manual_data',
                      kinds: Iterable[str] = MANUAL_KINDS) -> dict:
    """Merge licensed user-supplied ``pt.txt``, ``en.txt`` and ``code.txt`` files.

    The function is idempotent, so rerunning the Colab cell never keeps appending
    the same manual rows. Empty files are reported but do not count as data.
    """
    root = Path(manual_dir)
    report = {}
    for kind in kinds:
        if kind not in MANUAL_KINDS:
            raise ValueError(f'unsupported manual group {kind!r}')
        path = root / f'{kind}.txt'
        rows = _bounded_lines(path) if path.exists() else []
        prior = groups.setdefault(kind, [])
        seen = set(prior)
        added = []
        for row in rows:
            if row not in seen:
                seen.add(row)
                added.append(row)
        prior.extend(added)
        report[kind] = {'path': str(path), 'exists': path.exists(),
                        'rows_read': len(rows), 'rows_added': len(added)}
    return report


def group_counts(groups: dict) -> dict[str, int]:
    """Stable compact group counts for notebook diagnostics."""
    return {kind: len(rows or []) for kind, rows in sorted(groups.items())}


def collect(cache_dir='data_cache', max_rows=4000, max_bytes=8_000_000):
    """Stream bounded samples from explicitly licensed candidates, then cache JSON.

    Dataset streams can fail or require gated access; absent groups remain empty.
    No restricted, unknown, or ambiguously-labelled license is silently accepted.
    ``manifest.json`` records every attempted source/config and is provenance for
    the final model upload.
    """
    if not isinstance(max_rows, int) or max_rows <= 0:
        raise ValueError('max_rows must be a positive integer')
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError('max_bytes must be a positive integer')
    from datasets import load_dataset
    from huggingface_hub import HfApi

    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    result = {kind: [] for kind in CANDIDATES}
    manifest = []
    for kind, ids in CANDIDATES.items():
        for name in ids:
            if len(result[kind]) >= max_rows:
                break
            record = {'name': name, 'kind': kind, 'config': dataset_config(name, kind), 'split': 'train'}
            try:
                info = api.dataset_info(name)
                license_ = license_of(info)
                record.update(
                    license=license_ or 'unknown',
                    language=language_of(info),
                    size_bytes=getattr(info, 'size_in_bytes', None),
                )
                if license_ not in ALLOWED:
                    record['status'] = 'skipped: unverified or non-permissive license'
                    continue
                if kind == 'react':
                    record['status'] = 'skipped: chat messages are not ReAct trajectories'
                    continue
                stream = load_dataset(name, record['config'], split='train', streaming=True)
                seen_bytes = sum(len(text.encode('utf-8')) for text in result[kind])
                rows_seen = rows_added = 0
                for row in stream:
                    rows_seen += 1
                    text = (_translation_text(row, kind)
                            if name == 'Helsinki-NLP/opus-100' else extract(row, kind))
                    if text:
                        text = text.strip()[:4096]
                        n_bytes = len(text.encode('utf-8'))
                        if seen_bytes + n_bytes > max_bytes:
                            record['stop_reason'] = 'byte_limit'
                            break
                        result[kind].append(text)
                        rows_added += 1
                        seen_bytes += n_bytes
                    if len(result[kind]) >= max_rows:
                        record['stop_reason'] = 'row_limit'
                        break
                    if rows_seen >= max_rows * 10:
                        record['stop_reason'] = 'scan_limit'
                        break
                record.update(status='streamed', rows=len(result[kind]), rows_added=rows_added,
                              rows_seen=rows_seen, cached_bytes=seen_bytes)
            except Exception as exc:
                record['status'] = 'skipped: ' + type(exc).__name__ + ': ' + str(exc)[:200]
            finally:
                manifest.append(record)
                print(json.dumps(record, ensure_ascii=False, default=str))
    (root / 'samples.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf8')
    (root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf8')
    return result
