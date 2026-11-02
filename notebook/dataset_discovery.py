"""Automatic data acquisition for the Phase 19 notebook.

Nothing here asks the user to upload anything.  Every ``pt`` / ``en`` / ``code``
/ ``react`` row is *found* by the notebook:

1. curated, reviewed Hugging Face corpora with known configs,
2. a live Hugging Face Hub search per group (``discover`` feeds ``collect``),
3. the real corpora already checked into this repository
   (``data/github_raw/*.jsonl`` with per-record SPDX licenses) and this
   repository's own source files for code bytes.

Only declared, reviewed permissive licenses are streamed.  Gated, private,
ambiguously licensed and non-permissive sources are skipped, and chat messages
are never relabelled as ReAct trajectories.  Every attempt -- including failures
-- is written to ``data_cache/manifest.json`` so attribution can be reviewed
before the model is published.  Pre-existing files under ``manual_data/`` are
merged silently if present; no Colab file picker is ever opened.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

QUERIES = ['pt-br', 'portuguese', 'brazilian', 'code', 'programming', 'python',
           'javascript', 'minecraft', 'game commands', 'openassistant', 'tool', 'agent']

# Per-group search terms.  ``discover()`` returns these hits and ``collect()``
# actually tries them; previously discovery only printed a table and the
# collector ignored it, which is why a failed curated source ended in an upload
# prompt instead of another dataset.
KIND_QUERIES = {
    'pt': ['portuguese text corpus', 'pt-br corpus', 'brazilian portuguese', 'portuguese web text'],
    'en': ['english text corpus', 'fineweb edu', 'openwebtext'],
    'code': ['code instruction dataset', 'code alpaca', 'python code dataset'],
    'react': ['tool use trajectories', 'react agent dataset'],
}
DEEP_QUERIES = [
    'portuguese', 'portugues', 'pt-br', 'brazilian portuguese text',
    'portuguese wikipedia', 'portuguese books', 'c4 portuguese',
    'english corpus', 'english web text', 'fineweb', 'wikipedia english',
    'code instruction', 'code alpaca', 'python functions', 'javascript code',
    'programming problems', 'source code',
]

# Curated candidates, each with a *declared* permissive license on the Hub.
# These are starting points, not availability guarantees: every one is
# re-checked with ``dataset_info`` before a single row is streamed.  Sources
# whose Hub card says ``license:unknown`` (Helsinki-NLP/opus-100) or an
# ambiguous ``license:cc`` (HuggingFaceH4/CodeAlpaca_20K) are deliberately
# absent -- they used to be skipped silently and leave a group empty.
CANDIDATES = {
    'pt': ['allenai/c4', 'Madras1/corpus-ptbr-v1', 'HuggingFaceFW/fineweb-2'],
    'en': ['HuggingFaceFW/fineweb-edu', 'allenai/c4'],
    'code': ['flwrlabs/code-alpaca-20k', 'm-a-p/CodeFeedback-Filtered-Instruction',
             'theblackcat102/evol-codealpaca-v1', 'google-research-datasets/mbpp'],
    'react': [],  # real ReAct rows come from data/github_raw/github_react.jsonl
}
DATASET_CONFIGS = {
    ('allenai/c4', 'pt'): 'pt',
    ('allenai/c4', 'en'): 'en',
    ('HuggingFaceFW/fineweb-edu', 'en'): 'sample-10BT',
    ('HuggingFaceFW/fineweb-2', 'pt'): 'por_Latn',
    # Kept for older configs still referenced by downstream experiments.
    ('Helsinki-NLP/opus-100', 'pt'): 'en-pt',
    ('Helsinki-NLP/opus-100', 'en'): 'en-pt',
    ('uonlp/CulturaX', 'pt'): 'pt',
}

# Licenses as declared on the Hub when these sources were reviewed
# (2026-11-02).  ``dataset_info`` re-validates at run time; this table makes
# the review auditable and stops an undeclared-license source from being
# reintroduced as "curated" without a check.
CURATED_LICENSES = {
    'allenai/c4': 'odc-by',
    'Madras1/corpus-ptbr-v1': 'odc-by',
    'HuggingFaceFW/fineweb-2': 'odc-by',
    'HuggingFaceFW/fineweb-edu': 'odc-by',
    'flwrlabs/code-alpaca-20k': 'apache-2.0',
    'm-a-p/CodeFeedback-Filtered-Instruction': 'apache-2.0',
    'theblackcat102/evol-codealpaca-v1': 'apache-2.0',
    'google-research-datasets/mbpp': 'cc-by-4.0',
}

# ODC-BY is attribution-only; its attribution requirement is retained in
# data_cache/manifest.json and uploaded with the model card.
ALLOWED = {
    'apache-2.0', 'mit', 'cc0-1.0', 'cc-by-4.0', 'cc-by-3.0',
    'bsd-3-clause', 'bsd-2-clause', 'isc', 'unlicense', 'odc-by', 'odc-by-1.0',
}
MANUAL_KINDS = ('pt', 'en', 'code')
MIN_ROWS = 200              # enough rows per group to stop searching that group
MAX_CANDIDATES = 12         # hard bound on Hub sources tried per group
MAX_UNVERIFIED = 4          # hits whose license still needs dataset_info
MAX_SCAN_ROWS = 40_000      # hard bound on rows inspected per source
MAX_LOCAL_FILES = 400       # hard bound on repository files scanned
FIRST_PARTY = 'first-party (no third-party license claimed)'

_PT_WORDS = {'não', 'você', 'obrigado', 'então', 'porque', 'está', 'são', 'para',
             'com', 'mais', 'como', 'mas', 'isso', 'meu', 'uma', 'dos', 'das',
             'olá', 'bom', 'dia', 'estou', 'bem', 'português', 'brasil', 'aqui',
             'fazer', 'pode', 'tem', 'ele', 'ela', 'nós', 'também', 'sobre',
             'quando', 'quem', 'qual', 'onde', 'ser', 'ter', 'sendo', 'entre'}
_CODE_FENCE = re.compile(r'```[a-zA-Z0-9_+-]*\s*\n(.*?)```', re.S)


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
    value = value.rstrip('/').split('/')[-1]  # license URLs in croissant tags
    aliases = {
        'apache 2.0': 'apache-2.0', 'apache-2.0 license': 'apache-2.0',
        'apache2.0': 'apache-2.0', 'mit license': 'mit',
        'cc by 4.0': 'cc-by-4.0', 'cc by 3.0': 'cc-by-3.0',
        'odc-by 1.0': 'odc-by-1.0', 'odc-by-1.0 license': 'odc-by-1.0',
        'cc0': 'cc0-1.0', 'bsd': 'bsd-3-clause',
    }
    return aliases.get(value, value)


def _tag_license(info) -> str:
    """Some repos only declare ``license:mit`` as a Hub tag."""
    tags = getattr(info, 'tags', None) or []
    for tag in tags:
        if isinstance(tag, str) and tag.lower().startswith('license:'):
            return _normalise_license(tag.split(':', 1)[1])
    return ''


def license_of(info) -> str:
    card = getattr(info, 'card_data', None) or getattr(info, 'cardData', None) or {}
    return _normalise_license(_card_value(card, 'license', '')) or _tag_license(info)


def language_of(info):
    card = getattr(info, 'card_data', None) or getattr(info, 'cardData', None) or {}
    return _card_value(card, 'language', getattr(info, 'language', 'unknown'))


def dataset_config(name: str, kind: str) -> str | None:
    """The intentional streaming configuration for a known candidate."""
    return DATASET_CONFIGS.get((name, kind))


def _first(row: dict, keys: Iterable[str]):
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _looks_portuguese(text: str) -> bool:
    words = set(re.findall(r'[a-zà-ÿ]+', text.lower()))
    return len(words & _PT_WORDS) >= 2


def _translation_text(row, kind):
    translation = row.get('translation')
    if not isinstance(translation, dict):
        return None
    keys = ('pt', 'pt-br', 'por', 'pt_BR') if kind == 'pt' else ('en', 'en-us', 'eng')
    for key in keys:
        value = translation.get(key)
        if isinstance(value, str) and len(value.strip()) >= 30:
            return value.strip()
    return None


def extract(row, kind):
    """Turn a supported dataset row into a bounded, trainable text record."""
    if not isinstance(row, dict):
        return None
    if kind == 'code':
        question = _first(row, ('instruction', 'query', 'prompt', 'problem', 'question'))
        answer = _first(row, ('output', 'response', 'answer', 'completion',
                              'solution', 'code', 'content', 'text'))
        if isinstance(answer, list):
            answer = '\n'.join(str(item.get('content', '')) for item in answer
                              if isinstance(item, dict)) or None
        # A code dataset may contain only a bare source snippet.  The old
        # keyword filter dropped valid instruction/output rows and was one of
        # the reasons the ``code`` group came back empty.
        if question and answer and question != answer:
            return ('Question: ' + question + '\nCode: ' + answer)[:4096]
        # A bare column is code only if it actually contains code-ish syntax;
        # otherwise a prose dataset found by a "code" query would poison the mix.
        if answer and len(answer) >= 30 and re.search(
                r'def |function|import |class |=>|#include|SELECT |;|\{|\}|=', answer):
            return answer[:4096]
        return None
    if kind == 'react':
        # OASST-style chat messages are not tool trajectories; do not mislabel.
        text = _first(row, ('text', 'trajectory', 'content', 'conversation'))
        if text and 'Action:' in text and ('Observation:' in text or 'Thought:' in text):
            return text[:4096]
        return None
    translation = _translation_text(row, kind)
    if translation:
        return translation[:4096]
    for key in ('text', 'content', 'sentence', 'document', 'article', 'body'):
        value = row.get(key)
        if isinstance(value, dict):
            value = (_first(value, ('pt', 'pt-br', 'por')) if kind == 'pt'
                     else _first(value, ('en', 'en-us')))
        if isinstance(value, str) and len(value.strip()) >= 30:
            text = value.strip()[:4096]
            if kind == 'pt' and not _looks_portuguese(text):
                # Do not silently train an English corpus as Portuguese.
                return None
            return text
    return None


def _bounded_lines(path: Path) -> list[str]:
    """Read newline-delimited UTF-8 records, rejecting broken input explicitly."""
    try:
        with path.open('r', encoding='utf8', errors='strict') as handle:
            return [line.strip()[:4096] for line in handle if line.strip()]
    except UnicodeDecodeError as exc:
        raise ValueError(f'{path} is not valid UTF-8: {exc}') from exc


def _query_kinds(query, known=None):
    """Map any search term -- including deeper retry terms -- to data groups."""
    q = query.lower()
    kinds = set((known or {}).get(query, set()))
    if any(t in q for t in ('portug', 'pt-br', 'brazil', 'opus')):
        kinds.add('pt')
    if any(t in q for t in ('english', 'fineweb', 'web text', 'wikipedia english')):
        kinds.add('en')
    if any(t in q for t in ('code', 'python', 'javascript', 'program', 'alpaca', 'instruction')):
        kinds.add('code')
    if any(t in q for t in ('react', 'tool use', 'trajector', 'agent')):
        kinds.add('react')
    return sorted(kinds)


def discover(limit=5, api=None, queries=None):
    """Search the Hub and return *usable* candidate rows (not just a printout)."""
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError('limit must be a positive integer')
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    query_kinds = {}
    for kind, terms in KIND_QUERIES.items():
        for term in terms:
            query_kinds.setdefault(term, set()).add(kind)
    all_queries = list(dict.fromkeys((queries or []) + QUERIES +
                                     [q for terms in KIND_QUERIES.values() for q in terms]))[:40]
    reports = []
    for query in all_queries:
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
                    'downloads': getattr(row, 'downloads', 0) or 0,
                    'accepted_license': lic in ALLOWED,
                    'kinds': _query_kinds(query, query_kinds),
                })
        except Exception as exc:
            print('Search failed:', query, type(exc).__name__, str(exc)[:200])
    for row in reports:
        print(json.dumps(row, ensure_ascii=False, default=str))
    return reports


def search_candidates(reports, max_per_kind=MAX_CANDIDATES):
    """Turn discovery hits into per-group candidate IDs, ranked by popularity.

    Hits with an accepted license come first.  A bounded number of hits with
    missing/ambiguous metadata still goes through, because ``dataset_info``
    often exposes a license that ``list_datasets`` did not; those are skipped at
    collection time if they turn out to be non-permissive or gated.
    """
    buckets = {kind: [] for kind in KIND_QUERIES}
    for row in reports:
        for kind in row.get('kinds', []):
            buckets.setdefault(kind, []).append(row)
    rank = lambda r: (-int(r.get('downloads') or 0), r['name'])
    result = {}
    for kind, rows in buckets.items():
        curated = set(CANDIDATES.get(kind, []))
        accepted = sorted([r for r in rows if r.get('accepted_license')], key=rank)
        unverified = sorted([r for r in rows if not r.get('accepted_license')],
                            key=rank)[:MAX_UNVERIFIED]
        seen, names = set(), []
        for row in accepted + unverified:
            name = row['name']
            if name in curated or name in seen or '/' not in name:
                continue
            seen.add(name)
            names.append(name)
            if len(names) >= max_per_kind:
                break
        result[kind] = names
    return result


def _config_candidates(name, kind, info):
    """Ordered configs: known one, default, then declared configs."""
    options = []
    for value in [dataset_config(name, kind), None]:
        if value not in options:
            options.append(value)
    card = getattr(info, 'card_data', None) or getattr(info, 'cardData', None) or {}
    configs = _card_value(card, 'configs', None) or []
    for config in configs[:3]:
        value = config.get('config_name') if isinstance(config, dict) else config
        if isinstance(value, str) and value not in options:
            options.append(value)
    return options


def _open_stream(load_dataset, name, config, split='train'):
    """Load a streaming split, tolerating datasets without a ``train`` split."""
    try:
        stream = load_dataset(name, config, split=split, streaming=True)
    except TypeError:
        stream = load_dataset(name, config, streaming=True)
    except Exception:
        stream = load_dataset(name, config, streaming=True)
    if hasattr(stream, 'keys') and not hasattr(stream, '__next__'):
        if split in stream:
            stream = stream[split]
        elif stream:
            stream = stream[next(iter(stream.keys()))]
        else:
            raise ValueError('dataset exposes no splits')
    return iter(stream)


def _language_ok(info, kind):
    """Reject a source whose declared language contradicts the target group."""
    lang = language_of(info)
    values = [str(x).lower() for x in (lang if isinstance(lang, (list, tuple)) else [lang])]
    if not values or 'unknown' in values:
        return True
    # Only a short, explicit language list can contradict the target group;
    # hugely multilingual cards are decided by extraction instead.
    if len(values) > 3:
        return True
    if kind == 'pt' and not any(v.startswith(('pt', 'por')) for v in values):
        return False
    if kind == 'en' and not any(v.startswith(('en', 'eng')) for v in values):
        return False
    return True


def _stream_into(result, name, kind, config, load_dataset, max_rows,
                 max_bytes, min_rows, seen, record):
    """Stream one bounded source and return the number of rows added."""
    added = 0
    rows_seen = 0
    used_bytes = sum(len(text.encode('utf8')) for text in result[kind])
    for row in _open_stream(load_dataset, name, config):
        rows_seen += 1
        if rows_seen > MAX_SCAN_ROWS:
            record['stop_reason'] = 'scan_limit'
            break
        text = extract(row, kind)
        if not text:
            # Give up on a source that yields nothing instead of scanning it
            # forever; a wrong config usually fails immediately.
            if added == 0 and rows_seen >= min_rows * 10:
                record['stop_reason'] = 'no_usable_rows'
                break
            continue
        text = text.strip()[:4096]
        if text in seen[kind]:
            continue
        n_bytes = len(text.encode('utf8'))
        if used_bytes + n_bytes > max_bytes:
            record['stop_reason'] = 'byte_limit'
            break
        result[kind].append(text)
        seen[kind].add(text)
        used_bytes += n_bytes
        added += 1
        if len(result[kind]) >= max_rows:
            record['stop_reason'] = 'row_limit'
            break
        if len(result[kind]) >= min_rows:
            record['stop_reason'] = 'min_rows'
            break
    record.update(rows=len(result[kind]), rows_added=added, rows_seen=rows_seen,
                  cached_bytes=used_bytes)
    return added


def _hub_source(result, manifest, api, load_dataset, name, kind, max_rows,
                max_bytes, min_rows, seen):
    """Verify a Hub dataset's license, then try configs until rows arrive.

    Exactly one manifest row is written per source; individual config attempts
    are nested under ``config_attempts`` so provenance stays readable.
    """
    record = {'source': 'huggingface', 'name': name, 'kind': kind,
              'config': dataset_config(name, kind), 'split': 'train'}
    try:
        info = api.dataset_info(name)
        license_ = license_of(info)
        gated = getattr(info, 'gated', False) or getattr(info, 'private', False)
        record.update(license=license_ or 'unknown', language=language_of(info),
                      size_bytes=getattr(info, 'size_in_bytes', None))
        if license_ not in ALLOWED:
            record['status'] = 'skipped: unverified or non-permissive license'
        elif gated:
            record['status'] = 'skipped: gated or private dataset'
        elif not _language_ok(info, kind):
            record['status'] = 'skipped: declared language does not match group'
        else:
            attempts = []
            record['status'] = 'skipped: no usable rows in any config'
            for config in _config_candidates(name, kind, info):
                attempt = {'config': config}
                try:
                    added = _stream_into(result, name, kind, config, load_dataset,
                                         max_rows, max_bytes, min_rows, seen, attempt)
                    attempt['status'] = 'streamed' if added else 'skipped: no usable rows'
                except Exception as exc:
                    added = 0
                    attempt['status'] = ('skipped: ' + type(exc).__name__ + ': ' +
                                         str(exc)[:200])
                attempts.append(attempt)
                if added:
                    record.update(status='streamed', config=config,
                                  rows_added=added, rows_seen=attempt.get('rows_seen'),
                                  cached_bytes=attempt.get('cached_bytes'))
                    break
            record['config_attempts'] = attempts
    except Exception as exc:
        record['status'] = 'skipped: ' + type(exc).__name__ + ': ' + str(exc)[:200]
    print(json.dumps(record, ensure_ascii=False, default=str))
    manifest.append(record)


def _add(result, kind, text, seen, used, max_rows, max_bytes):
    text = text.strip()[:4096]
    if not text or text in seen[kind] or len(result[kind]) >= max_rows:
        return 0
    n_bytes = len(text.encode('utf8'))
    if used[kind] + n_bytes > max_bytes:
        return 0
    result[kind].append(text)
    seen[kind].add(text)
    used[kind] += n_bytes
    return 1


def _local_sources(result, manifest, repo_root, seen, used, max_rows, max_bytes):
    """Search the checkout for real, licensed corpora and first-party code."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    raw = root / 'data' / 'github_raw'
    licenses = {}
    try:
        licenses = json.loads((raw / 'manifest.json').read_text(encoding='utf8')).get('licenses', {})
    except Exception:
        pass
    changed = {kind: 0 for kind in result}

    stories = raw / 'github_stories.jsonl'
    if stories.exists():
        for line in _bounded_lines(stories)[:6000]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _normalise_license(record.get('license')) not in ALLOWED:
                continue  # NOASSERTION and unknown licenses are not training data
            title = str(record.get('title') or record.get('description') or '').strip()
            body = str(record.get('body') or record.get('message') or record.get('content') or '').strip()
            text = ('Title: ' + title + '\n' + body).strip() if title else body
            changed['en'] += _add(result, 'en', text, seen, used, max_rows, max_bytes)
            for block in _CODE_FENCE.findall(body)[:2]:
                changed['code'] += _add(result, 'code', block, seen, used, max_rows, max_bytes)
        manifest.append({'source': 'local-corpus', 'name': 'data/github_raw/github_stories.jsonl',
                         'license': 'per-record SPDX (MIT/Apache-2.0/BSD-3-Clause only)',
                         'status': 'scanned', 'rows_added': {'en': changed['en'], 'code': changed['code']}})

    react = raw / 'github_react.jsonl'
    if react.exists():
        for line in _bounded_lines(react)[:6000]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _normalise_license(licenses.get(record.get('source'))) not in ALLOWED:
                continue
            text = record.get('text')
            if isinstance(text, str):
                changed['react'] += _add(result, 'react', text, seen, used, max_rows, max_bytes)
        manifest.append({'source': 'local-corpus', 'name': 'data/github_raw/github_react.jsonl',
                         'license': 'per-source SPDX from manifest.json',
                         'status': 'scanned', 'rows_added': {'react': changed['react']}})

    # First-party source files are real code bytes, not invented text.  They are
    # recorded separately because no third-party license is claimed for them.
    py_files, added_code = [], 0
    for path in sorted(root.rglob('*.py')):
        if any(part in ('.git', 'data_cache', '__pycache__', 'checkpoints', '.venv')
               for part in path.parts):
            continue
        if len(py_files) >= MAX_LOCAL_FILES:
            break
        try:
            text = path.read_text(encoding='utf8')[:4096]
        except (OSError, UnicodeDecodeError):
            continue
        if len(text) < 80:
            continue
        if _add(result, 'code', text, seen, used, max_rows, max_bytes):
            py_files.append(str(path.relative_to(root)))
            added_code += 1
    if py_files:
        manifest.append({'source': 'local-first-party', 'name': 'repository *.py sources',
                         'license': FIRST_PARTY, 'status': 'scanned',
                         'rows_added': {'code': added_code}, 'files': py_files[:20]})


def _ensure(result, kinds):
    for kind in kinds:
        result.setdefault(kind, [])
    return result


def merge_manual_data(groups: dict, manual_dir='data_cache/manual_data',
                      kinds: Iterable[str] = MANUAL_KINDS) -> dict:
    """Merge licensed user-supplied ``pt.txt``, ``en.txt`` and ``code.txt`` files.

    Kept for optional, *silent* use: files already present in the cache are
    merged idempotently, but the notebook never opens an upload picker.
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


def collect(cache_dir='data_cache', max_rows=4000, max_bytes=8_000_000,
            min_rows=MIN_ROWS, search=True, reports=None, groups=None,
            repo_root=None, search_limit=8, max_candidates=MAX_CANDIDATES,
            kinds=None):
    """Find and cache real training rows.  No user upload, ever.

    Curated corpora are tried first, then the datasets returned by a live Hub
    search, then the real corpora checked into this repository.  Sources with
    unknown/ambiguous licenses are skipped and every attempt is recorded in
    ``manifest.json``.  ``groups`` may be an existing dict to extend in place
    (used by ``fill_missing``), and ``reports`` may reuse a prior search.
    """
    if not isinstance(max_rows, int) or max_rows <= 0:
        raise ValueError('max_rows must be a positive integer')
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError('max_bytes must be a positive integer')
    if not isinstance(min_rows, int) or min_rows <= 0:
        raise ValueError('min_rows must be a positive integer')
    from datasets import load_dataset
    from huggingface_hub import HfApi

    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    all_kinds = set(CANDIDATES) | set(KIND_QUERIES) | set(MANUAL_KINDS)
    target_kinds = (set(kinds) & all_kinds) if kinds else all_kinds
    result = _ensure(groups if groups is not None else {}, all_kinds)
    seen = {kind: set(result[kind]) for kind in result}
    used = {kind: sum(len(t.encode('utf8')) for t in result[kind]) for kind in result}
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text(encoding='utf8'))
            if isinstance(prior, list):
                manifest = [row for row in prior if isinstance(row, dict)
                            and row.get('kind') != 'summary']
            else:
                manifest = []
        except (json.JSONDecodeError, OSError):
            manifest = []
    else:
        manifest = []

    if search:
        if reports is None:
            reports = discover(limit=search_limit, api=api)
        searched = search_candidates(reports, max_per_kind=max_candidates)
    else:
        searched = {}

    for kind in sorted(target_kinds):
        names = list(dict.fromkeys(list(CANDIDATES.get(kind, [])) + searched.get(kind, [])))
        for name in names[:len(CANDIDATES.get(kind, [])) + max_candidates]:
            if len(result[kind]) >= min_rows or len(result[kind]) >= max_rows:
                break
            _hub_source(result, manifest, api, load_dataset, name, kind,
                        max_rows, max_bytes, min_rows, seen)
            used[kind] = sum(len(t.encode('utf8')) for t in result[kind])
        print(kind, 'rows after Hub search:', len(result[kind]))

    _local_sources(result, manifest, repo_root, seen, used, max_rows, max_bytes)
    manual = merge_manual_data(result, root / 'manual_data')
    if any(row['rows_added'] for row in manual.values()):
        manifest.append({'source': 'local-optional-files', 'name': 'data_cache/manual_data',
                         'license': 'owner-supplied', 'status': 'merged', 'rows_added': manual})

    counts = group_counts(result)
    manifest.append({'kind': 'summary', 'source': 'search', 'group_counts': counts,
                     'hub_attempts': sum(1 for row in manifest if row.get('source') == 'huggingface'),
                     'upload_required': False,
                     'missing_groups': [k for k in MANUAL_KINDS if not counts.get(k)]})
    (root / 'samples.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf8')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                             encoding='utf8')
    return result


def fill_missing(groups, cache_dir='data_cache', kinds=MANUAL_KINDS,
                 search_limit=15, max_candidates=MAX_CANDIDATES, **kwargs):
    """Retry with a deeper automatic search for any still-empty group.

    Called by the notebook's preflight cell instead of a file picker.  It never
    asks for data; if a group is still empty it fails with a diagnostic that
    names the manifest and the likely cause (Hub/network availability).
    """
    missing = [kind for kind in kinds if not groups.get(kind)]
    if not missing:
        return {'attempted': False, 'missing': []}
    print('Groups still empty after the first pass:', ', '.join(missing))
    print('Searching more Hub datasets and the repository corpus; no upload is required.')
    reports = discover(limit=search_limit, queries=DEEP_QUERIES)
    collect(cache_dir=cache_dir, groups=groups, reports=reports, kinds=missing,
            search_limit=search_limit, max_candidates=max_candidates, **kwargs)
    still = [kind for kind in kinds if not groups.get(kind)]
    if still:
        raise RuntimeError(
            'Automatic dataset search could not obtain real rows for: ' + ', '.join(still) +
            '. See data_cache/manifest.json for every source attempted and why it was '
            'skipped (license, gating, config or network). Check that this runtime can '
            'reach huggingface.co and rerun this cell; the notebook does not request '
            'uploaded files and will not substitute fabricated language data.')
    return {'attempted': True, 'missing': missing, 'group_counts': group_counts(groups)}
