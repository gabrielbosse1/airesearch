"""HF discovery and bounded streaming. Only verified permissive licenses are used."""
import json
import os
import re

QUERIES = ['pt-br', 'portuguese', 'brazilian', 'code', 'programming', 'python',
           'javascript', 'minecraft', 'game commands', 'openassistant', 'tool', 'agent']
# These are candidates, not a license or availability guarantee. bert-base is a
# model, not a dataset. en2ptbridge must pass the exact same dataset checks.
CANDIDATES = {
    'pt': ['maritaca-ai/en2ptbridge', 'Helsinki-NLP/opus-100', 'uonlp/CulturaX'],
    'en': ['Helsinki-NLP/opus-100', 'HuggingFaceFW/fineweb-edu'],
    'code': ['m-a-p/CodeFeedback-Filtered-Instruction', 'HuggingFaceH4/CodeAlpaca_20K'],
    'react': ['OpenAssistant/oasst1'],
}
ALLOWED = {'apache-2.0', 'mit', 'cc0-1.0', 'cc-by-4.0', 'cc-by-3.0',
           'bsd-3-clause', 'bsd-2-clause', 'isc', 'unlicense'}

def license_of(info):
    card = getattr(info, 'card_data', None) or {}
    license_ = card.get('license') if hasattr(card, 'get') else getattr(card, 'license', None)
    return str(license_ or '').lower()

def discover(limit=5):
    from huggingface_hub import HfApi
    api = HfApi()
    reports = []
    print('Note: pierreguillou/bert-base-cased-pt-br is a model, not a corpus dataset.')
    for q in QUERIES:
        try:
            rows = list(api.list_datasets(search=q, limit=limit, full=True))
            for row in rows:
                lic = license_of(row)
                reports.append({'query': q, 'name': row.id, 'license': lic or 'unknown',
                    'language': (getattr(row, 'card_data', None) or {}).get('language', 'unknown'),
                    'size_bytes': getattr(row, 'size_in_bytes', None),
                    'accepted_license': lic in ALLOWED})
        except Exception as exc:
            print('Search failed:', q, type(exc).__name__, str(exc)[:200])
    for row in reports:
        print(json.dumps(row, ensure_ascii=False, default=str))
    return reports

def extract(row, kind):
    if kind == 'code':
        q = row.get('instruction') or row.get('query') or row.get('prompt') or row.get('problem')
        a = row.get('output') or row.get('response') or row.get('answer') or row.get('completion')
        if isinstance(a, list):
            a = '\n'.join(str(x.get('content', '')) for x in a if isinstance(x, dict))
        if not isinstance(q, str) or not isinstance(a, str): return None
        if not re.search(r'python|javascript|\bjs\b|\bdef\b|\bfunction\b', q+' '+a, re.I): return None
        return 'Question: '+q+'\nCode: '+a
    if kind == 'react':
        # OASST messages are not tool trajectories; do not mislabel them as such.
        return None
    for key in ('text', 'content', 'sentence', 'translation'):
        v = row.get(key)
        if isinstance(v, dict): v = v.get('pt') or v.get('pt-br') or v.get('en')
        if isinstance(v, str) and len(v.strip()) >= 30: return v
    return None

def collect(cache_dir='data_cache', max_rows=4000, max_bytes=8_000_000):
    """Stream bounded samples from explicitly licensed candidates, then cache JSON.

    Dataset streams can fail or require gated access; absent groups remain empty.
    No restricted or unknown license is silently accepted.
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi
    os.makedirs(cache_dir, exist_ok=True)
    api = HfApi(); result = {k: [] for k in CANDIDATES}; manifest = []
    for kind, ids in CANDIDATES.items():
        for name in ids:
            if len(result[kind]) >= max_rows: break
            record = {'name': name, 'kind': kind}
            try:
                info = api.dataset_info(name)
                lic = license_of(info)
                record.update(license=lic or 'unknown', language=(getattr(info, 'card_data', None) or {}).get('language', 'unknown'), size_bytes=getattr(info, 'size_in_bytes', None))
                if lic not in ALLOWED:
                    record['status'] = 'skipped: unverified or non-permissive license'
                    continue
                if kind == 'react':
                    record['status'] = 'skipped: chat messages are not ReAct trajectories'
                    continue
                # CulturaX has language configs; explicitly request Portuguese.
                config = 'pt' if name == 'uonlp/CulturaX' else ('en-pt' if name == 'Helsinki-NLP/opus-100' else None)
                stream = load_dataset(name, config, split='train', streaming=True)
                seen_bytes = sum(len(x.encode('utf-8')) for x in result[kind])
                rows_seen = 0
                for row in stream:
                    rows_seen += 1
                    text = (row.get('translation', {}).get('pt' if kind == 'pt' else 'en') if name == 'Helsinki-NLP/opus-100' else extract(row, kind))
                    if text:
                        text = text[:4096]
                        n = len(text.encode('utf-8'))
                        if seen_bytes + n > max_bytes: break
                        result[kind].append(text); seen_bytes += n
                    if len(result[kind]) >= max_rows or rows_seen >= max_rows * 10: break
                record.update(status='streamed', rows=len(result[kind]), cached_bytes=seen_bytes)
            except Exception as exc:
                record['status'] = 'skipped: ' + type(exc).__name__ + ': ' + str(exc)[:200]
            finally:
                manifest.append(record); print(json.dumps(record, ensure_ascii=False, default=str))
    with open(os.path.join(cache_dir, 'samples.json'), 'w', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, 'manifest.json'), 'w', encoding='utf8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return result
