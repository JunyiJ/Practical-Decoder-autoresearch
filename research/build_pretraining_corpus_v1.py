"""Bounded, revision-pinned corpus preparation; never launches training.

Run with the project Python, from the repository root. Text payloads stay under
data/raw (gitignored). This is independent of frozen Shakespeare campaigns.
"""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import random
import re
import shutil
import time
import unicodedata
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import fsspec
import numpy as np
import pyarrow.parquet as pq
import requests
import tiktoken

SOURCES = {
    'fineweb_edu': {
        'repo': 'HuggingFaceFW/fineweb-edu',
        'revision': '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9',
        'prefix': 'sample/10BT/', 'weight': .8,
        'license': 'ODC-By-1.0; Common Crawl terms; underlying content rights remain applicable',
    },
    'wikipedia': {
        'repo': 'wikimedia/wikipedia',
        'revision': 'b04c8d1ceb2f5cd4588862100d08de323dccfbaa',
        'prefix': '20231101.en/', 'weight': .2,
        'license': 'CC-BY-SA-3.0 / GFDL per pinned dataset card; retain article URLs',
    },
}
SEED = 20261108
MAX_TOKENS = 8192
MIN_TOKENS = 256
SPLITS = ('train', 'val', 'test')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_url(url):
    p = urlsplit(url)
    return urlunsplit(('https', p.netloc.lower().removeprefix('www.'),
                       p.path.rstrip('/'), '', ''))


def split_for(url):
    bucket = int(hashlib.sha256(f'{SEED}:{canonical_url(url)}'.encode()).hexdigest()[:8], 16) % 10000
    return 'test' if bucket < 100 else 'val' if bucket < 200 else 'train'


def clean_text(text):
    text = unicodedata.normalize('NFC', text).replace('\r\n', '\n').replace('\r', '\n')
    text = ''.join(c for c in text if c in '\n\t' or not unicodedata.category(c).startswith('C'))
    text = '\n'.join(re.sub(r'[\t ]+', ' ', line).strip() for line in text.splitlines())
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def quality_reason(text):
    if not 500 <= len(text) <= 80000:
        return 'character_length'
    if '\ufffd' in text:
        return 'replacement_character'
    if len(re.findall(r'<[^>]{1,200}>', text)) > 5:
        return 'html_residue'
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())
    if len(words) < 100 or sum(c.isalpha() for c in text) / len(text) < .55:
        return 'not_prose'
    if len(set(words)) / len(words) < .12:
        return 'low_word_diversity'
    lines = [line.lower() for line in text.splitlines() if len(line) > 20]
    if lines and 1 - len(set(lines)) / len(lines) > .2:
        return 'repeated_lines'
    grams = list(zip(words, words[1:], words[2:]))
    if grams and 1 - len(set(grams)) / len(grams) > .25:
        return 'repeated_trigrams'
    return None


def fingerprint(text):
    normalized = ' '.join(text.casefold().split())
    exact = hashlib.sha256(normalized.encode()).hexdigest()
    words = re.findall(r'\w+', normalized)
    # SimHash of unique word trigrams, no stochastic sampling.
    shingles = {' '.join(words[i:i+3]) for i in range(len(words)-2)}
    hashes = b''.join(hashlib.blake2b(s.encode(), digest_size=8).digest() for s in shingles)
    bits = np.unpackbits(np.frombuffer(hashes, dtype=np.uint8).reshape(-1, 8), axis=1)
    sim = int.from_bytes(np.packbits(bits.sum(axis=0) * 2 >= len(shingles)).tobytes(), 'big')
    return exact, sim


class Deduper:
    """Global exact/URL and conservative approximate SimHash deduplication."""
    def __init__(self):
        self.exact = set()
        self.urls = set()
        self.bands = defaultdict(list)

    def duplicate(self, url, exact, sim, length):
        if exact in self.exact:
            return 'duplicate_text'
        if canonical_url(url) in self.urls:
            return 'duplicate_url'
        for band in range(4):
            for other, n in self.bands[band, (sim >> (16 * band)) & 65535]:
                if min(n, length) / max(n, length) >= .8 and (sim ^ other).bit_count() <= 3:
                    return 'near_duplicate'
        return None

    def add(self, url, exact, sim, length):
        self.exact.add(exact)
        self.urls.add(canonical_url(url))
        for band in range(4):
            self.bands[band, (sim >> (16 * band)) & 65535].append((sim, length))


class CrossSplitOverlap:
    """Reject copied passages across splits using 64-token shingles.

    Store every 32nd training shingle and all evaluation shingles; query every
    shingle of incoming documents. Keeping all evaluation shingles also catches
    matches to sampled training windows when evaluation arrives before training.
    This catches every exact shared passage of >=95 tokens, and some shorter
    passages. It is not a claim of complete semantic decontamination.
    """
    def __init__(self):
        self.shingles = {s: set() for s in SPLITS}

    def hashes(self, ids):
        raw = np.asarray(ids, dtype='<u2').tobytes()
        return [hashlib.blake2b(raw[2*i:2*(i+64)], digest_size=16).digest()
                for i in range(len(ids)-63)]

    def duplicate(self, split, hashes):
        return any(h in self.shingles[s] for s in SPLITS if s != split for h in hashes)

    def add(self, split, hashes):
        self.shingles[split].update(hashes[::32] if split == 'train' else hashes)


def source_rows(source, directory, log):
    spec = SOURCES[source]
    base = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}"
    api = f"https://huggingface.co/api/datasets/{spec['repo']}/revision/{spec['revision']}"
    response = requests.get(api, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    if metadata['sha'] != spec['revision']:
        raise ValueError('Source revision mismatch')
    files = sorted(x['rfilename'] for x in metadata['siblings']
                   if x['rfilename'].startswith(spec['prefix']) and x['rfilename'].endswith('.parquet'))
    rng = random.Random(f'{SEED}:{source}')
    rng.shuffle(files)
    files = files[:4]
    card = requests.get(base + '/README.md', timeout=60)
    card.raise_for_status()
    (directory / f'{source}-card.md').write_bytes(card.content)
    log[source] = dict(spec, selected_shards=files, row_groups=[], card_sha256=sha_file(directory / f'{source}-card.md'))
    with ExitStack() as stack:
        readers = []
        for path in files:
            f = stack.enter_context(fsspec.open(base + '/' + path, 'rb', block_size=1024*1024, cache_type='readahead'))
            reader = pq.ParquetFile(f)
            groups = list(range(reader.num_row_groups))
            rng.shuffle(groups)
            readers.append((path, reader, groups))
        for round_index in range(max(len(x[2]) for x in readers)):
            for path, reader, groups in readers:
                if round_index >= len(groups):
                    continue
                group = groups[round_index]
                log[source]['row_groups'].append({'shard': path, 'row_group': group})
                records = reader.read_row_group(group).to_pylist()
                order = list(range(len(records)))
                rng.shuffle(order)
                for row_index in order:
                    yield records[row_index], {'shard': path, 'row_group': group, 'row_in_group': row_index}


def build(output, train_tokens, eval_tokens):
    # Max retained output forecast: text+tokens+metadata+verification, with ample
    # slack. Range reads stay in RAM; no full parquet cache is written to disk.
    forecast = max(1024**3, (train_tokens + 2*eval_tokens) * 40)
    if shutil.disk_usage(output.parent).free < forecast + 2*1024**3:
        raise RuntimeError(f'Need {forecast + 2*1024**3} free bytes including reserve')
    output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output / 'builder-source.py')
    started = time.time()
    enc = tiktoken.get_encoding('gpt2')
    ranks = sorted((token.hex(), rank) for token, rank in enc._mergeable_ranks.items())
    manifest = {'format': 'rsi-pretraining-corpus-v1', 'status': 'building', 'seed': SEED,
                    'requested_tokens': {'train': train_tokens, 'val': eval_tokens, 'test': eval_tokens},
                    'forecast_bytes': forecast, 'reserve_bytes': 2*1024**3,
                    'tokenizer': {'name': 'gpt2', 'vocab_size': enc.n_vocab, 'eot': enc.eot_token,
                                   'vocabulary_sha256': hashlib.sha256(json.dumps(ranks).encode()).hexdigest()},
                    'versions': {x: importlib.metadata.version(x) for x in ('numpy', 'tiktoken', 'pyarrow', 'fsspec')},
                    'builder_sha256': sha_file(__file__), 'sources': {}, 'counts': {}, 'rejections': {}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    deduper = Deduper()
    overlap = CrossSplitOverlap()
    counts = defaultdict(Counter)
    rejected = defaultdict(Counter)
    domains = Counter()
    offsets = Counter()
    documents = Counter()
    domain_limit = max(10000, train_tokens // 100)
    with ExitStack() as stack:
        token_files = {s: stack.enter_context((output / f'{s}.bin').open('wb')) for s in SPLITS}
        doc_files = {s: stack.enter_context(gzip.open(output / f'{s}.jsonl.gz', 'wt', encoding='utf8')) for s in SPLITS}
        indexes = {s: stack.enter_context((output / f'{s}.index.jsonl').open('w')) for s in SPLITS}
        for source, spec in SOURCES.items():
            targets = {s: int(manifest['requested_tokens'][s]*spec['weight']) for s in SPLITS}
            rows = source_rows(source, output, manifest['sources'])
            try:
                for seen, (row, location) in enumerate(rows, 1):
                    if seen > 100000:
                        raise RuntimeError('Bounded scan exhausted before completing quotas')
                    if all(counts[source][s] >= targets[s] for s in SPLITS):
                        break
                    url = row['url']
                    domain = urlsplit(canonical_url(url)).hostname
                    split = split_for(url)
                    if counts[source][split] >= targets[split]:
                        rejected[source]['split_quota_full'] += 1
                        continue
                    if source == 'fineweb_edu':
                        if row.get('language') != 'en' or row.get('language_score', 0) < .8 or row.get('int_score', 0) < 3:
                            rejected[source]['upstream_quality'] += 1
                            continue
                        if any(domain == d or domain.endswith('.'+d) for d in ('wikipedia.org', 'wikimedia.org')):
                            rejected[source]['wikipedia_mirror_domain'] += 1
                            continue
                        if domains[domain] >= domain_limit:
                            rejected[source]['domain_cap'] += 1
                            continue
                    text = clean_text(row['text'])
                    reason = quality_reason(text)
                    if reason:
                        rejected[source][reason] += 1
                        continue
                    ids = enc.encode(text, disallowed_special=())
                    if not MIN_TOKENS <= len(ids) <= MAX_TOKENS:
                        rejected[source]['token_length'] += 1
                        continue
                    exact, sim = fingerprint(text)
                    reason = deduper.duplicate(url, exact, sim, len(ids))
                    if reason:
                        rejected[source][reason] += 1
                        continue
                    shingles = overlap.hashes(ids)
                    if overlap.duplicate(split, shingles):
                        rejected[source]['cross_split_passage_overlap'] += 1
                        continue
                    deduper.add(url, exact, sim, len(ids))
                    overlap.add(split, shingles)
                    ids.append(enc.eot_token)
                    tokens = len(ids)
                    record = {'source': source, 'id': row['id'], 'url': url, 'title': row.get('title'),
                                  'location': location, 'split': split, 'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                                  'normalized_sha256': exact, 'simhash': f'{sim:016x}', 'tokens': tokens,
                                  'edu_score': row.get('score'), 'language_score': row.get('language_score'),
                                  'dump': row.get('dump')}
                    doc_files[split].write(json.dumps(record, ensure_ascii=False)+'\n')
                    indexes[split].write(json.dumps({'offset': offsets[split], 'tokens': tokens, 'source': source,
                                                       'normalized_sha256': exact, 'url': url})+'\n')
                    token_files[split].write(np.asarray(ids, dtype='<u2').tobytes())
                    offsets[split] += tokens
                    documents[split] += 1
                    counts[source][split] += tokens
                    domains[domain] += tokens
                    if sum(documents.values()) % 500 == 0:
                        print(json.dumps({'documents': sum(documents.values()), 'tokens': dict(offsets), 'source': source,
                                              'elapsed_s': round(time.time()-started)}), flush=True)
            finally:
                rows.close()
            if any(counts[source][s] < targets[s] for s in SPLITS):
                raise RuntimeError('Source exhausted before completing quotas')
    manifest.update(status='complete', counts={s: dict(v) for s,v in counts.items()},
                    rejections={s: dict(v) for s,v in rejected.items()}, tokens=dict(offsets), documents=dict(documents),
                    unique_hostnames=len(domains), top_hostnames=domains.most_common(20),
                    elapsed_seconds=time.time()-started, free_bytes_after=shutil.disk_usage(output).free)
    manifest['files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': sha_file(p)}
                         for p in sorted(output.iterdir()) if p.name != 'manifest.json'}
    manifest['total_payload_bytes'] = sum(x['bytes'] for x in manifest['files'].values())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('status','tokens','documents','total_payload_bytes','elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/raw/rsi-general-10m-v1'))
    parser.add_argument('--train-tokens', type=int, default=10_000_000)
    parser.add_argument('--eval-tokens', type=int, default=100_000)
    args = parser.parse_args()
    if args.train_tokens < 100000 or args.eval_tokens < 1000:
        parser.error('Minimum 100K training and 1K tokens per evaluation split')
    build(args.output, args.train_tokens, args.eval_tokens)
