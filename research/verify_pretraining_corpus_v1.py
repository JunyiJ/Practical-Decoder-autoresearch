"""Verify all local corpus documents and token bytes without scoring a model."""
import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tiktoken

from research.build_pretraining_corpus_v1 import canonical_url, sha_file, split_for
from src.data.pretraining_corpus import PretrainingCorpus


def verify(directory):
    started = time.time()
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    for name, expected in manifest['files'].items():
        path = directory / name
        assert path.stat().st_size == expected['bytes'], name
        assert sha_file(path) == expected['sha256'], name
    enc = tiktoken.get_encoding('gpt2')
    ranks = sorted((token.hex(), rank) for token, rank in enc._mergeable_ranks.items())
    assert hashlib.sha256(json.dumps(ranks).encode()).hexdigest() == manifest['tokenizer']['vocabulary_sha256']
    exacts, urls = set(), set()
    bands = defaultdict(list)
    counts, sources = {}, defaultdict(Counter)
    for split in ('train', 'val', 'test'):
        array = np.memmap(directory / f'{split}.bin', mode='r', dtype='<u2')
        assert len(array) == manifest['tokens'][split]
        assert int(array.max()) < enc.n_vocab
        offset, documents = 0, 0
        with gzip.open(directory / f'{split}.jsonl.gz', 'rt') as docs, (directory / f'{split}.index.jsonl').open() as index:
            for line in docs:
                row = json.loads(line)
                entry = json.loads(next(index))
                text = row['text']
                digest = hashlib.sha256(' '.join(text.casefold().split()).encode()).hexdigest()
                assert hashlib.sha256(text.encode()).hexdigest() == row['text_sha256']
                assert digest == row['normalized_sha256'] == entry['normalized_sha256']
                url = canonical_url(row['url'])
                assert digest not in exacts and url not in urls
                exacts.add(digest)
                urls.add(url)
                assert split_for(row['url']) == split == row['split']
                assert entry['offset'] == offset and entry['tokens'] == row['tokens']
                assert entry['source'] == row['source'] and entry['url'] == row['url']
                ids = enc.encode(text, disallowed_special=()) + [enc.eot_token]
                assert len(ids) == row['tokens']
                assert np.array_equal(array[offset:offset+len(ids)], ids)
                # Independently check every retained signature against preceding
                # ones, including cross-split pairs, under the declared heuristic.
                sim = int(row['simhash'], 16)
                for band in range(4):
                    key = band, (sim >> (16*band)) & 65535
                    for previous, length in bands[key]:
                        assert not (min(length, len(ids)-1)/max(length, len(ids)-1) >= .8
                                    and (sim ^ previous).bit_count() <= 3)
                    bands[key].append((sim, len(ids)-1))
                offset += len(ids)
                documents += 1
                sources[row['source']][split] += len(ids)
            assert next(index, None) is None
        assert offset == len(array)
        assert documents == manifest['documents'][split]
        counts[split] = {'documents': documents, 'tokens': offset}
    assert {k:dict(v) for k,v in sources.items()} == manifest['counts']
    # Independent passage audit: compare ALL 64-token windows in each later
    # split against stride-32 windows in preceding splits. Shared runs of at
    # least 95 exact tokens must collide; many shorter runs also collide.
    previous_shingles = set()
    passage_matches = {}
    for split in ('train', 'val', 'test'):
        array = np.memmap(directory / f'{split}.bin', mode='r', dtype='<u2')
        own_shingles = set()
        matches = 0
        for line in (directory / f'{split}.index.jsonl').read_text().splitlines():
            entry = json.loads(line)
            raw = array[entry['offset']:entry['offset']+entry['tokens']-1].tobytes()
            for i in range(entry['tokens']-64):
                h = hashlib.blake2b(raw[2*i:2*(i+64)], digest_size=16).digest()
                matches += h in previous_shingles
                if i % 32 == 0:
                    own_shingles.add(h)
        passage_matches[split] = matches
        assert matches == 0, (split, 'copied passages across splits', matches)
        previous_shingles.update(own_shingles)
    loader = PretrainingCorpus(directory)
    twin = PretrainingCorpus(directory)
    for split in ('train', 'val'):
        x, y = loader.get_batch(split)
        tx, ty = twin.get_batch(split)
        assert x.shape == y.shape == (8, 128)
        assert (x == tx).all() and (y == ty).all()
        assert (x[:, 1:] == y[:, :-1]).all()
    try:
        loader.get_batch('test')
    except ValueError:
        pass
    else:
        raise AssertionError('Training loader exposed final test split')
    result = {'status': 'passed', 'manifest_sha256': sha_file(directory/'manifest.json'),
                  'checks': ['all_payload_hashes_and_sizes', 'all_document_token_roundtrips',
                          'token_range_and_eot', 'complete_index_coverage',
                          'global_normalized_exact_text_and_url_uniqueness',
                          'retained_simhash_duplicate_threshold', 'deterministic_document_split',
                          'source_token_totals', 'cross_split_passage_audit',
                          'deterministic_train_val_batches', 'test_access_rejected'],
                  'passage_matches': passage_matches,
                  'counts': counts, 'elapsed_seconds': time.time()-started,
                  'limitation': 'SimHash is approximate; no claim of complete semantic deduplication or factual verification.'}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    verify(sys.argv[1] if len(sys.argv)>1 else 'data/raw/rsi-general-10m-v1')
