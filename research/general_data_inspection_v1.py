"""Read-only training/validation integrity checks; test payload stays unopened."""
import gzip
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/raw/rsi-general-10m-v1'
OUT = ROOT / 'research/tokenizer-evidence/general-data-inspection-001.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    enc = tiktoken.get_encoding('gpt2')
    ranks = sorted((token.hex(), rank) for token, rank in enc._mergeable_ranks.items())
    vocabulary = hashlib.sha256(json.dumps(ranks).encode()).hexdigest()
    assert vocabulary == manifest['tokenizer']['vocabulary_sha256']
    recipe = json.loads((ROOT / 'research/tokenizer-evidence/public-inference-correction-001/adoption/recipe.json').read_text())
    assert vocabulary == recipe['model_signature_json'][3][3]
    verified = []
    for name, record in manifest['files'].items():
        if name.startswith('test.'):
            continue
        path = DATA / name
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], name
        verified.append(name)
    result = dict(manifest_sha256=sha(DATA / 'manifest.json'), verified_files=verified,
                  tokenizer_matches_selected_model=True, test_payload_opened=False,
                  training_launched=False, splits={})
    indexes = {}
    arrays = {}
    for split in ('train', 'val'):
        rows = [json.loads(line) for line in (DATA / f'{split}.index.jsonl').read_text().splitlines()]
        tokens = np.fromfile(DATA / f'{split}.bin', dtype='<u2')
        assert len(tokens) == manifest['tokens'][split]
        assert len(rows) == manifest['documents'][split]
        assert int(tokens.max()) < enc.n_vocab
        offset = 0
        for row in rows:
            assert row['offset'] == offset
            offset += row['tokens']
            assert tokens[offset - 1] == enc.eot_token
        assert offset == len(tokens)
        lengths = [r['tokens'] for r in rows]
        counts = Counter()
        for row in rows:
            counts[row['source']] += row['tokens']
        result['splits'][split] = dict(tokens=len(tokens), documents=len(rows),
            source_tokens=dict(counts), min_doc_tokens=min(lengths),
            median_doc_tokens=statistics.median(lengths), max_doc_tokens=max(lengths),
            contiguous_offsets=True, all_document_end_markers_valid=True)
        indexes[split], arrays[split] = rows, tokens
    overlap = set(r['normalized_sha256'] for r in indexes['train']) & set(r['normalized_sha256'] for r in indexes['val'])
    assert not overlap
    result['train_validation_exact_normalized_hash_overlap'] = len(overlap)
    sample_indices = sorted(set(np.linspace(0, len(indexes['train']) - 1, 24, dtype=int).tolist() + [0,100,1000,4000,8000,9500]))
    checked = []
    with gzip.open(DATA / 'train.jsonl.gz', 'rt') as stream:
        for i, line in enumerate(stream):
            if i not in sample_indices:
                continue
            record = json.loads(line)
            assert hashlib.sha256(record['text'].encode()).hexdigest() == record['text_sha256']
            tokens = enc.encode(record['text'], disallowed_special=()) + [enc.eot_token]
            row = indexes['train'][i]
            assert len(tokens) == row['tokens']
            assert np.array_equal(tokens, arrays['train'][row['offset']:row['offset']+row['tokens']])
            checked.append(i)
    result['training_documents_retokenized_exactly'] = checked
    result['source_order'] = 'Stored as FineWeb-Edu followed by Wikipedia; training must sample/shuffle across the full stream.'
    result['limitations'] = ['No independent near-duplicate or passage-overlap rerun.',
        'No test contents read or scored; test counts are manifest metadata.',
        'Training-text spot inspection only; no comprehensive factual-quality audit.',
        'New-domain training quality, throughput and memory are unmeasured.']
    result['budget_arithmetic'] = dict(old_train_tokens=267688,
        size_ratio=len(arrays['train'])/267688,
        old_run_target_tokens=1200*1024,
        old_run_fraction_of_new_corpus=1200*1024/len(arrays['train']),
        updates_for_one_corpus_equivalent_at_1024_targets=int(np.ceil(len(arrays['train'])/1024)),
        note='Token exposure, not guaranteed coverage under sampling with replacement.')
    result['next_action'] = 'A separately bounded data-loader and throughput pilot before choosing a full training budget. Autonomous iteration remains paused.'
    with OUT.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
