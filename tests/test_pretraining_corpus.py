import json

import numpy as np
import pytest
import torch

from research.build_pretraining_corpus_v1 import (
    CrossSplitOverlap,
    Deduper,
    canonical_url,
    fingerprint,
    quality_reason,
    split_for,
)
from src.data.pretraining_corpus import PretrainingCorpus


def test_url_variants_stay_in_same_split():
    a = 'http://www.example.com/article/?tracking=x#section'
    b = 'https://example.com/article'
    assert canonical_url(a) == b
    assert split_for(a) == split_for(b)


def test_duplicate_matching_and_repetition_filter():
    a, sim = fingerprint('The river flows through the green valley. The hills are steep.')
    b, _ = fingerprint('THE RIVER FLOWS THROUGH THE GREEN VALLEY.\nThe hills are steep.')
    assert a == b
    d = Deduper()
    d.add('https://example.com/a', a, sim, 1000)
    assert d.duplicate('https://mirror.com/a', b, sim, 1000) == 'duplicate_text'
    assert d.duplicate('https://example.com/a?x=1', 'different', sim ^ 255, 1000) == 'duplicate_url'
    assert d.duplicate('https://other.com', 'different', sim ^ 1, 999) == 'near_duplicate'
    assert quality_reason('word word word '*100) == 'low_word_diversity'


def test_passage_overlap_detects_shifted_copy_across_splits():
    overlap = CrossSplitOverlap()
    ids = list(range(200))
    overlap.add('train', overlap.hashes(ids))
    assert overlap.duplicate('val', overlap.hashes([600]*17 + ids[25:125] + [700]*17))
    assert not overlap.duplicate('train', overlap.hashes(ids))
    assert not overlap.duplicate('test', overlap.hashes(list(range(1000, 1200))))
    reverse = CrossSplitOverlap()
    reverse.add('val', reverse.hashes([600]*17 + ids[32:96] + [700]*17))
    assert reverse.duplicate('train', reverse.hashes(ids))


def fixture_corpus(path):
    manifest = {'format': 'rsi-pretraining-corpus-v1', 'status': 'complete',
                    'tokenizer': {'vocab_size': 50257}, 'tokens': {}}
    for split in ('train', 'val'):
        docs = [np.arange(10, 20, dtype='<u2'), np.arange(100, 115, dtype='<u2')]
        np.concatenate(docs).tofile(path/f'{split}.bin')
        entries = [{'offset': 0, 'tokens': 10}, {'offset': 10, 'tokens': 15}]
        (path/f'{split}.index.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in entries))
        manifest['tokens'][split] = 25
    (path/'manifest.json').write_text(json.dumps(manifest))


def test_batches_never_cross_documents_and_val_has_private_rng(tmp_path):
    fixture_corpus(tmp_path)
    loader = PretrainingCorpus(tmp_path, batch_size=1000, block_size=4, seed=21)
    twin = PretrainingCorpus(tmp_path, batch_size=1000, block_size=4, seed=21)
    loader.get_batch('val')
    x, y = loader.get_batch('train')
    tx, ty = twin.get_batch('train')
    assert torch.equal(x, tx) and torch.equal(y, ty)
    assert torch.all(y-x == 1)  # Crossing the fixture's 19 -> 100 gap fails.
    assert x.min() == 10 and y.max() == 114  # First and last eligible windows.
    # Uniform valid windows: second document has 11 of 17 possible starts.
    assert .59 < float((x[:, 0] >= 100).float().mean()) < .70
    with pytest.raises(ValueError, match='test'):
        loader.get_batch('test')


def test_truncated_payload_and_short_documents_rejected(tmp_path):
    fixture_corpus(tmp_path)
    with pytest.raises(ValueError, match='eligible'):
        PretrainingCorpus(tmp_path, block_size=20)
    (tmp_path/'train.bin').write_bytes(b'\x01\x00')
    with pytest.raises(ValueError, match='size'):
        PretrainingCorpus(tmp_path)
