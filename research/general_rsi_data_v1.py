"""Pinned general corpus; no access path to the test split."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from research.block_norm_v1.runtime.data.loaders import _get_batch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/raw/rsi-general-10m-v1'
MANIFEST_SHA = 'b1849a5fed1fb648f507e407e5794946bba64328f519e6b0d6c360fa4e2e4fa8'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GeneralData:
    def __init__(self, train_seed, validation_seed, device):
        assert device in ('cpu', 'mps') and train_seed != validation_seed
        assert sha(DATA / 'manifest.json') == MANIFEST_SHA
        self.manifest = json.loads((DATA / 'manifest.json').read_text())
        self.indexes, arrays = {}, {}
        for split in ('train', 'val'):
            for suffix in ('bin', 'index.jsonl'):
                name = f'{split}.{suffix}'
                rec = self.manifest['files'][name]
                assert (DATA / name).stat().st_size == rec['bytes']
                assert sha(DATA / name) == rec['sha256']
            values = np.fromfile(DATA / f'{split}.bin', dtype='<u2').astype(np.int64)
            assert len(values) == self.manifest['tokens'][split] and int(values.max()) < 50257
            rows = [json.loads(line) for line in (DATA / f'{split}.index.jsonl').read_text().splitlines()]
            offset = 0
            for row in rows:
                assert row['offset'] == offset and row['tokens'] > 1
                offset += row['tokens']
                assert values[offset - 1] == 50256
            assert offset == len(values)
            arrays[split] = torch.from_numpy(values)
            self.indexes[split] = rows
        self.view = SimpleNamespace(train_data=arrays['train'], val_data=arrays['val'],
            splits=arrays, generators={'train': torch.Generator().manual_seed(train_seed),
                                      'val': torch.Generator().manual_seed(validation_seed)},
            batch_size=8, block_size=128, batch_layout='paired_blocks512', device=device)

    def batch(self):
        return _get_batch(self.view, 'train')

    def validation_chunks(self):
        """Each within-document next-token target once, terminal EOT included."""
        values = self.view.val_data
        for doc, row in enumerate(self.indexes['val']):
            start, size = row['offset'], row['tokens']
            for offset in range(0, size - 1, 128):
                count = min(128, size - 1 - offset)
                x = values[start+offset:start+offset+count][None, :]
                y = values[start+offset+1:start+offset+count+1][None, :]
                yield doc, row['source'], x, y
