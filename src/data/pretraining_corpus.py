"""Disk-backed, document-bounded batches for the new corpus, not frozen runs."""
import json
from pathlib import Path

import numpy as np
import torch


class PretrainingCorpus:
    def __init__(self, directory, batch_size=8, block_size=128, device='cpu', seed=0):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        if self.manifest['status'] != 'complete' or self.manifest['format'] != 'rsi-pretraining-corpus-v1':
            raise ValueError('A completed corpus is required')
        if batch_size < 1 or block_size < 1:
            raise ValueError('Positive batch and context sizes required')
        self.batch_size, self.block_size, self.device = batch_size, block_size, device
        self.vocab_size = self.manifest['tokenizer']['vocab_size']
        self.data, self.starts, self.cumulative, self.generators = {}, {}, {}, {}
        # The test split is intentionally not opened by the training interface.
        for i, split in enumerate(('train', 'val')):
            path = self.directory / f'{split}.bin'
            if path.stat().st_size != 2*self.manifest['tokens'][split]:
                raise ValueError('Token file size differs from manifest')
            self.data[split] = np.memmap(path, mode='r', dtype='<u2')
            docs = [json.loads(line) for line in (self.directory / f'{split}.index.jsonl').read_text().splitlines()]
            starts, sizes = [], []
            for doc in docs:
                if doc['tokens'] > block_size:
                    starts.append(doc['offset'])
                    sizes.append(doc['tokens'] - block_size)
            if not sizes:
                raise ValueError(f'No eligible documents in {split}')
            self.starts[split] = np.asarray(starts, dtype=np.int64)
            self.cumulative[split] = np.cumsum(sizes, dtype=np.int64)
            self.generators[split] = np.random.default_rng(seed+i)

    def get_batch(self, split='train'):
        if split not in self.data:
            raise ValueError('Only train and val are available; test is sealed for final evaluation')
        cumulative = self.cumulative[split]
        draws = self.generators[split].integers(0, cumulative[-1], size=self.batch_size)
        docs = np.searchsorted(cumulative, draws, side='right')
        before = np.where(docs == 0, 0, cumulative[np.maximum(docs-1, 0)])
        starts = self.starts[split][docs] + draws - before
        # Sampling is uniform over valid windows, and never joins two documents.
        array = np.stack([self.data[split][s:s+self.block_size+1] for s in starts]).astype(np.int64)
        batch = torch.from_numpy(array)
        return batch[:, :-1].to(self.device), batch[:, 1:].to(self.device)
