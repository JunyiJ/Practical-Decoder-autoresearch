import math
import hashlib
import importlib.metadata
import json

import tiktoken
import torch
from torch.utils.data import Dataset


def _configure_splits(loader, data, train_fraction, val_fraction, seed, eval_seed):
    train_fraction, val_fraction = float(train_fraction), float(val_fraction)
    total = train_fraction + val_fraction
    if (not math.isfinite(total) or train_fraction <= 0 or val_fraction <= 0
            or total > 1.0 + 1e-12):
        raise ValueError("train_fraction and val_fraction must be positive and sum to at most 1")
    train_end = int(train_fraction * len(data))
    # Preserve the legacy 90/10 split, including its rounding behavior.
    val_end = len(data) if math.isclose(total, 1.0, abs_tol=1e-12, rel_tol=0) else int(total * len(data))
    loader.train_fraction = train_fraction
    loader.val_fraction = val_fraction
    loader.train_data = data[:train_end]
    loader.val_data = data[train_end:val_end]
    loader.splits = {'train': loader.train_data, 'val': loader.val_data}
    loader.eval_seed = seed + 1 if eval_seed is None else int(eval_seed)
    loader.generators = {
        'train': torch.Generator().manual_seed(seed),
        'val': torch.Generator().manual_seed(loader.eval_seed),
    }
    if val_end < len(data):
        loader.test_data = data[val_end:]
        loader.splits['test'] = loader.test_data
        loader.generators['test'] = torch.Generator().manual_seed(loader.eval_seed + 1)


def _get_batch(loader, split):
    if split not in loader.splits:
        raise ValueError(f"split must be one of {tuple(loader.splits)}")
    data = loader.splits[split]
    if split == 'train' and getattr(loader, 'batch_layout', 'independent') in {'paired_blocks512', 'paired_batch512'}:
        max_start = len(data) - 512
        if max_start <= 0:
            raise ValueError('Training split too small for paired 512-token blocks')
        # Fixed two-block draws preserve the token stream across batch grouping.
        groups = loader.batch_size * loader.block_size // 1024
        starts = torch.cat([torch.randint(max_start, (2,), generator=loader.generators[split])
                            for _ in range(groups)])
        x = torch.stack([data[i:i+512] for i in starts]).reshape(loader.batch_size, loader.block_size)
        y = torch.stack([data[i+1:i+513] for i in starts]).reshape(loader.batch_size, loader.block_size)
        return x.to(loader.device), y.to(loader.device)
    max_start = len(data) - loader.block_size
    if max_start <= 0:
        raise ValueError(f"Dataset split {split!r} is too small for the configured block_size")
    ix = torch.randint(max_start, (loader.batch_size,), generator=loader.generators[split])
    x = torch.stack([data[i:i+loader.block_size] for i in ix])
    y = torch.stack([data[i+1:i+loader.block_size+1] for i in ix])
    return x.to(loader.device), y.to(loader.device)


class TinyDataLoader:
    def __init__(self, data_path, batch_size, block_size, device, seed=0,
                 train_fraction=.9, val_fraction=.1, eval_seed=None, batch_layout='independent'):
        if batch_layout not in {'independent', 'paired_blocks512', 'paired_batch512'}:
            raise ValueError('Unknown batch_layout')
        if batch_layout == 'paired_blocks512' and (block_size not in {128,256,512} or batch_size*block_size != 1024):
            raise ValueError('paired_blocks512 requires context128/256/512 and1024 targets/update')
        if batch_layout == 'paired_batch512' and (block_size != 128 or batch_size not in {8,16,32}):
            raise ValueError('paired_batch512 requires context128 and batch8/16/32')
        self.batch_layout = batch_layout
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch:i for i, ch in enumerate(chars)}
        self.itos = {i:ch for i, ch in enumerate(chars)}

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        _configure_splits(self, data, train_fraction, val_fraction, seed, eval_seed)

    def get_batch(self, split='train'):
        return _get_batch(self, split)

    def decode(self, ids):
        return "".join([self.itos[int(i)] for i in ids])

    def encode(self, text):
        return [self.stoi[c] for c in text]

class TextDataset(Dataset):
    def __init__(self, data_path, batch_size, block_size, device, seed=0,
                 train_fraction=.9, val_fraction=.1, eval_seed=None,
                 batch_layout='independent', split_unit='tokens'):
        if batch_layout not in {'independent', 'paired_blocks512', 'paired_batch512'}:
            raise ValueError('Unknown batch_layout')
        if batch_layout == 'paired_blocks512' and (block_size not in {128,256,512} or batch_size*block_size != 1024):
            raise ValueError('paired_blocks512 requires context128/256/512 and1024 targets/update')
        if batch_layout == 'paired_batch512' and (block_size != 128 or batch_size not in {8,16,32}):
            raise ValueError('paired_batch512 requires context128 and batch8/16/32')
        if split_unit not in {'tokens', 'raw_text'}:
            raise ValueError('Unknown split_unit')
        self.batch_layout = batch_layout
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.enc = tiktoken.get_encoding("gpt2")
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        if split_unit == 'tokens':  # Preserve the old standalone loader API.
            self.data = torch.tensor(self.enc.encode(text), dtype=torch.long)
            _configure_splits(self, self.data, train_fraction, val_fraction, seed, eval_seed)
        else:
            # Freeze raw-text boundaries before BPE, so no merge crosses splits.
            _configure_splits(self, text, train_fraction, val_fraction, seed, eval_seed)
            self.split_texts = dict(self.splits)
            self.splits = {k: torch.tensor(self.encode(v), dtype=torch.long)
                           for k, v in self.split_texts.items()}
            self.train_data, self.val_data = self.splits['train'], self.splits['val']
            if 'test' in self.splits:
                self.test_data = self.splits['test']
            self.data = torch.cat(list(self.splits.values()))
        # Bind checkpoints to the actual vocabulary, regex, implementation and splits.
        # Old test doubles have no vocabulary internals; factory runs always use tiktoken.
        if hasattr(self.enc, '_mergeable_ranks'):
            ranks = [(token.hex(), rank) for token, rank in self.enc._mergeable_ranks.items()]
            spec = dict(name='gpt2', implementation='tiktoken',
                        version=importlib.metadata.version('tiktoken'),
                        vocabulary_sha256=hashlib.sha256(json.dumps(sorted(ranks)).encode()).hexdigest(),
                        pattern=self.enc._pat_str, special_tokens=self.enc._special_tokens,
                        special_token_policy='ordinary_text', split_unit=split_unit,
                        vocab_size=self.vocab_size)
            self.tokenizer_contract = spec

    def encode(self, text):
        return self.enc.encode(text, disallowed_special=())

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x.to(self.device), y.to(self.device)

    @property
    def vocab_size(self):
        return self.enc.n_vocab

    def get_batch(self, split="train"):
        return _get_batch(self, split)

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.enc.decode(ids)
