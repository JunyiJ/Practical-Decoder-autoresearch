"""Train-only paired512 draws, identical flattened targets for both lengths."""
import json
from types import SimpleNamespace
import numpy as np
import torch
from research.general_rsi_data_v1 import DATA,MANIFEST_SHA,sha
from research.block_norm_v1.runtime.data.loaders import _get_batch
class TrainData:
    def __init__(self,seed,context,device='mps'):
        assert type(context) is int and context in (128,256)
        assert sha(DATA/'manifest.json')==MANIFEST_SHA
        manifest=json.loads((DATA/'manifest.json').read_text());rec=manifest['files']['train.bin']
        assert (DATA/'train.bin').stat().st_size==rec['bytes'] and sha(DATA/'train.bin')==rec['sha256']
        values=torch.from_numpy(np.fromfile(DATA/'train.bin',dtype='<u2').astype(np.int64))
        assert len(values)==manifest['tokens']['train'] and int(values.max())<50257
        self.view=SimpleNamespace(train_data=values,splits={'train':values},generators={'train':torch.Generator().manual_seed(seed)},batch_size=1024//context,block_size=context,batch_layout='paired_blocks512',device=device)
    def batch(self): return _get_batch(self.view,'train')
