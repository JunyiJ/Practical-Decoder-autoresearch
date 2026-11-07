"""Standalone native train/val T+2 data view; SOURCE ONLY until new data gates.

Metadata inspection/decoding and private generator construction are separate.
No model, old worker/validator, holdout encoder, scorer or checkpoint I/O exists.
"""
import copy
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path

POLICY='future_token_physical_train_val_sequence_view_v1'
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001/quality002-plan.json'
REFERENCE_SHA256='d15be136c3ac75a70b29ad7afa2a091d2aaf7572307aecfffb0ffc1e592a5730'
CORPUS_CAP=8*1024**2


def require(condition,message):
    if not condition:raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False)


def ids_digest(values):
    # Exact original output_bias.digest JSON representation, not tensor-byte SHA.
    require(type(values) in (list,tuple) and all(type(v) is int and 0<=v<50257 for v in values),
        'Exact native integer ID sequence required')
    return hashlib.sha256(canonical(values).encode()).hexdigest()


def reference_spec():
    """Read only the pinned original metadata, never corpus/tokenizer/RNG."""
    raw=REFERENCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==REFERENCE_SHA256,'Original physical data manifest differs')
    d=json.loads(raw)['data']
    require(d['path']==str(ROOT/'data/raw/shakespeare.txt') and d['train_fraction']==.8 and d['val_fraction']==.1,
        'Original corpus path/raw fractions required')
    require((d['raw_characters'],d['raw_train_end'],d['raw_val_end'])==(1115394,892315,1003854),
        'Original raw-text split boundaries required')
    require((d['train_tokens'],d['validation_tokens'],d['validation_targets'],d['validation_target_bytes'])==(267688,34279,34278,111536),
        'Original physical token/target/byte inventory required')
    require(d['tokenizer']['name']=='gpt2' and d['tokenizer']['vocab_size']==50257
        and d['tokenizer']['split_unit']=='raw_text' and not d['holdout_scored'] and not d['holdout_tokenized'],
        'Original ordinary GPT2 train/val-only identity required')
    windows=d['validation_window_target_counts']
    require(len(windows)==268 and windows[:-1]==[128]*267 and windows[-1]==102 and sum(windows)==34278,
        'Full physical primary validation coverage must remain independent')
    return dict(policy=POLICY,reference_manifest_sha256=REFERENCE_SHA256,
        corpus=dict(path=d['path'],sha256=d['corpus_sha256'],raw_characters=d['raw_characters'],
                    raw_train_end=d['raw_train_end'],raw_val_end=d['raw_val_end'],train_fraction=.8,val_fraction=.1),
        tokenizer=copy.deepcopy(d['tokenizer']),
        splits=dict(train=dict(tokens=d['train_tokens'],ids_sha256=d['train_ids_sha256']),
                    val=dict(tokens=d['validation_tokens'],ids_sha256=d['validation_ids_sha256'])),
        sequence_layout=dict(context=128,right_targets=2,batch=8,train_anchor_span=514,
            train_eligible_starts=267175,train_last_anchor=267174,val_eligible_starts=34150,val_last_start=34149,
            support='new_Tplus2_eligibility_for_all_fresh_arms;not_historical_stream_equivalence'),
        physical_primary_validation=dict(targets=34278,target_ids_sha256=d['validation_target_ids_sha256'],
            target_bytes=111536,context=128,window_target_counts=copy.deepcopy(windows),
            statistical_block_target_counts=copy.deepcopy(d['statistical_block_target_counts']),
            scope='identity_only_not_a_scorer;all34278targets_CPU_and_MPS_CE3e-5_required_separately'),
        holdout_encoded=False,holdout_sampled=False,holdout_scored=False)


def _read_corpus(spec):
    path=Path(spec['corpus']['path'])
    with path.open('rb') as f:raw=f.read(CORPUS_CAP+1)
    require(len(raw)<=CORPUS_CAP and hashlib.sha256(raw).hexdigest()==spec['corpus']['sha256'],
        'Bounded original corpus bytes required')
    return raw


def tokenizer_contract(enc):
    """Exact inherited codec contract; no alternate vocabulary or tokenization."""
    ranks=sorted((token.hex(),rank) for token,rank in enc._mergeable_ranks.items())
    return dict(name='gpt2',implementation='tiktoken',version=importlib.metadata.version('tiktoken'),
        vocabulary_sha256=hashlib.sha256(json.dumps(ranks).encode()).hexdigest(),pattern=enc._pat_str,
        special_tokens=copy.deepcopy(enc._special_tokens),special_token_policy='ordinary_text',
        split_unit='raw_text',vocab_size=enc.n_vocab)


@dataclass(frozen=True)
class Decoded:
    train_ids:tuple
    val_ids:tuple
    contract:dict

    def validate(self):
        require(type(self) is Decoded and self.contract==reference_spec(),'Own exact physical decoded identity required')
        for name,values in (('train',self.train_ids),('val',self.val_ids)):
            wanted=self.contract['splits'][name]
            require(type(values) is tuple and len(values)==wanted['tokens'] and ids_digest(values)==wanted['ids_sha256'],
                'Own physical token IDs differ: '+name)
        require(ids_digest(self.val_ids[1:])==self.contract['physical_primary_validation']['target_ids_sha256'],
            'Full original next-token validation targets differ')
        layout=self.contract['sequence_layout']
        require(len(self.train_ids)-513==layout['train_eligible_starts']
            and layout['train_last_anchor']+513==len(self.train_ids)-1
            and len(self.val_ids)-129==layout['val_eligible_starts']
            and layout['val_last_start']+129==len(self.val_ids)-1,'Own physical right-two eligibility differs')


def decode():
    """Only after a NEW physical-data runtime admission; no generator constructed.

    The complete raw file is identity-checked, but ONLY its original train and
    validation substrings are passed to BPE. No holdout substring is encoded.
    """
    spec=reference_spec();raw=_read_corpus(spec);text=raw.decode('utf8')
    require(len(text)==spec['corpus']['raw_characters'],'Original UTF8 character inventory required')
    a=int(.8*len(text));b=int((.8+.1)*len(text))
    require((a,b)==(spec['corpus']['raw_train_end'],spec['corpus']['raw_val_end']),
        'Split raw text BEFORE independent train/val encoding')
    import tiktoken
    enc=tiktoken.get_encoding('gpt2')
    require(tokenizer_contract(enc)==spec['tokenizer'],'Original installed codec/vocabulary contract differs')
    train=enc.encode(text[:a],disallowed_special=())
    val=enc.encode(text[a:b],disallowed_special=())
    require(_read_corpus(spec)==raw,'Corpus changed during independent train/val encoding')
    result=Decoded(tuple(train),tuple(val),copy.deepcopy(spec))
    result.validate()
    return result


class View:
    """Own CPU sequence/position interface; not the historical (x,y) batch API."""
    def __init__(self,decoded,*,train_seed,eval_seed):
        # Explicit role preflight precedes importing/constructing private streams.
        require(type(train_seed) is int and type(eval_seed) is int and 0<=train_seed<2**63
            and 0<=eval_seed<2**63 and train_seed!=eval_seed,'Two distinct newly audited private sampler roles required')
        require(type(decoded) is Decoded,'Own separately qualified physical data required')
        decoded.validate()
        import torch
        from research.future_token_sampling_v1 import Sampler
        self.decoded=decoded
        train=torch.tensor(decoded.train_ids,dtype=torch.int64,device='cpu')
        val=torch.tensor(decoded.val_ids,dtype=torch.int64,device='cpu')
        self.sampler=Sampler(train,val,train_seed=train_seed,eval_seed=eval_seed)
        self.signature=dict(policy=POLICY,physical_identity=hashlib.sha256(canonical(decoded.contract).encode()).hexdigest(),
            sampler=copy.deepcopy(self.sampler.signature),batch_interface='explicit_sequence8x130_plus_positions_CPU')
        self._signature_json=canonical(self.signature)
        self.validate()

    def validate(self):
        from research.future_token_sampling_v1 import Sampler
        require(type(self.decoded) is Decoded and type(self.sampler) is Sampler,'Own physical data/private sampler required')
        self.decoded.validate();self.sampler.validate()
        require(canonical(self.signature)==self._signature_json and self.signature['policy']==POLICY
            and self.signature['physical_identity']==hashlib.sha256(canonical(self.decoded.contract).encode()).hexdigest()
            and self.signature['sampler']==self.sampler.signature,'Own physical view/sampler signature differs')
        for name in ('train','val'):
            require(ids_digest(self.sampler.data[name].tolist())==self.decoded.contract['splits'][name]['ids_sha256'],
                'Private sampler must still own the original physical IDs: '+name)

    def next_sequence(self,split='train'):
        require(type(split) is str and split in ('train','val'),'No holdout or foreign physical stream exists')
        self.validate()
        out=self.sampler.next(split)
        return dict(policy=POLICY,physical_identity=self.signature['physical_identity'],sampler_policy=out['policy'],
            **{k:v for k,v in out.items() if k!='policy'})

    def state_dict(self):
        self.validate()
        return dict(kind='future_token_physical_sequence_view_state_v1',signature=copy.deepcopy(self.signature),
            sampler_state=self.sampler.state_dict())

    def load_state_dict(self,state):
        self.validate()
        require(type(state) is dict and set(state)=={'kind','signature','sampler_state'}
            and state['kind']=='future_token_physical_sequence_view_state_v1' and state['signature']==self.signature,
            'Exact own physical sequence-view state required')
        # Sampler preflights/stages BOTH streams before destination mutation.
        self.sampler.load_state_dict(state['sampler_state'])
