"""Training-only punctuation-bounded spans; not a grammatical sentence parser."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/raw/rsi-general-10m-v1'
OUT=ROOT/'research/tokenizer-evidence/general-rsi-pseudo-spans-001'
KIND='punctuation_bounded_span_sampler_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory():
    import fcntl
    import shutil
    import tiktoken
    import time
    start=time.monotonic()
    assert not OUT.exists()
    assert shutil.disk_usage(ROOT).free>=10*2**30+64*2**20
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        OUT.mkdir(parents=True)
        from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
        with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=180,rss_limit_bytes=2*2**30,reserve_bytes=10*2**30):
            manifest=json.loads((DATA/'manifest.json').read_text());inputs={}
            for name in ['train.bin','train.index.jsonl']:
                inputs[name]=sha(DATA/name);assert inputs[name]==manifest['files'][name]['sha256']
            values=np.fromfile(DATA/'train.bin',dtype='<u2')
            docs=[json.loads(l) for l in (DATA/'train.index.jsonl').read_text().splitlines()]
            enc=tiktoken.get_encoding('gpt2')
            pieces=[enc.decode_single_token_bytes(i) for i in range(50257)]
            terminal=np.array([p.rstrip(b'\"\')]}').endswith((b'.',b'!',b'?')) for p in pieces])
            whitespace=np.array([bool(p) and p[0] in b' \t\r\n' for p in pieces])
            rows=[];counts={};rejected={'first_span_no_preceding_context':0,'too_short':0,'too_long':0,'unterminated_tail':0}
            for doc,d in enumerate(docs):
                n=d['tokens']-1;o=d['offset'];assert values[o+n]==50256
                seq=values[o:o+n];ends=(np.flatnonzero(terminal[seq[:-1]] & whitespace[seq[1:]])+1).tolist()
                if terminal[int(seq[-1])]:ends.append(n)
                else:rejected['unterminated_tail']+=1
                begin=0
                for end in ends:
                    size=end-begin
                    if begin==0:rejected['first_span_no_preceding_context']+=1
                    elif size<4:rejected['too_short']+=1
                    elif size>36:rejected['too_long']+=1
                    else:
                        prefix=min(16,begin);assert prefix+3*size<=129
                        rows.append([doc,o+begin,size,prefix]);counts[d['source']]=counts.get(d['source'],0)+1
                    begin=end
            array=np.asarray(rows,dtype='<u4');assert array.ndim==2 and array.shape[1]==4 and len(array)>0
            with (OUT/'inventory.npz').open('xb') as f:np.savez_compressed(f,spans=array)
            result=dict(kind=KIND,source_sha256=sha(__file__),inputs=inputs,inventory_sha256=sha(OUT/'inventory.npz'),
                columns=['document_index','absolute_start','span_tokens','preceding_context_tokens'],
                segmentation='Adjacent token-aligned endings in .!? after stripping closing quote/bracket bytes; following token starts ASCII whitespace or document ends. Discard unterminated tails and initial span without preceding context. Heuristic spans, not guaranteed sentences.',
                eligibility='4..36 tokens inclusive;1..16 preceding tokens from same document; no EOT; at least3 complete copies in129 tokens.',
                sampling='Uniform over eligible spans with replacement using dedicated CPU torch.Generator; no source reweighting. Ordinary-CE sampler independent.',
                packing='Prefix plus floor((129-prefix)/span_length) complete copies; EOT padding to129. x=first128,y=last128. Penalize aligned targets in copies2..K only; prefix, first copy and padding masked.',
                corpus_documents=len(docs),eligible_spans=len(array),eligible_documents=len(set(array[:,0].tolist())),source_counts=counts,
                length_quantiles=np.quantile(array[:,2],[0,.25,.5,.75,1]).tolist(),rejected=rejected,
                test_opened=False,validation_opened=False,model_scores=0,seconds=time.monotonic()-start)
            with (OUT/'inventory.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
            print(json.dumps(result,indent=2))


class Sampler:
    def __init__(self,seed):
        import torch
        self.manifest=json.loads((OUT/'inventory.json').read_text());self.binding=sha(OUT/'inventory.json')
        assert self.manifest['source_sha256']==sha(__file__)
        for name,h in self.manifest['inputs'].items():assert sha(DATA/name)==h
        assert sha(OUT/'inventory.npz')==self.manifest['inventory_sha256']
        self.values=np.fromfile(DATA/'train.bin',dtype='<u2');self.spans=np.load(OUT/'inventory.npz')['spans']
        self.generator=torch.Generator().manual_seed(seed);self.seed=seed;self.draws=0

    def batch(self,device='cpu'):
        import torch
        indices=torch.randint(len(self.spans),(8,),generator=self.generator).tolist();self.draws+=8
        xs=[];ys=[];previous=[];masks=[];details=[]
        for index in indices:
            doc,start,length,prefix=map(int,self.spans[index]);span=self.values[start:start+length].tolist()
            context=self.values[start-prefix:start].tolist();copies=(129-prefix)//length
            sequence=context+span*copies;valid_end=len(sequence)
            assert copies>=3 and 50256 not in sequence
            sequence += [50256]*(129-len(sequence))
            mask=[prefix+length<=p<valid_end for p in range(1,129)]
            prev=[p-length-1 if keep else 0 for p,keep in zip(range(1,129),mask)]
            x=sequence[:-1];y=sequence[1:]
            assert all(not keep or (j<k and y[j]==y[k]) for k,(j,keep) in enumerate(zip(prev,mask)))
            xs.append(x);ys.append(y);previous.append(prev);masks.append(mask)
            details.append(dict(inventory_index=index,document=doc,length=length,prefix=prefix,copies=copies,penalty_targets=sum(mask),padding=129-valid_end))
        return (torch.tensor(xs,dtype=torch.long,device=device),torch.tensor(ys,dtype=torch.long,device=device),
                torch.tensor(previous,dtype=torch.long,device=device),torch.tensor(masks,dtype=torch.bool,device=device),details)

    def state_dict(self):
        return dict(kind=KIND,binding=self.binding,seed=self.seed,draws=self.draws,rng=self.generator.get_state().clone())

    def load_state_dict(self,state):
        assert state['kind']==KIND and state['binding']==self.binding and state['seed']==self.seed
        assert type(state['draws']) is int and state['draws']>=0 and state['draws']%8==0
        self.generator.set_state(state['rng']);self.draws=state['draws']


def repetition_loss(logits,targets,previous,mask):
    """Batch mean over eligible targets. Previous-copy anchors have no gradient."""
    import torch
    assert logits.ndim==3 and logits.dtype in (torch.float32,torch.float64)
    assert targets.shape==previous.shape==mask.shape==logits.shape[:2]
    assert targets.dtype==previous.dtype==torch.long and mask.dtype==torch.bool
    assert bool(mask.any()) and bool(torch.isfinite(logits).all())
    assert bool((targets>=0).all()) and bool((targets<logits.shape[-1]).all())
    assert bool((previous>=0).all()) and bool((previous<targets.shape[1]).all())
    positions=torch.arange(targets.shape[1],device=targets.device)[None,:]
    assert bool((previous[mask]<positions.expand_as(targets)[mask]).all())
    assert torch.equal(targets.gather(1,previous)[mask],targets[mask])
    probabilities=logits.log_softmax(-1).gather(-1,targets[...,None]).squeeze(-1).exp()
    anchor=probabilities.gather(1,previous).detach()
    difference=(probabilities[mask]-.5*anchor[mask]).abs()
    capped=difference.clamp(max=1-torch.finfo(logits.dtype).eps)
    return -torch.log1p(-capped).mean(),dict(targets=int(mask.sum()),saturated=int((difference>1-torch.finfo(logits.dtype).eps).sum()))


if __name__=='__main__':inventory()
