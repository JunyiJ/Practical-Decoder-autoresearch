"""Read-only endpoint table geometry, not inference or a new quality experiment."""
import collections
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/tokenizer-evidence/shared-table-geometry-001'
PANEL = ROOT/'research/tokenizer-evidence/output-anchored-tying-001'
LABELS = ('0','1-9','10-99','100-999','1000+')


def read(path): return json.loads(path.read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2): h.update(block)
    return h.hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def bin_for(count):
    return '0' if count == 0 else '1-9' if count < 10 else '10-99' if count < 100 else '100-999' if count < 1000 else '1000+'
def put(path, value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    assert len(raw) <= 4*1024**2
    with path.open('xb') as stream: stream.write(raw);stream.flush();os.fsync(stream.fileno())


def run(plan):
    import numpy as np
    import torch
    import tiktoken
    torch.set_num_threads(4)
    rng = torch.get_rng_state().clone()
    for path, expected in plan['sources'].items(): assert sha(ROOT/path) == expected
    assert sha(PANEL/'final-review.json') == plan['review_sha256']
    review = read(PANEL/'final-review.json'); assert review['accepted']
    expected = read(PANEL/'00/config.json')['data']
    assert sha(PANEL/'00/config.json') == plan['config_sha256']
    assert importlib.metadata.version('tiktoken') == expected['tokenizer']['version']
    encoding = tiktoken.get_encoding('gpt2')
    assert hashlib.sha256(json.dumps(sorted((t.hex(),r) for t,r in encoding._mergeable_ranks.items())).encode()).hexdigest() == expected['tokenizer']['vocabulary_sha256']
    corpus=Path(expected['path']); assert sha(corpus) == expected['corpus_sha256']
    text=corpus.read_text(); assert len(text) == expected['raw_characters']
    train=encoding.encode(text[:expected['raw_train_end']],disallowed_special=())
    validation=encoding.encode(text[expected['raw_train_end']:expected['raw_val_end']],disallowed_special=())
    assert digest(train)==expected['train_ids_sha256'] and digest(validation)==expected['validation_ids_sha256']
    assert digest(validation[1:])==expected['validation_target_ids_sha256'] and len(validation)-1==34278
    frequencies=collections.Counter(train); targets=collections.Counter(validation[1:])
    groups=np.array([LABELS.index(bin_for(frequencies[i])) for i in range(50257)])
    occurrence_weights=np.array([targets[i] for i in range(50257)],dtype=np.int64)
    results=[]
    for pair in plan['pairs']:
        saved=[]
        for arm in ('matched','tied'):
            record=pair[arm];path=Path(record['path'])
            assert path.is_file() and not path.is_symlink() and path.stat().st_nlink==1
            assert path.stat().st_size==record['size'] and sha(path)==record['sha256']
            payload=torch.load(path,map_location='cpu',weights_only=True)
            assert payload['kind']=='h200_inference_weights' and payload['spec']['arm']==arm
            assert payload['spec']['roles']['native_initializer']==pair['initializer']
            assert payload['plan_sha256']==plan['panel_plan_sha256']
            saved.append(payload['weights'])
        q=saved[0]['transformer.wte.weight'];w=saved[0]['lm_head.weight'];s=saved[1]['transformer.wte.weight']
        assert 'lm_head.weight' not in saved[1]
        assert all(t.shape==(50257,256) and t.dtype==torch.float32 and not t.requires_grad and torch.isfinite(t).all() for t in (q,w,s))
        metrics=collections.defaultdict(list)
        # Common raw coordinates: matched Q and W and shared S all began at native W.
        # Both input paths divide their stored table by k; output does not.
        with torch.inference_mode():
            for start in range(0,50257,1024):
                a,b,c=[t[start:start+1024].to(torch.float64) for t in (q,w,s)]
                na,nb,nc=[torch.linalg.vector_norm(t,dim=1) for t in (a,b,c)]
                assert all(bool((n>0).all()) for n in (na,nb,nc))
                values={
                    'cos_matched_input_output':(a*b).sum(1)/(na*nb),
                    'cos_shared_matched_input':(c*a).sum(1)/(nc*na),
                    'cos_shared_matched_output':(c*b).sum(1)/(nc*nb),
                    'shared_input_norm_ratio':nc/na,
                    'shared_output_norm_ratio':nc/nb,
                    'matched_effective_input_norm':na*math.sqrt(192),
                    'matched_output_norm':nb,
                    'shared_effective_input_norm':nc*math.sqrt(192),
                    'shared_output_norm':nc,
                    'matched_role_relative_distance':torch.linalg.vector_norm(a-b,dim=1)/nb,
                    'shared_input_relative_distance':torch.linalg.vector_norm(c-a,dim=1)/na,
                    'shared_output_relative_distance':torch.linalg.vector_norm(c-b,dim=1)/nb,
                }
                for name,value in values.items():
                    assert bool(torch.isfinite(value).all())
                    if name.startswith('cos_'): assert bool((value.abs()<=1+1e-12).all())
                    metrics[name].append(value.numpy())
        metrics={name:np.concatenate(values) for name,values in metrics.items()}
        rows=[]
        for index,label in enumerate(LABELS):
            mask=groups==index;weights=occurrence_weights[mask];assert mask.any() and weights.sum()>0
            row=dict(bin=label,vocabulary_types=int(mask.sum()),validation_types=int((weights>0).sum()),validation_targets=int(weights.sum()),metrics={})
            for name,values in metrics.items():
                subset=values[mask]
                row['metrics'][name]=dict(type_mean=float(subset.mean()),type_median=float(np.median(subset)),
                    type_p10=float(np.quantile(subset,.1)),type_p90=float(np.quantile(subset,.9)),
                    validation_target_weighted_mean=float((subset*weights).sum()/weights.sum()))
            rows.append(row)
        assert sum(r['vocabulary_types'] for r in rows)==50257 and sum(r['validation_targets'] for r in rows)==34278
        for name,values in metrics.items():
            weighted=sum(r['metrics'][name]['validation_target_weighted_mean']*r['validation_targets'] for r in rows)/34278
            assert abs(weighted-float((values*occurrence_weights).sum()/34278))<1e-11
        results.append(dict(initializer=pair['initializer'],bins=rows))
        for arm in ('matched','tied'): assert sha(Path(pair[arm]['path']))==pair[arm]['sha256']
        del saved,payload,q,w,s,metrics
    assert torch.equal(rng,torch.get_rng_state()) and sha(corpus)==expected['corpus_sha256']
    for path, expected in plan['sources'].items(): assert sha(ROOT/path)==expected
    return dict(accepted=True,plan_sha256=sha(OUT/'plan.json'),pairs=results,
        no_model_construction=True,no_training=True,no_inference=True,no_new_losses=True,
        no_initializer_replay=True,cpu_rng_unchanged=True,holdout_encoded=False,
        scope=plan['scope'],limitations=plan['limitations'])


if __name__=='__main__':
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(OUT/'plan.json')
        assert not (OUT/'guard.json').exists()
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=2*1024**3):
                result=run(plan)
            put(OUT/'result.json',result)
        except BaseException:
            put(OUT/'failure.json',dict(traceback=traceback.format_exc(),original_attempt_closed=True));raise
