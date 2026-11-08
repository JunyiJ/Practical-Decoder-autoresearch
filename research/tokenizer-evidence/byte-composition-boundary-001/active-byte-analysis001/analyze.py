"""Count active byte rows in pinned training data; no model or RNG execution."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
BASE=OUT.parent


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def rec(p):return dict(path=str(p),size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


plan=json.loads((BASE/'quality001/plan.json').read_text());expected=plan['data']
for asset in plan['codebook_assets']:
    p=Path(asset['path']);assert rec(p)['sha256']==asset['sha256'] and p.stat().st_size==asset['size']
os.environ['TIKTOKEN_CACHE_DIR']=str(Path(plan['codebook_assets'][0]['path']).parent)
import tiktoken
enc=tiktoken.get_encoding('gpt2');assert importlib.metadata.version('tiktoken')==expected['tokenizer']['version']
assert hashlib.sha256(json.dumps(sorted((raw.hex(),rank) for raw,rank in enc._mergeable_ranks.items())).encode()).hexdigest()==expected['tokenizer']['vocabulary_sha256']
p=Path(expected['path']);raw=p.read_bytes();assert rec(p)['sha256']==expected['corpus_sha256']
text=raw.decode('utf8');ids=enc.encode(text[:expected['raw_train_end']],disallowed_special=())
assert len(ids)==expected['train_tokens'] and digest(ids)==expected['train_ids_sha256']
by_id={rank:set(raw) for raw,rank in enc._mergeable_ranks.items()};by_id[50256]={258}
def active(tokens):return sorted({256,257}.union(*(by_id[t] for t in tokens)))
windows=[dict(start=i,tokens=min(1024,len(ids)-i),active_rows=len(active(ids[i:i+1024]))) for i in range(0,len(ids),1024)]
counts=[w['active_rows'] for w in windows if w['tokens']==1024]
result=dict(kind='fixed_training_byte_row_analysis',sources=[rec(Path(__file__)),rec(BASE/'quality001/plan.json'),rec(p),rec(ROOT/'research/byte_embedding_v1.py'),rec(ROOT/'research/byte_composer_v2.py')],
    train_tokens=len(ids),train_ids_sha256=digest(ids),all_training_active_rows=active(ids),prefix1024_active_rows=active(ids[:1024]),
    nonoverlapping_windows=windows,complete_window_summary=dict(count=len(counts),minimum=min(counts),median=statistics.median(counts),maximum=max(counts)),
    original_gradient_rows=259,original_position_chunk=2048,padded_width=130,
    no_model_execution=True,no_random_draws=True,validation_encoded=False,holdout_encoded=False,
    limits=['Nonoverlapping1024-token windows describe input sparsity; not expected counts under the two-block sampler.','Row-count reductions are arithmetic opportunity, not runtime or numerical-equivalence evidence.','No padded positions or chunks are removed; pad and begin both map to row256 with the existing mask.'])
with (OUT/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(training_active_rows=len(result['all_training_active_rows']),prefix_active_rows=len(result['prefix1024_active_rows']),complete_window_summary=result['complete_window_summary'])))
