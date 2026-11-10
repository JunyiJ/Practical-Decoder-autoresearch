"""Static sufficient-condition collisions of the current max-pool byte composer."""
import collections
import datetime
import hashlib
import time
import numpy as np
from research.general_rsi_transfer_data_v1 import ROOT,P,D,read,put,sha
from research.byte_input_model_v3.codebook import build_codebook

OUT=ROOT/'research/tokenizer-evidence/general-rsi-transfer-diagnosis-001'

def main():
    start=time.monotonic();plan=read(P/'protocol.json');assets=plan['inference']['codebook_assets']
    for a in assets:assert sha(a['path'])==a['sha256']
    book=build_codebook(*(open(a['path'],'rb').read() for a in assets),'raw_bytes')
    codes=np.frombuffer(book.codes,dtype='<u2').reshape(50257,130);lengths=np.frombuffer(book.lengths,dtype='<u2')
    # Read and bind the actual widths without importing a model or running torch.
    import ast
    source=ROOT/'research/byte_composer_v2.py';tree=ast.parse(source.read_text())
    widths=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WIDTHS' for t in n.targets))
    assert widths==(1,2,3,4,5,6)
    groups=collections.defaultdict(list)
    for token in range(50257):
        seq=tuple(map(int,codes[token,:int(lengths[token])]))
        fingerprint=tuple(tuple(sorted({seq[j:j+w] for j in range(max(0,len(seq)-w+1))})) for w in widths)
        groups[fingerprint].append(token)
    collisions=sorted([g for g in groups.values() if len(g)>1],key=lambda g:g[0]);union={t for g in collisions for t in g}
    train=ROOT/'data/raw/rsi-general-10m-v1/train.bin';manifest=read(train.parent/'manifest.json');assert sha(train)==manifest['files']['train.bin']['sha256']
    train_values=np.fromfile(train,dtype='<u2');train_counts=np.bincount(train_values,minlength=50257)
    dataset=read(P/'dataset.json');assert sha(D/'excerpts.bin')==dataset['payloads'][0]['sha256'];values=np.fromfile(D/'excerpts.bin',dtype='<u2')
    counts={};totals={}
    for domain in plan['domains']:
        inp=np.concatenate([values[r['offset']:r['offset']+r['tokens']-1] for r in dataset['documents'] if r['source']==domain]);counts[domain]=np.bincount(inp,minlength=50257);totals[domain]=len(inp)
    rows=[]
    for g in collisions:
        rows.append(dict(tokens=[dict(token=t,raw_bytes_hex=bytes(codes[t,1:int(lengths[t])-1]).hex(),byte_length=int(lengths[t])-2,training_stream_occurrences=int(train_counts[t]),transfer_scored_input_occurrences={d:int(c[t]) for d,c in counts.items()}) for t in g]))
    summary=dict(collision_groups=len(collisions),affected_vocabulary_ids=len(union),training_stream_occurrences=int(sum(train_counts[t] for t in union)),training_stream_total=len(train_values),transfer_scored_input_occurrences={d:int(sum(c[t] for t in union)) for d,c in counts.items()},transfer_scored_input_totals=totals)
    put(OUT/'composer-collision-audit.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),composer_source_sha256=sha(source),codebook_builder_sha256=sha(ROOT/'research/byte_input_model_v3/codebook.py'),codebook_identity=book.identity,train_sha256=sha(train),dataset_sha256=sha(P/'dataset.json'),widths=list(widths),summary=summary,groups=rows,elapsed_seconds=time.monotonic()-start,reasoning='For each filter width, identical sets of valid byte-symbol windows produce the same set of pre-pool activations for any fixed filter weights and biases. Max pooling discards multiplicity/order. Equal pooled vectors imply equal subsequent deterministic highway/projection outputs. Boundary symbols and the empty-width case are included in the fingerprint. This is a sufficient algebraic condition, not an exhaustive test of learned numerical collisions.',scope='Static audit only; no model forward pass or training. Actual finite-precision checkpoint equality is not measured here. Training-stream counts are not sampled exposure. Input collisions can remove distinctions but do not prove that they cause transfer loss or that a proposed repair improves quality. These are properties of the accepted architecture, not grounds to invalidate or rerun completed studies. No new quality gate or original test access.',test_opened=False))
    print(summary)
if __name__=='__main__':main()
