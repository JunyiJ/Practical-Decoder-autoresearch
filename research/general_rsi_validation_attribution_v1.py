"""Retain token losses from all original validation parents; no new quality gate."""
import argparse
import fcntl
import gc
import json
import math
import os
import signal
import time
import traceback
import numpy as np
from research.general_rsi_transfer_data_v1 import ROOT,read,put,sha,contract,TransferData
from research import general_rsi_transfer_worker_v1 as loader
from research.general_rsi_transfer_token_attribution_v1 import label_pieces,NAMES
OUT=ROOT/'research/tokenizer-evidence/general-rsi-validation-attribution-001'
DATA=ROOT/'data/raw/rsi-general-10m-v1'
PRIOR=ROOT/'research/tokenizer-evidence/general-rsi-equal-time-001'

def inputs():
    manifest=read(DATA/'manifest.json')
    for name in ['val.bin','val.index.jsonl']:assert sha(DATA/name)==manifest['files'][name]['sha256']
    values=np.fromfile(DATA/'val.bin',dtype='<u2');rows=[json.loads(x) for x in (DATA/'val.index.jsonl').read_text().splitlines()]
    assert len(rows)==108 and len(values)==100524
    return values,rows

def prepare():
    import tiktoken
    values,rows=inputs();enc=tiktoken.get_encoding('gpt2');labels=[];offsets=[0];docids=[]
    for i,row in enumerate(rows):
        seq=values[row['offset']:row['offset']+row['tokens']]
        assert seq[-1]==50256 and not np.any(seq[:-1]==50256)
        # EOT is a special target, not literal source markup. It belongs to other.
        lab=np.concatenate([label_pieces([enc.decode_single_token_bytes(int(t)) for t in seq[:-1]]),np.array([3],dtype='u1')])[1:]
        assert len(lab)==len(seq)-1 and lab[-1]==3
        labels.extend(lab);offsets.append(offsets[-1]+len(lab));docids.extend([i]*len(lab))
    data=TransferData(values,rows);observed=[[] for _ in rows];chunks=0
    for i,source,x,y in data.validation_chunks():
        assert len(y[0])<=128 and source==rows[i]['source'];observed[i].extend(y[0].tolist());chunks+=1
    for row,got in zip(rows,observed,strict=True):assert got==values[row['offset']+1:row['offset']+row['tokens']].tolist()
    assert offsets[-1]==100416 and len(labels)==len(docids)==100416
    with (OUT/'labels.npz').open('xb') as f:np.savez_compressed(f,labels=np.array(labels,dtype='u1'),offsets=np.array(offsets,dtype='<u4'),document_ids=np.array(docids,dtype='<u2'))
    put(OUT/'qualification.json',dict(worker_sha256=sha(__file__),reused_labeler_sha256=sha(ROOT/'research/general_rsi_transfer_token_attribution_v1.py'),labels_sha256=sha(OUT/'labels.npz'),passed=True,parents=108,targets=100416,chunks=chunks,counts=np.bincount(labels,minlength=4).tolist(),checks=['exact original target inventory including108 EOT targets','variable parent lengths flatten/recover with109 offsets','all four existing label categories; explicit EOT assigned other','first context token excluded per parent'],new_model_scores=0,test_opened=False))

def capture(model,data):
    import torch
    import torch.nn.functional as F
    from research.general_rsi_training_v2 import FP32RMSNormInference
    versions=tuple(p._version for _,p in model.ordered_parameters());mode=model.training;device=model.validate();values=[[] for _ in data.indexes['val']]
    try:
        model.eval()
        with torch.no_grad(),FP32RMSNormInference():
            for doc,source,x,y in data.validation_chunks():
                logits=model.primary_logits(x.to(device)).values[0]
                loss=F.cross_entropy(logits,y[0].to(device),reduction='none')
                assert bool(torch.isfinite(loss).all()) and bool((loss>=0).all())
                values[doc].extend(loss.cpu().double().tolist())
    finally:model.train(mode)
    assert versions==tuple(p._version for _,p in model.ordered_parameters())
    assert all(len(v)==r['tokens']-1 for v,r in zip(values,data.indexes['val'],strict=True))
    return np.array([x for v in values for x in v],dtype='<f4')

def run(attempt):
    import torch
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    cfg=read(OUT/'plan.json');assert cfg['worker_sha256']==sha(__file__)
    for f,h in cfg['inputs_sha256'].items():assert sha(ROOT/f)==h
    remaining=cfg['total_guard_seconds']-sum(read(f)['elapsed_seconds'] for f in OUT.glob('guard-*.json'));assert remaining>0
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with CpuPrerequisiteGuard(OUT/f'guard-{attempt:02d}.json',ROOT,timeout_seconds=remaining,rss_limit_bytes=4*1024**3,reserve_bytes=10*1024**3):
            dest=OUT/f'attempt-{attempt:02d}';dest.mkdir()
            try:
                plan=contract();loader.validate_sources(plan);book=loader.setup(plan)
                values,rows=inputs();data=TransferData(values,rows,loader.check)
                lab=np.load(OUT/'labels.npz');labels=lab['labels'];offsets=lab['offsets'];docids=lab['document_ids'];sources=np.array([r['source'] for r in rows])[docids]
                statefile=ROOT/'research_state/general-rsi-state.json';s=read(statefile);s.update(status='VALIDATION_ATTRIBUTION_RUNNING',active_worker_pid=os.getpid(),active_session=None,next_action='Retain all six original-validation token-loss arrays and reproduce all648 parent scores before descriptive category analysis.');loader.replace(statefile,s)
                for rec in cfg['models']:
                    loader.check();start=time.monotonic();model=loader.load_model(rec,book,plan);loss=capture(model,data);assert len(loss)==100416
                    old=read(PRIOR/f"slot-{rec['slot']:02d}/time_ema-validation.json");deltas=[]
                    for i,r in enumerate(old['documents']):
                        assert r['document']==i and r['source']==rows[i]['source'] and r['count']==offsets[i+1]-offsets[i]
                        deltas.append(abs(math.fsum(map(float,loss[offsets[i]:offsets[i+1]]))-r['loss_sum'])/r['count'])
                    assert max(deltas)<1e-6,'original aggregate mismatch'
                    path=dest/f"slot-{rec['slot']:02d}-losses.npz"
                    with path.open('xb') as f:np.savez_compressed(f,losses=loss)
                    groups={}
                    for source in ['all','fineweb_edu','wikipedia']:
                        mask=np.ones(len(loss),dtype=bool) if source=='all' else sources==source
                        groups[source]={name:dict(count=int((mask&(labels==k)).sum()),loss_sum=math.fsum(loss[mask&(labels==k)].astype('f8').tolist())) for k,name in enumerate(NAMES)}
                    put(dest/f"slot-{rec['slot']:02d}.json",dict(checkpoint=rec,worker_sha256=sha(__file__),plan_sha256=sha(OUT/'plan.json'),losses_sha256=sha(path),max_original_parent_ce_abs_delta=max(deltas),groups=groups,elapsed_seconds=time.monotonic()-start,mps_driver_bytes=torch.mps.driver_allocated_memory(),test_opened=False))
                    print('Completed slot',rec['slot'],'max original parent CE delta',max(deltas),flush=True)
                    del model,loss;gc.collect();torch.mps.empty_cache()
                put(dest/'complete.json',dict(slots=[4,5,6,7,8,9],test_opened=False));s=read(statefile);s.update(status='VALIDATION_ATTRIBUTION_REVIEW_PENDING',active_worker_pid=None,active_session=None);loader.replace(statefile,s)
            except BaseException as e:
                put(dest/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),source_sha256=sha(__file__)));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--attempt',type=int,choices=[1,2],default=1);args=a.parse_args()
    signal.signal(signal.SIGTERM,loader.stop);signal.signal(signal.SIGINT,loader.stop)
    prepare() if args.action=='prepare' else run(args.attempt)
