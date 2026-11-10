"""Diagnostic retention of FP32 token losses on the already scored transfer panel."""
import argparse
import datetime
import fcntl
import gc
import math
import os
import re
import signal
import time
import traceback
import numpy as np
from research.general_rsi_transfer_data_v1 import ROOT,P,D,read,put,sha,contract,TransferData
from research import general_rsi_transfer_worker_v1 as loader
OUT=ROOT/'research/tokenizer-evidence/general-rsi-transfer-diagnosis-001'
NAMES=['whitespace_only','grid_line_nonwhitespace','greek_placeholder_nonwhitespace','other']

def label_pieces(pieces):
    raw=b''.join(pieces);grid=np.zeros(len(raw),dtype=bool);greek=grid.copy();at=0
    for line in raw.splitlines(keepends=True):
        if (line.lstrip().startswith(b'|') and line.rstrip().endswith(b'|')) or re.fullmatch(rb'\s*\+(?:[-=]+\+)+\s*',line):grid[at:at+len(line)]=True
        at+=len(line)
    for m in re.finditer(rb'\{~GREEK[^\n]*?~\}',raw):greek[m.start():m.end()]=True
    labels=[];at=0
    for piece in pieces:
        end=at+len(piece);assert end>at
        if all(c in b' \t\n\r\v\f' for c in piece):label=0
        elif grid[at:end].all():label=1
        elif greek[at:end].all():label=2
        else:label=3
        labels.append(label);at=end
    assert at==len(raw)
    return np.asarray(labels,dtype='u1')

def prepare():
    import tiktoken
    # Boundary fixtures address only the new partition, not the old model loader.
    assert label_pieces([b'   ',b'hello',b'\n']).tolist()==[0,3,0]
    assert label_pieces([b'| ',b'x',b' |\n']).tolist()==[1,1,1]
    assert label_pieces([b'{~GREEK ',b'ALPHA',b'~}',b' x']).tolist()==[2,2,2,3]
    assert label_pieces([b'|x|\noutside']).tolist()==[3]
    dataset=read(P/'dataset.json');assert sha(D/'excerpts.bin')==dataset['payloads'][0]['sha256']
    values=np.fromfile(D/'excerpts.bin',dtype='<u2');enc=tiktoken.get_encoding('gpt2');labels=[]
    for row in dataset['documents']:
        seq=values[row['offset']:row['offset']+row['tokens']]
        labels.append(label_pieces([enc.decode_single_token_bytes(int(x)) for x in seq])[1:])
    labels=np.stack(labels);assert labels.shape==(64,3200) and np.all(labels<4)
    with (OUT/'target-labels.npz').open('xb') as f:np.savez_compressed(f,labels=labels)
    put(OUT/'partition-qualification.json',dict(source_sha256=sha(__file__),passed=True,checks=['pure whitespace precedence','entire token inside grid line','entire token inside Greek placeholder','straddling token assigned other','64x3200 targets exactly once; first context token excluded'],label_names=NAMES,counts=np.bincount(labels.ravel(),minlength=4).tolist(),labels_sha256=sha(OUT/'target-labels.npz'),test_opened=False))

def capture(model,data):
    import torch
    import torch.nn.functional as F
    from research.general_rsi_training_v2 import FP32RMSNormInference
    before=tuple(p._version for _,p in model.ordered_parameters());mode=model.training;device=model.validate();values=[[] for _ in range(64)]
    try:
        model.eval()
        with torch.no_grad(),FP32RMSNormInference():
            for doc,source,x,y in data.validation_chunks():
                logits=model.primary_logits(x.to(device)).values[0]
                loss=F.cross_entropy(logits,y[0].to(device),reduction='none')
                assert bool(torch.isfinite(loss).all()) and bool((loss>=0).all())
                values[doc].extend(loss.cpu().double().tolist())
    finally:model.train(mode)
    assert before==tuple(p._version for _,p in model.ordered_parameters())
    assert all(len(x)==3200 for x in values)
    return np.asarray(values,dtype='<f4')

def run(attempt):
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    import torch
    cfg=read(OUT/'attribution-plan.json');assert cfg['worker_sha256']==sha(__file__)
    for f,h in cfg['inputs_sha256'].items():assert sha(ROOT/f)==h
    spent=sum(read(f)['elapsed_seconds'] for f in OUT.glob('attribution-guard-*.json'));remaining=cfg['total_guard_seconds']-spent;assert remaining>0
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with CpuPrerequisiteGuard(OUT/f'attribution-guard-{attempt:02d}.json',ROOT,timeout_seconds=remaining,rss_limit_bytes=4*1024**3,reserve_bytes=10*1024**3):
            dest=OUT/f'attribution-attempt-{attempt:02d}';dest.mkdir()
            try:
                plan=contract();loader.validate_sources(plan);book=loader.setup(plan)
                dataset=read(P/'dataset.json');data=TransferData(np.fromfile(D/'excerpts.bin',dtype='<u2'),read(D/'index.json'),loader.check)
                labels=np.load(OUT/'target-labels.npz')['labels'];statefile=ROOT/'research_state/general-rsi-state.json';state=read(statefile)
                state.update(status='TRANSFER_ATTRIBUTION_RUNNING',active_worker_pid=os.getpid(),active_session=None,next_action='Finish diagnostic token-loss retention and exact aggregate reconciliation for all six existing transfer checkpoints; no acceptance change.');loader.replace(statefile,state)
                for rec in plan['models']:
                    loader.check();start=time.monotonic();model=loader.load_model(rec,book,plan);loss=capture(model,data)
                    old=read(P/f"evaluation-attempt-01/slot-{rec['slot']:02d}.json")['score'];deltas=[]
                    for i,row in enumerate(old['documents']):
                        assert row['document']==i and row['count']==3200
                        deltas.append(abs(math.fsum(map(float,loss[i]))-row['loss_sum'])/3200)
                    assert max(deltas)<1e-6,'original aggregate mismatch'
                    path=dest/f"slot-{rec['slot']:02d}-losses.npz"
                    with path.open('xb') as f:np.savez_compressed(f,losses=loss)
                    groups={}
                    for domain in plan['domains']:
                        rows=np.array([i for i,r in enumerate(dataset['documents']) if r['source']==domain]);v=loss[rows].astype('f8');lab=labels[rows]
                        groups[domain]={name:dict(count=int((lab==k).sum()),loss_sum=math.fsum(v[lab==k].tolist())) for k,name in enumerate(NAMES)}
                    put(dest/f"slot-{rec['slot']:02d}.json",dict(checkpoint=rec,worker_sha256=sha(__file__),plan_sha256=sha(OUT/'attribution-plan.json'),losses_sha256=sha(path),max_original_parent_ce_abs_delta=max(deltas),groups=groups,elapsed_seconds=time.monotonic()-start,mps_driver_bytes=torch.mps.driver_allocated_memory(),test_opened=False))
                    print('Completed diagnostic slot',rec['slot'],'max original CE difference',max(deltas),flush=True)
                    del model,loss;gc.collect();torch.mps.empty_cache()
                put(dest/'complete.json',dict(slots=[4,5,6,7,8,9],test_opened=False))
                state=read(statefile);state.update(status='TRANSFER_ATTRIBUTION_REVIEW_PENDING',active_worker_pid=None,active_session=None);loader.replace(statefile,state)
            except BaseException as e:
                put(dest/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),source_sha256=sha(__file__)));raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('--attempt',type=int,choices=[1,2],default=1);args=a.parse_args()
    signal.signal(signal.SIGTERM,loader.stop);signal.signal(signal.SIGINT,loader.stop)
    prepare() if args.action=='prepare' else run(args.attempt)
