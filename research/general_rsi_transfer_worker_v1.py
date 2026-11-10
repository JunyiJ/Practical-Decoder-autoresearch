"""Load frozen equal-time endpoints and score transfer data without training."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
from research.general_rsi_transfer_data_v1 import ROOT,P,D,BINDING,sha,read,put,contract,TransferData

STOP=False

def stop(signum,frame):
    global STOP
    STOP=True

def check():
    import torch
    assert not STOP and not read(ROOT/'research_state/general-rsi-state.json').get('stop_requested',False),'human_stop'
    assert torch.mps.driver_allocated_memory()<6*1024**3,'MPS resource limit'

def replace(path,obj):
    temp=Path(path).with_suffix('.tmp')
    with temp.open('w') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def validate_sources(plan):
    prior_path=ROOT/'research/tokenizer-evidence/general-rsi-equal-time-001/plan.json'
    assert sha(prior_path)==plan['checkpoint_source_plan_sha256']
    prior=read(prior_path)
    for f,h in prior['sources'].items():assert sha(ROOT/f)==h,f
    for asset in plan['inference']['codebook_assets']:assert sha(asset['path'])==asset['sha256']
    return prior

def setup(plan):
    import torch
    from research.byte_input_model_v6 import build_codebook
    assert torch.__version__==plan['inference']['torch_version'] and torch.backends.mps.is_available()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    torch.set_num_threads(4)
    return build_codebook(*(Path(a['path']).read_bytes() for a in plan['inference']['codebook_assets']),'raw_bytes')

def load_model(rec,book,plan):
    import torch
    from research.byte_input_model_v6 import ByteInputModel,Spec
    p=Path(rec['path']);assert not p.is_symlink() and p.stat().st_size==rec['size'] and sha(p)==rec['sha256']
    payload=torch.load(p,map_location='cpu',weights_only=True)
    assert payload['kind']=='general_rsi_equal_time_inference_v1' and payload['endpoint']=='time_ema'
    assert payload['binding']==plan['checkpoint_source_plan_sha256'] and payload['slot']['slot']==rec['slot'] and payload['slot']['arm']==rec['arm']
    cfg=payload['slot'];roles=cfg['roles']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(roles['native_initializer'])
        spec=Spec('native') if rec['arm']=='native' else Spec('raw_bytes',roles['composer_initializer'],book.identity)
        model=ByteInputModel(spec,None if rec['arm']=='native' else book)
    assert model.identity==payload['model_identity'] and model._own_expected_buffer_hashes==payload['immutable_buffers']
    items=dict(model.ordered_parameters());assert list(items)==list(payload['parameters'])
    with torch.no_grad():
        for name,param in items.items():
            saved=payload['parameters'][name]
            assert param.shape==saved.shape and saved.dtype==torch.float32 and torch.isfinite(saved).all()
            param.copy_(saved)
    del payload
    model=model.to('mps');model.validate();return model

def qualify(plan):
    import torch
    from research.general_rsi_training_v2 import score
    prior=validate_sources(plan);book=setup(plan)
    old=ROOT/'data/raw/rsi-general-10m-v1';manifest=read(old/'manifest.json')
    for name in ['val.bin','val.index.jsonl']:assert sha(old/name)==manifest['files'][name]['sha256']
    first=json.loads((old/'val.index.jsonl').read_text().splitlines()[0])
    ids=np.fromfile(old/'val.bin',dtype='<u2',count=first['tokens'])
    data=TransferData(ids,[dict(offset=0,tokens=len(ids),source=first['source'])],check)
    results=[]
    for rec in plan['models'][:2]:
        model=load_model(rec,book,plan);t=time.monotonic();got=score(model,data)
        expected=read(ROOT/f"research/tokenizer-evidence/general-rsi-equal-time-001/slot-{rec['slot']:02d}/time_ema-validation.json")['documents'][0]
        assert got['count']==expected['count']
        delta=abs(got['documents'][0]['loss_sum']-expected['loss_sum'])/expected['count']
        assert delta<1e-6
        results.append(dict(slot=rec['slot'],arm=rec['arm'],known_validation_document=0,count=got['count'],absolute_ce_delta=delta,tolerance=1e-6,seconds=time.monotonic()-t))
        del model;gc.collect();torch.mps.empty_cache()
    put(P/'checkpoint-loader-qualification.json',dict(passed=True,protocol_sha256=BINDING,worker_sha256=sha(__file__),data_adapter_sha256=sha(ROOT/'research/general_rsi_transfer_data_v1.py'),results=results,original_frozen_sources_verified=True,new_transfer_text_scored=False,test_opened=False))
    print(json.dumps(results),flush=True)

def execute(plan,attempt):
    import torch
    from research.general_rsi_training_v2 import score
    admission=read(P/'scoring-admission.json')
    assert admission['protocol_sha256']==BINDING and admission['worker_sha256']==sha(__file__)
    assert admission['data_adapter_sha256']==sha(ROOT/'research/general_rsi_transfer_data_v1.py')
    assert admission['dataset_sha256']==sha(P/'dataset.json')
    validate_sources(plan);book=setup(plan)
    dataset=read(P/'dataset.json')
    for rec in dataset['payloads']:assert sha(ROOT/rec['path'])==rec['sha256']
    ids=np.fromfile(D/'excerpts.bin',dtype='<u2');rows=read(D/'index.json');data=TransferData(ids,rows,check)
    assert len(rows)==64 and len(ids)-len(rows)==204800
    out=P/f'evaluation-attempt-{attempt:02d}';out.mkdir()
    statefile=ROOT/'research_state/general-rsi-state.json';state=read(statefile)
    state.update(status='TRANSFER_EVALUATION_RUNNING',active_worker_pid=os.getpid(),active_slot=None,next_action='Review complete frozen six-checkpoint transfer results; do not change source selection, gates or seed roles.')
    replace(statefile,state)
    for rec in plan['models']:
        check();t=time.monotonic();model=load_model(rec,book,plan);loaded=time.monotonic()
        put(out/f"slot-{rec['slot']:02d}-start.json",dict(checkpoint=rec,protocol_sha256=BINDING,at=time.time(),pid=os.getpid()))
        print('Scoring slot',rec['slot'],rec['arm'],flush=True)
        result=score(model,data)
        torch.mps.synchronize()
        assert result['count']==204800 and all(v['count']==102400 for v in result['sources'].values())
        put(out/f"slot-{rec['slot']:02d}.json",dict(checkpoint=rec,protocol_sha256=BINDING,dataset_sha256=admission['dataset_sha256'],worker_sha256=sha(__file__),score=result,load_seconds=loaded-t,score_seconds=time.monotonic()-loaded,mps_driver_bytes=torch.mps.driver_allocated_memory(),test_opened=False))
        print('Completed slot',rec['slot'],json.dumps(result['sources']),flush=True)
        del model;gc.collect();torch.mps.empty_cache()
    put(out/'complete.json',dict(protocol_sha256=BINDING,slots=[r['slot'] for r in plan['models']],test_opened=False))
    state=read(statefile);state.update(status='TRANSFER_EVALUATION_COMPLETE_REVIEW_PENDING',active_worker_pid=None,active_session=None,next_action='Verify six transfer score inventories, apply frozen per-domain multiplicity-adjusted rules, reconcile evidence and report scoped result.');replace(statefile,state)

def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['qualify','score']);a.add_argument('--attempt',type=int,default=1);args=a.parse_args();assert args.attempt in [1,2]
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    plan=contract()
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spent=sum(read(f)['elapsed_seconds'] for f in P.glob('*-guard-*.json'));remaining=plan['resource']['total_local_guard_seconds']-spent;assert remaining>0
        name='loader' if args.action=='qualify' else 'evaluation'
        with CpuPrerequisiteGuard(P/f'{name}-guard-{args.attempt:02d}.json',ROOT,timeout_seconds=remaining,rss_limit_bytes=plan['resource']['rss_limit_bytes'],reserve_bytes=plan['resource']['reserve_bytes']):
            try:qualify(plan) if args.action=='qualify' else execute(plan,args.attempt)
            except BaseException as e:
                put(P/f'{name}-failure-{args.attempt:02d}.json',dict(type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),worker_sha256=sha(__file__),protocol_sha256=BINDING));raise
if __name__=='__main__':main()
