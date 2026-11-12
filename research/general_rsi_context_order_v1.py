"""Length/position/multiset-preserving intervention on retained contexts only."""
import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time
import traceback
from research.general_rsi_repetition_probability_v1 import sha,read,put,append,statistics

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'research/tokenizer-evidence/general-rsi-repetition-probability-001'
OUT=ROOT/'research/tokenizer-evidence/general-rsi-context-order-001'
SEED=62111205020001
STOP=False

def stop(*_):
    global STOP
    STOP=True


def permute(tokens,lo,hi,target,key):
    assert 0<=lo<=hi<=target-2 and target<len(tokens)
    perm=list(range(lo,hi));seed=int(hashlib.sha256(f'{SEED}:{key}'.encode()).hexdigest(),16)
    random.Random(seed).shuffle(perm)
    changed=tokens.copy();changed[lo:hi]=[tokens[i] for i in perm]
    assert len(changed)==len(tokens) and Counter(changed[:target])==Counter(tokens[:target])
    assert changed[target-2:]==tokens[target-2:] and changed[:lo]==tokens[:lo]
    return changed,perm


def prepare():
    assert not OUT.exists();OUT.mkdir(parents=True)
    old=read(OLD/'plan.json');analysis=read(OLD/'analysis.json')
    records=[json.loads(l) for l in (OLD/'attempt-01/rows.jsonl').read_text().splitlines()]
    sourcegen={(r['slot'],r['id']):r for r in old['generations']}
    scores={(r['slot'],r['kind'],r.get('id',r.get('document'))):r for r in records}
    windows={w['document']:w for w in old['training_windows']};cases=[]
    for loop in analysis['loop_rows']:
        slot=loop['slot'];r=sourcegen[slot,loop['id']];prefix=r['prefix'];seq=prefix+r['tokens']
        period=loop['period'];n=len(seq);target=n-period;lo=n-3*period;hi=target-2
        key=f"loop:{slot}:{r['id']}";changed,perm=permute(seq,lo,hi,target,key)
        expected=scores[slot,'generation',r['id']]['tokens'][-period:]
        cases.append(dict(kind='loop',slot=slot,id=r['id'],key=key,original=seq,permuted=changed,
            range=[lo,hi],permutation=perm,prefix_tokens=len(prefix),target_positions=list(range(target,n)),
            expected=expected,period=period,changed_positions=sum(a!=b for a,b in zip(seq,changed))))
    for ref in analysis['natural_rows']:
        slot=ref['slot'];doc=ref['document'];w=windows[doc];seq=w['tokens'];target=ref['repeat_endpoint'];lo=ref['first_endpoint']-2;hi=target-2
        key=f'natural:{doc}:{target}';changed,perm=permute(seq,lo,hi,target,key)
        expected=[scores[slot,'training_window',doc]['tokens'][target-1]]
        cases.append(dict(kind='natural',slot=slot,id=f'{doc}:{target}',key=key,source=w['source'],original=seq,permuted=changed,
            range=[lo,hi],permutation=perm,prefix_tokens=1,target_positions=[target],expected=expected,
            changed_positions=sum(a!=b for a,b in zip(seq,changed))))
    assert len(cases)==281 and len([r for r in cases if r['kind']=='loop'])==23
    for r in cases:
        assert all(r['original'][i]==x['token'] for i,x in zip(r['target_positions'],r['expected'],strict=True))
    # Empty and one-position edits must remain honest no-op controls.
    assert permute([1,2,3,4],0,0,2,'empty')[0]==[1,2,3,4]
    assert permute([1,2,3,4],0,1,3,'single')[0]==[1,2,3,4]
    z,p=permute(list(range(10)),1,6,8,'multi');assert Counter(z[:8])==Counter(range(8)) and z[6:]==list(range(6,10))
    sources=dict(old['sources']);sources['research/general_rsi_context_order_v1.py']=sha(__file__)
    inputs=dict(old['inputs'])
    for f in [OLD/'plan.json',OLD/'analysis.json',OLD/'attempt-01/rows.jsonl',OLD/'attempt-01/complete.json']:
        inputs[str(f.relative_to(ROOT))]=sha(f)
    plan=dict(kind='retained_context_order_intervention_v1',sources=sources,inputs=inputs,models=old['models'],
        checkpoint_source_plan_sha256=old['checkpoint_source_plan_sha256'],torch_version=old['torch_version'],cases=cases,
        seed=SEED,permutation_policy='Python Random(SHA256(seed:key)).shuffle one index permutation; natural key excludes model; no rerolls. Half-open ranges. Preserve last2 tokens before target and all later tokens. Retain no-ops.',
        comparisons='Counterfactual-minus-intact target log probability, strict rank and argmax; average targets within loop then loops equally within each model. Natural endpoint cases separate; descriptive only. Source order/semantics changes remain inseparable.',
        exact_original_greedy_required=True,intact_log_probability_abs_tolerance=1e-6,
        total_worker_seconds=600,rss_limit_bytes=4*2**30,mps_limit_bytes=6*2**30,reserve_bytes=10*2**30,output_cap_bytes=64*2**20,
        new_generations=0,new_checkpoints=0,test_opened=False)
    put(OUT/'plan.json',plan)
    put(OUT/'admission.json',dict(plan_sha256=sha(OUT/'plan.json'),cases=281,paired_condition_sequences=562,
        loop_incremental_calls=2*sum(len(r['original'])-r['prefix_tokens'] for r in cases if r['kind']=='loop'),
        natural_full_prefix_calls=516,new_invariant_checks_passed=3,all281_case_invariants_passed=True,
        prior_guard_seconds=68.50267541711219,forecast_worker_seconds=180,ceiling_seconds=600,
        forecast_basis='Prior72 incremental generations plus192 full windows took68.5s; now46 incremental generations plus516 full windows.180s includes extra load/hash/I/O margin. Not a speed claim.',
        free_bytes=shutil.disk_usage(ROOT).free,required_free_bytes=10*2**30+64*2**20,complete_retained_output_cap_bytes=64*2**20))
    print('Frozen281 cases,562 conditions; no-op cases',sum(r['changed_positions']==0 for r in cases),flush=True)


def run():
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    plan=read(OUT/'plan.json');binding=sha(OUT/'plan.json');assert read(OUT/'admission.json')['plan_sha256']==binding
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free>=plan['reserve_bytes']+plan['output_cap_bytes']
        dest=OUT/'attempt-01';dest.mkdir()
        with CpuPrerequisiteGuard(dest/'guard.json',ROOT,timeout_seconds=600,rss_limit_bytes=plan['rss_limit_bytes'],reserve_bytes=plan['reserve_bytes']):
            try:
                import torch
                from research.general_rsi_transfer_worker_v1 import load_model
                from research.byte_input_cache_v4 import Session
                from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
                for name,h in {**plan['sources'],**plan['inputs']}.items():assert sha(ROOT/name)==h,name
                assert torch.__version__==plan['torch_version'] and torch.backends.mps.is_available()
                assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0';torch.set_num_threads(4)
                def check():
                    assert not STOP and not read(ROOT/'research_state/general-rsi-state.json').get('stop_requested',False),'human_stop'
                    assert torch.mps.driver_allocated_memory()<plan['mps_limit_bytes']
                    assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<plan['output_cap_bytes']
                count=0;max_delta=0.
                for rec in plan['models']:
                    check();model=load_model(rec,None,plan);model.eval();versions=tuple(p._version for _,p in model.ordered_parameters())
                    for r in [c for c in plan['cases'] if c['slot']==rec['slot']]:
                        check();out={}
                        for condition in ['original','permuted']:
                            seq=r[condition];session=Session(model,1);values=[]
                            if r['kind']=='loop':
                                inputs=torch.tensor([seq[:r['prefix_tokens']]],device='mps')
                                for pos in range(r['prefix_tokens'],len(seq)):
                                    check()
                                    with torch.no_grad(),FP32RMSNormInference():logits=session.logits(inputs,session.position)[0,-1].cpu()
                                    if pos in r['target_positions']:values.append(statistics(logits,seq[pos]))
                                    inputs=torch.tensor([[seq[pos]]],device='mps')
                            else:
                                with torch.no_grad(),FP32RMSNormInference():logits=session.logits(torch.tensor([seq[:-1]],device='mps'),0)[0].cpu()
                                values=[statistics(logits[pos-1],seq[pos]) for pos in r['target_positions']]
                            out[condition]=values
                        append(dest/'rows.jsonl',dict(slot=r['slot'],kind=r['kind'],id=r['id'],changed_positions=r['changed_positions'],**out))
                        deltas=[abs(a['log_probability']-b['log_probability']) for a,b in zip(out['original'],r['expected'],strict=True)]
                        max_delta=max(max_delta,max(deltas));assert max(deltas)<=plan['intact_log_probability_abs_tolerance'],'intact replay mismatch'
                        assert all(a['argmax']==b['argmax'] for a,b in zip(out['original'],r['expected'],strict=True))
                        if r['changed_positions']==0:assert out['original']==out['permuted'],'no-op replay mismatch'
                        count+=1
                    assert versions==tuple(p._version for _,p in model.ordered_parameters())
                    print('Completed slot',rec['slot'],'cases',count,flush=True)
                    del model,session,logits;gc.collect();torch.mps.empty_cache()
                assert count==281
                put(dest/'complete.json',dict(plan_sha256=binding,cases=count,sequences=2*count,max_intact_log_probability_abs_delta=max_delta,
                    rows_sha256=sha(dest/'rows.jsonl'),test_opened=False,new_generations=0))
            except BaseException as e:
                put(dest/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),plan_sha256=binding));raise


if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args();prepare() if a.action=='prepare' else run()
