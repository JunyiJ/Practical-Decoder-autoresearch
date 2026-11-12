"""Frozen inference-only replay; no optimizer, new decoding or holdout access."""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-repetition-probability-001'
PRIOR=ROOT/'research/tokenizer-evidence/general-rsi-native-horizon-001'
STOP=False


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def append(p,v):
    with Path(p).open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def stop(*_):
    global STOP
    STOP=True


def targets(prefix,tokens):
    assert prefix and tokens and len(prefix)+len(tokens)<=128
    seq=prefix+tokens
    return seq[:-1],tokens,list(range(len(prefix)-1,len(seq)-1))


def statistics(logits,target):
    import torch
    assert logits.ndim==1 and bool(torch.isfinite(logits).all())
    x=logits.float().cpu()
    return dict(token=target,log_probability=float(torch.log_softmax(x,dim=-1)[target]),
                argmax=int(x.argmax()),rank=1+int((x>x[target]).sum()))


def prepare():
    import numpy as np
    import tiktoken
    assert not OUT.exists();OUT.mkdir(parents=True)
    prior=read(PRIOR/'plan.json');inputs={str((PRIOR/'plan.json').relative_to(ROOT)):sha(PRIOR/'plan.json')}
    sources=dict(prior['sources'])
    for name in ['research/general_rsi_repetition_probability_v1.py','research/general_rsi_transfer_worker_v1.py',
                 'research/general_rsi_transfer_data_v1.py','src/research/cpu_prerequisite_guard.py']:
        sources[name]=sha(ROOT/name)
    models=[];generations=[];enc=tiktoken.get_encoding('gpt2')
    for slot in [1,3,5]:
        for name in ['review.json','result.json','generation/result.json']:
            p=PRIOR/f'slot-{slot:02d}'/name;inputs[str(p.relative_to(ROOT))]=sha(p)
        assert read(PRIOR/f'slot-{slot:02d}/review.json')['accepted_execution']
        result=read(PRIOR/f'slot-{slot:02d}/result.json')
        rec=next(x['weights'] for x in result['endpoints'] if x['endpoint']=='time_ema')
        assert sha(rec['path'])==rec['sha256']
        models.append(dict(**rec,slot=slot,arm='native'))
        gen=read(PRIOR/f'slot-{slot:02d}/generation/result.json')
        for r in gen['rows']:
            assert r['checkpoint_sha256']==rec['sha256']
            generations.append(dict(slot=slot,id=r['id'],decoder=r['decoder'],prefix=enc.encode(r['prompt']['text']),
                tokens=r['generated_token_ids'],terminal_loop_phrase_lengths=r['metrics']['terminal_loop_phrase_lengths']))
    data=ROOT/'data/raw/rsi-general-10m-v1';manifest=read(data/'manifest.json')
    for name in ['train.bin','train.index.jsonl']:
        assert sha(data/name)==manifest['files'][name]['sha256'];inputs[str((data/name).relative_to(ROOT))]=sha(data/name)
    docs=[json.loads(x) for x in (data/'train.index.jsonl').read_text().splitlines()]
    values=np.fromfile(data/'train.bin',dtype='<u2');windows=[]
    for source in ['fineweb_edu','wikipedia']:
        chosen=sorted(enumerate(docs),key=lambda x:(x[1]['normalized_sha256'],x[0]))
        chosen=[(i,d) for i,d in chosen if d['source']==source][:32]
        for i,d in chosen:
            n=d['tokens']-1;assert n>=64;offset=d['offset']+(n-64)//2
            window=values[offset:offset+64].tolist();assert 50256 not in window
            windows.append(dict(document=i,source=source,normalized_sha256=d['normalized_sha256'],offset=offset,tokens=window))
    assert len(generations)==72 and len(windows)==64
    assert targets([10,11],[12,13])==([10,11,12],[12,13],[1,2])
    assert targets([10],[11])==([10],[11],[0])
    plan=dict(kind='repetition_probability_diagnostic_v1',sources=sources,inputs=inputs,models=models,
        checkpoint_source_plan_sha256=sha(PRIOR/'plan.json'),torch_version=prior['torch_version'],
        generations=generations,training_windows=windows,
        evaluation='Original generation uses exact cached prefix/incremental path, teacher-forcing retained tokens. Natural windows use cached full-prefix logits for targets1..63. Corrected FP32 RMSNorm; CPU FP32 log_softmax. Rank=1+number strictly greater logits; ties share rank.',
        analysis='Smallest frozen terminal suffix period; last3 complete copies; mean log probability and third-minus-first paired token differences; per-model counts and means. Descriptive only, no causal/adoption claim. Natural repeated-trigram endpoints compared with first same-trigram endpoint within each64-token window.',
        total_worker_seconds=600,rss_limit_bytes=4*2**30,mps_limit_bytes=6*2**30,reserve_bytes=10*2**30,
        retained_output_cap_bytes=64*2**20,new_checkpoints=0,new_generations=0,test_opened=False)
    put(OUT/'plan.json',plan)
    put(OUT/'preparation.json',dict(source_sha256=sha(__file__),plan_sha256=sha(OUT/'plan.json'),
        indexing_checks_passed=2,sequences=264,generated_target_tokens=sum(len(r['tokens']) for r in generations),natural_targets=3*64*63,
        free_bytes=shutil.disk_usage(ROOT).free,forecast_incremental_bytes=64*2**20,required_free_bytes=10*2**30+64*2**20))
    print('Prepared264 sequences; no model probabilities inspected.',flush=True)


def run():
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    plan=read(OUT/'plan.json');binding=sha(OUT/'plan.json')
    assert read(OUT/'preparation.json')['plan_sha256']==binding
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free>=plan['reserve_bytes']+plan['retained_output_cap_bytes']
        dest=OUT/'attempt-01';dest.mkdir()
        with CpuPrerequisiteGuard(dest/'guard.json',ROOT,timeout_seconds=600,rss_limit_bytes=plan['rss_limit_bytes'],reserve_bytes=plan['reserve_bytes']):
            try:
                import torch
                from research import general_rsi_transfer_worker_v1 as loader
                from research.byte_input_cache_v4 import Session
                from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
                start=time.monotonic()
                for name,h in {**plan['sources'],**plan['inputs']}.items():assert sha(ROOT/name)==h,name
                assert torch.__version__==plan['torch_version'] and torch.backends.mps.is_available()
                assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0';torch.set_num_threads(4)
                # Independent scalar oracle for probability/rank and ties.
                checkstats=statistics(torch.tensor([0.,math.log(2),math.log(2)]),1)
                assert abs(math.exp(checkstats['log_probability'])-.4)<1e-7 and checkstats['rank']==1 and checkstats['argmax']==1
                put(dest/'indexing-check.json',dict(passed=True,oracle='p=2/5, strict rank1 with tied maxima; prefix/target offset cases frozen at preparation',plan_sha256=binding))
                def check():
                    assert not STOP and not read(ROOT/'research_state/general-rsi-state.json').get('stop_requested',False),'human_stop'
                    assert torch.mps.driver_allocated_memory()<plan['mps_limit_bytes']
                    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<plan['retained_output_cap_bytes']
                total=0;greedy_targets=0
                for rec in plan['models']:
                    check();model=loader.load_model(rec,None,plan);model.eval()
                    versions=tuple(p._version for _,p in model.ordered_parameters())
                    for r in [x for x in plan['generations'] if x['slot']==rec['slot']]:
                        check();session=Session(model,1);inputs=torch.tensor([r['prefix']],device='mps');stats=[]
                        for token in r['tokens']:
                            check()
                            with torch.no_grad(),FP32RMSNormInference():logits=session.logits(inputs,session.position)[0,-1].cpu()
                            st=statistics(logits,token);stats.append(st)
                            if r['decoder']=='greedy' and st['argmax']!=token:
                                append(dest/'rows.jsonl',dict(kind='failed_greedy_replay',slot=rec['slot'],id=r['id'],tokens=stats));raise AssertionError('original greedy argmax mismatch')
                            inputs=torch.tensor([[token]],device='mps')
                        greedy_targets+=len(stats) if r['decoder']=='greedy' else 0
                        append(dest/'rows.jsonl',dict(kind='generation',slot=rec['slot'],id=r['id'],decoder=r['decoder'],tokens=stats,
                            terminal_loop_phrase_lengths=r['terminal_loop_phrase_lengths']))
                        total+=1
                    for w in plan['training_windows']:
                        check();session=Session(model,1);x,y,positions=targets(w['tokens'][:1],w['tokens'][1:])
                        with torch.no_grad(),FP32RMSNormInference():logits=session.logits(torch.tensor([x],device='mps'),0)[0].cpu()
                        stats=[statistics(logits[pos],tok) for tok,pos in zip(y,positions,strict=True)]
                        append(dest/'rows.jsonl',dict(kind='training_window',slot=rec['slot'],document=w['document'],source=w['source'],tokens=stats))
                        total+=1
                    assert versions==tuple(p._version for _,p in model.ordered_parameters())
                    print('Completed slot',rec['slot'],'sequences',total,flush=True)
                    del model,session,logits;gc.collect();torch.mps.empty_cache()
                assert total==264 and greedy_targets==2304
                put(dest/'complete.json',dict(plan_sha256=binding,sequences=total,all_original_greedy_targets_match=greedy_targets,
                    seconds=time.monotonic()-start,rows_sha256=sha(dest/'rows.jsonl'),new_generations=0,test_opened=False))
            except BaseException as exc:
                put(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),plan_sha256=binding));raise


if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.action=='prepare' else run()
