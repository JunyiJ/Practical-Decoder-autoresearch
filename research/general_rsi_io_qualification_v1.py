"""Qualify only new durable clock and fixed generation integration boundaries."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-io-qualification-001'


def put(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def prepare():
    from research.general_rsi_training_v2 import sha
    prior=json.loads((ROOT/'research/tokenizer-evidence/general-rsi-time-qualification-001/plan.json').read_text())
    sources=dict(prior['sources'])
    for name in ('research/general_rsi_time_journal_v1.py','research/general_rsi_generation_v2.py',
                 'research/general_rsi_io_qualification_v1.py','research/general_rsi_generation_metrics_v1.py',
                 'research/general-rsi-generation-eval-v1.json'):
        sources[name]=sha(ROOT/name)
    inputs={}
    for arm in prior['arms']:
        path=ROOT/f'research/tokenizer-evidence/general-rsi-time-qualification-001/{arm}-result.json'
        inputs[arm]=dict(report=str(path.relative_to(ROOT)),sha256=sha(path),checkpoint=json.loads(path.read_text())['checkpoint'])
    plan=dict(kind='equal_time_io_engineering_qualification_v1',sources=sources,inputs=inputs,
        context_plan='research/tokenizer-evidence/general-rsi-time-qualification-001/plan.json',
        context_binding=sha(ROOT/'research/tokenizer-evidence/general-rsi-time-qualification-001/plan.json'),
        generation_protocol='research/general-rsi-generation-eval-v1.json',
        tests=['Synthetic complete/replay charge and malformed journal rejection','Real subprocess interrupted interval and required supervisor exit bound',
               'Both MPS engineering checkpoints: all24 first fixed generations, unchanged complete training state',
               'One real update per arm through durable interval clock'],
        timeout_seconds=600,rss_limit_bytes=4*1024**3,mps_limit_bytes=6*1024**3,
        reserve_bytes=10*1024**3,retained_output_cap_bytes=32*1024**2,
        scientific_selection=False,test_opened=False,no_new_training_seeds=True)
    assert not OUT.exists();OUT.mkdir();put(OUT/'plan.json',plan);print(sha(OUT/'plan.json'),flush=True)


def run(plan,binding):
    import torch
    from research.general_rsi_time_journal_v1 import IntervalJournal,accounted_intervals,recovery_charge
    from research.general_rsi_generation_v2 import generate,select_token,blind
    from research.general_rsi_equal_time_training_v1 import Context,read_checkpoint,sha
    from research.general_rsi_qualification_v1 import equal
    from research.byte_input_model_v6 import build_codebook
    synthetic=OUT/'synthetic-complete.jsonl'
    ticks=iter((100,120,130,160))
    j=IntervalJournal(synthetic,1,1,0,lambda:next(ticks));assert j.tick()==100 and j.tick()==120
    j=IntervalJournal(synthetic,1,2,20,lambda:next(ticks));j.tick();j.tick()
    rows=[json.loads(l) for l in synthetic.read_text().splitlines()]
    charge=recovery_charge(rows,1,20);assert charge['duration_ns']==30 and not charge['conservative']
    for invalid in ([rows[1]], [rows[0],rows[0]], [dict(rows[0],event='unknown')]):
        try:accounted_intervals(invalid)
        except (AssertionError,ValueError):pass
        else:raise AssertionError('Invalid journal accepted')
    child_log=OUT/'synthetic-interrupted.jsonl'
    child=subprocess.Popen([sys.executable,'-c',
        "import os,sys;from research.general_rsi_time_journal_v1 import IntervalJournal;j=IntervalJournal(sys.argv[1],2,1,0);j.tick();os._exit(23)",str(child_log)],cwd=ROOT)
    code=child.wait();exit_ns=time.perf_counter_ns();assert code==23
    put(OUT/'synthetic-supervisor-exit.json',dict(pid=child.pid,returncode=code,exit_ns=exit_ns,same_boot=True))
    interrupted=[json.loads(l) for l in child_log.read_text().splitlines()]
    try:accounted_intervals(interrupted)
    except ValueError:pass
    else:raise AssertionError('Unknown unfinished work silently ignored')
    bound=recovery_charge(interrupted,0,0,exit_ns=exit_ns)
    assert bound['conservative'] and bound['duration_ns']>0
    put(OUT/'journal-check.json',dict(complete_replay_charge=charge,interrupted_charge=bound,invalid_rejected=3,missing_exit_rejected=True))
    # Synthetic decoder seeds and blinded-sheet identities; no model rerolls.
    logits=torch.arange(50257,dtype=torch.float32)/10000
    a=[select_token(logits,'temperature_0.8_top_k_40',torch.Generator().manual_seed(42)) for _ in range(2)]
    assert a[0]==a[1] and select_token(logits,'greedy',None)==50256
    fake=[dict(id=str(i),prompt=dict(text='synthetic',consistency_reference='synthetic'),continuation='synthetic',checkpoint_sha256=str(i),decoder='greedy') for i in range(144)]
    sheets,mapping=blind(fake)
    assert len({r['id'] for r in sheets})==144 and len({r['original_id'] for r in mapping})==144
    assert all('checkpoint_sha256' not in r and 'decoder' not in r for r in sheets)
    put(OUT/'decoder-synthetic-check.json',dict(seed_reset=True,greedy=True,blind_mapping_bijection=True))
    prior=json.loads((ROOT/plan['context_plan']).read_text())
    assert sha(ROOT/plan['context_plan'])==plan['context_binding']
    assert torch.__version__==prior['torch_version'] and torch.backends.mps.is_available()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0';torch.set_num_threads(4)
    raw=[]
    for asset in prior['codebook_assets']:
        assert sha(asset['path'])==asset['sha256'];raw.append(Path(asset['path']).read_bytes())
    book=build_codebook(*raw,'raw_bytes');protocol=json.loads((ROOT/plan['generation_protocol']).read_text())
    reports=[]
    for arm in prior['arms']:
        assert not json.loads((ROOT/'research_state/general-rsi-state.json').read_text()).get('stop_requested',False)
        spec=plan['inputs'][arm];assert sha(ROOT/spec['report'])==spec['sha256']
        context=Context(arm,prior['lr'],prior['budget_ns'],prior['half_life_ns'],prior['roles'],book,plan['context_binding'])
        saved=read_checkpoint(spec['checkpoint']);context.restore(saved);del saved
        before=context.capture()
        started=time.perf_counter()
        rows=generate(context.model,protocol,OUT/f'{arm}-generations',spec['checkpoint'],binding)
        generation_seconds=time.perf_counter()-started
        equal(context.capture(),before);del before
        journal=OUT/f'{arm}-interval.jsonl'
        j=IntervalJournal(journal,1,context.step+1,context.clock.spent_ns)
        row=context.update(now_ns=j.tick)
        events=[json.loads(l) for l in journal.read_text().splitlines()]
        intervals=accounted_intervals(events)
        assert len(intervals)==1 and intervals[0]['spent_after_ns']==context.clock.spent_ns
        assert intervals[0]['duration_ns']==row['end_ns']-row['begin_ns']
        assert torch.mps.driver_allocated_memory()<plan['mps_limit_bytes']
        report=dict(arm=arm,generation_seconds=generation_seconds,first_outputs=len(rows),
            generated_tokens=sum(len(r['generated_token_ids']) for r in rows),
            complete_training_state_unchanged_after_generation=True,durable_clock_interval=intervals[0],
            scientific_selection=False)
        put(OUT/f'{arm}-result.json',report);reports.append(report);print(json.dumps(report),flush=True)
        del context;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
    return dict(qualified=True,arms=reports,test_opened=False,scientific_selection=False,
        scope='Journal and generation integration only. These48 engineering completions are retained and excluded from the future144 scientific completions.')


def main():
    from research.general_rsi_training_v2 import sha
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('binding',nargs='?');args=p.parse_args()
    if args.action=='prepare':return prepare()
    assert sha(OUT/'plan.json')==args.binding;plan=json.loads((OUT/'plan.json').read_text())
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free>=plan['reserve_bytes']+plan['retained_output_cap_bytes']
        for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,path
        put(OUT/'attempt.json',dict(pid=os.getpid(),binding=args.binding,started=time.time()))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=plan['timeout_seconds'],rss_limit_bytes=plan['rss_limit_bytes'],reserve_bytes=plan['reserve_bytes']):result=run(plan,args.binding)
            for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,path
            assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=plan['retained_output_cap_bytes']
            put(OUT/'result.json',result);print('QUALIFIED',flush=True)
        except BaseException:
            put(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
