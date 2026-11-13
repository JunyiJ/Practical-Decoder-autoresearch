"""Serial equal-token depth panel. Every invocation executes one fixed slot."""
import argparse,fcntl,gc,json,os,shutil,signal,subprocess,time,traceback
from pathlib import Path
from research.general_rsi_context_length_worker_v1 import put,replace,stopped,stop,remaining_guard_seconds
from research.general_rsi_training_v2 import sha,write_checkpoint,read_checkpoint,score
from research.general_rsi_time_journal_v1 import append
from research.general_rsi_pseudo_update_diagnostic_v1 import check_stop,digest
ROOT=Path(__file__).resolve().parents[1]
PANEL=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-quality-001'
RUNS=ROOT/'research_runs/general-rsi-cross-size-quality-001'

def retire(receipt,journal,dependency,weights):
    p=Path(receipt['path']);assert p.resolve()==p and p.parent==weights and p.name.startswith('restart-')
    assert weights.is_relative_to(ROOT/'research_runs') and not p.is_symlink()
    st=p.stat();assert st.st_nlink==1 and st.st_size==receipt['size'] and sha(p)==receipt['sha256']
    c=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True)
    assert c.returncode==1 and not c.stdout and not c.stderr
    row=dict(receipt=receipt,inode=st.st_ino,closed_dependency=dependency,no_open_consumers=True)
    append(journal,dict(event='before_unlink',**row))
    assert p.stat().st_ino==st.st_ino and sha(p)==receipt['sha256'];p.unlink()
    append(journal,dict(event='unlinked',**row))

class CheckedData:
    def __init__(self,data):self.data=data;self.view=data.view;self.indexes=data.indexes
    def validation_chunks(self):
        for row in self.data.validation_chunks():check_stop();yield row

def execute(plan,slot,binding,attempt,out,weights,*,engineering_data=None,pause_at=None):
    import numpy as np
    import torch
    from research.byte_input_model_v6 import build_codebook
    from research.general_rsi_cross_size_training_v1 import Context
    from research.general_rsi_cross_size_endpoint_v1 import export,load
    from research.general_rsi_cross_size_generation_v1 import generate
    from research.general_rsi_data_v1 import GeneralData
    from research.general_rsi_transfer_data_v1 import TransferData
    assert torch.__version__==plan['torch_version'] and torch.backends.mps.is_available()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0';torch.set_num_threads(4)
    config=plan['slots'][slot];scientific=plan['scientific_training']
    assert (engineering_data is None and pause_at is None) if scientific else engineering_data is not None
    raw=[]
    for a in plan['codebook_assets']:assert sha(a['path'])==a['sha256'];raw.append(Path(a['path']).read_bytes())
    book=build_codebook(*raw,'raw_bytes')
    ctx=Context(config['arm'],config['depth'],config['roles'],config['extra_group_seeds'],book,binding,config['horizon'])
    if scientific:
        data=GeneralData(config['roles']['train_sampler'],config['roles']['validation_sampler'],'cpu')
        for p,h in plan['broader_payloads'].items():assert sha(ROOT/p)==h
        base=(ROOT/next(p for p in plan['broader_payloads'] if p.endswith('/index.json'))).parent
        broader=TransferData(np.fromfile(base/'excerpts.bin',dtype='<u2'),json.loads((base/'index.json').read_text()))
    else:data=engineering_data(ctx);broader=data
    data=CheckedData(data);broader=CheckedData(broader)
    latest=None;checkpoint_seconds=0
    def checkpoint():
        nonlocal latest,checkpoint_seconds
        tick=time.perf_counter();payload=ctx.capture();expected=digest(payload)
        receipt=write_checkpoint(weights/f'restart-a{attempt:02d}-s{ctx.step:05d}.pt',payload,cap=640*2**20)
        del payload
        verified=read_checkpoint(receipt);assert digest(verified)==expected;del verified
        prior=latest;latest=dict(receipt=receipt,attempt=attempt,step=ctx.step,complete_state_digest=expected)
        replace(out/'latest.json',latest)
        if prior:retire(prior['receipt'],out/'deletion-journal.jsonl',latest,weights)
        checkpoint_seconds+=time.perf_counter()-tick
    if (out/'latest.json').exists():
        latest=json.loads((out/'latest.json').read_text());assert latest['attempt']==attempt-1
        # Automatic restart is limited to a recorded clean update boundary. Any
        # partial journal or lost work needs a separate evidence-grounded repair.
        previous=json.loads((out/f'stopped-{attempt-1:02d}.json').read_text())
        assert previous['checkpoint']==latest and previous['step']==latest['step']
        ctx.restore(read_checkpoint(latest['receipt']));assert digest(ctx.capture())==latest['complete_state_digest']
        put(out/f'recovery-{attempt:02d}.json',dict(exact_complete_state=True,checkpoint=latest,additional_training_targets=0))
    else:assert attempt==1;checkpoint()
    while ctx.step<config['horizon']:
        if stopped() or (pause_at is not None and ctx.step==pause_at):
            if latest['step']!=ctx.step:checkpoint()
            row=dict(status='stopped',step=ctx.step,checkpoint=latest)
            put(out/f'stopped-{attempt:02d}.json',row);return row
        check_stop();torch.mps.synchronize();tick=time.perf_counter()
        row=ctx.update();torch.mps.synchronize();seconds=time.perf_counter()-tick
        memory=torch.mps.driver_allocated_memory();assert memory<plan['mps_limit_bytes']
        append(out/f'steps-{attempt:02d}.jsonl',dict(**row,seconds=seconds,mps_bytes=memory))
        if ctx.step%128==0:
            progress=dict(pid=os.getpid(),slot=slot,step=ctx.step,horizon=config['horizon'],loss=row['loss'],seconds=seconds)
            replace(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        if ctx.step%plan['checkpoint_interval']==0:checkpoint()
        if ctx.step in plan['probe_steps']:
            path=out/f'probe-{ctx.step:05d}.json';assert not path.exists()
            put(path,score(ctx.model,data,plan['probe_documents']))
    if latest['step']!=ctx.step:checkpoint()
    assert ctx.ema and ctx.snapshots==1+(config['horizon']-ctx.ema_start)//10
    final=[]
    # Export both endpoints before discarding optimizer state, then load and
    # score the standalone artifacts through the actual deployment path.
    for endpoint in ('raw','h200'):
        receipt=export(ctx,weights/f'{endpoint}.pt',endpoint)
        final.append(dict(endpoint=endpoint,weights=receipt))
    context_config=ctx.config;snapshots=ctx.snapshots;history=ctx.history
    put(out/'training-history.json',history)
    del ctx;gc.collect();torch.mps.empty_cache()
    for e in final:
        check_stop();model=load(e['weights'],context_config,e['endpoint'],book)
        tick=time.perf_counter();scored=score(model,data);put(out/f"{e['endpoint']}-validation.json",scored)
        transferred=score(model,broader);put(out/f"{e['endpoint']}-broader.json",transferred)
        assert scored['count']==plan['validation_targets'] and transferred['count']==plan['broader_targets']
        e.update(ce=scored['ce'],count=scored['count'],sources=scored['sources'],broader_ce=transferred['ce'],score_seconds=time.perf_counter()-tick)
        if e['endpoint']=='h200':
            tick=time.perf_counter();generate(model,plan['generation_protocol'],out/'generation',e['weights'],binding)
            e['generation_seconds']=time.perf_counter()-tick
        del model;gc.collect();torch.mps.empty_cache()
    updates=[json.loads(line) for p in sorted(out.glob('steps-*.jsonl')) for line in p.read_text().splitlines()]
    assert [r['step'] for r in updates]==list(range(1,config['horizon']+1))
    result=dict(status='completed',binding=binding,slot=config,steps=config['horizon'],snapshots=snapshots,
        training_seconds=sum(r['seconds'] for r in updates),executed_targets=len(updates)*1024,
        checkpoint_seconds_this_attempt=checkpoint_seconds,endpoints=final,scientific_training=scientific,test_opened=False)
    put(out/'result.json',result)
    for e in final:assert sha(e['weights']['path'])==e['weights']['sha256']
    retire(latest['receipt'],out/'deletion-journal.jsonl',dict(result=str(out/'result.json'),sha256=sha(out/'result.json')),weights)
    replace(out/'latest.json',dict(status='retired_after_completion',result=str(out/'result.json')))
    return result

def run(args):
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    assert sha(PANEL/'plan.json')==args.binding
    plan=json.loads((PANEL/'plan.json').read_text());assert plan['scientific_training'] and 0<=args.slot<18
    out=PANEL/f'slot-{args.slot:02d}';weights=RUNS/f'slot-{args.slot:02d}'
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop();assert not (out/'result.json').exists()
        for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
        for i in range(args.slot):
            prior=PANEL/f'slot-{i:02d}';r=json.loads((prior/'review.json').read_text())
            assert r['accepted_execution'] and r['result_sha256']==sha(prior/'result.json')
        completed=sum(p.stat().st_size for p in RUNS.glob('slot-*/*.pt') if p.name in ('raw.pt','h200.pt'))
        assert shutil.disk_usage(ROOT).free>=plan['reserve_bytes']+max(0,plan['incremental_peak_bytes']-completed)
        spent=sum(json.loads(p.read_text())['elapsed_seconds'] for p in PANEL.glob('slot-*/guard-*.json'))
        slot_spent=sum(json.loads(p.read_text())['elapsed_seconds'] for p in out.glob('guard-*.json'))
        attempt=1+len(list(out.glob('attempt-*.json')))
        remaining=remaining_guard_seconds(plan,args.slot,slot_spent,spent,attempt)
        out.mkdir(exist_ok=True);weights.mkdir(parents=True,exist_ok=True)
        put(out/f'attempt-{attempt:02d}.json',dict(pid=os.getpid(),binding=args.binding,started=time.time()))
        statepath=ROOT/'research_state/general-rsi-state.json';state=json.loads(statepath.read_text())
        state.update(status='ACTIVE_SCIENTIFIC_WORKER',phase='cross_size_confirmation',active_worker_pid=os.getpid(),active_slot=args.slot,cross_size_panel_launched=True)
        replace(statepath,state);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            with CpuPrerequisiteGuard(out/f'guard-{attempt:02d}.json',ROOT,timeout_seconds=remaining,rss_limit_bytes=plan['rss_limit_bytes'],reserve_bytes=plan['reserve_bytes']):
                result=execute(plan,args.slot,args.binding,attempt,out,weights)
            for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
            state=json.loads(statepath.read_text());state.update(status='ACTIVE_SLOT_COMPLETED' if result['status']=='completed' else 'STOPPED',active_worker_pid=None,next_action='Review cross-size execution, endpoints, first outputs and cumulative costs. Continue fixed18-slot order after acceptance.')
            replace(statepath,state)
        except BaseException:
            put(out/f'failure-{attempt:02d}.json',dict(traceback=traceback.format_exc()))
            state=json.loads(statepath.read_text());state.update(status='ACTIVE_RECOVERY_REQUIRED',active_worker_pid=None,next_action='Diagnose preserved cross-size failure; no automatic replay of uncheckpointed updates or overwrite of endpoint/generation artifacts. Keep frozen seeds and gates.')
            replace(statepath,state);raise

def main():
    p=argparse.ArgumentParser();p.add_argument('binding');p.add_argument('slot',type=int);run(p.parse_args())
if __name__=='__main__':main()
