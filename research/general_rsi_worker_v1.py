"""One frozen scientific slot; complete restart, bounded execution, final evidence."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / 'research/tokenizer-evidence/general-rsi-panel-001'
RUNS = ROOT / 'research_runs/general-rsi-panel-001'
STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def put(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def replace_json(path, value):
    path = Path(path); temp = path.with_suffix('.json.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def retire(receipt, journal, reason, dependency_receipt):
    """Exact owned transient checkpoint only; caller must hold research lock."""
    from research.general_rsi_training_v1 import sha
    path = Path(receipt['path'])
    assert path.is_relative_to(ROOT / 'research_runs') and path.resolve() == path
    assert (path.parent.name.startswith('slot-') and path.parent.parent==RUNS) or path.parent.name in ('general-rsi-qualification-001','general-rsi-pilot-001')
    assert not path.is_symlink() and path.is_file()
    st = path.stat()
    assert st.st_nlink == 1 and st.st_size == receipt['size'] and sha(path) == receipt['sha256']
    consumers = subprocess.run(['/usr/sbin/lsof','-t','--',str(path)],capture_output=True,text=True)
    assert consumers.returncode == 1 and not consumers.stdout and not consumers.stderr
    entry = dict(receipt=receipt, inode=st.st_ino, reason=reason,
                 closed_dependency=dependency_receipt, checked_no_open_consumers=True)
    with Path(journal).open('a') as stream:
        stream.write(json.dumps(dict(event='before_unlink',**entry))+'\n'); stream.flush(); os.fsync(stream.fileno())
    assert path.stat().st_ino == st.st_ino and path.stat().st_size == st.st_size
    path.unlink()
    with Path(journal).open('a') as stream:
        stream.write(json.dumps(dict(event='unlinked',**entry))+'\n'); stream.flush(); os.fsync(stream.fileno())


def generate(model, prompts):
    import torch
    import tiktoken
    from research.byte_input_cache_v4 import Session
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    enc = tiktoken.get_encoding('gpt2')
    mode = model.training; model.eval()
    generator = torch.Generator().manual_seed(42)
    samples=[]
    try:
        for decoding in ('greedy','temperature_0.8_top_k_40'):
            for prompt in prompts:
                ids=enc.encode(prompt);assert len(ids)+64<=128
                session=Session(model,1)
                inputs=torch.tensor([ids],dtype=torch.long,device=model.validate())
                continuation=[]
                for _ in range(64):
                    with torch.no_grad(),FP32RMSNormInference():
                        logits=session.logits(inputs,session.position)[0,-1].cpu()
                    if decoding=='greedy': token=int(logits.argmax())
                    else:
                        values,indices=torch.topk(logits/.8,40)
                        token=int(indices[torch.multinomial(torch.softmax(values,dim=-1),1,generator=generator)])
                    continuation.append(token)
                    if token==enc.eot_token:break
                    inputs=torch.tensor([[token]],dtype=torch.long,device=model.validate())
                samples.append(dict(prompt=prompt,decoding=decoding,seed=42,
                    tokens=continuation,continuation=enc.decode(continuation),
                    stop_reason='eos' if continuation[-1]==enc.eot_token else 'length_limit'))
    finally:model.train(mode)
    return samples


def launch(plan, slot, plan_sha, out, weights):
    import torch
    from research.general_rsi_training_v1 import Context, write_checkpoint, read_checkpoint, score, sha, clone_tree
    from research.byte_input_model_v6 import build_codebook
    assert torch.__version__==plan['torch_version'] and torch.backends.mps.is_available()
    torch.set_num_threads(4)
    config=dict(plan['slots'][slot])
    if config['phase']=='confirmation':
        selection=json.loads((PANEL/'selection.json').read_text())
        assert selection['plan_sha256']==plan_sha
        for record in selection['screen_results']:
            assert sha(record['path'])==record['sha256']
        config['lr']=selection['selected_lrs'][config['arm']]
    raw=[]
    for asset in plan['codebook_assets']:
        assert sha(asset['path'])==asset['sha256']
        raw.append(Path(asset['path']).read_bytes())
    book=build_codebook(*raw,'raw_bytes')
    context=Context(config['arm'],config['lr'],config['horizon'],config['roles'],book,plan_sha)
    latest=None
    if (out/'latest.json').exists():
        latest=json.loads((out/'latest.json').read_text())
        payload=read_checkpoint(latest);context.restore(payload);del payload
    elif config['generation']:
        put(out/'generation-initial.json',generate(context.model,plan['prompts']))
    def checkpoint():
        nonlocal latest
        path=weights/f'restart-{context.step:05d}.pt'
        payload=context.capture()
        receipt=write_checkpoint(path,payload)
        del payload
        # Full deserialization verifies the published archive before predecessor retirement.
        verified=read_checkpoint(receipt)
        assert verified['step']==context.step and verified['config']==context.config
        del verified
        prior=latest
        replace_json(out/'latest.json',receipt)
        latest=receipt
        if prior is not None:
            retire(prior,out/'deletion-journal.jsonl','Superseded transient restart; successor hash/read verified and format restart-qualified.',receipt)
    started=time.perf_counter()
    while context.step < config['horizon']:
        if STOP or json.loads((ROOT/'research_state/general-rsi-state.json').read_text()).get('stop_requested',False):
            if latest is None or not latest['path'].endswith(f'restart-{context.step:05d}.pt'):checkpoint()
            put(out/f'stopped-{context.step:05d}.json',dict(step=context.step,checkpoint=latest))
            return dict(status='stopped',step=context.step)
        tick=time.perf_counter(); row=context.update();torch.mps.synchronize()
        row=dict(row,seconds=time.perf_counter()-tick,mps_driver_bytes=torch.mps.driver_allocated_memory())
        assert row['mps_driver_bytes'] < plan['mps_limit_bytes']
        with (out/'steps.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n');stream.flush()
        if context.step%128==0:
            replace_json(out/'progress.json',dict(pid=os.getpid(),step=context.step,horizon=config['horizon'],last_loss=row['loss'],elapsed_seconds=time.perf_counter()-started))
            print(json.dumps(dict(slot=slot,step=context.step,loss=row['loss'],seconds=row['seconds'])),flush=True)
        if context.step%1024==0:
            checkpoint()
            if context.step < config['horizon']:
                diagnostic=score(context.model,context.data,None if config['phase']=='confirmation' and context.step==2048 else plan['probe_documents'])
                put(out/f'validation-{context.step:05d}.json',diagnostic)
    if latest is None or not latest['path'].endswith(f'restart-{context.step:05d}.pt'):checkpoint()
    final=[]
    for endpoint in ('raw','h200'):
        if endpoint=='h200':
            with torch.no_grad():
                for n,p in context.model.ordered_parameters():p.copy_(context.ema[n])
        payload=dict(kind='general_rsi_inference_weights_v1',plan_sha256=plan_sha,slot=config,
            model_identity=context.model.identity,endpoint=endpoint,completed_steps=context.step,
            parameters=clone_tree(dict(context.model.ordered_parameters())),
            immutable_buffers=context.model._own_expected_buffer_hashes)
        receipt=write_checkpoint(weights/f'{endpoint}.pt',payload,cap=128*1024**2)
        del payload
        scored=score(context.model,context.data)
        put(out/f'{endpoint}-validation.json',scored)
        if config['generation']:put(out/f'generation-{endpoint}.json',generate(context.model,plan['prompts']))
        final.append(dict(endpoint=endpoint,weights=receipt,ce=scored['ce'],count=scored['count'],sources=scored['sources']))
    result=dict(status='completed',plan_sha256=plan_sha,slot=config,steps=context.step,
        ema_snapshots=context.snapshots,elapsed_seconds=time.perf_counter()-started,endpoints=final,
        optimizer_restart_qualified=True,test_opened=False)
    put(out/'result.json',result)
    # The complete run is closed and both independent final inference endpoints are retained.
    for row in final:assert sha(row['weights']['path'])==row['weights']['sha256']
    retire(latest,out/'deletion-journal.jsonl','Completed run; final RAW/H200 standalone weights and metrics verified. No further optimizer consumer.',
           dict(path=str(out/'result.json'),sha256=sha(out/'result.json')))
    replace_json(out/'latest.json',dict(status='retired_after_completion',final_result=str(out/'result.json')))
    return result


def main():
    from research.general_rsi_training_v1 import sha
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    parser=argparse.ArgumentParser();parser.add_argument('plan_sha');parser.add_argument('slot',type=int);args=parser.parse_args()
    assert sha(PANEL/'plan.json')==args.plan_sha
    plan=json.loads((PANEL/'plan.json').read_text());assert 0<=args.slot<len(plan['slots'])
    out=PANEL/f'slot-{args.slot:02d}';weights=RUNS/f'slot-{args.slot:02d}'
    assert not (out/'result.json').exists()
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,path
        # Forecast is reduced only by final payloads already retained, never by unrun slots.
        completed_bytes=sum(p.stat().st_size for p in RUNS.glob('slot-*/*.pt')) if RUNS.exists() else 0
        remaining=max(0,plan['incremental_peak_bytes']-completed_bytes)
        assert shutil.disk_usage(ROOT).free>=plan['reserve_bytes']+remaining
        out.mkdir(exist_ok=True);weights.mkdir(parents=True,exist_ok=True)
        attempt=1+len(list(out.glob('attempt-*.json')))
        put(out/f'attempt-{attempt:02d}.json',dict(pid=os.getpid(),started=time.time(),plan_sha256=args.plan_sha))
        state_path=ROOT/'research_state/general-rsi-state.json'
        state=json.loads(state_path.read_text());assert not state.get('stop_requested',False)
        state.update(status='ACTIVE_SCIENTIFIC_WORKER',active_worker_pid=os.getpid(),active_slot=args.slot,scientific_panel_launched=True)
        replace_json(state_path,state)
        signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
        try:
            spent=sum(json.loads(p.read_text())['elapsed_seconds'] for p in PANEL.glob('slot-*/guard-*.json'))
            remaining_seconds=plan['tranche_seconds']-spent
            assert remaining_seconds>0, 'Frozen tranche time budget exhausted'
            with CpuPrerequisiteGuard(out/f'guard-{attempt:02d}.json',ROOT,timeout_seconds=min(plan['slot_timeout_seconds'],remaining_seconds),
                    rss_limit_bytes=plan['rss_limit_bytes'],reserve_bytes=plan['reserve_bytes']):
                result=launch(plan,args.slot,args.plan_sha,out,weights)
            for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,path
            state=json.loads(state_path.read_text());state.update(status='ACTIVE_SLOT_COMPLETED' if result['status']=='completed' else 'STOPPED',active_worker_pid=None,next_action='Review the latest slot result and guard. Continue frozen slot order; select LR only after all four screens. Never replace scientific seeds or alter gates.')
            replace_json(state_path,state)
            print(json.dumps(dict(status=result['status'],slot=args.slot)),flush=True)
        except BaseException:
            put(out/f'failure-{attempt:02d}.json',dict(traceback=traceback.format_exc()))
            state=json.loads(state_path.read_text());state.update(status='ACTIVE_RECOVERY_REQUIRED',active_worker_pid=None,next_action='Diagnose exact slot failure and preserve evidence; keep frozen roles, comparison and acceptance rules.')
            replace_json(state_path,state);raise


if __name__=='__main__':main()
