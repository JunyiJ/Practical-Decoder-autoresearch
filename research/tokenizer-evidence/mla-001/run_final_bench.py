"""Three serial, bounded MHA/MLA comparisons, only after completed training and audits."""
import json,os,signal,subprocess,sys,shutil
from pathlib import Path
from src.research.runner import session_lock,source_fingerprint,atomic_json
from src.research.tokenizer_study import runtime_identity
from src.research.protocol import file_digest,digest
root=Path.cwd();run=root/'research_runs/tokenizer-mla-001'
plan=json.loads((run/'plan.json').read_text());state=json.loads((run/'state.json').read_text())
assert state['status']=='completed' and len(state['records'])==10 and state['plan_sha256']==digest(plan)
assert source_fingerprint()==plan['source'] and runtime_identity()==plan['runtime']
assert file_digest(root/'research/TOKENIZER_MLA_PROTOCOL.md')==plan['protocol_sha256']
for r in state['records']:
    review=json.loads((root/'research/tokenizer-evidence/mla-001'/r['id']/'review.json').read_text())
    assert review['status']=='passed' and review['result_sha256']==file_digest(run/r['id']/'result.json')
out=run/'final-inference';out.mkdir(exist_ok=False)
progress=root/'research_state/tokenizer-quality-progress.json';d=json.loads(progress.read_text())
d.update(status='running_mla_final_inference',timing_controller_pid=os.getpid());atomic_json(progress,d)
with session_lock(root/'research_state/tokenizer-worker-lock') as fd:
    for seed in (2911,2912,2913):
        if (run/'STOP').exists():raise RuntimeError('Stopped before timing')
        assert shutil.disk_usage(root).free>8*1024**3 and source_fingerprint()==plan['source']
        rows={r['family']:r for r in state['records'] if r['stage']=='confirm' and r['seed']==seed}
        assert set(rows)=={'mha','mla'}
        for r in rows.values():assert file_digest(Path(r['h200']['checkpoint']))==r['h200']['checkpoint_sha256']
        name=f'w256-s{seed}';target=out/(name+'.json')
        args=[sys.executable,'-m','src.research.tokenizer_attention_bench','--mha',rows['mha']['h200']['checkpoint'],'--mla',rows['mla']['h200']['checkpoint'],'--output',str(target)]
        print('START '+name,flush=True)
        with (out/(name+'.log')).open('x') as log:
            proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(fd,))
            def stop(signum,frame):
                (run/'STOP').write_text('Stop requested during timing\n');os.killpg(proc.pid,signal.SIGTERM)
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            try:code=proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=124
        if code or not target.exists():
            atomic_json(out/'failure.json',dict(pair=name,exit_code=code));raise RuntimeError(f'{name} failed {code}')
        print('DONE '+name,flush=True)
    atomic_json(out/'completion.json',dict(status='completed',pairs=3,source_unchanged=source_fingerprint()==plan['source']))
