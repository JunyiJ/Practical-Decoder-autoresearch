"""Finite width256 GQA duration/schedule experiment after the reviewed GQA round."""
import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from ..models.tokenizer_architecture import architecture_contract
from .protocol import digest, file_digest
from .runner import atomic_json, session_lock, source_fingerprint
from .tokenizer_study import ROOT, runtime_identity, worker

RECIPE = ROOT/'research/tokenizer-evidence/gqa-001/adopted-width256-serving-recipe.json'
PROTOCOL = ROOT/'research/TOKENIZER_HORIZON_PROTOCOL.md'
DURATIONS = (1200, 1800, 3000)


def config(output, seed, steps, device='mps'):
    if type(steps) is not int or steps not in (*DURATIONS, 60):
        raise ValueError('Undeclared tokenizer duration')
    cfg = copy.deepcopy(json.loads(RECIPE.read_text())['config'])
    t = cfg['training']
    t.update(seed=seed, device=device, max_iters=steps, eval_every=300 if steps != 60 else 60,
             checkpoint_dir=str(output/'checkpoints'), metrics_file=str(output/'metrics.jsonl'),
             summary_file=str(output/'training-summary.json'))
    t['lr_schedule'].update(total_steps=steps, start_step=2*steps//3)
    t['weight_averaging'].update(total_steps=steps, start_step=2*steps//3, interval=10)
    assert t['learning_rate'] == .00125 and cfg['model']['dim'] == 256 and cfg['model']['attn_type'] == 'gqa'
    architecture_contract(cfg['model'])
    return cfg


def specs(plan, selection=None):
    if plan['pilot']:
        rows = [('pilot', 2791, 60)]
    else:
        rows = [('screen', 2801, n) for n in plan['durations']]
        if selection is not None and selection != 3000:
            rows += [('confirm',s,n) for i,s in enumerate(plan['confirm_seeds'])
                     for n in ((3000,selection) if i%2==0 else (selection,3000))]
    return [{'id': f'{stage}-gqa-w256-s{s}-n{n}','stage': stage,'family': 'gqa','width': 256,'seed': s,'steps': n,'lr': .00125}
            for stage,s,n in rows]


def choose(records, plan):
    rows = [r for r in records if r['stage']=='screen']
    if sorted((r['seed'],r['steps']) for r in rows) != [(2801,n) for n in sorted(plan['durations'])]:
        raise ValueError('Incomplete or duplicate duration screening')
    losses = {r['steps']:r['h200']['mean_ce'] for r in rows}
    candidates = [n for n in losses if n != 3000 and losses[n]-losses[3000] < -.02]
    return min(candidates,key=lambda n:(losses[n],n)) if candidates else 3000


def conclusion(records, selection, plan):
    if selection == 3000:
        return {'adopted': False,'reason': 'No shorter screen candidate met the frozen gain threshold.','intervals': []}
    from .coverage_eval import paired_blocks
    fresh = [r for r in records if r['stage']=='confirm']
    expected = sorted((s,n) for s in plan['confirm_seeds'] for n in (selection,3000))
    if sorted((r['seed'],r['steps']) for r in fresh) != expected:
        raise ValueError('Incomplete or duplicate fresh confirmation')
    arms = {n:{r['seed']:r['h200'] for r in records if r['stage']=='confirm' and r['steps']==n}
            for n in (selection,3000)}
    if any(sorted(a)!=plan['confirm_seeds'] for a in arms.values()):
        raise ValueError('Incomplete fresh confirmation')
    intervals = [paired_blocks(arms[selection],arms[3000],n) for n in (2048,4096,8192)]
    passes = (intervals[0]['mean_delta'] < -.02 and all(d<0 for d in intervals[0]['paired_seed_deltas'].values())
              and all(p['conditional_text_bootstrap_95pct'][1]<0 for p in intervals))
    return {'adopted': False,'quality_eligible': passes,'selected_steps': selection,'intervals': intervals,
                'reason': 'Eligibility only; complete artifact review before scoped schedule adoption. Test closed.'}


def prepare(output, pilot=False):
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit clean source before preparing a plan')
    output.mkdir(parents=True,exist_ok=False)
    cfg = config(output/'template',2791 if pilot else 2801,60 if pilot else 3000)
    plan = {'version': 1,'pilot': pilot,'source': source_fingerprint(),'runtime': runtime_identity(),
                'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'config': cfg,'durations': list(DURATIONS),'confirm_seeds': [2811,2812,2813],
                'recipe_sha256': file_digest(RECIPE),'protocol_sha256': file_digest(PROTOCOL),
                'corpus_sha256': file_digest(Path(cfg['data']['path'])),'job_seconds': 600 if pilot else 7200,
                'min_free_disk_mb': 8192,'max_output_mb': 1024 if pilot else 6144}
    if not pilot:
        resource = ROOT/'research/tokenizer-evidence/horizon-pilot-001/resource-review.json'
        review = json.loads(resource.read_text())
        assert review['status']=='passed' and review['source']==plan['source']
        assert review['projected_bytes_with_headroom']+8192*1024**2 < shutil.disk_usage(output).free
        plan['resource_review_sha256']=file_digest(resource)
    atomic_json(output/'plan.json',plan)
    atomic_json(output/'state.json',{'status': 'prepared','selection': None,'records': [],'plan_sha256': digest(plan)})
    return plan


def run(output):
    plan=json.loads((output/'plan.json').read_text());state=json.loads((output/'state.json').read_text())
    if state['status'] in ('completed','failed','paused'):return state
    if state['plan_sha256']!=digest(plan):raise ValueError('Frozen plan changed')
    def verify():
        if (source_fingerprint()!=plan['source'] or runtime_identity()!=plan['runtime']
                or file_digest(RECIPE)!=plan['recipe_sha256'] or file_digest(PROTOCOL)!=plan['protocol_sha256']
                or file_digest(Path(plan['config']['data']['path']))!=plan['corpus_sha256']):
            raise ValueError('Frozen source/runtime/recipe/protocol/corpus changed')
        if not plan['pilot'] and file_digest(ROOT/'research/tokenizer-evidence/horizon-pilot-001/resource-review.json')!=plan['resource_review_sha256']:
            raise ValueError('Frozen resource review changed')
    verify()
    def stop(signum, frame):(output/'STOP').write_text(f'Stop signal {signum}\n')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    lock=ROOT/'research_state/tokenizer-worker-lock';lock.mkdir(exist_ok=True)
    with session_lock(output) as study_fd,session_lock(lock) as worker_fd:
        while True:
            verify()
            if (output/'STOP').exists():
                state['status']='paused';atomic_json(output/'state.json',state);return state
            done={r['id'] for r in state['records']};pending=[r for r in specs(plan,state['selection']) if r['id'] not in done]
            if not pending:
                if not plan['pilot'] and state['selection'] is None:
                    state['selection']=choose(state['records'],plan)
                    atomic_json(output/'selection.json',{'selected_steps': state['selection']})
                    atomic_json(output/'state.json',state);continue
                state.update(status='completed',active=None,conclusion=None if plan['pilot'] else conclusion(state['records'],state['selection'],plan))
                atomic_json(output/'state.json',state);return state
            if shutil.disk_usage(output).free/1024**2<plan['min_free_disk_mb']:raise RuntimeError('Disk reserve reached')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())/1024**2>plan['max_output_mb']:raise RuntimeError('Round disk cap reached')
            spec=pending[0];trial=output/spec['id'];trial.mkdir(exist_ok=False)
            cfg=config(trial,spec['seed'],spec['steps'])
            atomic_json(trial/'request.json',{'spec': spec,'config': cfg,'result_path': str(trial/'result.json')})
            state.update(status='running',active=spec,controller_pid=os.getpid());atomic_json(output/'state.json',state)
            print('START '+spec['id'],flush=True)
            with (trial/'worker.log').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-m','src.research.tokenizer_horizon_study','worker','--output',str(trial/'request.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(study_fd,worker_fd))
                state['worker_pid']=proc.pid;atomic_json(output/'state.json',state)
                try:code=proc.wait(timeout=plan['job_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    code=124
            if code or not (trial/'result.json').exists():
                state['records'].append(dict(**spec,status='failed',exit_code=code));state['status']='failed'
                atomic_json(output/'state.json',state);return state
            result=json.loads((trial/'result.json').read_text());state['records'].append(result);state['active']=None
            atomic_json(output/'state.json',state);print(f'DONE {spec["id"]}: H200 CE={result["h200"]["mean_ce"]:.6f}',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','worker']);p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();output=a.output.resolve()
    if a.command=='prepare':prepare(output,a.pilot)
    elif a.command=='worker':worker(output)
    else:
        try:run(output)
        except Exception as error:
            atomic_json(output/'controller-error.json',{'error': type(error).__name__,'message': str(error)});raise


if __name__=='__main__':main()
