"""Finite MHA/GQA comparison on the adopted GPT-2 backbone."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from .protocol import digest, file_digest
from .runner import atomic_json, session_lock, source_fingerprint
from .tokenizer_study import ROOT, make_config, runtime_identity, trial_config, worker
from ..models.tokenizer_architecture import FIXED, architecture_contract


def config(output, family='mha', width=64, seed=2701, lr=.0025, steps=3000, device='mps'):
    cfg = make_config(output, width, seed, lr, steps, device)
    prior = cfg['model']['output_bias_prior']
    cfg['model'] = dict(FIXED, dim=width, n_heads=width//32, attn_type=family,
                        attention_width_policy='mha_head32' if family == 'mha' else 'gqa_half_head32',
                        depth_width_policy='depth4_unscaled' if family == 'mha' else 'depth4_gqa_unscaled',
                        output_bias_prior=prior)
    architecture_contract(cfg['model'])
    return cfg


def specs(plan, selection=None):
    if plan['pilot']:
        rows = [('pilot', w, 2691, f, .0025) for w in plan['widths'] for f in ('mha','gqa')]
    else:
        rows = [('screen',64,s,f,lr) for i,s in enumerate(plan['screen_seeds'])
                for f,lr in ([(f,lr) for f in ('mha','gqa') for lr in plan['learning_rates']][::1 if i%2==0 else -1])]
        if selection is not None:
            rows += [('confirm',w,s,f,selection[f]) for w in plan['widths']
                     for i,s in enumerate(plan['confirm_seeds']) for f in (('mha','gqa') if i%2==0 else ('gqa','mha'))]
    return [dict(id=f'{stage}-{f}-w{w}-s{s}-lr{lr:g}',stage=stage,family=f,width=w,seed=s,lr=lr)
            for stage,w,s,f,lr in rows]


def choose(records, plan):
    chosen = {}
    for family in ('mha','gqa'):
        means = {}
        for lr in plan['learning_rates']:
            rows = [r for r in records if r['stage']=='screen' and r['family']==family and r['lr']==lr]
            if sorted(r['seed'] for r in rows) != sorted(plan['screen_seeds']):
                raise ValueError('Incomplete or duplicate family screen')
            means[lr] = sum(r['h200']['mean_ce'] for r in rows)/len(rows)
        chosen[family] = min(means, key=lambda lr:(means[lr],lr))
    return chosen


def quality_conclusion(records, plan):
    from .coverage_eval import paired_blocks
    rows = []
    for width in plan['widths']:
        arms = {f:{r['seed']:r['h200'] for r in records if r['stage']=='confirm' and r['width']==width and r['family']==f}
                for f in ('mha','gqa')}
        if any(sorted(a)!=sorted(plan['confirm_seeds']) for a in arms.values()):
            raise ValueError('Incomplete confirmation')
        intervals = [paired_blocks(arms['gqa'],arms['mha'],n) for n in (2048,4096,8192)]
        neutral = (intervals[0]['mean_delta'] <= plan['neutrality_margin']
                   and all(x<=plan['neutrality_margin'] for x in intervals[0]['paired_seed_deltas'].values())
                   and all(x['conditional_text_bootstrap_95pct'][1]<=plan['neutrality_margin'] for x in intervals))
        rows.append(dict(width=width,quality_eligible=neutral,intervals=intervals))
    return dict(comparisons=rows,adopted=False,reason='Quality gate only; cached correctness and measured inference-cost gate plus artifact review still required.')


def prepare(output, pilot=False):
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit clean source before freezing a plan')
    output.mkdir(parents=True,exist_ok=False)
    recipe = ROOT/'research/tokenizer-evidence/quality-001/adopted-recipe.json'
    plan = dict(version=1,output=str(output),pilot=pilot,source=source_fingerprint(),
                runtime=runtime_identity(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                config=config(output/'template'),learning_rates=[.0025,.00125],widths=[64,256],
                screen_seeds=[2701,2702],confirm_seeds=[2711,2712,2713],
                steps=40 if pilot else 3000,neutrality_margin=.01,min_cache_reduction=.4,
                job_seconds=600 if pilot else 7200,min_free_disk_mb=8192,max_output_mb=8192,
                adopted_recipe_sha256=file_digest(recipe),protocol_sha256=file_digest(ROOT/'research/TOKENIZER_GQA_PROTOCOL.md'))
    plan['corpus_sha256']=file_digest(Path(plan['config']['data']['path']))
    atomic_json(output/'plan.json',plan)
    atomic_json(output/'state.json',dict(status='prepared',selection=None,records=[],plan_sha256=digest(plan)))
    return plan


def run(output):
    plan=json.loads((output/'plan.json').read_text());state=json.loads((output/'state.json').read_text())
    if state['status'] in ('completed','failed','paused'):
        return state
    if state['plan_sha256']!=digest(plan) or source_fingerprint()!=plan['source'] or runtime_identity()!=plan['runtime']:
        raise ValueError('Frozen plan/source/runtime changed')
    if (file_digest(ROOT/'research/tokenizer-evidence/quality-001/adopted-recipe.json')!=plan['adopted_recipe_sha256']
            or file_digest(ROOT/'research/TOKENIZER_GQA_PROTOCOL.md')!=plan['protocol_sha256']):
        raise ValueError('Frozen prior/protocol changed')
    def stop(signum, frame):
        (output/'STOP').write_text(f'Stop signal {signum}\n')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    lock=ROOT/'research_state/tokenizer-worker-lock';lock.mkdir(exist_ok=True)
    with session_lock(output) as study_fd,session_lock(lock) as worker_fd:
        while True:
            if (output/'STOP').exists():
                state['status']='paused';atomic_json(output/'state.json',state);return state
            if source_fingerprint()!=plan['source'] or file_digest(Path(plan['config']['data']['path']))!=plan['corpus_sha256']:
                raise ValueError('Frozen source/corpus changed')
            queue=specs(plan,state['selection']);done={r['id'] for r in state['records']}
            pending=[r for r in queue if r['id'] not in done]
            if not pending:
                if not plan['pilot'] and state['selection'] is None:
                    state['selection']=choose(state['records'],plan);atomic_json(output/'selection.json',state['selection'])
                    atomic_json(output/'state.json',state);continue
                state.update(status='completed',active=None,conclusion=None if plan['pilot'] else quality_conclusion(state['records'],plan))
                atomic_json(output/'state.json',state);return state
            if shutil.disk_usage(output).free/1024**2<plan['min_free_disk_mb']:
                raise RuntimeError('Disk reserve reached')
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())/1024**2>plan['max_output_mb']:
                raise RuntimeError('Round disk cap reached')
            spec=pending[0];trial=output/spec['id'];trial.mkdir(exist_ok=False)
            base=copy.deepcopy(plan['config']);f=spec['family']
            base['model'].update(attn_type=f,attention_width_policy='mha_head32' if f=='mha' else 'gqa_half_head32',
                                 depth_width_policy='depth4_unscaled' if f=='mha' else 'depth4_gqa_unscaled')
            cfg=trial_config(base,trial,spec['width'],spec['seed'],spec['lr'],plan['steps'])
            architecture_contract(cfg['model'])
            atomic_json(trial/'request.json',dict(spec=spec,config=cfg,result_path=str(trial/'result.json')))
            state.update(status='running',active=spec,controller_pid=os.getpid());atomic_json(output/'state.json',state)
            print('START '+spec['id'],flush=True)
            with (trial/'worker.log').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-m','src.research.tokenizer_attention_study','worker','--output',str(trial/'request.json')],
                                      cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(study_fd,worker_fd))
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
            atomic_json(output/'state.json',state)
            print(f'DONE {spec["id"]}: H200 CE={result["h200"]["mean_ce"]:.6f}',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','worker']);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();output=args.output.resolve()
    if args.command=='prepare':prepare(output,args.pilot)
    elif args.command=='run':
        try:run(output)
        except Exception as error:
            atomic_json(output/'controller-error.json',dict(error=type(error).__name__,message=str(error)))
            raise
    else:worker(output)


if __name__=='__main__':main()
