"""Bounded optimizer transfer round: freeze width-64 LR choices before transfer.

Uses the same trusted worker, final-budget scorer, process limits and checkpoint
verification as team suites. No Codex agents and no held-out test access.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from statistics import mean, stdev
import subprocess
import time

from .protocol import digest, file_digest
from .runner import atomic_json, session_lock, validate_result
from .team.suite import (_launch, _request, source_digest, validate_suite,
                         _fingerprint, _preflight, _verify_checkpoints)

FAMILIES = {
    'adam_sp_conventional': ('adamw', 'sp', 'conventional', .1),
    'adam_sp_independent': ('adamw', 'sp', 'independent', .00006),
    'adam_mup_conventional': ('adamw', 'mup', 'conventional', .1),
    'adam_mup_independent': ('adamw', 'mup', 'independent', .00006),
    'muon_sp_independent': ('muon', 'sp', 'independent', .00006),
    'muon_mup_independent': ('muon', 'mup', 'independent', .00006),
    'reference_zero_decay': ('adamw', 'sp', 'conventional', 0.),
}


def make_plan(root, config):
    root=Path(root).resolve()
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise ValueError('Commit source changes before freezing the study')
    # All seven families share committed trainer and model implementations.
    baseline=copy.deepcopy(config)
    baseline['data']['path']=str(Path(baseline['data']['path']).resolve())
    plan={'version':1,'source_root':str(root),'source_sha256':source_digest(root),
          'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
          'data_sha256':file_digest(Path(baseline['data']['path'])),
          'config':baseline,'families':FAMILIES,'learning_rates':[.001,.002,.003],
          'seeds':[11,22,33],'widths':[64,128,256],'eval_seed':2026,
          'min_improvement':.005,'max_trials':105,'session_seconds':14400,
          'trial_seconds':600,'max_disk_mb':8192,
          'selection':'Lowest mean final width-64 validation CE within each family; ties use lower LR. Freeze all choices before any transfer.',
          'scope':'Adaptive development; fixed depth and tokens; held-out test closed. Not a convergence or universal optimum claim.'}
    return json.loads(json.dumps(plan))


def suite_spec(plan,family,lr):
    cfg=copy.deepcopy(plan['config'])
    optimizer,param,decay,wd=plan['families'][family]
    cfg['model'].update(parameterization=param,base_width=64)
    cfg['training'].update(optimizer=optimizer,decay_mode=decay,weight_decay=wd,learning_rate=lr)
    arm={'root':plan['source_root'],'source_sha256':plan['source_sha256'],'config':cfg}
    return validate_suite({'version':1,'hypothesis':f'Controlled optimizer family {family}',
        'baseline':arm,'candidate':copy.deepcopy(arm),
        'targets':[{'name':f'width-{w}','width':w,'n_heads':w//32} for w in plan['widths']],
        'seeds':plan['seeds'],'eval_seed':plan['eval_seed'],'min_improvement':plan['min_improvement'],
        'audit_test':False,'data_sha256':plan['data_sha256'],
        'limits':{'max_trials':plan['max_trials'],'session_seconds':plan['session_seconds'],
                  'trial_seconds':plan['trial_seconds'],'max_disk_mb':plan['max_disk_mb'],
                  'max_failures':1,'max_consecutive_failures':1}})


def local_jobs(plan):
    jobs=[]
    # Interleave families by seed and LR to reduce monotonic host-time confounding.
    for lr in plan['learning_rates']:
        for seed in plan['seeds']:
            families=list(plan['families'])
            if seed==22: families.reverse()
            for family in families:
                if family=='reference_zero_decay' and lr!=.002: continue
                jobs.append((family,lr,64,seed))
    return jobs


def freeze_selection(plan,records):
    choices={}
    for family in plan['families']:
        scores=[]
        lrs=[.002] if family=='reference_zero_decay' else plan['learning_rates']
        for lr in lrs:
            rows=[r for r in records if r['family']==family and r['lr']==lr and r['width']==64]
            if sorted(r['seed'] for r in rows)!=sorted(plan['seeds']) or any(r['status']!='completed' for r in rows):
                raise ValueError('Cannot select from incomplete local evidence')
            scores.append({'lr':lr,'mean_ce':mean(r['metric'] for r in rows)})
        selected=min(scores,key=lambda s:(s['mean_ce'],s['lr']))
        choices[family]={**selected,'grid':scores,'at_grid_boundary':selected['lr'] in (min(lrs),max(lrs)) if len(lrs)>1 else False}
    return {'choices':choices,'local_evidence_sha256':digest(records),'selected_by':'width-64 only',
            'plan_sha256':digest(plan)}


def summarize(plan,state):
    choices=state['selection']['choices']; records=state['records']; result={}
    for family,choice in choices.items():
        result[family]={'selected_lr':choice['lr'],'at_grid_boundary':choice['at_grid_boundary'],'widths':{}}
        for width in plan['widths']:
            rows=[r for r in records if r['family']==family and r['lr']==choice['lr'] and r['width']==width]
            refs=[r for r in records if r['family']=='reference_zero_decay' and r['width']==width]
            if len(rows)!=len(plan['seeds']) or len(refs)!=len(plan['seeds']): raise ValueError('Missing paired results')
            refs={r['seed']:r for r in refs}; deltas=[r['metric']-refs[r['seed']]['metric'] for r in rows]
            result[family]['widths'][str(width)]={
                'mean_ce':mean(r['metric'] for r in rows),'mean_delta_vs_reference':mean(deltas),
                'sample_sd_delta':stdev(deltas),'paired_deltas':dict(zip([str(r['seed']) for r in rows],deltas)),
                'supported_vs_reference':mean(deltas)<-plan['min_improvement'] and all(d<0 for d in deltas),
                'mean_training_seconds':mean(r['training_time_sec'] for r in rows),
                'mean_peak_process_rss_mb':mean(r['rss_mb'] for r in rows)}
    return {'version':1,'plan_sha256':digest(plan),'selection':state['selection'],'families':result,
            'records':records,'holdout_consumed':False,'scope':plan['scope'],
            'limitations':['Only three base learning rates; boundary winners need wider-grid testing.',
              'Decay coefficient is matched at the historical baseline LR; it is not separately tuned per optimizer.',
              'Repeated development seeds and corpus; this is not independent confirmation.',
              'Muon is the RMS-matched hybrid; muP cancels its sqrt(width) factor for hidden matrices.',
              'Full training horizons are fixed, not convergence- or compute-matched.']}


def run(plan,output,*,suite_builder=suite_spec,job_builder=local_jobs,
        selection_builder=freeze_selection,report_builder=summarize):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    with session_lock(output) as lock_fd:
        path=output/'state.json'
        if path.exists():
            state=json.loads(path.read_text())
            if state['plan_sha256']!=digest(plan): raise ValueError('Frozen study plan changed')
            if any(r['status']!='completed' for r in state['records']):
                raise ValueError('Preserved interrupted/failed trial requires inspection; no retry')
            if state['status']=='failed': raise ValueError('Failed study requires inspection')
            if state['status']=='completed': return state
        else:
            state={'plan_sha256':digest(plan),'records':[],'selection':None,'status':'running','active_seconds':0}
        prior=state['active_seconds']; start=time.monotonic()
        def save(reserve=0):
            state['active_seconds']=prior+time.monotonic()-start+reserve
            atomic_json(path,state)
        def trial(job):
            family,lr,width,seed=job
            tid=f'{family}_lr{lr:g}_w{width}_s{seed}'
            if any(r['trial_id']==tid for r in state['records']): return
            if (output/'STOP').exists(): raise InterruptedError('Human stop requested')
            if len(state['records'])>=plan['max_trials']: raise ValueError('Trial budget exhausted')
            remaining=plan['session_seconds']-prior-(time.monotonic()-start)
            if remaining<=0: raise ValueError('Study time budget exhausted')
            spec=suite_builder(plan,family,lr); _preflight(spec); contract=_fingerprint(spec)
            if state.setdefault('runtime',contract['runtime']) != contract['runtime']:
                raise ValueError('Study runtime changed')
            _verify_checkpoints(state['records'],output)
            directory=output/'trials'/tid; directory.mkdir(parents=True,exist_ok=False)
            row={'family':family,'lr':lr,'width':width,'seed':seed,'trial_id':tid,
                 'arm':'candidate','stage':'local' if width==64 else 'transfer',
                 'target':f'width-{width}','n_heads':width//32,'status':'running'}
            request=_request(spec,row,directory); atomic_json(directory/'request.json',request)
            state['records'].append(row); allowance=min(remaining,plan['trial_seconds']); save(allowance)
            before=time.monotonic()
            try:
                status,error=_launch(directory/'request.json',output,allowance,plan['max_disk_mb'],lock_fd,root=plan['source_root'])
                row.update(status=status,error=error)
                if _fingerprint(spec)!=contract: raise ValueError('Evidence changed during trial')
                if status!='completed': raise RuntimeError(f'{tid}: {error}')
                raw=json.loads((directory/'result.json').read_text()); cfg=request['config']
                checked=validate_result(raw,request,{'device':cfg['training']['device'],
                     'training':{**cfg['training'],'block_size':cfg['model']['block_size']}},directory)
                if raw.get('runtime')!={'torch':contract['runtime']['packages']['torch'],
                    'cpu_threads':request['cpu_threads'],'requested_device':cfg['training']['device']}:
                    raise ValueError('Worker runtime changed')
                row.update(checked)
            except BaseException as exc:
                row.update(status='failed',error=str(exc)); raise
            finally:
                row['duration_seconds']=time.monotonic()-before; save()
        try:
            for job in job_builder(plan): trial(job)
            local=[r for r in state['records'] if r['width']==64]
            selected=selection_builder(plan,local)
            if state['selection'] is None:
                state['selection']=selected
                atomic_json(output/'selection.json',selected)
                os.chmod(output/'selection.json',0o444)
                save()
            elif state['selection']!=selected or json.loads((output/'selection.json').read_text())!=selected:
                raise ValueError('Frozen small-width selection changed')
            for width in plan['widths'][1:]:
                for seed_index,seed in enumerate(plan['seeds']):
                    families=list(plan['families'])
                    if seed_index % 2: families.reverse()
                    for family in families:
                        trial((family,state['selection']['choices'][family]['lr'],width,seed))
            _verify_checkpoints(state['records'],output)
            atomic_json(output/'report.json',report_builder(plan,state))
            state['status']='completed'
        except BaseException as exc:
            state['status']='paused' if isinstance(exc,InterruptedError) else 'failed'
            state['error']=str(exc); raise
        finally: save()
        return state


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-training',action='store_true')
    args=p.parse_args()
    if not args.allow_training: p.error('Explicit --allow-training required')
    plan=json.loads(args.plan.read_text())
    if file_digest(Path(plan['config']['data']['path']))!=plan['data_sha256']: raise ValueError('Corpus changed')
    run(plan,args.output)

if __name__=='__main__': main()
