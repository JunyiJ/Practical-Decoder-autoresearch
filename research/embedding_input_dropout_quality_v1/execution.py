"""Frozen finite LR screen and three-pair confirmation, serial and no rescue."""
import argparse,gc,json,shutil
from pathlib import Path
import numpy as np
import torch
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.embedding_input_dropout_worker_v2.execution import sources,checked,Guard
from research.embedding_input_dropout_worker_v2.dispatch import launch
from research.embedding_input_dropout_worker_v2.validation import verify_artifact
from research.embedding_input_dropout_campaign_v1.pilot import RUN as PILOT_ROOT,campaign_sources
from .contract import ARMS,DESIGN
from research.final_ln_scale_quality_v3.support import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from .contract import B,RUN,PLAN,FREEZE,FLOOR,TARGET,ENTRY,quality_sources,screen_specs,fresh_specs,select,weight_credit
from .retention import retire,publish
from .statistics import review as statistics_review
from .benchmark import measure

CONFIRM=B/'quality001-confirmation-plan.json';SELECTION=B/'quality001-selection.json'
def read_anchor(slot,*,retired=False):
    anchor=checked(reference(B/f'quality001-accepted-{slot:02d}.json'));row=checked(anchor['accepted'])
    require(anchor['original_freeze']==reference(FREEZE),'Accepted scientific freeze anchor differs')
    require(row['slot']==slot and row['acceptance']['status']=='accepted','Accepted prefix identity differs')
    worker=checked(row['worker']);require(worker['request']==row['request'],'Worker request identity differs')
    for k in ('parent','h200'):
        if Path(worker[k]['path']).exists():verify_artifact(worker[k])
        else:
            require(retired and slot<6,'Missing selected/fresh payload forbidden')
            out=B/'quality001-loser-retirement001';completed=checked(reference(out/'completed.json'));manifest=checked(completed['manifest'])
            require(any(r['path']==worker[k]['path'] and r['sha256']==worker[k]['sha256'] and r['size']==worker[k]['size'] for r in manifest['delete']),'Absent losing tensor lacks exact permanent-retirement receipt')
    return row,worker

def bind():
    seal=checked(reference(FREEZE));plan=checked(seal['plan'])
    require(seal['plan']==reference(PLAN) and plan['root']==str(RUN) and not (RUN/'STOP').exists(),'Open original scientific freeze required')
    require(plan['sources']==sources() and plan['quality_sources']==quality_sources() and plan['pilot_sources']==campaign_sources()
        and plan['environment']==environment() and plan['design']==DESIGN and plan['required_free']==FLOOR
        and plan['runner']==reference(__file__) and plan['seconds']==7200 and plan['compare_cpu'] is False,'Scientific source/environment/design differs')
    require(plan['specs']==screen_specs(plan['seeds']),'Screen configuration differs')
    for ref in plan['prerequisites'].values():checked(ref)
    require(plan['seeds']==checked(plan['prerequisites']['census001/result.json'])['seeds'],'Original prospective seeds differ')
    return plan

def used_weights():
    total=0
    for root in (PILOT_ROOT,RUN):
        if not root.exists():continue
        expected={}
        for p in root.glob('evidence/*/*/accepted.json'):
            if root==PILOT_ROOT:
                row=checked(reference(p));worker=checked(row['worker']);out=B/'quality001-pilot-retirement001'
            else:
                row,worker=read_anchor(int(p.parent.name),retired=True);out=B/'quality001-loser-retirement001'
            for k in ('parent','h200'):
                ref=worker[k];path=Path(ref['path'])
                if path.exists():verify_artifact(ref);expected[str(path.relative_to(root))]=ref['size']
                else:
                    receipt=checked(reference(out/'completed.json'));manifest=checked(receipt['manifest'])
                    require(any(r['path']==ref['path'] and r['sha256']==ref['sha256'] and r['size']==ref['size'] for r in manifest['delete']),'Missing panel payload lacks exact retirement evidence')
        seen=snapshot(root);actual={n:v['size'] for n,v in seen['files'].items() if v['group'].endswith('/weights')}
        total+=weight_credit(actual,expected)
    return total

def remaining():
    used=used_weights();free=shutil.disk_usage(ROOT).free;required=max(8*1024**3,FLOOR-used)
    return dict(free_bytes=free,actual_retained_panel_weight_bytes=used,required_free=required,cleanup_target=required+2*1024**3,
        wake_free=required+ENTRY,mandatory_deficit=max(0,required+ENTRY-free),target_deficit=max(0,required+2*1024**3-free),capacity_pass=free>=required+ENTRY)

def prepare():
    require(not RUN.exists() and not PLAN.exists() and not FREEZE.exists(),'New finite scientific attempt required')
    pilot=checked(reference(B/'pilot001-review.json'));require(pilot['status']=='two_resource_pilots_passed','All actual resource pilots required')
    finite=checked(reference(B/'quality-contracts001.json'));require(finite['status']=='passed' and finite['quality_sources']==quality_sources(),'Independent finite scientific contracts required')
    census=checked(reference(B/'census001/result.json'));require(census['status']=='reserved_unused','Original prospective nonuse census required')
    # Pilot seed/evaluation are used. Screen/fresh/bootstrap are not permitted in any
    # earlier numerical config, including saved cfg inside raw/checkpoint metadata.
    reserved=[census['seeds']['screen'],*census['seeds']['fresh']]
    for parent in (ROOT/'research_runs').iterdir():
        if not parent.is_dir():continue
        for cfg in parent.glob('evidence/*/*/config.json'):
            d=json.loads(cfg.read_text());require(d.get('training',{}).get('seed') not in reserved,'Reserved scientific seed already used')
    observed=remaining();require(observed['capacity_pass'],'Complete current scientific high-water budget required')
    plan=dict(kind='embedding_input_feature_dropout_finite_two_arm_scientific_screen_v1',root=str(RUN),runner=reference(__file__),sources=sources(),quality_sources=quality_sources(),pilot_sources=campaign_sources(),
        environment=environment(),specs=screen_specs(census['seeds']),seeds=census['seeds'],design=DESIGN,seconds=7200,compare_cpu=False,
        required_free=FLOOR,cleanup_target=TARGET,initial_capacity=observed,
        budget_note='Tested stage high-water8x560MiB=2pilots+6screens or2selected+6fresh;256MiBmetadata+64MiBsidecars+8GiBreserve+1267904512Bexternal+1458176Bentry. Only actual retained accepted current-panel bytes subtracted once;no future deletion/hardlink credit. Additional measured entry margin required before launch.',
        prerequisites={n:reference(B/n) for n in ['pilot001-review.json','pilot-integration001.json','generated-retirement001/completed.json','census001/result.json','quality-contracts001.json']})
    RUN.mkdir();publish(PLAN,plan);publish(FREEZE,dict(plan=reference(PLAN),scientific_design=DESIGN,phase='LR screen; fresh selection algorithm and seeds already fixed'))
    print(json.dumps(dict(status='scientific_comparison_frozen',freeze=reference(FREEZE),capacity=observed)),flush=True)

def prefix():
    seen=[]
    for slot in range(12):
        base=RUN/f'evidence/quality/{slot:02d}'
        if (base/'accepted.json').exists():seen.append(slot)
        elif base.exists():raise ValueError('Attempted incomplete slot cannot rerun; inspect actual process/STOP')
    require(seen==list(range(len(seen))),'Noncontiguous accepted prefix');return len(seen)

def select_screen(plan):
    rows=[]
    for slot in range(6):
        row,w=read_anchor(slot);require(w['config']==plan['specs'][slot]['config'],'Screen scientific config differs')
        rows.append(dict(slot=slot,arm=row['arm'],lr=w['config']['training']['learning_rate'],ce=row['acceptance']['full_validation']['ce'],accepted=reference(RUN/f'evidence/quality/{slot:02d}/accepted.json')))
    choice,audits=select(rows);publish(SELECTION,dict(kind='frozen_per_arm_LR_selection',plan=reference(PLAN),rows=rows,choices=choice,audits=audits,
        no_fresh_seed_used=True,no_extra_screen=True,only_adoption_candidate='input_feature_p10',control_retained_if_candidate_fails='input_zero'))
    return checked(reference(SELECTION))

def verify_selection(plan):
    chosen=checked(reference(SELECTION));rows=[]
    for slot in range(6):
        row,w=read_anchor(slot,retired=True);require(w['config']==plan['specs'][slot]['config'],'Original screen config differs')
        rows.append(dict(slot=slot,arm=row['arm'],lr=w['config']['training']['learning_rate'],ce=row['acceptance']['full_validation']['ce'],accepted=reference(RUN/f'evidence/quality/{slot:02d}/accepted.json')))
    choices,audits=select(rows)
    require(chosen['plan']==reference(PLAN) and chosen['rows']==rows and chosen['choices']==choices and chosen['audits']==audits,'Frozen original LR selection differs')
    return chosen

def bound_confirmation(plan):
    selection=verify_selection(plan);confirm=checked(reference(CONFIRM))
    require(confirm['specs']==fresh_specs(plan['seeds'],selection['choices']) and confirm['selection']==reference(SELECTION)
        and confirm['original_scientific_freeze']==reference(FREEZE) and confirm['sources']==plan['sources']
        and confirm['quality_sources']==plan['quality_sources'] and confirm['environment']==plan['environment']
        and confirm['root']==str(RUN) and confirm['design']==DESIGN and confirm['required_free']==FLOOR
        and confirm['seconds']==7200 and confirm['compare_cpu'] is False,'Resolved original fresh panel changed')
    return confirm

def retire_pilots(plan=None):
    # Source-identical accepted worker, finite scientific contracts and actual
    # pilot parent audits close every pilot tensor consumer. New trajectories
    # initialize fresh and never read these payloads.
    out=B/'quality001-pilot-retirement001'
    if (out/'completed.json').exists():return checked(reference(out/'completed.json'))
    bindings=[];refs=[reference(B/'quality-contracts001.json'),reference(B/'pilot001-review.json')]
    finite=checked(reference(B/'quality-contracts001.json'));require(finite['quality_sources']==quality_sources() and finite['pilot_dependencies_closed'] is True,'Actual pilot dependency closure required')
    if plan is not None:refs.append(reference(PLAN))
    for slot in range(2):
        row=checked(reference(PILOT_ROOT/f'evidence/pilots/{slot:02d}/accepted.json'));w=checked(row['worker'])
        require(row['acceptance']['status']=='accepted' and w['step']==20,'Actual complete pilot required')
        bindings.extend(w[n] for n in ('parent','h200'));refs.extend([reference(PILOT_ROOT/f'evidence/pilots/{slot:02d}/accepted.json'),row['worker']])
    with session_lock(PILOT_ROOT):
        require(checked(reference(PILOT_ROOT/'STOP'))['status']=='resource_pair_complete','Original completed pilot STOP must remain intact')
        refs.append(reference(PILOT_ROOT/'STOP'))
        with CpuPrerequisiteGuard(B/'quality001-pilot-retirement001-guard.json',ROOT,timeout_seconds=900,poll_seconds=.25):
            return retire(out,bindings,refs,kind='pilots')

def confirmation(plan):
    selection=verify_selection(plan);out=B/'quality001-loser-retirement001';selected_slots={v['selected_slot'] for v in selection['audits'].values()}
    require(len(selected_slots)==2,'One retained selected recipe per arm required')
    if not (out/'completed.json').exists():
        bindings=[];protected=[];refs=[reference(PLAN),reference(SELECTION)]
        for slot in range(6):
            row,w=read_anchor(slot);refs.extend([reference(RUN/f'evidence/quality/{slot:02d}/accepted.json'),row['worker']])
            if slot in selected_slots:protected.extend(w[n]['path'] for n in ('parent','h200'))
            else:bindings.extend(w[n] for n in ('parent','h200'))
        with CpuPrerequisiteGuard(B/'quality001-loser-retirement001-guard.json',ROOT,timeout_seconds=900,poll_seconds=.25):
            retire(out,bindings,refs,kind='losing_screens',selected=protected)
    observed=remaining();require(observed['capacity_pass'],'Complete remaining fresh confirmation budget required')
    confirm=dict(kind='resolved_original_scientific_panel_fresh_confirmation_v1',root=str(RUN),sources=plan['sources'],quality_sources=plan['quality_sources'],pilot_sources=plan['pilot_sources'],
        environment=plan['environment'],specs=fresh_specs(plan['seeds'],selection['choices']),seconds=7200,compare_cpu=False,
        original_scientific_freeze=reference(FREEZE),selection=reference(SELECTION),retirement=reference(out/'completed.json'),
        design=DESIGN,seeds=plan['seeds'],required_free=FLOOR,capacity=observed,provenance='Same prospectively declared two-arm study; no score pooling, reselection, seed change or extra tuning.')
    publish(CONFIRM,confirm);print(json.dumps(dict(status='selection_and_full_fresh_confirmation_frozen',choices=selection['choices'],capacity=observed)),flush=True)

def advance(global_fd):
    plan=bind()
    with session_lock(RUN) as r:
        writer=MetadataWriter(RUN,(global_fd,r));n=prefix()
        if n==0:retire_pilots(plan)
        if n==6:
            if not SELECTION.exists():select_screen(plan)
            if not CONFIRM.exists():confirmation(plan)
        if n==12:return False
        active_plan=plan if n<6 else bound_confirmation(plan);spec=active_plan['specs'][n if n<6 else n-6]
        for prior in range(n):
            row,w=read_anchor(prior,retired=True)
            expected=plan['specs'][prior] if prior<6 else active_plan['specs'][prior-6]
            require(w['config']==expected['config'] and row['acceptance']['step']==1200 and row['acceptance']['targets']==1228800
                and row['acceptance']['ema_snapshots']==41,'Prior accepted scientific endpoint differs')
        observed=remaining()
        if not observed['capacity_pass']:
            print(json.dumps(dict(status='capacity_wait_before_unattempted_slot',slot=n,capacity=observed)),flush=True)
            return False
        row=launch(writer,reference(PLAN if n<6 else CONFIRM),spec,observed['required_free'])
        publish(B/f'quality001-accepted-{n:02d}.json',dict(accepted=reference(RUN/f'evidence/quality/{n:02d}/accepted.json'),phase='screen' if n<6 else 'fresh',original_freeze=reference(FREEZE)))
        return True

def independent_statistics(pairs,seed,result):
    # Explicit per-draw grouping/reduction is separate from vectorized production.
    for control in ('input_zero',):
        for width,bound in zip((2048,4096,8192),result['contrasts'][control]['conditional_text_upper_bounds']):
            cells=[];counts=[]
            for start in range(0,34278,width):
                indices=list(range(start//2048,min(17,(start+width)//2048)))
                counts.append(min(width,34278-start));cells.append([sum(p['input_feature_p10']['block_loss_sums'][i]-p[control]['block_loss_sums'][i] for i in indices) for p in pairs])
            rng=np.random.Generator(np.random.PCG64(seed));values=[]
            for _ in range(2000):
                draw=rng.integers(0,len(cells),len(cells));denom=sum(counts[i] for i in draw)
                values.append(sum(sum(cells[i][j] for i in draw)/denom for j in range(3))/3)
            value=float(np.quantile(values,1-.05/3,method='linear'));require(abs(value-bound['upper'])<1e-12,'Independent bootstrap/grouping differs')

def final():
    plan=bind();require(prefix()==12 and not (B/'quality001-final-review.json').exists(),'Complete unreviewed three-pair panel required')
    selection=verify_selection(plan);confirm=bound_confirmation(plan);rows=[];pairs=[]
    for pair in range(3):
        values={}
        for slot in range(6+pair*2,8+pair*2):
            row,w=read_anchor(slot);require(w['config']==confirm['specs'][slot-6]['config'],'Fresh scientific configuration differs')
            require(w['config']['training']['seed']==plan['seeds']['fresh'][pair],'Fresh pairing differs');values[row['arm']]=row['acceptance']['full_validation'];rows.append(row)
        require(set(values)==set(ARMS),'Complete fresh pair required');pairs.append(values)
    with Guard(B/'quality001-final-review-guard.json',7200):
        result=statistics_review(pairs,plan['seeds']['bootstrap']);independent_statistics(pairs,plan['seeds']['bootstrap'],result)
        inference=measure(rows);publish(B/'quality001-fresh-inference.json',dict(rows=inference,all_fresh_pairs=True,selection=False,holdout_scored=False))
        require(sources()==plan['sources'] and quality_sources()==plan['quality_sources'],'Frozen source changed at closure')
    report=dict(status='complete_fresh_panel_reviewed',freeze=reference(FREEZE),selection=reference(SELECTION),confirmation=reference(CONFIRM),
        fresh_acceptances=[reference(RUN/f'evidence/quality/{i:02d}/accepted.json') for i in range(6,12)],statistics=result,
        independent_bootstrap_agreement_atol=1e-12,inference=reference(B/'quality001-fresh-inference.json'),adopted=False,
        next_action='Separate evidence/adoption review if all strict gates pass; otherwise retain incumbent. Then choose next evidence-qualified improvement using actual curves/stability/cost and primary literature.')
    publish(B/'quality001-final-review.json',report)
    publish(RUN/'STOP',dict(kind='completed_finite_scientific_study',review=reference(B/'quality001-final-review.json'),never_rerun_training_inference_review=True))
    print(json.dumps(dict(status='complete_fresh_panel_reviewed',all_strict_gates_pass=result['all_strict_gates_pass'],statistics=result)),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','advance','run-all','final','retire-pilots']);args=p.parse_args()
    def once(mode):
        with session_lock(GLOBAL_LOCK.parent) as g:
            idle();environment()
            try:
                if mode=='prepare':prepare();return False
                if mode=='retire-pilots':retire_pilots();return False
                if mode=='advance':return advance(g)
                if mode=='final':
                    with session_lock(RUN):final();return False
            except Exception as exc:
                if RUN.exists() and not (RUN/'STOP').exists():publish(RUN/'STOP',dict(error=repr(exc),mode=mode,no_retry=True))
                raise
    if args.mode=='run-all':
        while once('advance'):pass
        if not (RUN/'STOP').exists() and prefix()==12:once('final')
    else:once(args.mode)
