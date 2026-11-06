"""Three finite real-worker resource pilots and independent launch review."""
import argparse,json,shutil
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,config,plain,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_worker_v1.execution import sources,checked
from research.body_allocation_worker_v1.dispatch import launch
from research.body_allocation_worker_v1.data import load
from research.final_ln_scale_quality_v3.support import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from .rules import ARMS,WEIGHT_CAPS,RAW_CAPS,EXPORT_CAPS,FLOOR,TARGET,DESIGN,capacity
B=ROOT/'research/tokenizer-evidence/body-allocation-preparation-001';RUN=ROOT/'research_runs/tokenizer-body-allocation-pilot-v1-001';PLAN=B/'pilot001-plan.json'
def exclusive(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def campaign_sources():
    return {str(p.relative_to(ROOT)):reference(p)['sha256'] for p in sorted(Path(__file__).parent.glob('*.py'))}
def bind():
    plan=checked(reference(PLAN));require(not (RUN/'STOP').exists() and plan['sources']==sources()
        and plan['campaign_sources']==campaign_sources() and plan['environment']==environment()
        and plan['design']==DESIGN and plan['required_free']==FLOOR,'Pilot source/design/environment differs')
    for ref in plan['prerequisites'].values():checked(ref)
    return plan
def prepare():
    require(not RUN.exists() and not PLAN.exists(),'New pilot attempt required')
    integration=checked(reference(B/'integration001.json'));require(integration['status']=='generated_model_and_worker_dependencies_closed','Worker integration required')
    retirement=checked(reference(B/'generated-retirement001/completed.json'));require(retirement['status']=='completed','Actual prerequisite retirement required, no conditional credit')
    census=checked(reference(B/'census001/result.json'));require(census['status']=='reserved_unused','Unused seeds required')
    observed=capacity(shutil.disk_usage(ROOT).free);require(observed['capacity_pass'],'Complete prospective panel capacity required before pilots')
    specs=[]
    for slot,arm in enumerate(ARMS):
        cfg=config(arm,20,census['seeds']['pilot']);cfg.training.eval_seed=census['seeds']['evaluation'];base=RUN/f'evidence/pilots/{slot:02d}'
        cfg.training.checkpoint_dir=str(base/'checkpoints');cfg.training.metrics_file=str(base/'metrics.jsonl');cfg.training.summary_file=str(base/'worker.json')
        specs.append(dict(slot=slot,stage='pilots',arm=arm,config=plain(cfg),weight_cap=WEIGHT_CAPS[arm]))
    data=load(config('incumbent'),'cpu').data_identity
    plan=dict(kind='body_allocation_finite_resource_pilot_v1',root=str(RUN),runner=reference(__file__),sources=sources(),campaign_sources=campaign_sources(),environment=environment(),
        specs=specs,seconds=600,compare_cpu=True,design=DESIGN,seeds=census['seeds'],data=data,required_free=FLOOR,cleanup_target=TARGET,capacity=observed,
        prerequisites={n:reference(B/n) for n in ['integration001.json','generated-retirement001/completed.json','census001/result.json']},
        full_budget='5,972,688,896 retained weight high-water +268,435,456 metadata +1,267,904,512 measured external filesystem headroom +1,458,176 entry headroom +8,589,934,592 reserve. Streaming publication aliases one inode; no second disk serialization copy.',
        quality_launch_permitted=False,resource_projection='Require conservative 2×1200/20 times total measured pilot training-loop seconds <7200 for each arm, plus original actual worker/parent RSS/MPS/output guards.')
    RUN.mkdir();exclusive(PLAN,plan);exclusive(B/'pilot001-freeze.json',dict(plan=reference(PLAN),quality_launch_permitted=False));print(json.dumps(dict(status='resource_pilots_frozen',capacity=observed)),flush=True)
def run(slot):
    plan=bind();require(slot is not None,'Explicit unstarted slot required')
    for i in range(slot):checked(reference(B/f'pilot001-accepted-{i:02d}.json'))
    with session_lock(RUN) as r:
        writer=MetadataWriter(RUN,(global_fd,r));used=snapshot(RUN)['checkpoint_logical_bytes'];observed=capacity(shutil.disk_usage(ROOT).free,used)
        require(observed['capacity_pass'],'Complete remaining panel capacity required')
        row=launch(writer,reference(PLAN),plan['specs'][slot],observed['required_free'])
        exclusive(B/f'pilot001-accepted-{slot:02d}.json',dict(accepted=reference(RUN/f'evidence/pilots/{slot:02d}/accepted.json'),scientific_quality=False))
def review():
    plan=bind();require(not (B/'pilot001-review.json').exists(),'Closed resource review cannot rerun');rows=[]
    for slot,arm in enumerate(ARMS):
        row=checked(reference(RUN/f'evidence/pilots/{slot:02d}/accepted.json'));w=checked(row['worker']);a=row['acceptance']
        require(a['status']=='accepted' and a['step']==20 and a['targets']==20480 and a['ema_snapshots']==8,'Complete pilot required')
        projection=2*1200/20*w['training_seconds'];require(projection<7200,'Conservative original time gate failed')
        for name,cap in [('parent',RAW_CAPS[arm]),('h200',EXPORT_CAPS[arm])]:require(w[name]['size']<=cap,'Measured serialization exceeds frozen artifact cap')
        guards=[checked(row[k]) for k in ('worker_guard','parent_guard')]
        require(all(g['status']=='finished' and g['reason'] is None and g['observation']['peak_rss_bytes']<4*1024**3
            and g['observation']['peak_observed_mps_driver_bytes']<6*1024**3 for g in guards),'Actual pilot guard failed')
        mirror=B/f'pilot001-{arm}-evidence001';require(not mirror.exists(),'Existing evidence mirror');mirror.mkdir();mirrored=[]
        base=RUN/f'evidence/pilots/{slot:02d}'
        for p in sorted(base.rglob('*')):
            if p.is_file() and 'checkpoints' not in p.parts and not p.name.startswith('.'):
                to=mirror/p.relative_to(base);to.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,to);require(reference(p)['sha256']==reference(to)['sha256'],'Mirrored bytes differ');mirrored.append(reference(to))
        rows.append(dict(arm=arm,accepted=reference(base/'accepted.json'),projected_conservative_seconds=projection,training_seconds=w['training_seconds'],weights={n:w[n] for n in ('parent','h200')},metadata_mirror=mirrored,weight_copies=0))
    observed=capacity(shutil.disk_usage(ROOT).free,snapshot(RUN)['checkpoint_logical_bytes'])
    exclusive(B/'pilot001-review.json',dict(status='three_resource_pilots_passed',plan=reference(PLAN),rows=rows,capacity=observed,quality_frozen=False,
        next_action='Implement/freeze finite three-arm LR/selection/fresh confirmation and exact pilot/loser retirement; re-audit complete live capacity before every unstarted trajectory.'))
    print(json.dumps(dict(status='three_resource_pilots_passed',projections=[r['projected_conservative_seconds'] for r in rows],capacity=observed)),flush=True)
if __name__=='__main__':
    torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','review']);p.add_argument('--slot',type=int,choices=range(3));args=p.parse_args()
    with session_lock(GLOBAL_LOCK.parent) as global_fd:
        idle();environment()
        try:
            if args.mode=='prepare':prepare()
            elif args.mode=='run':run(args.slot)
            else:review()
        except Exception as exc:
            if RUN.exists() and not (RUN/'STOP').exists():exclusive(RUN/'STOP',dict(error=repr(exc),mode=args.mode,slot=args.slot,no_retry=True))
            raise
