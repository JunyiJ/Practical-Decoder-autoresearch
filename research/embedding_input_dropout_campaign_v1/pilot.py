"""Finite20-update pilots after actual generated dependency closure and retirement."""
import argparse,json,shutil
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference,environment
from research.embedding_input_dropout_resume_v1.contract import config
from research.embedding_input_dropout_worker_v2.execution import sources,checked
from research.embedding_input_dropout_worker_v2.dispatch import launch
from research.embedding_input_dropout_worker_v2.validation import verify_artifact
from research.embedding_input_dropout_worker_v1.data import load
from research.final_ln_scale_quality_v3.support import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .rules import ARMS,PROBABILITIES,PAIR_CAP,RAW_CAP,H200_CAP,FLOOR,TARGET,ENTRY,DESIGN,capacity
E=ROOT/'research/tokenizer-evidence/embedding-input-dropout-preparation-001';RUN=ROOT/'research_runs/tokenizer-embedding-input-dropout-pilot-v1-001';PLAN=E/'pilot001-plan.json'
def publish(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def campaign_sources():return {str(p.relative_to(ROOT)):reference(p)['sha256'] for p in sorted(Path(__file__).parent.glob('*.py'))}
def bind():
    p=checked(reference(PLAN));require(not(RUN/'STOP').exists() and p['sources']==sources() and p['campaign_sources']==campaign_sources() and p['environment']==environment() and p['design']==DESIGN,'Frozen pilot binding differs')
    for ref in p['prerequisites'].values():checked(ref)
    return p

def prepare():
    require(not RUN.exists() and not PLAN.exists(),'New unattempted finite pilot pair required')
    integration=checked(reference(E/'pilot-integration001.json'));require(integration['status']=='passed_generated_dependencies_closed_for_finite_resource_pilots' and integration['sources']==sources() and integration['campaign_sources']==campaign_sources(),'Actual generated endpoint/source integration required')
    retired=checked(reference(E/'generated-retirement001/completed.json'));require(retired['status']=='completed' and retired['deleted_files']==4,'Actual exact generated retirement required;no conditional credit')
    census=checked(reference(E/'census001/result.json'));require(census['status']=='reserved_unused','Actual prospective seed nonuse required')
    ready=capacity(shutil.disk_usage(ROOT).free);require(ready['capacity_pass'],'Complete live pilot-pair budget and entry margin required')
    specs=[]
    for slot,(arm,prob) in enumerate(zip(ARMS,PROBABILITIES)):
        cfg=config(prob,steps=20,seed=census['seeds']['pilot'],lr=.00125);cfg.training.eval_seed=census['seeds']['evaluation'];base=RUN/f'evidence/pilots/{slot:02d}'
        cfg.training.checkpoint_dir=str(base/'checkpoints');cfg.training.metrics_file=str(base/'metrics.jsonl');cfg.training.summary_file=str(base/'worker.json')
        specs.append(dict(slot=slot,stage='pilots',arm=arm,config=plain(cfg),weight_cap=PAIR_CAP))
    plan=dict(kind='input_embedding_dropout_finite_resource_pilot_v1',root=str(RUN),runner=reference(__file__),sources=sources(),campaign_sources=campaign_sources(),environment=environment(),specs=specs,
        seconds=600,compare_cpu=True,design=DESIGN,seeds=census['seeds'],data=load(config(0.)).data_identity,required_free=FLOOR,cleanup_target=TARGET,
        prerequisites={n:reference(E/n) for n in ['pilot-integration001.json','generated-retirement001/completed.json','census001/result.json']},
        budget='Original2x560MiB slotcaps+16MiBmetadata+8GiBreserve+1267904512Bexternal+1458176Bentryheadroom;actualacceptedbytes only. Atomic temporary file counts against slotcap,no second diskcopy.',quality_launch_permitted=False)
    RUN.mkdir();publish(PLAN,plan);publish(E/'pilot001-freeze.json',dict(plan=reference(PLAN),quality_launch_permitted=False));print(json.dumps(dict(status='two_resource_pilots_frozen',capacity=ready)),flush=True)

def run_all(global_fd):
    plan=bind()
    with session_lock(RUN) as run_fd:
        writer=MetadataWriter(RUN,(global_fd,run_fd));used=0
        for spec in plan['specs']:
            slot=spec['slot'];anchor=E/f'pilot001-accepted-{slot:02d}.json';base=RUN/f'evidence/pilots/{slot:02d}'
            if anchor.exists():
                a=checked(checked(reference(anchor))['accepted']);w=checked(a['worker'])
                for n in ('parent','h200'):verify_artifact(w[n]);used+=w[n]['size']
                continue
            require(not base.exists(),'Attempted slot without clean acceptance is terminal;no retry')
            ready=capacity(shutil.disk_usage(ROOT).free,used)
            if not ready['capacity_pass']:print(json.dumps(dict(status='capacity_wait_before_unstarted_slot',slot=slot,capacity=ready)),flush=True);return
            row=launch(writer,reference(PLAN),spec,ready['required_free']);w=checked(row['worker']);used+=sum(w[n]['size'] for n in ('parent','h200'))
            publish(anchor,dict(accepted=reference(base/'accepted.json'),scientific_quality=False))

def review():
    p=bind();require(not(E/'pilot001-review.json').exists(),'Closed resource review cannot rerun');rows=[]
    for slot,arm in enumerate(ARMS):
        row=checked(checked(reference(E/f'pilot001-accepted-{slot:02d}.json'))['accepted']);w=checked(row['worker']);a=row['acceptance']
        require(a['status']=='accepted' and a['step']==20 and a['targets']==20480 and a['ema_snapshots']==8 and a['parameter_tensors']==88,'Full independent resource endpoint required')
        projection=2*1200/20*w['training_seconds'];require(projection<7200,'Original conservative time projection failed')
        for n,cap in (('parent',RAW_CAP),('h200',H200_CAP)):verify_artifact(w[n]);require(w[n]['size']<=cap,'Serialized artifact exceeds original cap')
        for name in ('worker_guard','parent_guard'):
            g=checked(row[name]);require(g['status']=='finished' and g['reason'] is None and g['observation']['peak_rss_bytes']<4*1024**3 and g['observation']['peak_observed_mps_driver_bytes']<6*1024**3,'Original actual resource guard failed')
        rows.append(dict(arm=arm,accepted=reference(RUN/f'evidence/pilots/{slot:02d}/accepted.json'),conservative_projected_training_seconds=projection,training_seconds=w['training_seconds'],weights={n:w[n] for n in ('parent','h200')},cpu_mps_ce_difference=a['cpu_mps_ce_difference']))
    require(p['sources']==sources() and p['campaign_sources']==campaign_sources(),'Pilot sources changed')
    publish(E/'pilot001-review.json',dict(status='two_resource_pilots_passed',plan=reference(PLAN),rows=rows,quality_frozen=False,
        next_action='Re-audit actual pilots and finite quality/selection/retention contracts;derive complete live all-output high-water before any scientific freeze. Retain both pilots until a NEW exact closure audit.'))
    publish(RUN/'STOP',dict(status='resource_pair_complete',all_commands_closed=True));print(json.dumps(dict(status='two_resource_pilots_passed',projections=[r['conservative_projected_training_seconds'] for r in rows])),flush=True)
if __name__=='__main__':
    torch.set_num_threads(4);parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run-all','review']);args=parser.parse_args()
    with session_lock(GLOBAL_LOCK.parent) as fd:
        idle();environment()
        try:
            if args.mode=='prepare':prepare()
            elif args.mode=='run-all':run_all(fd)
            else:review()
        except Exception as exc:
            if RUN.exists() and not(RUN/'STOP').exists():publish(RUN/'STOP',dict(error=repr(exc),mode=args.mode,closed=True,no_retry=True))
            raise
