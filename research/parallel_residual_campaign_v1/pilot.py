"""Finite two-arm resource pilots using the already exercised real worker path."""
import argparse,copy,gc,hashlib,json,os,shutil,sys,time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.parallel_residual_campaign_v1 import evidence_contract as c,artifact_acceptance as a
from research.parallel_residual_worker_v2 import validation as v,inference as inf
from research.local_mix_v1.output import Sink,TRAJECTORY_CAP,RAW_CAP,EXPORT_CAP,WEIGHT_CAP
from research.parallel_residual_v1.runtime.train.train import run_training
from research.parallel_residual_v1.runtime.train.lr_schedule import schedule_contract
from research.parallel_residual_v1.runtime.train.weight_averaging import averaging_contract,export_averaged_checkpoint
from research.parallel_residual_v1.runtime.models.block_norm import contract as model_contract
from research.parallel_residual_worker_v2.validation_data import load as validation_data
from research.block_norm_v1 import campaign_contract as data_contract
from research.mla_score_quality_v1.support import idle as previous_idle
from research.block_norm_v1.fixture_dispatch import read,reference
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.block_norm_v1.pilot_execution import exclusive
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from src.research.tokenizer_wide_hybrid_process_v2 import run_process
B=c.ROOT/'research/tokenizer-evidence/parallel-residual-preparation-001'
ROOT=c.ROOT/'research_runs/tokenizer-parallel-residual-pilot-001'
from .rules import ARMS,TOPOLOGIES,LR,REQUIRED,QUALITY_DESIGN,require_capacity,validate_seeds
from research.parallel_residual_v1.integration import identity

FREEZE=B/'pilot001-freeze.json'
def idle(allowed=()):
    previous_idle(allowed)
    import subprocess
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        row=line.strip().split(None,1)
        if len(row)==2 and 'research.parallel_residual_campaign_v1' in row[1] and Path(row[1].split()[0]).name.lower().startswith('python'):
            assert int(row[0]) in {os.getpid(),*allowed},'Other parallel residual campaign worker live'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def clear():gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
class Guard(StrictMpsGuard):
    def __init__(self,path,seconds=600):
        assert 0<seconds<=600
        super().__init__(path,c.ROOT);self.limits['timeout_seconds']=seconds

def config(slot,seeds):
    assert type(slot) is int and slot in (0,1)
    validate_seeds(seeds)
    previous=read(read(ref(B/'trainer-worker003.json'))['plan'])
    cfg=copy.deepcopy(previous['configs'][slot]);adopted=data_contract.baseline()
    cfg['data']=copy.deepcopy(adopted['data']);cfg['model']['output_bias_prior']=copy.deepcopy(adopted['model']['output_bias_prior'])
    model_contract(cfg['model']);root=ROOT/f'evidence/pilots/{slot:02d}';t=cfg['training']
    t.update(seed=seeds['pilot'],eval_seed=seeds['evaluation'],max_iters=20,learning_rate=LR,eval_iters=10,eval_every=20,
        checkpoint_dir=str(root/'checkpoints'),metrics_file=str(root/'metrics.jsonl'),summary_file=str(root/'summary.json'))
    t['lr_schedule'].update(total_steps=20,start_step=13)
    t['weight_averaging'].update(total_steps=20,start_step=13,interval=1)
    assert identity(OmegaConf.create(cfg))['residual']==TOPOLOGIES[ARMS[slot]]
    return cfg

def bind():
    assert not (ROOT/'STOP').exists()
    seal=read(ref(FREEZE));plan=read(seal['plan'])
    assert plan['kind']=='fixed_lr_parallel_residual_two_resource_pilots_v1' and plan['quality_launch_permitted'] is False
    assert plan['root']==str(ROOT) and plan['runner']==ref(__file__)
    c.verify_sources(plan['sources']);assert plan['environment']==data_contract.environment() and plan['data']==data_contract.bind_data()
    assert plan['configs']==[config(i,plan['seeds']) for i in range(2)]
    assert plan['required_free']==REQUIRED and plan['limits']==data_contract.LIMITS
    assert plan['prospective_quality']==QUALITY_DESIGN
    for value in plan['prerequisites'].values():read(value)
    return plan

def prepare():
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with session_lock(GLOBAL_LOCK.parent):
        idle();assert not ROOT.exists() and not FREEZE.exists()
        with CpuPrerequisiteGuard(B/'pilot001-prepare-guard.json',c.ROOT,timeout_seconds=120):
            census=read(ref(B/'scientific-seed-census-001/result.json'));assert census['status']=='fresh_seeds_reserved' and not census['matches']
            assert sha(B/'scientific-seed-census-001/inventory.json')==census['inventory_sha256']
            prereqs={n:ref(B/n) for n in ['model001.json','trainer-worker003.json','trained-inference003.json','retention-boundary001.json','pilot-integration001.json','capacity001.json','scientific-seed-census-001/result.json']}
            assert read(prereqs['retention-boundary001.json'])['status']=='exact_fixed_lr_pilot_boundary_passed'
            assert read(prereqs['pilot-integration001.json'])['status']=='finite_pilot_config_and_parent_bridge_passed'
            sources=c.source_inventory();old=read(prereqs['trainer-worker003.json'])['sources'];assert all(sources.get(n)==h for n,h in old.items())
            configs=[config(i,census['seeds']) for i in range(2)];data=data_contract.bind_data();checks=[]
            for cfg in configs:
                o=OmegaConf.create(cfg);schedule=schedule_contract(o.training);ema=averaging_contract(o.training)
                assert schedule['total_steps']==ema['total_steps']==20 and schedule['start_step']==ema['start_step']==13
                d=validation_data(o);target=v.target_contract(d)
                assert set(d.splits)=={'train','val'} and target['targets']==data['validation_targets'] and target['target_ids_sha256']==data['validation_target_ids_sha256']
                assert target['target_bytes']==data['validation_target_bytes'] and target['window_counts']==data['validation_window_target_counts']
                checks.append(dict(targets=target['targets'],schedule=schedule,ema=ema))
            free=shutil.disk_usage(c.ROOT).free;require_capacity(free)
            plan=dict(kind='fixed_lr_parallel_residual_two_resource_pilots_v1',root=str(ROOT),runner=ref(__file__),sources=sources,configs=configs,seeds=census['seeds'],data=data,environment=data_contract.environment(),
                prerequisites=prereqs,limits=data_contract.LIMITS,raw_cap=RAW_CAP,export_cap=EXPORT_CAP,weight_cap=WEIGHT_CAP,trajectory_cap=TRAJECTORY_CAP,round_metadata=256*1024**2,required_free=REQUIRED,free_bytes=free,
                quality_launch_permitted=False,pilot_order=list(ARMS),steps=20,lr=LR,checks=checks,
                retention='Both successful pilots retained through independent acceptance, trained paired inference, resource review and final quality-source preparation; exact audited pilot retirement required before first fresh trajectory. No LR screens or selection. Every one of the six fresh trajectories remains retained.',
                prospective_quality=copy.deepcopy(QUALITY_DESIGN),
                failure_policy='Preserve failed/partial artifacts and source; STOP and no retry. No partial panel on capacity shortfall.',
                holdout='Train/validation evidence only; inherited training loader materializes test tokens but never scores test. Pilots excluded from scientific LR/quality selection.')
            exclusive(B/'pilot001-plan.json',plan);ROOT.mkdir();exclusive(FREEZE,dict(plan=ref(B/'pilot001-plan.json'),quality_launch_permitted=False))
            assert bind()==plan
        exclusive(B/'pilot001-preflight.json',dict(status='frozen_two_resource_pilots_only',freeze=ref(FREEZE),guard=ref(B/'pilot001-prepare-guard.json'),free_bytes=free,required_free=REQUIRED))
        print('finite resource pilot plan frozen; full quality plan remains closed',flush=True)
def worker(args):
    writer=MetadataWriter(ROOT,(args.global_fd,args.run_fd));base=f'evidence/pilots/{args.slot:02d}';root=ROOT/base
    rq=read(dict(path=str(root/'request.json'),sha256=args.request_sha));plan=read(rq['plan'])
    assert rq['parent_pid']==os.getppid() and rq['slot']==args.slot and plan['runner']==ref(__file__)
    c.verify_sources(plan['sources']);idle((os.getppid(),))
    assert data_contract.bind_data()==plan['data']
    with Guard(ROOT/f'control/worker-{args.slot:02d}-guard.json'):
        plain=plan['configs'][args.slot];cfg=OmegaConf.create(plain)
        writer.publish_json(base+'/config.json',plain);sink=Sink(writer,'pilots',args.slot,plain)
        try:summary=run_training(cfg,output_sink=sink)
        finally:sink.close()
        assert summary['completed_steps']==20 and summary['stop_reason']=='max_iters' and summary['device']=='mps'
        assert summary['weight_averaging']['snapshots']==8 and summary['final_metric']['tokens_processed']==20480
        clear();derived=root/'checkpoints/h200.pt'
        exported=export_averaged_checkpoint(sink.raw,derived,200,payload_writer=sink.averaged)
        writer.publish_json(base+'/export.json',exported);clear()
        files={k:ref(p) for k,p in dict(config=root/'config.json',parent=sink.raw,h200=derived,summary=root/'summary.json',metrics=root/'metrics.jsonl',export=root/'export.json').items()}
        request=dict(scope=a.SCOPE,sources=plan['sources'],config=plain,files=files)
        a.verify_request(request);binding={k:files[k] for k in ('config','parent','h200')}
        report=v.evaluate(binding,'mps');clear()
        writer.publish_json(base+'/worker.json',dict(request=ref(root/'request.json'),acceptance_request=request,validation=report,pid=os.getpid(),parent_pid=os.getppid()))
        c.verify_sources(plan['sources']);assert data_contract.bind_data()==plan['data']
    writer.publish_json(base+'/worker-guard.json',json.loads((ROOT/f'control/worker-{args.slot:02d}-guard.json').read_text()))
    print('bounded scientific resource pilot completed',ARMS[args.slot],flush=True)

def run(slot):
    assert type(slot) is int and slot in (0,1)
    with session_lock(GLOBAL_LOCK.parent) as global_fd:
        idle();plan=bind()
        with session_lock(ROOT) as run_fd:
            writer=MetadataWriter(ROOT,(global_fd,run_fd));basepath=f'evidence/pilots/{slot:02d}';root=ROOT/basepath
            assert not root.exists(),'No retry of pilot attempt'
            with Guard(ROOT/f'control/prefix-{slot:02d}-guard.json',120):
                for prior in range(slot):
                    p=ROOT/f'evidence/pilots/{prior:02d}';row=read(ref(p/'accepted.json'))
                    assert row['slot']==prior and row['plan']==ref(B/'pilot001-plan.json')
                    a.accept(row['acceptance_request'],row['validation']);clear()
            assert not any((ROOT/f'evidence/pilots/{i:02d}').exists() for i in range(slot+1,2))
            used=snapshot(ROOT)['checkpoint_logical_bytes'];require_capacity(shutil.disk_usage(ROOT).free,used)
            started=time.monotonic();writer.publish_json(basepath+'/request.json',dict(plan=ref(B/'pilot001-plan.json'),parent_pid=os.getpid(),slot=slot))
            argv=[sys.executable,'-m','research.parallel_residual_campaign_v1.pilot','worker','--slot',str(slot),'--global-fd',str(global_fd),'--run-fd',str(run_fd),'--request-sha',sha(root/'request.json')]
            process=run_process(writer,'pilots',slot,argv,c.ROOT,required_free_bytes=REQUIRED-used,output_limit_bytes=TRAJECTORY_CAP,timeout_seconds=600)
            assert process['status']=='completed' and read(ref(root/'worker-guard.json'))['status']=='finished'
            with Guard(ROOT/f'control/parent-{slot:02d}-guard.json',600-(time.monotonic()-started)):
                result=read(ref(root/'worker.json'));assert result['pid']==process['pid'] and result['parent_pid']==os.getpid()
                request=result['acceptance_request'];assert request['config']==plan['configs'][slot] and request['sources']==plan['sources']
                acceptance=a.accept(request,result['validation']);clear();assert bind()==plan
            writer.publish_json(basepath+'/parent-guard.json',read(ref(ROOT/f'control/parent-{slot:02d}-guard.json')))
            writer.publish_json(basepath+'/accepted.json',dict(slot=slot,arm=ARMS[slot],plan=ref(B/'pilot001-plan.json'),acceptance_request=request,validation=result['validation'],acceptance=acceptance,process=ref(root/'process/process.json'),parent_guard=ref(root/'parent-guard.json')))
            exclusive(B/f'pilot001-accepted-{slot:02d}.json',dict(accepted=ref(root/'accepted.json'),scientific_quality=False))
            print(json.dumps(dict(status='resource_pilot_accepted',slot=slot,arm=ARMS[slot],ce=result['validation']['ce'],audit=acceptance['audit'])),flush=True)

def review():
    with session_lock(GLOBAL_LOCK.parent):
        idle();plan=bind()
        with session_lock(ROOT) as fd, Guard(B/'pilot001-review-guard.json'):
            rows=[read(ref(ROOT/f'evidence/pilots/{i:02d}/accepted.json')) for i in range(2)]
            for row in rows:a.accept(row['acceptance_request'],row['validation']);clear()
            bindings={row['arm']:{k:row['acceptance_request']['files'][k] for k in ('config','parent','h200')} for row in rows}
            with Guard(B/'pilot001-inference-guard.json'):
                measured=inf.measure_pair(bindings,0);checked=inf.review(measured,bindings,0)
                exclusive(B/'pilot001-inference-measurements.json',measured);assert bind()==plan
            exclusive(B/'pilot001-review.json',dict(status='both_resource_pilots_and_inference_passed',plan=ref(B/'pilot001-plan.json'),accepted=[ref(ROOT/f'evidence/pilots/{i:02d}/accepted.json') for i in range(2)],inference=ref(B/'pilot001-inference-measurements.json'),inference_guard=ref(B/'pilot001-inference-guard.json'),review=checked,
                quality_adopted=False,quality_launch_permitted=False,free_bytes=shutil.disk_usage(ROOT).free,retained_weights=snapshot(ROOT)['checkpoint_logical_bytes'],
                next_action='Freeze finite quality source/plan using accepted resource budgets. Audit successful pilot retirement after dependencies close, before first fresh trajectory. Preserve every selected/fresh/failed pair.'))
            print('both resource pilots and paired inference passed; quality freeze next',flush=True)

if __name__=='__main__':
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','worker','review']);p.add_argument('--slot',type=int,choices=[0,1]);p.add_argument('--global-fd',type=int);p.add_argument('--run-fd',type=int);p.add_argument('--request-sha');args=p.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='worker':worker(args)
        elif args.mode=='run':run(args.slot)
        else:review()
    except Exception as exc:
        if args.mode!='worker':
            destination=B/f'pilot001-{args.mode}-{args.slot}-failure.json'
            if not destination.exists():exclusive(destination,dict(error=repr(exc),runner=ref(__file__),no_retry=True))
            if ROOT.exists() and not (ROOT/'STOP').exists():exclusive(ROOT/'STOP',dict(error=repr(exc),mode=args.mode,slot=args.slot,no_retry=True))
        raise
