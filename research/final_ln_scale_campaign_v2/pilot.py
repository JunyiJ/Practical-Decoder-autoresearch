"""Finite two-arm resource pilots using the already exercised real worker path."""
import argparse,copy,gc,hashlib,json,os,shutil,sys,time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.final_ln_scale_campaign_v2 import evidence_contract as c,artifact_acceptance as a
from research.final_ln_scale_worker_v1 import validation as v,inference as inf
from research.local_mix_v1.output import Sink,TRAJECTORY_CAP,RAW_CAP,EXPORT_CAP,WEIGHT_CAP
from research.final_ln_scale_worker_v1.training import run_training
from research.final_ln_scale_model_v1.runtime.train.lr_schedule import schedule_contract
from research.final_ln_scale_model_v1.runtime.train.weight_averaging import averaging_contract,export_averaged_checkpoint
from research.final_ln_scale_model_v1.runtime.models.block_norm import contract as model_contract
from research.final_ln_scale_worker_v1.validation_data import load as validation_data
from research.block_norm_v1 import campaign_contract as data_contract
from research.mla_score_quality_v1.support import idle as previous_idle
from research.block_norm_v1.fixture_dispatch import read,reference
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.block_norm_v1.pilot_execution import exclusive
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from src.research.tokenizer_wide_hybrid_process_v2 import run_process
B=c.ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001'
ROOT=c.ROOT/'research_runs/tokenizer-final-ln-scale-pilot-002'
from research.final_ln_scale_campaign_v1.rules import ARMS,DECAYS,COOLDOWNS,PILOT_STARTS,THRESHOLDS,BETAS,SCALE_KINDS,LR,REQUIRED,QUALITY_DESIGN,require_capacity,validate_seeds
from research.final_ln_scale_model_v1.integration import identity

FREEZE=B/'pilot002-freeze.json'
OLDROOT=c.ROOT/'research_runs/tokenizer-final-ln-scale-pilot-001'
def idle(allowed=()):
    previous_idle(allowed)
    import subprocess
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        row=line.strip().split(None,1)
        if len(row)==2 and 'research.final_ln_scale_campaign_v2' in row[1] and Path(row[1].split()[0]).name.lower().startswith('python'):
            assert int(row[0]) in {os.getpid(),*allowed},'Other final-scale campaign worker live'
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
    old=read(ref(B/'pilot001-plan.json'))
    assert seeds==old['seeds']
    cfg=copy.deepcopy(old['configs'][slot])
    if slot==1:
        root=ROOT/'evidence/pilots/01'
        cfg['training'].update(checkpoint_dir=str(root/'checkpoints'),metrics_file=str(root/'metrics.jsonl'),summary_file=str(root/'summary.json'))
    identity(OmegaConf.create(cfg));return cfg

def request_for(row,plan):
    # Resolve immutable worker bytes, then apply only the explicitly additive source bridge.
    worker=read(row['worker']);request=copy.deepcopy(worker['acceptance_request'])
    assert all(plan['sources'].get(n)==h for n,h in request['sources'].items())
    request['sources']=plan['sources']
    assert c.fingerprint(request)==row['acceptance_request_sha256']
    return request

def used_weights():
    return read(ref(B/'pilot002-plan.json'))['imported_weight_bytes']+snapshot(ROOT)['checkpoint_logical_bytes']

def bind():
    assert not (ROOT/'STOP').exists()
    seal=read(ref(FREEZE));plan=read(seal['plan'])
    assert plan['kind']=='fixed_lr_final_ln_scale_imported_control_resource_pilot_v2' and plan['quality_launch_permitted'] is False
    assert plan['root']==str(ROOT) and plan['runner']==ref(__file__)
    c.verify_sources(plan['sources']);assert plan['environment']==data_contract.environment() and plan['data']==data_contract.bind_data()
    assert plan['configs']==[config(i,plan['seeds']) for i in range(2)]
    assert plan['required_free']==REQUIRED and plan['limits']==data_contract.LIMITS
    assert plan['prospective_quality']==QUALITY_DESIGN
    for value in plan['prerequisites'].values():read(value)
    return plan

def prepare():
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with session_lock(GLOBAL_LOCK.parent) as global_fd:
        idle();assert not ROOT.exists() and not FREEZE.exists()
        with CpuPrerequisiteGuard(B/'pilot002-prepare-guard.json',c.ROOT,timeout_seconds=120):
            old=read(ref(B/'pilot001-plan.json'));failure=read(ref(B/'pilot001-run-0-failure.json'))
            assert 'metadata group cap' in failure['error'] and (OLDROOT/'STOP').exists()
            assert not (OLDROOT/'evidence/pilots/01').exists() and not (OLDROOT/'evidence/pilots/00/accepted.json').exists()
            original=OLDROOT/'evidence/pilots/00';worker_ref=ref(original/'worker.json');worked=read(worker_ref)
            process_ref=ref(original/'process/process.json');process=read(process_ref)
            assert process['status']=='completed' and process['returncode']==0
            original_request=worked['acceptance_request'];assert original_request['config']==old['configs'][0]
            for name in ('worker-guard.json','parent-guard.json'):
                guard=read(ref(original/name));assert guard['status']=='finished' and guard['reason'] is None
            imported=sum(Path(original_request['files'][k]['path']).stat().st_size for k in ('parent','h200'))
            sources=c.source_inventory();assert all(sources[n]==h for n,h in old['sources'].items())
            assert data_contract.environment()==old['environment'] and data_contract.bind_data()==old['data']
            configs=[config(i,old['seeds']) for i in range(2)]
            require_capacity(shutil.disk_usage(c.ROOT).free,imported)
            prerequisites={n:ref(B/n) for n in ['pilot001-plan.json','pilot001-freeze.json','pilot001-run-0-failure.json','pilot002-publication-preflight.json','scientific-seed-census-001/result.json']}
            plan=dict(kind='fixed_lr_final_ln_scale_imported_control_resource_pilot_v2',root=str(ROOT),runner=ref(__file__),sources=sources,configs=configs,seeds=old['seeds'],data=old['data'],environment=old['environment'],prerequisites=prerequisites,limits=data_contract.LIMITS,required_free=REQUIRED,imported_weight_bytes=imported,raw_cap=RAW_CAP,export_cap=EXPORT_CAP,weight_cap=WEIGHT_CAP,trajectory_cap=TRAJECTORY_CAP,round_metadata=256*1024**2,quality_launch_permitted=False,pilot_order=list(ARMS),steps=20,lr=LR,prospective_quality=QUALITY_DESIGN,original_closed_freeze=ref(B/'pilot001-freeze.json'),original_stopped_root=str(OLDROOT),imported_control_worker=worker_ref,imported_control_process=process_ref,scope='Import only completed learned20-update endpoint with ZERO repeated training or weight copies; train only the previously unstarted fixed-unit arm. Same scientific seeds/settings. Original metadata caps and all numeric/resource guards unchanged. Acceptance receipt references immutable worker instead of duplicating its full source inventory.')
            exclusive(B/'pilot002-plan.json',plan);ROOT.mkdir();exclusive(FREEZE,dict(plan=ref(B/'pilot002-plan.json'),quality_launch_permitted=False))
            assert bind()==plan
            request=copy.deepcopy(original_request);request['sources']=sources
            with Guard(B/'pilot002-import-guard.json'):
                acceptance=a.accept(request,worked['validation']);clear()
                assert acceptance['accepted'] and acceptance['audit']['completed_steps']==20 and acceptance['audit']['tokens_processed']==20480
                assert acceptance['audit']['ema_parameters']==88 and acceptance['audit']['ema_snapshots']==8
                assert ref(original/'worker.json')==worker_ref
            with session_lock(ROOT) as run_fd:
                writer=MetadataWriter(ROOT,(global_fd,run_fd))
                writer.publish_json('evidence/pilots/00/accepted.json',dict(slot=0,arm='learned',plan=ref(B/'pilot002-plan.json'),worker=worker_ref,acceptance_request_sha256=c.fingerprint(request),validation=worked['validation'],acceptance=acceptance,process=process_ref,worker_guard=ref(original/'worker-guard.json'),parent_guard=ref(B/'pilot002-import-guard.json'),original_parent_guard=ref(original/'parent-guard.json'),imported=True,repeated_training_updates=0,checkpoint_copies=0))
            exclusive(B/'pilot002-accepted-00.json',dict(accepted=ref(ROOT/'evidence/pilots/00/accepted.json'),imported=True,repeated_training_updates=0,checkpoint_copies=0,scientific_quality=False))
        exclusive(B/'pilot002-preflight.json',dict(status='separate_frozen_imported_control_and_unstarted_arm',freeze=ref(FREEZE),guard=ref(B/'pilot002-prepare-guard.json'),imported_control_updates=20,new_control_updates=0))
        print('completed learned endpoint independently imported; only unstarted slot01 may train',flush=True)

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
    assert type(slot) is int and slot==1,'Only the previously unstarted arm may train'
    with session_lock(GLOBAL_LOCK.parent) as global_fd:
        idle();plan=bind()
        with session_lock(ROOT) as run_fd:
            writer=MetadataWriter(ROOT,(global_fd,run_fd));basepath=f'evidence/pilots/{slot:02d}';root=ROOT/basepath
            assert not root.exists(),'No retry of pilot attempt'
            with Guard(ROOT/f'control/prefix-{slot:02d}-guard.json',120):
                for prior in range(slot):
                    p=ROOT/f'evidence/pilots/{prior:02d}';row=read(ref(p/'accepted.json'))
                    assert row['slot']==prior and row['plan']==ref(B/'pilot002-plan.json')
                    a.accept(request_for(row,plan),row['validation']);clear()
            assert not any((ROOT/f'evidence/pilots/{i:02d}').exists() for i in range(slot+1,2))
            used=used_weights();require_capacity(shutil.disk_usage(ROOT).free,used)
            started=time.monotonic();writer.publish_json(basepath+'/request.json',dict(plan=ref(B/'pilot002-plan.json'),parent_pid=os.getpid(),slot=slot))
            argv=[sys.executable,'-m','research.final_ln_scale_campaign_v2.pilot','worker','--slot',str(slot),'--global-fd',str(global_fd),'--run-fd',str(run_fd),'--request-sha',sha(root/'request.json')]
            process=run_process(writer,'pilots',slot,argv,c.ROOT,required_free_bytes=REQUIRED-used,output_limit_bytes=TRAJECTORY_CAP,timeout_seconds=600)
            assert process['status']=='completed' and read(ref(root/'worker-guard.json'))['status']=='finished'
            with Guard(ROOT/f'control/parent-{slot:02d}-guard.json',600-(time.monotonic()-started)):
                result=read(ref(root/'worker.json'));assert result['pid']==process['pid'] and result['parent_pid']==os.getpid()
                request=result['acceptance_request'];assert request['config']==plan['configs'][slot] and request['sources']==plan['sources']
                acceptance=a.accept(request,result['validation']);clear();assert bind()==plan
            writer.publish_json(basepath+'/parent-guard.json',read(ref(ROOT/f'control/parent-{slot:02d}-guard.json')))
            writer.publish_json(basepath+'/accepted.json',dict(slot=slot,arm=ARMS[slot],plan=ref(B/'pilot002-plan.json'),worker=ref(root/'worker.json'),acceptance_request_sha256=c.fingerprint(request),validation=result['validation'],acceptance=acceptance,process=ref(root/'process/process.json'),worker_guard=ref(root/'worker-guard.json'),parent_guard=ref(root/'parent-guard.json'),imported=False))
            exclusive(B/f'pilot002-accepted-{slot:02d}.json',dict(accepted=ref(root/'accepted.json'),scientific_quality=False))
            print(json.dumps(dict(status='resource_pilot_accepted',slot=slot,arm=ARMS[slot],ce=result['validation']['ce'],audit=acceptance['audit'])),flush=True)

def review():
    with session_lock(GLOBAL_LOCK.parent):
        idle();plan=bind()
        with session_lock(ROOT) as fd, Guard(B/'pilot002-review-guard.json'):
            rows=[read(ref(ROOT/f'evidence/pilots/{i:02d}/accepted.json')) for i in range(2)]
            for row in rows:a.accept(request_for(row,plan),row['validation']);clear()
            bindings={row['arm']:{k:request_for(row,plan)['files'][k] for k in ('config','parent','h200')} for row in rows}
            with Guard(B/'pilot002-inference-guard.json'):
                measured=inf.measure_pair(bindings,0);checked=inf.review(measured,bindings,0)
                exclusive(B/'pilot002-inference-measurements.json',measured);assert bind()==plan
            exclusive(B/'pilot002-review.json',dict(status='both_resource_pilots_and_inference_passed',plan=ref(B/'pilot002-plan.json'),accepted=[ref(ROOT/f'evidence/pilots/{i:02d}/accepted.json') for i in range(2)],inference=ref(B/'pilot002-inference-measurements.json'),inference_guard=ref(B/'pilot002-inference-guard.json'),review=checked,
                quality_adopted=False,quality_launch_permitted=False,free_bytes=shutil.disk_usage(ROOT).free,retained_weights=used_weights(),
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
            destination=B/f'pilot002-{args.mode}-{args.slot}-failure.json'
            if not destination.exists():exclusive(destination,dict(error=repr(exc),runner=ref(__file__),no_retry=True))
            if ROOT.exists() and not (ROOT/'STOP').exists():exclusive(ROOT/'STOP',dict(error=repr(exc),mode=args.mode,slot=args.slot,no_retry=True))
        raise
