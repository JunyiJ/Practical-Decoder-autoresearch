"""Finite six-trajectory worker with independent parent acceptance.

No implicit retry, resume, seed substitution or adoption. Existing pilots stay
closed. External anchors bind each accepted trajectory and the pilot retirement.
"""
import argparse,copy,gc,json,os,shutil,sys,time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from . import campaign_contract as c,campaign_rules as rules,artifact_acceptance as a
from research.decay_v1 import validation as v
from .support import exclusive,file_ref,cache_audit
from research.block_norm_v1.fixture_dispatch import reference,read
from research.local_mix_v1.validation_data import load as validation_data
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.local_mix_v1.runtime.train.train import run_training
from research.local_mix_v1.runtime.train.weight_averaging import export_averaged_checkpoint
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from .support import idle
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter
from src.research.tokenizer_wide_hybrid_process_v2 import run_process
from src.research.tokenizer_serial_process import _group_exists
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard,observation,violation
ROOT=c.ROOT/'research_runs/tokenizer-decay-quality-v1-001'
EVIDENCE=c.ROOT/'research/tokenizer-evidence/decay-preparation-001'
FREEZE=EVIDENCE/'quality001-freeze.json'
MODULE='research.decay_quality_v1.quality_execution';KIND='decay_quality_v1_finite_quality_v1'
ALLOWANCES=dict(zip(rules.ARMS,(603979776,603979776)))
FULL_FREE=rules.REQUIRED;SECONDS=7200
require=c.require


class QualityGuard(StrictMpsGuard):
    def __init__(self,path,root,seconds=SECONDS):
        require(0<seconds<=SECONDS,'Bounded remaining quality deadline required')
        super().__init__(path,root);self.limits['timeout_seconds']=seconds


def bind(writer,freeze_sha):
    require(writer.root==ROOT and not (ROOT/'STOP').exists(),'Dedicated unstopped quality root required')
    frozen=read(dict(path=str(FREEZE),sha256=freeze_sha));plan=read(frozen['plan'])
    require(plan['kind']==KIND and plan['root']==str(ROOT) and plan['status']=='frozen_quality_only','Frozen finite quality plan required')
    c.verify_sources(plan['sources']);c.frozen_predecessors(plan['sources'])
    require(c.environment()==plan['environment'] and c.bind_data()==plan['data'],'Frozen environment/data differ')
    require(plan['limits']==dict(c.LIMITS) and plan['allowances']==ALLOWANCES and plan['required_free_bytes']==FULL_FREE,'Resource contract differs')
    require(plan['fresh']==rules.fresh_specs(plan['seeds']) and plan['design']==rules.QUALITY_DESIGN,'Finite fixed-LR panel differs')
    # Explicit additive-source applicability; never rerun a closed pilot CLI.
    prior=read(plan['pilot_plan']);require(all(plan['sources'].get(k)==h for k,h in prior['sources'].items()),'Pilot numerical source changed')
    require(set(plan['sources'])-set(prior['sources'])==set(plan['added_sources']),'Unexpected source addition')
    for ref in plan['prerequisites'].values():read(ref)
    for ref in plan['final_sources'].values():require(file_ref(Path(ref['path']))==ref,'Final review source changed')
    from .pilot_retirement import verify_retirement
    verify_retirement(plan['pilot_retirement'])
    return plan


def anchor_path(slot):return EVIDENCE/f'quality001-accepted-{slot:02d}.json'


def state(plan,freeze_sha,*,check_files=True):
    records=[];receipts=[];anchors=[];data=None
    previous=dict(path=str(FREEZE),sha256=freeze_sha)
    present=[i for i in range(6) if anchor_path(i).exists()]
    require(present==list(range(len(present))),'Noncontiguous accepted prefix')
    for slot in present:
        anchor=read(reference(anchor_path(slot)));require(anchor['kind']==KIND and anchor['slot']==slot and anchor['previous']==previous,'External anchor chain differs')
        receipt=read(anchor['acceptance']);require(receipt['kind']==KIND and receipt['slot']==slot and receipt['freeze']==dict(path=str(FREEZE),sha256=freeze_sha),'Accepted trial identity differs')
        spec=plan['fresh'][slot]
        require(receipt['selection'] is None and receipt['cleanup'] is None,'No selection or fresh cleanup allowed')
        req=receipt['request'];c.validate_trial_config(req['config'],ROOT/f'evidence/quality/{slot:02d}',spec,plan['seeds'])
        accepted=receipt['artifact_acceptance']
        require(req['sources']==plan['sources'] and req['trial']==dict(spec=spec,seeds=plan['seeds']) and accepted['accepted']
            and accepted['request_sha256']==c.fingerprint(req) and accepted['source_sha256']==c.fingerprint(plan['sources'])
            and accepted['files']==req['files'] and accepted['audit']==dict(completed_steps=1200,tokens_processed=1228800,ema_parameters=88,ema_snapshots=41),'Accepted output contract differs')
        if data is None:data=validation_data(OmegaConf.create(req['config']))
        v.validate_report(accepted['validation'],{k:req['files'][k] for k in ('config','parent','h200')},data)
        require(receipt['target_contract']==accepted['validation']['target'],'Saved target contract differs')
        guard=read(anchor['parent_guard']);require(guard['status']=='finished' and guard['reason'] is None and 0<guard['limits']['timeout_seconds']<=SECONDS,'Parent guard differs')
        records.append(dict(spec,status='completed',h200_ce=accepted['validation']['ce']));receipts.append(receipt);anchors.append(anchor);previous=reference(anchor_path(slot))
    used=0
    for anchor in anchors:
        for ref in anchor['files']:
            if check_files:require(file_ref(Path(ref['path']),ref['path'].endswith('.pt'))==ref,'Accepted artifact changed')
            if ref['path'].endswith('.pt'):used+=Path(ref['path']).stat().st_size
    return dict(records=records,receipts=receipts,previous=previous,used_weights=used)


def next_spec(plan,current):
    slot=len(current['records']);require(slot<6,'Finite quality panel complete')
    return plan['fresh'][slot]


def worker_argv(writer,slot,freeze_sha):
    return [sys.executable,'-m',MODULE,'worker','--slot',str(slot),'--freeze-sha256',freeze_sha,'--global-fd',str(writer.lock_fds[0]),'--campaign-fd',str(writer.lock_fds[1])]


from research.local_mix_v1.output import Sink as BoundedSink
class Sink(BoundedSink):
    def __init__(self,writer,slot,cfg,arm):super().__init__(writer,'quality',slot,cfg)


def worker(writer,slot,freeze_sha,argv):
    require(type(slot) is int and 0<=slot<6,'Finite quality slot required');base=f'evidence/quality/{slot:02d}'
    with QualityGuard(ROOT/f'control/worker-{slot:02d}-guard.json',c.ROOT):
        plan=bind(writer,freeze_sha);current=state(plan,freeze_sha);require(slot==len(current['records']),'Next serial quality slot required')
        spec=next_spec(plan,current);cfg=c.trial_config(ROOT/base,spec,plan['seeds']);rq=read(reference(ROOT/base/'request.json'));parent=os.getppid()
        require(rq==dict(plan=read(dict(path=str(FREEZE),sha256=freeze_sha))['plan'],slot=slot,parent_pid=parent,argv=argv,previous=current['previous']) and argv==worker_argv(writer,slot,freeze_sha),'Worker parent/argv/request differs')
        idle(allowed=(parent,));writer.publish_json(base+'/config.json',cfg);sink=Sink(writer,slot,cfg,spec['arm'])
        try:summary=run_training(OmegaConf.create(copy.deepcopy(cfg)),output_sink=sink)
        finally:sink.close()
        require(summary['completed_steps']==1200 and summary['stop_reason']=='max_iters','Incomplete quality training')
        gc.collect();torch.mps.synchronize();torch.mps.empty_cache();avg=ROOT/base/'checkpoints/h200.pt'
        def export_writer(path,payload):require(path==avg,'Export path differs');sink.averaged(path,payload)
        export=export_averaged_checkpoint(sink.raw,avg,200,payload_writer=export_writer);writer.publish_json(base+'/export.json',export)
        files={k:file_ref(path,k in ('parent','h200')) for k,path in dict(config=ROOT/base/'config.json',parent=sink.raw,h200=avg,summary=ROOT/base/'summary.json',metrics=ROOT/base/'metrics.jsonl',export=ROOT/base/'export.json').items()}
        req=dict(scope=a.TRIAL_SCOPE,sources=plan['sources'],config=cfg,files=files,trial=dict(spec=spec,seeds=plan['seeds']));a.verify_request(req)
        validation=v.evaluate({k:files[k] for k in ('config','parent','h200')});cached=cache_audit(req);bind(writer,freeze_sha)
        require(os.getppid()==parent,'Parent disappeared')
        writer.publish_json(base+'/worker.json',dict(kind=KIND,pid=os.getpid(),parent_pid=parent,request=reference(ROOT/base/'request.json'),acceptance_request=req,validation=validation,cache=cached,holdout_scored=False))
    writer.publish_json(base+'/worker-guard.json',json.loads((ROOT/f'control/worker-{slot:02d}-guard.json').read_text()))


def run_once(writer,slot,freeze_sha):
    require(type(slot) is int and 0<=slot<6,'Finite quality slot required');base=f'evidence/quality/{slot:02d}'
    with CpuPrerequisiteGuard(ROOT/f'control/entry-{slot:02d}-guard.json',c.ROOT,timeout_seconds=120,poll_seconds=.25):
        plan=bind(writer,freeze_sha);current=state(plan,freeze_sha);require(slot==len(current['records']),'Serial accepted prefix required');spec=next_spec(plan,current)
        require(not (ROOT/base).exists(),'Existing quality attempt cannot retry')
        require(not any((ROOT/f'evidence/quality/{i:02d}').exists() for i in range(slot,6)),'Unaccepted or future slot exists')
        required=max(8*1024**3,FULL_FREE-current['used_weights']);idle();require(shutil.disk_usage(ROOT).free>=required,'Remaining full-panel capacity required')
        argv=worker_argv(writer,slot,freeze_sha);writer.publish_json(base+'/request.json',dict(plan=read(dict(path=str(FREEZE),sha256=freeze_sha))['plan'],slot=slot,parent_pid=os.getpid(),argv=argv,previous=current['previous']))
    start=time.monotonic()
    def check():
        reason=violation(observation(ROOT),time.monotonic()-start,4*1024**3,8*1024**3,SECONDS);require(reason is None,'Parent resource limit '+str(reason));require(torch.mps.driver_allocated_memory()<6*1024**3,'Parent driver limit')
    process=run_process(writer,'quality',slot,argv,c.ROOT,required_free_bytes=required,output_limit_bytes=ALLOWANCES[spec['arm']],timeout_seconds=SECONDS,parent_check=check);idle()
    with QualityGuard(ROOT/f'control/parent-{slot:02d}-guard.json',c.ROOT,SECONDS-(time.monotonic()-start)):
        bind(writer,freeze_sha);require(process['status']=='completed' and process['returncode']==0 and process['error'] is None and process['argv']==argv and process['parent_pid']==os.getpid() and not _group_exists(process['pid']) and not process['log_truncated'] and not any(k in process for k in ('cleanup_error','log_close_error','failure_observation_error')),'Worker did not complete cleanly')
        out=read(reference(ROOT/base/'worker.json'));guard=read(reference(ROOT/base/'worker-guard.json'));require(out['kind']==KIND and out['pid']==process['pid'] and out['parent_pid']==os.getpid() and out['request']==reference(ROOT/base/'request.json') and out['holdout_scored'] is False,'Worker identity differs')
        require(guard['status']=='finished' and guard['reason'] is None and guard['pid']==process['pid'] and guard['limits']==dict(rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,timeout_seconds=SECONDS,mps_driver_limit_bytes=6*1024**3),'Worker guard differs')
        req=out['acceptance_request'];expected=c.trial_config(ROOT/base,spec,plan['seeds']);require(req['sources']==plan['sources'] and req['config']==expected and req['trial']==dict(spec=spec,seeds=plan['seeds']),'Actual quality trial differs')
        accepted=a.accept(req,out['validation']);cached=cache_audit(req);require(c.canonical(cached)==c.canonical(out['cache']),'Cache replay differs')
        receipt=dict(kind=KIND,slot=slot,freeze=dict(path=str(FREEZE),sha256=freeze_sha),request=req,artifact_acceptance=accepted,target_contract=out['validation']['target'],cache=cached,process=reference(ROOT/base/'process/process.json'),selection=None,cleanup=None,adopted=False)
        writer.publish_json(base+'/accepted.json',receipt)
    require(time.monotonic()-start<SECONDS,'Total quality trial deadline exceeded')
    files=[file_ref(path,path.suffix=='.pt') for path in sorted((ROOT/base).rglob('*')) if path.is_file()]
    exclusive(anchor_path(slot),dict(kind=KIND,slot=slot,previous=current['previous'],acceptance=reference(ROOT/base/'accepted.json'),parent_guard=reference(ROOT/f'control/parent-{slot:02d}-guard.json'),files=files))
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker']);parser.add_argument('--slot',type=int);parser.add_argument('--freeze-sha256',required=True);parser.add_argument('--global-fd',type=int);parser.add_argument('--campaign-fd',type=int);args=parser.parse_args();torch.set_num_threads(4)
    if args.mode=='worker':worker(MetadataWriter(ROOT,(args.global_fd,args.campaign_fd)),args.slot,args.freeze_sha256,[sys.executable,'-m',MODULE,*sys.argv[1:]])
    else:
        with session_lock(GLOBAL_LOCK.parent) as g,session_lock(ROOT) as p:
            writer=MetadataWriter(ROOT,(g,p))
            try:result=run_once(writer,args.slot,args.freeze_sha256)
            except BaseException as exc:
                if not (ROOT/'STOP').exists():writer.publish_json('STOP',dict(error=repr(exc),mode=args.mode,slot=args.slot,no_retry=True))
                raise
            print(json.dumps(dict(status='quality_accepted',slot=args.slot)))


if __name__=='__main__':main()
