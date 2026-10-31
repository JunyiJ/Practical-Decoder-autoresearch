"""Finite quality worker, independent parent acceptance and exact screen retention.

No implicit retry, resume, seed substitution or adoption. Existing pilots stay
closed. External anchors bind each accepted trajectory and the cleanup boundary.
"""
import argparse,copy,gc,json,os,shutil,sys,time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from . import campaign_contract as c,campaign_rules as rules,artifact_acceptance as a,validation as v,quality_retention as retention
from .pilot_execution import exclusive,file_ref,cache_audit
from .fixture_dispatch import reference,read
from .validation_data import load as validation_data
from .mps_guard import StrictMpsGuard
from .runtime.train.train import run_training
from .runtime.train.weight_averaging import export_averaged_checkpoint
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter
from src.research.tokenizer_wide_hybrid_process_v2 import run_process
from src.research.tokenizer_serial_process import _group_exists
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard,observation,violation
ROOT=c.ROOT/'research_runs/tokenizer-block-norm-v2-quality-001'
EVIDENCE=c.ROOT/'research/tokenizer-evidence/block-norm-preparation-001'
FREEZE=EVIDENCE/'quality001-freeze.json'
MODULE='research.block_norm_v1.quality_execution';KIND='block_norm_v2_finite_quality_v1'
ALLOWANCES=dict(zip(rules.ARMS,(788953140,789009664)))
FULL_FREE=15170221264;SECONDS=7200
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
    require(plan['screens']==rules.screen_specs(plan['seeds']),'Finite screens differ')
    # Explicit additive-source applicability; never rerun a closed pilot CLI.
    prior=read(plan['pilot_plan']);require(all(plan['sources'].get(k)==h for k,h in prior['sources'].items()),'Pilot numerical source changed')
    require(set(plan['sources'])-set(prior['sources'])==set(plan['added_sources']),'Unexpected source addition')
    for ref in plan['prerequisites'].values():read(ref)
    for inventory_ref in plan['protected_pilot_inventories']:
        saved=read(inventory_ref);root=Path(saved['root'])
        require({str(f.relative_to(root)) for f in root.rglob('*') if f.is_file()}==set(saved['inventory']),'Protected pilot inventory changed')
        for name,row in saved['inventory'].items():retention.check_entry(str(root/name),row)
    return plan


def anchor_path(slot):return EVIDENCE/f'quality001-accepted-{slot:02d}.json'
def selection_path():return EVIDENCE/'quality001-selection.json'
def cleanup_path():return EVIDENCE/'quality001-screen-retention/receipt.json'


def state(plan,freeze_sha,*,check_files=True):
    records=[];receipts=[];anchors=[];selection=None;cleanup=None;data=None
    previous=dict(path=str(FREEZE),sha256=freeze_sha)
    present=[i for i in range(12) if anchor_path(i).exists()]
    require(present==list(range(len(present))),'Noncontiguous accepted prefix')
    for slot in present:
        anchor=read(reference(anchor_path(slot)));require(anchor['kind']==KIND and anchor['slot']==slot and anchor['previous']==previous,'External anchor chain differs')
        receipt=read(anchor['acceptance']);require(receipt['kind']==KIND and receipt['slot']==slot and receipt['freeze']==dict(path=str(FREEZE),sha256=freeze_sha),'Accepted trial identity differs')
        if slot>=6:
            selection=read(reference(selection_path()))
            require(selection['prefix']==[reference(anchor_path(i)) for i in range(6)] and selection['value']==rules.choose_rates(plan['seeds'],records[:6]),'Persisted LR selection differs')
            require(receipt['selection']==reference(selection_path()) and receipt['cleanup']==reference(cleanup_path()),'Fresh trial boundary differs')
        spec=plan['screens'][slot] if slot<6 else rules.confirmation_specs(plan['seeds'],selection['value'])[slot-6]
        req=receipt['request'];c.validate_trial_config(req['config'],ROOT/f'evidence/quality/{slot:02d}',spec,plan['seeds'])
        accepted=receipt['artifact_acceptance']
        require(req['sources']==plan['sources'] and req['trial']==dict(spec=spec,seeds=plan['seeds']) and accepted['accepted']
            and accepted['request_sha256']==c.fingerprint(req) and accepted['source_sha256']==c.fingerprint(plan['sources'])
            and accepted['files']==req['files'] and accepted['audit']==dict(completed_steps=1200,tokens_processed=1228800,ema_parameters=88 if spec['arm']=='rms_incumbent' else 96,ema_snapshots=41),'Accepted output contract differs')
        if data is None:data=validation_data(OmegaConf.create(req['config']))
        v.validate_report(accepted['validation'],{k:req['files'][k] for k in ('config','parent','h200')},data)
        require(receipt['target_contract']==accepted['validation']['target'],'Saved target contract differs')
        guard=read(anchor['parent_guard']);require(guard['status']=='finished' and guard['reason'] is None and 0<guard['limits']['timeout_seconds']<=SECONDS,'Parent guard differs')
        records.append(dict(spec,status='completed',h200_ce=accepted['validation']['ce']));receipts.append(receipt);anchors.append(anchor);previous=reference(anchor_path(slot))
    deleted={}
    if cleanup_path().exists():
        require(len(records)>=6 and selection_path().exists(),'Cleanup before complete screen')
        selection=read(reference(selection_path()));require(selection['value']==rules.choose_rates(plan['seeds'],records[:6]) and selection['prefix']==[reference(anchor_path(i)) for i in range(6)],'Cleanup selection differs')
        cleanup=dict(selection=reference(selection_path()),freeze=dict(path=str(FREEZE),sha256=freeze_sha),prefix=selection['prefix'])
        deleted=retention.verify(reference(cleanup_path()),cleanup)
        expected=rules.screening_retention(plan['seeds'],records[:6],selection['value'])
        allowed={receipts[i]['request']['files'][role]['path']:receipts[i]['request']['files'][role]['sha256'] for i in expected['losing_indices'] for role in ('parent','h200')}
        require({k:v['sha256'] for k,v in deleted.items()}==allowed,'Cleanup removed unexpected weights')
    used=0
    for anchor in anchors:
        for ref in anchor['files']:
            if ref['path'] in deleted:require(ref['sha256']==deleted[ref['path']]['sha256'],'Deleted-file binding differs');continue
            if check_files:require(file_ref(Path(ref['path']),ref['path'].endswith('.pt'))==ref,'Accepted artifact changed')
            if ref['path'].endswith('.pt'):used+=Path(ref['path']).stat().st_size
    return dict(records=records,receipts=receipts,previous=previous,selection=selection,cleanup=cleanup,used_weights=used)


def next_spec(plan,current):
    slot=len(current['records']);require(slot<12,'Finite quality panel complete')
    if slot<6:return plan['screens'][slot]
    require(current['cleanup'] is not None,'Screen retention boundary required before fresh confirmation')
    return rules.confirmation_specs(plan['seeds'],current['selection']['value'])[slot-6]


def worker_argv(writer,slot,freeze_sha):
    return [sys.executable,'-m',MODULE,'worker','--slot',str(slot),'--freeze-sha256',freeze_sha,'--global-fd',str(writer.lock_fds[0]),'--campaign-fd',str(writer.lock_fds[1])]


class Sink:
    def __init__(self,writer,slot,cfg,arm):
        self.writer,self.slot,self.cfg=writer,slot,cfg;self.base=f'evidence/quality/{slot:02d}';self.raw=None;self.stream=None
        self.weights=CheckpointWriter(writer,'quality',slot,ALLOWANCES[arm])
    def validate_config(self,cfg):require(c.canonical(OmegaConf.to_container(cfg,resolve=True))==c.canonical(self.cfg),'Sink config differs')
    def checkpoint(self,path,payload):
        path=Path(path);require(self.raw is None and path.parent==ROOT/self.base/'checkpoints' and payload['step']==1200,'One final1200 checkpoint required')
        self.weights.save(path.name,payload);self.raw=path
    def metric(self,path,value):
        require(Path(path)==ROOT/self.base/'metrics.jsonl','Metric path differs')
        if self.stream is None:self.stream=self.writer.open_append(self.base+'/metrics.jsonl')
        self.stream.write((c.canonical(value)+'\n').encode())
    def summary(self,path,value):
        require(Path(path)==ROOT/self.base/'training-summary.json','Summary path differs');self.writer.publish_json(self.base+'/training-summary.json',value)
    def close(self):
        if self.stream is not None:self.stream.close()


def worker(writer,slot,freeze_sha,argv):
    require(type(slot) is int and 0<=slot<12,'Finite quality slot required');base=f'evidence/quality/{slot:02d}'
    with QualityGuard(ROOT/f'control/worker-{slot:02d}-guard.json',c.ROOT):
        plan=bind(writer,freeze_sha);current=state(plan,freeze_sha);require(slot==len(current['records']),'Next serial quality slot required')
        spec=next_spec(plan,current);cfg=c.trial_config(ROOT/base,spec,plan['seeds']);rq=read(reference(ROOT/base/'request.json'));parent=os.getppid()
        require(rq==dict(plan=read(dict(path=str(FREEZE),sha256=freeze_sha))['plan'],slot=slot,parent_pid=parent,argv=argv,previous=current['previous']) and argv==worker_argv(writer,slot,freeze_sha),'Worker parent/argv/request differs')
        idle(allowed=(parent,));writer.publish_json(base+'/config.json',cfg);sink=Sink(writer,slot,cfg,spec['arm'])
        try:summary=run_training(OmegaConf.create(copy.deepcopy(cfg)),output_sink=sink)
        finally:sink.close()
        require(summary['completed_steps']==1200 and summary['stop_reason']=='max_iters','Incomplete quality training')
        gc.collect();torch.mps.synchronize();torch.mps.empty_cache();avg=ROOT/base/'checkpoints/h200.pt'
        def export_writer(path,payload):require(path==avg,'Export path differs');sink.weights.save(path.name,payload)
        export=export_averaged_checkpoint(sink.raw,avg,200,payload_writer=export_writer);writer.publish_json(base+'/export.json',export)
        files={k:file_ref(path,k in ('parent','h200')) for k,path in dict(config=ROOT/base/'config.json',parent=sink.raw,h200=avg,summary=ROOT/base/'training-summary.json',metrics=ROOT/base/'metrics.jsonl',export=ROOT/base/'export.json').items()}
        req=dict(scope=a.TRIAL_SCOPE,sources=plan['sources'],config=cfg,files=files,trial=dict(spec=spec,seeds=plan['seeds']));a.verify_request(req)
        validation=v.evaluate({k:files[k] for k in ('config','parent','h200')});cached=cache_audit(req);bind(writer,freeze_sha)
        require(os.getppid()==parent,'Parent disappeared')
        writer.publish_json(base+'/worker.json',dict(kind=KIND,pid=os.getpid(),parent_pid=parent,request=reference(ROOT/base/'request.json'),acceptance_request=req,validation=validation,cache=cached,holdout_scored=False))
    writer.publish_json(base+'/worker-guard.json',json.loads((ROOT/f'control/worker-{slot:02d}-guard.json').read_text()))


def run_once(writer,slot,freeze_sha):
    require(type(slot) is int and 0<=slot<12,'Finite quality slot required');base=f'evidence/quality/{slot:02d}'
    with CpuPrerequisiteGuard(ROOT/f'control/entry-{slot:02d}-guard.json',c.ROOT,timeout_seconds=120,poll_seconds=.25):
        plan=bind(writer,freeze_sha);current=state(plan,freeze_sha);require(slot==len(current['records']),'Serial accepted prefix required');spec=next_spec(plan,current)
        require(not (ROOT/base).exists(),'Existing quality attempt cannot retry')
        require(not any((ROOT/f'evidence/quality/{i:02d}').exists() for i in range(slot,12)),'Unaccepted or future slot exists')
        required=max(8*1024**3,FULL_FREE-current['used_weights']);idle();require(shutil.disk_usage(ROOT).free>=required,'Remaining full-panel capacity required')
        argv=worker_argv(writer,slot,freeze_sha);writer.publish_json(base+'/request.json',dict(plan=read(dict(path=str(FREEZE),sha256=freeze_sha))['plan'],slot=slot,parent_pid=os.getpid(),argv=argv,previous=current['previous']))
    start=time.monotonic()
    def check():
        reason=violation(observation(ROOT),time.monotonic()-start,4*1024**3,8*1024**3,SECONDS);require(reason is None,'Parent resource limit '+str(reason));require(torch.mps.driver_allocated_memory()<6*1024**3,'Parent driver limit')
    process=run_process(writer,'quality',slot,argv,c.ROOT,required_free_bytes=required,output_limit_bytes=ALLOWANCES[spec['arm']]+8*1024**2,timeout_seconds=SECONDS,parent_check=check);idle()
    with QualityGuard(ROOT/f'control/parent-{slot:02d}-guard.json',c.ROOT,SECONDS-(time.monotonic()-start)):
        bind(writer,freeze_sha);require(process['status']=='completed' and process['returncode']==0 and process['error'] is None and process['argv']==argv and process['parent_pid']==os.getpid() and not _group_exists(process['pid']) and not process['log_truncated'] and not any(k in process for k in ('cleanup_error','log_close_error','failure_observation_error')),'Worker did not complete cleanly')
        out=read(reference(ROOT/base/'worker.json'));guard=read(reference(ROOT/base/'worker-guard.json'));require(out['kind']==KIND and out['pid']==process['pid'] and out['parent_pid']==os.getpid() and out['request']==reference(ROOT/base/'request.json') and out['holdout_scored'] is False,'Worker identity differs')
        require(guard['status']=='finished' and guard['reason'] is None and guard['pid']==process['pid'] and guard['limits']==dict(rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,timeout_seconds=SECONDS,mps_driver_limit_bytes=6*1024**3),'Worker guard differs')
        req=out['acceptance_request'];expected=c.trial_config(ROOT/base,spec,plan['seeds']);require(req['sources']==plan['sources'] and req['config']==expected and req['trial']==dict(spec=spec,seeds=plan['seeds']),'Actual quality trial differs')
        accepted=a.accept(req,out['validation']);cached=cache_audit(req);require(c.canonical(cached)==c.canonical(out['cache']),'Cache replay differs')
        receipt=dict(kind=KIND,slot=slot,freeze=dict(path=str(FREEZE),sha256=freeze_sha),request=req,artifact_acceptance=accepted,target_contract=out['validation']['target'],cache=cached,process=reference(ROOT/base/'process/process.json'),selection=reference(selection_path()) if slot>=6 else None,cleanup=reference(cleanup_path()) if slot>=6 else None,adopted=False)
        writer.publish_json(base+'/accepted.json',receipt)
    require(time.monotonic()-start<SECONDS,'Total quality trial deadline exceeded')
    files=[file_ref(path,path.suffix=='.pt') for path in sorted((ROOT/base).rglob('*')) if path.is_file()]
    exclusive(anchor_path(slot),dict(kind=KIND,slot=slot,previous=current['previous'],acceptance=reference(ROOT/base/'accepted.json'),parent_guard=reference(ROOT/f'control/parent-{slot:02d}-guard.json'),files=files))
    return receipt


def boundary(writer,freeze_sha):
    with StrictMpsGuard(ROOT/'control/screen-boundary-guard.json',c.ROOT):
        plan=bind(writer,freeze_sha);current=state(plan,freeze_sha);require(len(current['records'])==6 and current['cleanup'] is None,'Exactly complete six-screen boundary')
        idle();require(not selection_path().exists() and not cleanup_path().parent.exists(),'Boundary attempts cannot retry')
        for receipt in current['receipts']:a.accept(receipt['request'],receipt['artifact_acceptance']['validation'])
        selection=rules.choose_rates(plan['seeds'],current['records']);prefix=[reference(anchor_path(i)) for i in range(6)]
        exclusive(selection_path(),dict(value=selection,prefix=prefix,freeze=dict(path=str(FREEZE),sha256=freeze_sha)))
        context=dict(selection=reference(selection_path()),freeze=dict(path=str(FREEZE),sha256=freeze_sha),prefix=prefix)
        def revalidate():
            bind(writer,freeze_sha);idle();require(read(context['selection'])['value']==selection,'Selection changed')
            for ref in prefix:read(ref)
            return context
        return retention.execute(writer,cleanup_path().parent,plan['seeds'],current['records'],selection,[r['request'] for r in current['receipts']],context,revalidate)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker','boundary']);parser.add_argument('--slot',type=int);parser.add_argument('--freeze-sha256',required=True);parser.add_argument('--global-fd',type=int);parser.add_argument('--campaign-fd',type=int);args=parser.parse_args();torch.set_num_threads(4)
    if args.mode=='worker':worker(MetadataWriter(ROOT,(args.global_fd,args.campaign_fd)),args.slot,args.freeze_sha256,[sys.executable,'-m',MODULE,*sys.argv[1:]])
    else:
        with session_lock(GLOBAL_LOCK.parent) as g,session_lock(ROOT) as p:
            writer=MetadataWriter(ROOT,(g,p));result=boundary(writer,args.freeze_sha256) if args.mode=='boundary' else run_once(writer,args.slot,args.freeze_sha256)
            print(json.dumps(dict(status='boundary_completed' if args.mode=='boundary' else 'quality_accepted',slot=args.slot)))


if __name__=='__main__':main()
