"""Finite one-pilot/three-quality dispatch using existing bounded IO and guards."""
import argparse,copy,gc,json,os,shutil,subprocess,sys,time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from . import contract as c,artifacts as a
from research.block_norm_v1.fixture_dispatch import reference,read
from research.block_norm_v1.pilot_execution import exclusive
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.block_norm_v1.runtime.train.train import run_training
from research.block_norm_v1.runtime.train.weight_averaging import export_averaged_checkpoint
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock,idle as inherited_idle
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter
from src.research.tokenizer_wide_hybrid_process_v2 import run_process
from src.research.tokenizer_serial_process import _group_exists
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard,observation,violation

ROOT=c.ROOT/'research_runs/tokenizer-ema-horizon-v1-001'
EVIDENCE=c.ROOT/'research/tokenizer-evidence/ema-horizon-preparation-001'
MODULE='research.ema_horizon_v1.execution'
require=c.require
def idle(allowed=()):
    inherited_idle(allowed)
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        row=line.strip().split(None,1)
        if len(row)==2 and 'research.ema_horizon_v1' in row[1] and Path(row[1].split()[0]).name.lower().startswith('python'):
            require(int(row[0]) in {os.getpid(),*allowed},'Other EMA worker is live')
def freeze_path(stage):return EVIDENCE/('pilot001-freeze.json' if stage=='pilots' else 'quality001-freeze.json')
def seconds(stage):return 600 if stage=='pilots' else 7200
def label(stage):return 'pilot' if stage=='pilots' else 'quality'
def anchor_path(stage,slot):return EVIDENCE/f'{label(stage)}001-accepted-{slot:02d}.json'
def ref_file(path):
    path=Path(path);limit=c.RAW_LIMIT if path.suffix=='.pt' else 2*1024**2
    return dict(path=str(path),sha256=c.inherited.file_sha(path,root=c.ROOT,max_bytes=limit))
class Guard(StrictMpsGuard):
    def __init__(self,path,seconds_left):
        require(0<seconds_left<=7200,'Bounded remaining deadline required')
        super().__init__(path,c.ROOT);self.limits['timeout_seconds']=seconds_left

def bind(writer,stage,sha):
    require(stage in ('pilots','quality') and writer.root==ROOT and not (ROOT/'STOP').exists(),'Dedicated unstopped EMA root required')
    seal=read(dict(path=str(freeze_path(stage)),sha256=sha));plan=read(seal['plan'])
    require(plan['kind']==c.KIND and plan['stage']==stage and plan['root']==str(ROOT),'Finite stage binding differs')
    require(plan['sources']==c.source_inventory() and plan['environment']==c.inherited.environment(),'Frozen source/environment differs')
    require(plan['data']==c.inherited.bind_data(),'Frozen data differs')
    require(plan['required_free_bytes']==c.REQUIRED_FREE and plan['trajectory_allowance_bytes']==c.TRAJECTORY_ALLOWANCE,'Frozen capacity differs')
    require(plan['limits']==dict(c.inherited.LIMITS),'Frozen resource limits differ')
    require(plan['specs']==[c.spec(label(stage),n,plan['seeds']) for n in range(1 if stage=='pilots' else 3)],'Finite specs differ')
    for n,cfg in enumerate(plan['configs']):c.validate_config(cfg,ROOT/f'evidence/{stage}/{n:02d}',label(stage),n,plan['seeds'])
    require(len(plan['configs'])==len(plan['specs']),'Complete configs required')
    for binding in plan['prerequisites'].values():read(binding)
    if stage=='quality':require(plan['pilot_acceptance']==reference(anchor_path('pilots',0)),'Accepted pilot required')
    return plan

def prefix(stage,slot):
    require(type(slot) is int and (slot==0 if stage=='pilots' else 0<=slot<3),'Finite slot required')
    prior=[('pilots',0)] if stage=='quality' else []
    prior.extend((stage,n) for n in range(slot))
    used=0;refs=[]
    for st,n in prior:
        anchor=read(reference(anchor_path(st,n)));receipt=read(anchor['acceptance'])
        require(receipt['stage']==st and receipt['slot']==n and receipt['artifact_acceptance']['accepted'],'Accepted predecessor differs')
        guard=read(anchor['parent_guard']);require(guard['status']=='finished' and guard['reason'] is None,'Predecessor parent guard failed')
        for ref in anchor['files']:
            require(ref_file(ref['path'])==ref,'Predecessor artifact changed')
            if ref['path'].endswith('.pt'):used+=Path(ref['path']).stat().st_size
        refs.append(reference(anchor_path(st,n)))
    require(not (ROOT/f'evidence/{stage}/{slot:02d}').exists(),'Existing attempt cannot retry')
    require(not any((ROOT/f'evidence/{stage}/{n:02d}').exists() for n in range(slot+1,1 if stage=='pilots' else 3)),'Future slot exists')
    return refs,used

def argv(writer,stage,slot,sha):return [sys.executable,'-m',MODULE,'worker','--stage',stage,'--slot',str(slot),'--freeze-sha256',sha,'--global-fd',str(writer.lock_fds[0]),'--campaign-fd',str(writer.lock_fds[1])]
class Sink:
    def __init__(self,writer,stage,slot,cfg):
        self.writer=writer;self.cfg=cfg;self.base=f'evidence/{stage}/{slot:02d}';self.raw=None;self.stream=None
        self.weights=CheckpointWriter(writer,stage,slot,c.TRAJECTORY_ALLOWANCE)
    def validate_config(self,cfg):require(OmegaConf.to_container(cfg,resolve=True)==self.cfg,'Sink config differs')
    def checkpoint(self,path,payload):
        path=Path(path);require(self.raw is None and path.parent==ROOT/self.base/'checkpoints' and payload['step']==self.cfg['training']['max_iters'],'Exactly one final raw checkpoint required')
        self.weights.save(path.name,payload);self.raw=path
    def metric(self,path,value):
        require(Path(path)==ROOT/self.base/'metrics.jsonl','Metric path differs')
        if self.stream is None:self.stream=self.writer.open_append(self.base+'/metrics.jsonl')
        self.stream.write((c.inherited.canonical(value)+'\n').encode())
    def summary(self,path,value):
        require(Path(path)==ROOT/self.base/'training-summary.json','Summary path differs');self.writer.publish_json(self.base+'/training-summary.json',value)
    def close(self):
        if self.stream is not None:self.stream.close()

def worker(writer,stage,slot,sha,actual):
    base=f'evidence/{stage}/{slot:02d}';parent=os.getppid()
    with Guard(ROOT/f'control/worker-{stage}-{slot:02d}.json',seconds(stage)):
        plan=bind(writer,stage,sha);rq=read(reference(ROOT/base/'request.json'))
        require(set(rq)=={'plan','stage','slot','parent_pid','argv','previous'} and rq['stage']==stage and rq['slot']==slot
            and actual==argv(writer,stage,slot,sha) and rq['argv']==actual and rq['parent_pid']==parent
            and rq['plan']==read(dict(path=str(freeze_path(stage)),sha256=sha))['plan'],'Worker entry binding differs')
        idle((parent,));cfg=plan['configs'][slot];writer.publish_json(base+'/config.json',cfg);sink=Sink(writer,stage,slot,cfg)
        try:summary=run_training(OmegaConf.create(copy.deepcopy(cfg)),output_sink=sink)
        finally:sink.close()
        require(summary['completed_steps']==cfg['training']['max_iters'] and summary['stop_reason']=='max_iters','Incomplete trajectory')
        gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
        files=dict(config=ROOT/base/'config.json',raw=sink.raw,summary=ROOT/base/'training-summary.json',metrics=ROOT/base/'metrics.jsonl')
        for half in c.HALF_LIVES:
            path=ROOT/base/f'checkpoints/h{half}.pt'
            def save(actual,payload):require(actual==path,'Export path differs');sink.weights.save(path.name,payload)
            exported=export_averaged_checkpoint(sink.raw,path,half,payload_writer=save)
            writer.publish_json(base+f'/export{half}.json',exported);files[f'h{half}']=path;files[f'export{half}']=ROOT/base/f'export{half}.json'
        req=dict(scope=a.SCOPE,sources=plan['sources'],stage=label(stage),slot=slot,seeds=plan['seeds'],config=cfg,files={k:ref_file(p) for k,p in files.items()})
        a.verify_request(req);validation=a.evaluate(req);cache=a.cache_audit(req)
        bind(writer,stage,sha);require(os.getppid()==parent,'Controller disappeared')
        writer.publish_json(base+'/worker.json',dict(kind=c.KIND,pid=os.getpid(),parent_pid=parent,request=reference(ROOT/base/'request.json'),acceptance_request=req,validation=validation,cache=cache,holdout_scored=False))
    writer.publish_json(base+'/worker-guard.json',json.loads((ROOT/f'control/worker-{stage}-{slot:02d}.json').read_text()))

def run_once(writer,stage,slot,sha):
    base=f'evidence/{stage}/{slot:02d}';limit=seconds(stage)
    with CpuPrerequisiteGuard(ROOT/f'control/entry-{stage}-{slot:02d}.json',c.ROOT,timeout_seconds=120,poll_seconds=.25):
        plan=bind(writer,stage,sha);previous,used=prefix(stage,slot);idle()
        required=max(8*1024**3,c.REQUIRED_FREE-used);require(shutil.disk_usage(ROOT).free>=required,'Full remaining panel capacity required')
        expected=argv(writer,stage,slot,sha)
        writer.publish_json(base+'/request.json',dict(plan=read(dict(path=str(freeze_path(stage)),sha256=sha))['plan'],stage=stage,slot=slot,parent_pid=os.getpid(),argv=expected,previous=previous))
    start=time.monotonic()
    def check():
        require(violation(observation(ROOT),time.monotonic()-start,4*1024**3,8*1024**3,limit) is None,'Parent resource limit')
        require(torch.mps.driver_allocated_memory()<6*1024**3,'Parent driver limit')
    process=run_process(writer,stage,slot,expected,c.ROOT,required_free_bytes=required,output_limit_bytes=c.TRAJECTORY_ALLOWANCE+8*1024**2,timeout_seconds=limit,parent_check=check)
    idle()
    with Guard(ROOT/f'control/parent-{stage}-{slot:02d}.json',limit-(time.monotonic()-start)):
        bind(writer,stage,sha)
        require(process['status']=='completed' and process['returncode']==0 and process['error'] is None and process['argv']==expected
            and process['parent_pid']==os.getpid() and not _group_exists(process['pid']) and process['log_truncated'] is False
            and not any(k in process for k in ('cleanup_error','log_close_error','failure_observation_error')),'Child did not exit cleanly')
        out=read(reference(ROOT/base/'worker.json'));guard=read(reference(ROOT/base/'worker-guard.json'))
        require(out['kind']==c.KIND and out['pid']==process['pid'] and out['parent_pid']==os.getpid() and out['request']==reference(ROOT/base/'request.json') and out['holdout_scored'] is False,'Worker provenance differs')
        require(guard['status']=='finished' and guard['reason'] is None and guard['pid']==process['pid'] and guard['limits']==dict(rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,timeout_seconds=limit,mps_driver_limit_bytes=6*1024**3),'Worker guard failed')
        req=out['acceptance_request'];require(req['sources']==plan['sources'] and req['config']==plan['configs'][slot] and req['seeds']==plan['seeds'] and req['stage']==label(stage) and req['slot']==slot,'Output config differs')
        accepted=a.accept(req,out['validation']);require(accepted['cache']==out['cache'],'Parent cache recomputation differs')
        writer.publish_json(base+'/accepted.json',dict(kind=c.KIND,stage=stage,slot=slot,freeze=dict(path=str(freeze_path(stage)),sha256=sha),previous=previous,
            request=reference(ROOT/base/'worker.json'),artifact_acceptance=accepted,process=reference(ROOT/base/'process/process.json'),quality_adopted=False))
    exclusive(anchor_path(stage,slot),dict(acceptance=reference(ROOT/base/'accepted.json'),parent_guard=reference(ROOT/f'control/parent-{stage}-{slot:02d}.json'),files=[ref_file(p) for p in sorted((ROOT/base).rglob('*')) if p.is_file()]))
    return dict(status='accepted',stage=stage,slot=slot)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--stage',choices=['pilots','quality'],required=True);p.add_argument('--slot',type=int,required=True);p.add_argument('--freeze-sha256',required=True);p.add_argument('--global-fd',type=int);p.add_argument('--campaign-fd',type=int);args=p.parse_args()
    torch.set_num_threads(4);require(args.slot==0 if args.stage=='pilots' else 0<=args.slot<3,'Finite slot required')
    if args.mode=='worker':worker(MetadataWriter(ROOT,(args.global_fd,args.campaign_fd)),args.stage,args.slot,args.freeze_sha256,[sys.executable,'-m',MODULE,*sys.argv[1:]])
    else:
        with session_lock(GLOBAL_LOCK.parent) as g,session_lock(ROOT) as lock:
            print(json.dumps(run_once(MetadataWriter(ROOT,(g,lock)),args.stage,args.slot,args.freeze_sha256)),flush=True)
if __name__=='__main__':main()
