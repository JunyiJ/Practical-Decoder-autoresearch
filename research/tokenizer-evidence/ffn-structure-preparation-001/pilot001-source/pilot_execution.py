"""Exactly two serial scientific-corpus resource pilots, with independent acceptance."""
import argparse, gc, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from . import pilot_contract as c, pilot_artifacts as a, validation as v
from .runtime.train.train import run_training
from .runtime.train.weight_averaging import export_averaged_checkpoint
from research.ema_horizon_v1.execution import idle as inherited_idle, exclusive
from research.block_norm_v1.fixture_dispatch import reference, read
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
from src.research.tokenizer_serial_process import run_process, _group_exists
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
MODULE='research.ffn_structure_v1.pilot_execution'
FREEZE=c.EVIDENCE/'pilot001-freeze.json'
require=c.require

def ref(path):
    path=Path(path);limit=c.RAW_LIMIT if path.suffix=='.pt' else 2*1024**2
    return dict(path=str(path),sha256=c.file_sha(path,root=c.ROOT,max_bytes=limit))
def idle(allowed=()):
    inherited_idle(allowed)
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        row=line.strip().split(None,1)
        if len(row)==2 and 'research.ffn_structure_v1' in row[1] and Path(row[1].split()[0]).name.lower().startswith('python'):
            require(int(row[0]) in {os.getpid(),*allowed},'Other FFN worker is live')
def clear():gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
def anchor(slot):return c.EVIDENCE/f'pilot001-accepted-{slot:02d}.json'
def bind(sha):
    require(not (c.RUN/'STOP').exists(),'STOP blocks dispatch')
    seal=read(dict(path=str(FREEZE),sha256=sha));plan=read(seal['plan'])
    require(plan['kind']=='ffn_structure_two_resource_pilots_v1' and plan['root']==str(c.RUN) and plan['quality_launch_permitted'] is False,'Only frozen FFN pilots permitted')
    c.verify_sources(plan['sources'])
    require(plan['environment']==c.base.environment() and plan['data']==c.base.bind_data(),'Frozen environment/corpus differs')
    require(plan['configs']==[c.config(i,plan['seeds']) for i in range(2)],'Finite configs differ')
    require(plan['limits']==c.base.LIMITS and plan['required_free_bytes']==c.REQUIRED_FREE,'Resource limits differ')
    require(plan['trajectory_allowance_bytes']==c.TRAJECTORY_ALLOWANCE,'Allocation differs')
    for r in plan['prerequisites'].values():read(r)
    return plan

def prefix(slot):
    c.output(slot);used=0;prior=[]
    for n in range(slot):
        receipt=read(ref(anchor(n)))
        require(receipt['slot']==n and receipt['status']=='accepted','Predecessor not accepted')
        require(read(receipt['parent_guard'])['status']=='finished','Predecessor guard failed')
        for item in receipt['files']:
            require(ref(item['path'])==item,'Accepted predecessor changed')
            used+=Path(item['path']).stat().st_size
        prior.append(ref(anchor(n)))
    require(all(not c.output(n).exists() for n in range(slot,2)),'Existing/future attempts cannot retry')
    return prior,used

class CappedWriter:
    def __init__(self,f,limit):self.f,self.limit=f,limit
    def tell(self):return self.f.tell()
    def flush(self):return self.f.flush()
    def write(self,data):
        require(self.tell()+memoryview(data).nbytes<=self.limit,'Serialization cap exceeded')
        return self.f.write(data)

def save_weight(path,payload,limit):
    with path.open('xb') as f:
        torch.save(payload,CappedWriter(f,limit));f.flush();os.fsync(f.fileno())

class Sink:
    def __init__(self,root,cfg):self.root,self.cfg,self.raw,self.count=root,cfg,None,0
    def validate_config(self,cfg):require(OmegaConf.to_container(cfg,resolve=True)==self.cfg,'Sink config differs')
    def checkpoint(self,path,payload):
        path=Path(path)
        require(path.parent==self.root/'checkpoints' and path.name.startswith('final_') and path.suffix=='.pt' and self.raw is None and payload['step']==20,'Only one final20 checkpoint allowed')
        save_weight(path,payload,c.RAW_LIMIT);self.raw=path
    def metric(self,path,value):
        require(Path(path)==self.root/'metrics.jsonl' and self.count<2 and value['step']==(0,19)[self.count],'Metric sequence differs')
        line=(c.canonical(value)+'\n').encode();existing=Path(path).stat().st_size if Path(path).exists() else 0
        require(existing+len(line)<=2*1024**2,'Metrics cap exceeded')
        with Path(path).open('xb' if not self.count else 'ab') as f:f.write(line);f.flush();os.fsync(f.fileno())
        self.count+=1
    def summary(self,path,value):
        require(Path(path)==self.root/'summary.json','Summary path differs');exclusive(Path(path),value)

class Guard(StrictMpsGuard):
    def __init__(self,path,seconds_left=600):
        require(0<seconds_left<=600,'Finite remaining pilot deadline required')
        super().__init__(path,c.ROOT);self.limits['timeout_seconds']=seconds_left

def worker(args):
    root=c.output(args.slot)
    for fd,path in [(args.global_fd,GLOBAL_LOCK),(args.run_fd,c.RUN/'.lock')]:
        st=os.fstat(fd);actual=path.stat();require((st.st_dev,st.st_ino)==(actual.st_dev,actual.st_ino),'Inherited lock differs')
    with Guard(root/'worker-guard.json'):
        plan=bind(args.freeze_sha);rq=read(dict(path=str(root/'request.json'),sha256=args.request_sha))
        require(rq['parent_pid']==os.getppid() and rq['slot']==args.slot and rq['freeze']==dict(path=str(FREEZE),sha256=args.freeze_sha),'Worker request differs')
        idle((os.getppid(),));cfg=plan['configs'][args.slot]
        exclusive(root/'config.json',cfg);(root/'checkpoints').mkdir();sink=Sink(root,cfg)
        summary=run_training(OmegaConf.create(cfg),output_sink=sink)
        require(summary['completed_steps']==20 and summary['stop_reason']=='max_iters' and sink.raw is not None and sink.count==2,'Incomplete pilot')
        clear();derived=root/'checkpoints/h200.pt'
        def writer(path,payload):require(Path(path)==derived,'Export path differs');save_weight(derived,payload,c.EXPORT_LIMIT)
        exported=export_averaged_checkpoint(sink.raw,derived,200,payload_writer=writer);exclusive(root/'export.json',exported);clear()
        files=dict(config=ref(root/'config.json'),parent=ref(sink.raw),h200=ref(derived),summary=ref(root/'summary.json'),metrics=ref(root/'metrics.jsonl'),export=ref(root/'export.json'))
        req=dict(scope=a.SCOPE,sources=plan['sources'],config=cfg,files=files,slot=args.slot,seeds=plan['seeds'])
        a.verify_request(req);binding={k:files[k] for k in ('config','parent','h200')}
        report=v.evaluate(binding,'mps');bind(args.freeze_sha)
        exclusive(root/'worker.json',dict(request=ref(root/'request.json'),pid=os.getpid(),parent_pid=os.getppid(),acceptance_request=req,validation=report))
    print('resource pilot completed',c.ARMS[args.slot],flush=True)

def run(args):
    with session_lock(GLOBAL_LOCK.parent) as g,session_lock(c.RUN) as r:
        idle();root=c.output(args.slot)
        with CpuPrerequisiteGuard(c.EVIDENCE/f'pilot001-entry-{args.slot:02d}-guard.json',c.ROOT,timeout_seconds=120,poll_seconds=.25):
            plan=bind(args.freeze_sha);previous,used=prefix(args.slot)
            required=c.REQUIRED_FREE-used;require(shutil.disk_usage(c.ROOT).free>=required,'Remaining full-panel capacity insufficient')
            root.mkdir()
            rq=dict(parent_pid=os.getpid(),slot=args.slot,freeze=dict(path=str(FREEZE),sha256=args.freeze_sha),previous=previous)
            exclusive(root/'request.json',rq)
        # Parent watchdog remains live during child dispatch and independent acceptance.
        with Guard(root/'parent-guard.json'):
            started=time.monotonic()
            argv=[sys.executable,'-m',MODULE,'worker','--slot',str(args.slot),'--freeze-sha',args.freeze_sha,'--request-sha',ref(root/'request.json')['sha256'],'--global-fd',str(g),'--run-fd',str(r)]
            process=run_process(argv,c.ROOT,root/'process',(g,r),required_free_bytes=required,
                output_limit_bytes=c.TRAJECTORY_ALLOWANCE,timeout_seconds=600-(time.monotonic()-started),watch_root=root)
            require(process['status']=='completed' and process['returncode']==0 and process['error'] is None and not _group_exists(process['pid']),'Child not cleanly reaped')
            idle();bind(args.freeze_sha);out=read(ref(root/'worker.json'));guard=read(ref(root/'worker-guard.json'))
            require(out['pid']==process['pid'] and out['parent_pid']==os.getpid() and out['request']==ref(root/'request.json'),'Worker provenance differs')
            require(guard['status']=='finished' and guard['reason'] is None and guard['pid']==process['pid'],'Worker guard failed')
            req=out['acceptance_request'];require(req['slot']==args.slot and req['config']==plan['configs'][args.slot] and req['seeds']==plan['seeds'],'Worker config differs')
            accepted=a.accept(req,out['validation']);clear();bind(args.freeze_sha)
            exclusive(root/'accepted.json',dict(status='accepted',slot=args.slot,arm=c.ARMS[args.slot],freeze=rq['freeze'],previous=previous,process=ref(root/'process/process.json'),artifact_acceptance=accepted,quality_adopted=False))
        exclusive(anchor(args.slot),dict(status='accepted',slot=args.slot,acceptance=ref(root/'accepted.json'),parent_guard=ref(root/'parent-guard.json'),
            files=[ref(p) for p in sorted(root.rglob('*')) if p.is_file()]))
        idle();print(json.dumps(dict(status='accepted',slot=args.slot,arm=c.ARMS[args.slot],quality_adopted=False)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--slot',type=int,required=True);p.add_argument('--freeze-sha',required=True);p.add_argument('--request-sha');p.add_argument('--global-fd',type=int);p.add_argument('--run-fd',type=int);args=p.parse_args()
    torch.set_num_threads(4);c.output(args.slot)
    if args.mode=='worker':worker(args);return
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    try:run(args)
    except BaseException as exc:
        if c.RUN.exists():
            (c.RUN/'STOP').touch(exist_ok=True)
            path=c.EVIDENCE/f'pilot001-slot{args.slot:02d}-failure.json'
            if not path.exists():exclusive(path,dict(error=repr(exc),slot=args.slot,source=ref(Path(__file__)),freeze_sha=args.freeze_sha))
        raise
if __name__=='__main__':main()
