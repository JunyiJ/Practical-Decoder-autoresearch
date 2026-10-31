"""Finite two-pilot execution using existing bounded IO/process primitives.

No quality dispatch, cleanup, implicit resume or retry. Each successful pilot has
an immutable external acceptance anchor; an interrupted slot stays blocked.
"""
import argparse,copy,gc,json,os,shutil,sys,time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from . import campaign_contract as c,campaign_rules as rules,artifact_acceptance as a,validation as v
from .mps_guard import StrictMpsGuard
from .runtime.train.train import run_training
from .runtime.train.weight_averaging import export_averaged_checkpoint
from .runtime.utils.checkpoint import load_model_weights,read_checkpoint
from .runtime.attention.cache import build_caches
from src.research.tokenizer_wide_hybrid_measurements import check_kernel
from src.research.tokenizer_wide_hybrid_timing import allocated_bytes,token_digest
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter
from src.research.tokenizer_wide_hybrid_process_v2 import run_process
from src.research.tokenizer_serial_process import _group_exists
from src.research.cpu_prerequisite_guard import observation,violation,CpuPrerequisiteGuard
from .fixture_dispatch import reference,read

ROOT=c.ROOT/'research_runs/tokenizer-block-norm-v2-pilots-001'
EVIDENCE=c.ROOT/'research/tokenizer-evidence/block-norm-preparation-001'
FREEZE=EVIDENCE/'pilot001-freeze.json'
MODULE='research.block_norm_v1.pilot_execution'
KIND='block_norm_v2_finite_pilots_v1'
FULL_FREE=15170221264
ALLOWANCES=(788953140,789009664)
require=c.require


def exclusive(path,value):
    raw=(c.canonical(value)+'\n').encode();require(len(raw)<2*1024**2,'Bounded external receipt required')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def bind(writer,freeze_sha):
    require(writer.root==ROOT,'Dedicated finite pilot root required')
    frozen=read(dict(path=str(FREEZE),sha256=freeze_sha));plan=read(frozen['plan'])
    require(plan['kind']==KIND and plan['root']==str(ROOT) and plan['status']=='frozen_pilots_only','Frozen pilots only')
    require(plan['limits']==dict(cpu_threads=4,seconds=600,rss_bytes=4*1024**3,driver_bytes=6*1024**3,reserve_bytes=8*1024**3,full_required_free_bytes=FULL_FREE,checkpoint_allowances=list(ALLOWANCES)),'Frozen resource limits differ')
    c.verify_sources(plan['sources']);c.frozen_predecessors(plan['sources'])
    require(c.environment()==plan['environment'] and c.bind_data()==plan['data'],'Frozen environment/data differ')
    for ref in plan['prerequisites'].values():read(ref)
    require(plan['specs']==rules.pilot_specs(plan['seeds']),'Finite pilot order differs')
    for slot,spec in enumerate(plan['specs']):c.validate_trial_config(plan['configs'][slot],ROOT/f'evidence/pilots/{slot:02d}',spec,plan['seeds'])
    require(not (ROOT/'STOP').exists(),'STOP requested')
    return plan


def worker_argv(writer,slot,freeze_sha):
    return [sys.executable,'-m',MODULE,'worker','--slot',str(slot),'--freeze-sha256',freeze_sha,
            '--global-fd',str(writer.lock_fds[0]),'--campaign-fd',str(writer.lock_fds[1])]


class Sink:
    def __init__(self,writer,slot,cfg):
        self.writer=writer;self.slot=slot;self.cfg=cfg;self.base=f'evidence/pilots/{slot:02d}';self.raw=None;self.stream=None
        self.weights=CheckpointWriter(writer,'pilots',slot,ALLOWANCES[slot])
    def validate_config(self,cfg):require(c.canonical(OmegaConf.to_container(cfg,resolve=True))==c.canonical(self.cfg),'Sink config differs')
    def checkpoint(self,path,payload):
        path=Path(path);require(self.raw is None and path.parent==ROOT/self.base/'checkpoints' and payload['step']==20,'One final pilot checkpoint required')
        self.weights.save(path.name,payload);self.raw=path
    def metric(self,path,value):
        require(Path(path)==ROOT/self.base/'metrics.jsonl','Metric path differs')
        if self.stream is None:self.stream=self.writer.open_append(self.base+'/metrics.jsonl')
        self.stream.write((c.canonical(value)+'\n').encode())
    def summary(self,path,value):
        require(Path(path)==ROOT/self.base/'training-summary.json','Summary path differs')
        self.writer.publish_json(self.base+'/training-summary.json',value)
    def close(self):
        if self.stream is not None:self.stream.close()


def file_ref(path,weight=False):return dict(path=str(path),sha256=c.file_sha(path,root=c.ROOT,max_bytes=(512 if weight else 2)*1024**2))


def cache_audit(request):
    binding={k:request['files'][k] for k in ('config','parent','h200')}
    model,data,cfg=v.load(binding,'mps');ids=data.train_data[:96].tolist();rows=[]
    try:
        for endpoint in ('parent','h200'):
            load_model_weights(model,read_checkpoint(binding[endpoint]['path']))
            for batch in (1,8):
                fixed=torch.tensor(ids,device='mps',dtype=torch.int64).unsqueeze(0).repeat(batch,1)
                factory=lambda:build_caches(cfg,'mps',batch_size=batch,dtype=torch.float32)
                result=check_kernel(model,fixed,factory);cache=factory()
                state=allocated_bytes(b for layer in cache for b in layer.buffers());require(state==196608*batch,'Pilot cache storage differs')
                rows.append(dict(endpoint=endpoint,batch=batch,state_bytes=state,**result));del cache,fixed
        return dict(training96_sha256=token_digest(ids),rows=rows)
    finally:del model,data,cfg;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()


def worker(writer,slot,freeze_sha,actual_argv):
    base=f'evidence/pilots/{slot:02d}';sink=None
    with StrictMpsGuard(ROOT/f'control/worker-{slot:02d}-guard.json',c.ROOT):
        plan=bind(writer,freeze_sha);rq=read(reference(ROOT/base/'request.json'))
        require(rq==dict(plan=read(dict(path=str(FREEZE),sha256=freeze_sha))['plan'],slot=slot,parent_pid=os.getppid(),argv=actual_argv),'Parent/request/argv differs')
        require(actual_argv==worker_argv(writer,slot,freeze_sha),'Worker argv differs')
        idle(allowed=(os.getppid(),));parent=os.getppid();cfg=plan['configs'][slot]
        writer.publish_json(base+'/config.json',cfg)
        sink=Sink(writer,slot,cfg)
        try:summary=run_training(OmegaConf.create(copy.deepcopy(cfg)),output_sink=sink)
        finally:sink.close()
        require(summary['completed_steps']==20 and summary['stop_reason']=='max_iters','Pilot incomplete')
        gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
        avg=ROOT/base/'checkpoints/h200.pt'
        def export_writer(path,payload):
            require(path==avg,'H200 output path differs');sink.weights.save(path.name,payload)
        export=export_averaged_checkpoint(sink.raw,avg,200,payload_writer=export_writer)
        writer.publish_json(base+'/export.json',export)
        files={k:file_ref(path,k in ('parent','h200')) for k,path in dict(config=ROOT/base/'config.json',parent=sink.raw,h200=avg,
             summary=ROOT/base/'training-summary.json',metrics=ROOT/base/'metrics.jsonl',export=ROOT/base/'export.json').items()}
        acceptance=dict(scope=a.TRIAL_SCOPE,sources=plan['sources'],config=cfg,files=files,trial=dict(spec=plan['specs'][slot],seeds=plan['seeds']))
        a.verify_request(acceptance)
        validation=v.evaluate({k:files[k] for k in ('config','parent','h200')});cached=cache_audit(acceptance)
        bind(writer,freeze_sha);require(os.getppid()==parent,'Parent disappeared')
        writer.publish_json(base+'/worker.json',dict(kind=KIND,pid=os.getpid(),parent_pid=parent,request=reference(ROOT/base/'request.json'),
            acceptance_request=acceptance,validation=validation,cache=cached,quality_training_launched=False,holdout_scored=False))
    writer.publish_json(base+'/worker-guard.json',json.loads((ROOT/f'control/worker-{slot:02d}-guard.json').read_text()))


def run_once(writer,slot,freeze_sha):
    require(type(slot) is int and slot in (0,1),'Exactly two pilot slots')
    with CpuPrerequisiteGuard(ROOT/f'control/entry-{slot:02d}-guard.json',c.ROOT,timeout_seconds=120,poll_seconds=.25):
        plan=bind(writer,freeze_sha);base=f'evidence/pilots/{slot:02d}'
        require(not (ROOT/base).exists(),'Existing pilot attempt cannot retry')
        previous=dict(path=str(FREEZE),sha256=freeze_sha)
        if slot:
            prior=read(reference(EVIDENCE/'pilot001-accepted-00.json'))
            require(prior['previous']==previous,'External pilot acceptance anchor differs')
            require(read(prior['acceptance'])['slot']==0,'Previous pilot not accepted')
            for ref in prior['files']:read(ref) if ref['path'].endswith('.json') else require(file_ref(Path(ref['path']),ref['path'].endswith('.pt'))==ref,'Previous pilot file changed')
            previous=reference(EVIDENCE/'pilot001-accepted-00.json')
        idle();require(shutil.disk_usage(ROOT).free>=FULL_FREE,'Full projected campaign capacity required')
        expected=worker_argv(writer,slot,freeze_sha)
        writer.publish_json(base+'/request.json',dict(plan=read(dict(path=str(FREEZE),sha256=freeze_sha))['plan'],slot=slot,parent_pid=os.getpid(),argv=expected))
    started=time.monotonic()
    def check():
        reason=violation(observation(ROOT),time.monotonic()-started,4*1024**3,8*1024**3,600)
        require(reason is None,'Parent resource limit: '+str(reason))
        require(torch.mps.driver_allocated_memory()<6*1024**3,'Parent MPS limit')
    process=run_process(writer,'pilots',slot,expected,c.ROOT,required_free_bytes=FULL_FREE,
        output_limit_bytes=ALLOWANCES[slot]+8*1024**2,timeout_seconds=600,parent_check=check)
    idle()
    with StrictMpsGuard(ROOT/f'control/parent-{slot:02d}-guard.json',c.ROOT):
        bind(writer,freeze_sha)
        require(process['status']=='completed' and process['returncode']==0 and process['error'] is None
            and process['argv']==expected and process['parent_pid']==os.getpid() and not _group_exists(process['pid'])
            and process['log_truncated'] is False and not any(k in process for k in ('cleanup_error','log_close_error','failure_observation_error')),'Pilot process did not complete cleanly')
        out=read(reference(ROOT/base/'worker.json'));guard=read(reference(ROOT/base/'worker-guard.json'))
        require(out['kind']==KIND and out['pid']==process['pid'] and out['parent_pid']==os.getpid()
                and out['request']==reference(ROOT/base/'request.json') and out['holdout_scored'] is False,'Worker output identity differs')
        require(guard['status']=='finished' and guard['reason'] is None and guard['pid']==process['pid']
                and guard['limits']==dict(rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,timeout_seconds=600,mps_driver_limit_bytes=6*1024**3),'Worker guard did not finish')
        req=out['acceptance_request'];require(req['sources']==plan['sources'] and req['config']==plan['configs'][slot]
            and req['trial']==dict(spec=plan['specs'][slot],seeds=plan['seeds']),'Actual output trial differs')
        accepted=a.accept(req,out['validation']);cached=cache_audit(req)
        require(c.canonical(cached)==c.canonical(out['cache']),'Parent cache recomputation differs')
        receipt=dict(kind=KIND,slot=slot,freeze=previous,artifact_acceptance=accepted,cache=cached,process=reference(ROOT/base/'process/process.json'),worker_guard=reference(ROOT/base/'worker-guard.json'),quality_plan_frozen=False,cleanup_authorized=False)
        writer.publish_json(base+'/accepted.json',receipt)
    # Publish the external durable anchor only after the parent guard closes.
    files=[file_ref(path,path.suffix=='.pt') for path in sorted((ROOT/base).rglob('*')) if path.is_file()]
    exclusive(EVIDENCE/f'pilot001-accepted-{slot:02d}.json',dict(kind=KIND,previous=previous,acceptance=reference(ROOT/base/'accepted.json'),
        parent_guard=reference(ROOT/f'control/parent-{slot:02d}-guard.json'),files=files))
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker']);parser.add_argument('--slot',required=True,type=int)
    parser.add_argument('--freeze-sha256',required=True);parser.add_argument('--global-fd',type=int);parser.add_argument('--campaign-fd',type=int);args=parser.parse_args()
    torch.set_num_threads(4);require(args.slot in (0,1),'Finite pilot slot')
    if args.mode=='worker':
        writer=MetadataWriter(ROOT,(args.global_fd,args.campaign_fd))
        worker(writer,args.slot,args.freeze_sha256,[sys.executable,'-m',MODULE,*sys.argv[1:]])
    else:
        with session_lock(GLOBAL_LOCK.parent) as g,session_lock(ROOT) as p:
            result=run_once(MetadataWriter(ROOT,(g,p)),args.slot,args.freeze_sha256)
            print(json.dumps(dict(status='pilot_accepted',slot=args.slot,audit=result['artifact_acceptance']['audit'])))


if __name__=='__main__':main()
