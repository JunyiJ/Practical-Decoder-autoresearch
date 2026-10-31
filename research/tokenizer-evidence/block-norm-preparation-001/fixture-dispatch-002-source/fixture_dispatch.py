"""Serial readonly fixture workers with real parent recomputation; no training CLI.

Reuses the frozen bounded pipe supervisor and lock-aware metadata writer. This
is deliberately limited to retained two-step fixtures, not a campaign ledger.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import torch
from . import artifact_acceptance as a, campaign_contract as c, validation as v
from .mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import observation, violation
from src.research.tokenizer_activation_entry import idle
from src.research.tokenizer_serial_process import _group_exists
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_process_v2 import run_process

MODULE='research.block_norm_v1.fixture_dispatch'
KIND='block_norm_readonly_fixture_dispatch_v1'
RESERVE=8*1024**3
OUTPUT=8*1024**2
require=c.require


def stem(slot):
    require(type(slot) is int and slot in (0,1),'Two readonly fixture slots only')
    return f'evidence/pilots/{slot:02d}'


def reference(path):
    return dict(path=str(path),sha256=c.file_sha(path,root=c.ROOT,max_bytes=2*1024**2))


def read(ref):
    require(type(ref) is dict and set(ref)=={'path','sha256'},'Exact document reference required')
    require(reference(Path(ref['path']))==ref,'Document hash differs')
    return json.loads(Path(ref['path']).read_text())


def argv(writer,slot,seal_ref):
    return [sys.executable,'-m',MODULE,'--root',str(writer.root),'--slot',str(slot),
            '--seal-sha256',seal_ref['sha256'],'--global-fd',str(writer.lock_fds[0]),
            '--campaign-fd',str(writer.lock_fds[1])]


def verify_entry(writer,slot,seal_ref,actual_argv):
    base=stem(slot)
    require(seal_ref['path']==str(writer.root/base/'seal.json'),'Unexpected seal location')
    seal=read(seal_ref)
    require(set(seal)=={'kind','slot','parent_pid','request','environment','lock_fds'},'Exact dispatch seal required')
    require(seal['kind']==KIND and seal['slot']==slot,'Dispatch scope/slot differs')
    require(type(seal['parent_pid']) is int and seal['parent_pid']==os.getppid(),'Live parent identity differs')
    require(seal['lock_fds']==list(writer.lock_fds),'Inherited lock descriptors differ')
    require(actual_argv==argv(writer,slot,seal_ref),'Exact worker argv required')
    require(not (writer.root/'STOP').exists(),'STOP requested')
    require(seal['environment']==c.environment(),'Worker environment differs')
    request=read(seal['request']);a.verify_request(request)
    require(request['config']['training']['max_iters']==2,'Only retained two-step fixture permitted')
    require(not (writer.root/base/'worker.json').exists(),'Worker attempt already completed')
    return seal,request


def worker(writer,slot,seal_ref,actual_argv):
    base=stem(slot);guard_path=writer.root/f'control/worker-{slot:02d}-guard.json'
    with StrictMpsGuard(guard_path,c.ROOT):
        seal,request=verify_entry(writer,slot,seal_ref,actual_argv)
        idle(allowed=(seal['parent_pid'],))
        binding={k:request['files'][k] for k in ('config','parent','h200')}
        report=v.evaluate(binding)
        require(os.getppid()==seal['parent_pid'],'Parent disappeared')
        a.verify_request(request)
        result=dict(kind=KIND,pid=os.getpid(),parent_pid=os.getppid(),argv=actual_argv,
                    seal=seal_ref,request=seal['request'],environment=c.environment(),
                    validation=report,optimizer_updates=0,holdout_scored=False)
        writer.publish_json(base+'/worker.json',result)
    writer.publish_json(base+'/worker-guard.json',json.loads(guard_path.read_text()))


def verify_completion(writer,slot,seal_ref,expected_argv,process,result,guard):
    base=stem(slot);seal=read(seal_ref)
    require(process['status']=='completed' and process['returncode']==0 and process['error'] is None,'Worker did not complete')
    require(process['argv']==expected_argv and process['parent_pid']==os.getpid(),'Process identity differs')
    require(type(process['pid']) is int and process['pid']>1 and not _group_exists(process['pid']),'Child process/group still live')
    require(process['capture_kind']=='bounded_pipe_v2' and process['log_truncated'] is False,'Bounded pipe completion required')
    require(process['timeout_seconds']==600 and process['required_free_bytes']==RESERVE+OUTPUT
            and process['output_limit_bytes']==OUTPUT and process['log_limit_bytes']==4*1024**2,'Process limits differ')
    require(not any(k in process for k in ('cleanup_error','log_close_error','failure_observation_error')),'Process cleanup failed')
    require(set(result)=={'kind','pid','parent_pid','argv','seal','request','environment','validation','optimizer_updates','holdout_scored'},'Exact worker report required')
    require(result['kind']==KIND and result['pid']==process['pid'] and result['parent_pid']==os.getpid()
            and result['argv']==expected_argv,'Worker report identity differs')
    require(result['seal']==seal_ref and result['request']==seal['request'],'Worker request reference differs')
    require(result['environment']==seal['environment']==c.environment(),'Worker/parent environment differs')
    require(type(result['optimizer_updates']) is int and result['optimizer_updates']==0 and result['holdout_scored'] is False,'Readonly scope violated')
    require(guard['status']=='finished' and guard['reason'] is None and guard['pid']==process['pid'],'Worker guard did not finish')
    require(guard['limits']==dict(rss_limit_bytes=4*1024**3,reserve_bytes=RESERVE,timeout_seconds=600,mps_driver_limit_bytes=6*1024**3),'Worker guard limits differ')
    sample=guard['observation']
    require(0<guard['elapsed_seconds']<600 and sample['peak_rss_bytes']<4*1024**3
            and sample['peak_observed_mps_driver_bytes']<6*1024**3 and sample['free_bytes']>=RESERVE,'Worker resources crossed limit')
    require(not (writer.root/'STOP').exists(),'STOP before acceptance')
    return read(seal['request'])


def dispatch_fixture(writer,slot,request):
    base=stem(slot)
    require(not (writer.root/'STOP').exists(),'STOP requested')
    require(not (writer.root/base).exists(),'Reserved/completed/failed slot cannot be dispatched again')
    if slot==1:
        previous=writer.root/stem(0)/'accepted.json'
        prior=read(reference(previous))
        require(prior['kind']==KIND and prior['artifact_acceptance']['accepted'] is True,'Previous worker must be accepted')
    a.verify_request(request)
    require(request['config']['training']['max_iters']==2,'Only retained two-step fixture permitted')
    idle();started=time.monotonic()
    def check():
        why=violation(observation(writer.root),time.monotonic()-started,4*1024**3,RESERVE,600)
        require(why is None,'Parent supervisor resource limit: '+str(why))
        require(torch.mps.driver_allocated_memory()<6*1024**3,'Parent MPS driver limit')
    check()
    request_path=writer.publish_json(base+'/request.json',request)
    seal= dict(kind=KIND,slot=slot,parent_pid=os.getpid(),request=reference(request_path),
               environment=c.environment(),lock_fds=list(writer.lock_fds))
    seal_path=writer.publish_json(base+'/seal.json',seal);seal_ref=reference(seal_path)
    expected_argv=argv(writer,slot,seal_ref)
    # No hard-exit watchdog around a live child: the supervisor terminates and
    # reaps its own process group on parent_check/time/disk/STOP/log failures.
    process=run_process(writer,'pilots',slot,expected_argv,c.ROOT,
        required_free_bytes=RESERVE+OUTPUT,output_limit_bytes=OUTPUT,timeout_seconds=600,parent_check=check)
    idle()
    with StrictMpsGuard(writer.root/f'control/parent-{slot:02d}-guard.json',c.ROOT):
        result=read(reference(writer.root/base/'worker.json'))
        guard=read(reference(writer.root/base/'worker-guard.json'))
        checked=verify_completion(writer,slot,seal_ref,expected_argv,process,result,guard)
        accepted=a.accept(checked,result['validation'])
        require(not (writer.root/'STOP').exists(),'STOP before publication')
        receipt=dict(kind=KIND,slot=slot,seal=seal_ref,
            process=reference(writer.root/base/'process/process.json'),worker=reference(writer.root/base/'worker.json'),
            guard=reference(writer.root/base/'worker-guard.json'),artifact_acceptance=accepted,
            quality_training_launched=False,quality_adopted=False)
        writer.publish_json(base+'/accepted.json',receipt)
    gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--slot',type=int,required=True)
    parser.add_argument('--seal-sha256',required=True);parser.add_argument('--global-fd',type=int,required=True);parser.add_argument('--campaign-fd',type=int,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    writer=MetadataWriter(Path(args.root),(args.global_fd,args.campaign_fd))
    seal_ref=dict(path=str(writer.root/stem(args.slot)/'seal.json'),sha256=args.seal_sha256)
    worker(writer,args.slot,seal_ref,[sys.executable,'-m',MODULE,*sys.argv[1:]])


if __name__=='__main__':main()
