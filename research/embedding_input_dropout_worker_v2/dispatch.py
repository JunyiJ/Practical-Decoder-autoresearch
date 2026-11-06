"""Serial fresh launch and independent parent acceptance; no implicit retry."""
import gc,json,os,shutil,sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.final_ln_scale_quality_v3.support import idle
from src.research.tokenizer_wide_hybrid_process_v2 import run_process
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .execution import sources,checked,Guard
from .validation import accept
from src.research.cpu_prerequisite_guard import observation

def launch(writer,plan_ref,spec,required):
    plan=checked(plan_ref);root=writer.root;stage=spec['stage'];slot=spec['slot'];base=f'evidence/{stage}/{slot:02d}'
    require(plan['sources']==sources() and plan['environment']==environment() and spec in plan['specs'],'Dispatch source/environment/config differs')
    require(not (root/'STOP').exists() and not (root/base).exists(),'Attempted/closed slot cannot rerun')
    require(shutil.disk_usage(ROOT).free>=required>=8*1024**3,'Complete remaining capacity required')
    idle()
    writer.publish_json(base+'/request.json',dict(plan=plan_ref,spec=spec,parent_pid=os.getpid(),slot=slot,stage=stage))
    rq=reference(root/base/'request.json')
    argv=[sys.executable,'-m','research.embedding_input_dropout_worker_v2.execution','--root',str(root),'--stage',stage,'--slot',str(slot),
        '--global-fd',str(writer.lock_fds[0]),'--run-fd',str(writer.lock_fds[1]),'--request-sha',rq['sha256']]
    def parent_check():
        require(observation(ROOT)['peak_rss_bytes']<4*1024**3 and torch.mps.driver_allocated_memory()<6*1024**3,'Parent RSS/MPS resource guard failed')
    process=run_process(writer,stage,slot,argv,ROOT,required_free_bytes=required,output_limit_bytes=spec['weight_cap']+10*1024**2,timeout_seconds=plan['seconds'],parent_check=parent_check)
    require(process['status']=='completed' and process['returncode']==0,'Worker did not finish cleanly');idle()
    guard=checked(reference(root/f'control/worker-{stage}-{slot:02d}-guard.json'))
    require(guard['status']=='finished' and guard['reason'] is None,'Worker resource guard failed')
    worker_ref=reference(root/base/'worker.json');summary=checked(worker_ref)
    require(summary['request']==rq,'Worker request provenance differs')
    config=checked(reference(root/base/'config.json'));require(config==spec['config'],'Published config differs')
    binding={n:dict(path=summary[n]['path'],sha256=summary[n]['sha256']) for n in ('parent','h200')};binding['config']=reference(root/base/'config.json')
    with Guard(root/f'control/parent-{stage}-{slot:02d}-guard.json',plan['seconds']),torch.random.fork_rng(devices=[]):
        result=accept(summary,OmegaConf.create(config),binding,compare_cpu=plan['compare_cpu'])
    require(plan['sources']==sources(),'Sources changed during acceptance')
    row=dict(slot=slot,arm=spec['arm'],request=rq,worker=worker_ref,process=reference(root/base/'process/process.json'),
        worker_guard=reference(root/f'control/worker-{stage}-{slot:02d}-guard.json'),parent_guard=reference(root/f'control/parent-{stage}-{slot:02d}-guard.json'),acceptance=result)
    writer.publish_json(base+'/accepted.json',row);snapshot(root)
    print(json.dumps(dict(status='independently_accepted',slot=slot,arm=spec['arm'],steps=summary['step'],ce=result['full_validation']['ce'])),flush=True)
    gc.collect();torch.mps.empty_cache();return row
