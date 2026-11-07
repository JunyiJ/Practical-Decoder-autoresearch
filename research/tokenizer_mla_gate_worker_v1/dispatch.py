import gc
import json
import os
import shutil
import sys
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_mla_gate_resume_v1.contract import runtime
from src.research.tokenizer_wide_hybrid_process_v2 import run_process
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from src.research.cpu_prerequisite_guard import observation
from .execution import sources,checked,Guard,idle
from .validation import accept


def launch(writer,plan_ref,spec,required):
    plan=checked(plan_ref);root=writer.root;stage=spec['stage'];slot=spec['slot'];base=f'evidence/{stage}/{slot:02d}'
    require(plan['sources']==sources() and plan['environment']==runtime() and spec in plan['specs'] and plan['seconds']==600 and plan['compare_cpu'] is True and plan['scientific_quality'] is False,'Dispatch source/environment/prerequisite scope differs')
    require(not (root/'STOP').exists() and not (root/base).exists(),'Attempted/closed slot cannot rerun')
    require(shutil.disk_usage(ROOT).free>=required>=8*1024**3,'Complete remaining capacity required');idle()
    writer.publish_json(base+'/request.json',dict(plan=plan_ref,spec=spec,parent_pid=os.getpid(),slot=slot,stage=stage));rq=reference(root/base/'request.json')
    argv=[sys.executable,'-m','research.tokenizer_mla_gate_worker_v1.execution','--root',str(root),'--stage',stage,'--slot',str(slot),'--global-fd',str(writer.lock_fds[0]),'--run-fd',str(writer.lock_fds[1]),'--request-sha',rq['sha256']]
    def parent_check():require(observation(ROOT)['peak_rss_bytes']<4*1024**3 and torch.mps.driver_allocated_memory()<6*1024**3,'Parent resource guard failed')
    process=run_process(writer,stage,slot,argv,ROOT,required_free_bytes=required,output_limit_bytes=spec['weight_cap']+10*1024**2,timeout_seconds=600,parent_check=parent_check)
    require(process['status']=='completed' and process['returncode']==0,'Worker did not finish cleanly');idle()
    guard=checked(reference(root/f'control/worker-{stage}-{slot:02d}-guard.json'));require(guard['status']=='finished' and guard['reason'] is None,'Worker resource guard failed')
    summary_ref=reference(root/base/'worker.json');summary=checked(summary_ref);require(summary['request']==rq,'Worker request ancestry differs')
    for name in ['parent','h200']:require(summary[name]['path']==str(root/base/'checkpoints'/(name+'.pt')),'Checkpoint must be the exact new slot artifact')
    config=checked(reference(root/base/'config.json'));require(config==spec['config'],'Published config differs')
    binding={n:dict(path=summary[n]['path'],sha256=summary[n]['sha256']) for n in ['parent','h200']};binding['config']=reference(root/base/'config.json')
    with Guard(root/f'control/parent-{stage}-{slot:02d}-guard.json'),torch.random.fork_rng(devices=[]):
        mps_rng=torch.mps.get_rng_state().clone()
        try:result=accept(summary,OmegaConf.create(config),binding,compare_cpu=True)
        finally:torch.mps.set_rng_state(mps_rng)
    require(plan['sources']==sources(),'Sources changed during parent acceptance')
    row=dict(slot=slot,arm=spec['arm'],request=rq,worker=summary_ref,process=reference(root/base/'process/process.json'),worker_guard=reference(root/f'control/worker-{stage}-{slot:02d}-guard.json'),parent_guard=reference(root/f'control/parent-{stage}-{slot:02d}-guard.json'),acceptance=result)
    writer.publish_json(base+'/accepted.json',row);snapshot(root)
    print(json.dumps(dict(status='independently_accepted',slot=slot,arm=spec['arm'],steps=summary['step'],validation_targets=34278)),flush=True)
    gc.collect();torch.mps.empty_cache();return row
