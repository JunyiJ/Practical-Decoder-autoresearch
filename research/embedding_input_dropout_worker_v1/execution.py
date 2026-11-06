"""Shared fresh-worker entry; a prospective plan supplies exact configurations."""
import argparse,gc,json,os
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import source_inventory,environment,reference
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter
from .training import run

def sources():
    from research.embedding_input_dropout_resume_v1.contract import sources as predecessor_sources
    result=predecessor_sources()
    for ns in ('embedding_input_dropout_resume_guard_v1','embedding_input_dropout_worker_v1'):
        for p in sorted((ROOT/'research'/ns).glob('*.py')):result[str(p.relative_to(ROOT))]=reference(p)['sha256']
    return result

def checked(ref):
    require(reference(ref['path'])==ref,'Evidence reference changed');return json.loads(Path(ref['path']).read_text())

class Guard(StrictMpsGuard):
    def __init__(self,path,seconds=7200):
        require(seconds in (120,600,7200),'Declared original guard horizon required')
        super().__init__(path,ROOT);self.limits['timeout_seconds']=seconds

def worker(args):
    torch.set_num_threads(4);environment();root=Path(args.root).resolve()
    require(root.is_relative_to(ROOT/'research_runs') and not (root/'STOP').exists(),'Open research run required')
    writer=MetadataWriter(root,(args.global_fd,args.run_fd));base=f'evidence/{args.stage}/{args.slot:02d}'
    rq=checked(dict(path=str(root/base/'request.json'),sha256=args.request_sha));plan=checked(rq['plan'])
    require(rq['parent_pid']==os.getppid() and rq['slot']==args.slot and rq['stage']==args.stage
        and plan['root']==str(root) and plan['sources']==sources() and plan['environment']==environment(),'Worker request/source/environment differs')
    spec=rq['spec'];cfg=OmegaConf.create(spec['config'])
    require(spec in plan['specs'],'Unplanned numerical configuration')
    with Guard(root/f'control/worker-{args.stage}-{args.slot:02d}-guard.json',plan['seconds']):
        writer.publish_json(base+'/config.json',spec['config']);sink=CheckpointWriter(writer,args.stage,args.slot,spec['weight_cap'])
        with writer.open_append(base+'/metrics.jsonl') as stream:
            def emit(row):
                stream.write((json.dumps(row,allow_nan=False)+'\n').encode());print(json.dumps(row),flush=True)
            summary=run(cfg,sink,emit)
        summary['request']=reference(root/base/'request.json')
        writer.publish_json(base+'/worker.json',summary)
        require(sources()==plan['sources'],'Runtime source changed during training')
    print('Fresh bounded input-embedding-dropout worker completed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=['pilots','quality'],required=True)
    p.add_argument('--slot',type=int,required=True);p.add_argument('--global-fd',type=int,required=True);p.add_argument('--run-fd',type=int,required=True);p.add_argument('--request-sha',required=True)
    worker(p.parse_args())
