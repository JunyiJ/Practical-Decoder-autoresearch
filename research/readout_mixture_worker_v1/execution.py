import argparse
import json
import os
from pathlib import Path
import subprocess
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.readout_mixture_resume_v1.contract import runtime
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter
from .training import run


def checked(ref):
    require(reference(ref['path'])==ref,'Evidence reference changed');return json.loads(Path(ref['path']).read_text())


def sources():
    e=ROOT/'research/tokenizer-evidence/post-gate-boundary-001'
    proof=checked(reference(e/'readout-resume001/STOP'))
    require(proof['status']=='complete_generated_readout_resume_scorer_proof','Complete own resume/scorer proof required')
    pins=dict(checked(reference(e/'readout-worker-seed-census001/plan.json'))['sources']);require(len(pins)==1102,'All1101 predecessors plus worker role census required')
    for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Immutable predecessor changed: '+p)
    for p in sorted(Path(__file__).parent.glob('*.py')):pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
    for name in ['readout-worker001.py','readout-worker-source001.py']:
        p=e/name;pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
    require(len(pins)==1109,'Five own worker files and frozen builder/source gate required');return pins


def idle():
    from research.final_ln_scale_quality_v3.support import idle as inherited_idle
    inherited_idle()
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        row=line.strip().split(None,1)
        if len(row)==2 and Path(row[1].split()[0]).name.lower().startswith('python') and any(v in row[1] for v in ('research.tokenizer_','research.readout_','readout-worker','readout-pilot','readout-resume')):
            require(int(row[0])==os.getpid(),'Another numerical research process is active')


class Guard(StrictMpsGuard):
    def __init__(self,path,seconds=600):
        require(seconds==600,'Original prerequisite horizon600s required');super().__init__(path,ROOT)


def worker(args):
    torch.set_num_threads(4);runtime();root=Path(args.root).resolve()
    require(root.is_relative_to(ROOT/'research_runs') and not (root/'STOP').exists(),'Open own research root required')
    writer=MetadataWriter(root,(args.global_fd,args.run_fd));base=f'evidence/{args.stage}/{args.slot:02d}'
    rq=checked(dict(path=str(root/base/'request.json'),sha256=args.request_sha));plan=checked(rq['plan'])
    require(rq['parent_pid']==os.getppid() and rq['slot']==args.slot and rq['stage']==args.stage and plan['root']==str(root)
        and plan['sources']==sources() and plan['environment']==runtime() and plan['seconds']==600 and plan['compare_cpu'] is True,'Worker request/source/environment differs')
    spec=rq['spec'];cfg=OmegaConf.create(spec['config'])
    from research.body_allocation_model_v1.contract import fingerprint,plain
    require(plain(cfg.readout_execution)==dict(kind='readout_prerequisite_execution_v1',sources_sha256=fingerprint(plan['sources']),runner_sha256=plan['runner']['sha256']),'Complete execution binding differs')
    require(spec in plan['specs'] and cfg.training.max_iters in (4,20) and plan['scientific_quality'] is False,'Finite prerequisite-only request required')
    with Guard(root/f'control/worker-{args.stage}-{args.slot:02d}-guard.json'):
        writer.publish_json(base+'/config.json',spec['config']);sink=CheckpointWriter(writer,args.stage,args.slot,spec['weight_cap'])
        with writer.open_append(base+'/metrics.jsonl') as stream:
            def emit(row):stream.write((json.dumps(row,allow_nan=False)+'\n').encode());print(json.dumps(row),flush=True)
            summary=run(cfg,sink,emit)
        summary['request']=reference(root/base/'request.json');writer.publish_json(base+'/worker.json',summary)
        require(sources()==plan['sources'],'Frozen worker sources changed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=['pilots','quality'],required=True);p.add_argument('--slot',type=int,required=True)
    p.add_argument('--global-fd',type=int,required=True);p.add_argument('--run-fd',type=int,required=True);p.add_argument('--request-sha',required=True);worker(p.parse_args())
