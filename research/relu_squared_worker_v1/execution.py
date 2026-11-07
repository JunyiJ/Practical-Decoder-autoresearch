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
from research.relu_squared_resume_v1.contract import runtime
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter
from .training import run


def checked(ref):
    require(reference(ref['path'])==ref,'Evidence reference changed');return json.loads(Path(ref['path']).read_text())


def sources():
    e=ROOT/'research/tokenizer-evidence/post-projection-boundary-001'
    proof=checked(reference(e/'relu-squared-resume001/STOP'))
    require(proof['status']=='complete_serialized_ffn_resume_scorer_proof','Accepted own serialized resume/scorer proof required')
    require(checked(reference(e/'relu-squared-resume001/mps-guard.json'))['status']=='finished','Original serialized proof guard required')
    require(checked(reference(e/'relu-squared-resume001-review-guard.json'))['status']=='finished','Independent serialized review must finish')
    require(checked(reference(e/'relu-squared-resume001-review.json'))['status']=='closed_accepted_generated_serialized_ffn_state_scorer_pair_fixture_only','Independent accepted serialized review required')
    pins=dict(checked(reference(e/'relu-squared-resume001/plan.json'))['sources']);require(len(pins)==1191,'All accepted1184model+5state+2proof predecessors required')
    for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Immutable predecessor changed: '+p)
    for name in ['__init__.py','training.py','validation.py','dispatch.py','execution.py']:
        p=Path(__file__).parent/name;pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
    for name in ['relu-squared-worker001.py','relu-squared-worker-source001.py','relu-squared-worker-seed-census001.py']:
        p=e/name;pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
    p=e/'real-worker-source001/review-source001.py';pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
    require(len(pins)==1200,'Exact five worker files/builder/source gate/twelve-role census required');return pins


def idle():
    from research.final_ln_scale_quality_v3.support import idle as inherited_idle
    inherited_idle()
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        row=line.strip().split(None,1)
        if len(row)==2 and Path(row[1].split()[0]).name.lower().startswith('python') and any(v in row[1] for v in ('research.tokenizer_','research.readout_','research.relu_squared_','readout-worker','readout-pilot','readout-resume','relu-squared-worker','relu-squared-resume')):
            require(int(row[0])==os.getpid(),'Another numerical research process is active')


class Guard(StrictMpsGuard):
    def __init__(self,path,seconds=600):
        require(seconds==600,'Original prerequisite horizon600s required');super().__init__(path,ROOT)


def worker(args):
    torch.set_num_threads(4);runtime();root=Path(args.root).resolve()
    require(root in {ROOT/'research_runs/tokenizer-relu-squared-worker-v1-001',ROOT/'research_runs/tokenizer-relu-squared-pilot-v1-001'} and not (root/'STOP').exists(),'Open own research root required')
    writer=MetadataWriter(root,(args.global_fd,args.run_fd));base=f'evidence/{args.stage}/{args.slot:02d}'
    rq=checked(dict(path=str(root/base/'request.json'),sha256=args.request_sha));plan=checked(rq['plan'])
    require(rq['parent_pid']==os.getppid() and rq['slot']==args.slot and rq['stage']==args.stage and plan['root']==str(root)
        and plan['sources']==sources() and plan['environment']==runtime() and plan['seconds']==600 and plan['compare_cpu'] is True,'Worker request/source/environment differs')
    spec=rq['spec'];cfg=OmegaConf.create(spec['config'])
    from research.body_allocation_model_v1.contract import fingerprint,plain
    require(plain(cfg.relu_squared_execution)==dict(kind='ffn_prerequisite_execution_v1',sources_sha256=fingerprint(plan['sources']),runner_sha256=plan['runner']['sha256']),'Complete execution binding differs')
    require(spec in plan['specs'] and cfg.training.max_iters in (4,20) and plan['scientific_quality'] is False,'Finite prerequisite-only request required')
    with Guard(root/f'control/worker-{args.stage}-{args.slot:02d}-guard.json'):
        writer.publish_json(base+'/config.json',spec['config']);sink=CheckpointWriter(writer,args.stage,args.slot,spec['weight_cap'])
        with writer.open_append(base+'/metrics.jsonl') as stream:
            def emit(row):stream.write((json.dumps(row,allow_nan=False)+'\n').encode());print(json.dumps(row),flush=True)
            summary=run(cfg,sink,emit)
        summary['request']=reference(root/base/'request.json');writer.publish_json(base+'/worker.json',summary)
        require(sources()==plan['sources'],'Frozen worker sources changed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=['pilots'],required=True);p.add_argument('--slot',type=int,required=True)
    p.add_argument('--global-fd',type=int,required=True);p.add_argument('--run-fd',type=int,required=True);p.add_argument('--request-sha',required=True);worker(p.parse_args())
