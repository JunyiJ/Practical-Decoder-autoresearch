"""Full-horizon worker accepts only its own frozen serial scientific request."""
import argparse,json,os
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_moe_resume_v1.contract import runtime as environment
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter
from . import contract as C
from .training import run

def worker(args):
 torch.set_num_threads(4);environment();root=Path(args.root).resolve();require(root==C.RUN and not(root/'STOP').exists() and args.stage=='quality' and 0<=args.slot<18,'Open exact scientific root/slot required')
 writer=MetadataWriter(root,(args.global_fd,args.run_fd));base=f'evidence/quality/{args.slot:02d}'
 rq=C.checked(dict(path=str(root/base/'request.json'),sha256=args.request_sha));plan=C.checked(rq['plan']);freeze=C.checked(reference(C.FREEZE))
 require(freeze['plan']==reference(C.PLAN) and freeze['design']==C.DESIGN,'Original scientific seal required')
 require(rq['parent_pid']==os.getppid() and rq['slot']==args.slot and rq['stage']=='quality' and plan['root']==str(root) and plan['sources']==C.sources() and plan['environment']==environment(),'Frozen source/environment/request differs')
 if args.slot<9:require(plan['kind']==C.SCREEN_KIND and rq['plan']==freeze['plan'] and plan['specs']==C.screen_specs(plan['seeds']),'Exact original frozen screen required')
 else:
  selected=C.checked(reference(C.SELECTION));require(plan['kind']==C.FRESH_KIND and rq['plan']==reference(C.CONFIRM) and plan['original_freeze']==reference(C.FREEZE) and plan['selection']==reference(C.SELECTION) and plan['specs']==C.fresh_specs(plan['seeds'],selected['choices']),'Exact resolved original fresh panel required')
 require(plan['design']==C.DESIGN and plan['seconds']==7200 and plan['compare_cpu'] is True and plan['required_free']==C.FLOOR,'Original full scientific gates required')
 require(all((root/f'evidence/quality/{i:02d}/accepted.json').exists() for i in range(args.slot)) and all(not(root/f'evidence/quality/{i:02d}').exists() for i in range(args.slot+1,18)),'Exactly one next serial trajectory required')
 spec=rq['spec'];expected=plan['specs'][args.slot if args.slot<9 else args.slot-9];require(spec==expected,'Unplanned scientific configuration')
 cfg=OmegaConf.create(spec['config']);require(cfg.training.max_iters==1200,'Complete original horizon required')
 with C.Guard(root/f'control/worker-quality-{args.slot:02d}-guard.json',7200):
  writer.publish_json(base+'/config.json',spec['config']);sink=CheckpointWriter(writer,'quality',args.slot,spec['weight_cap'])
  with writer.open_append(base+'/metrics.jsonl') as stream:
   def emit(row):stream.write((json.dumps(row,allow_nan=False)+'\n').encode());print(json.dumps(row),flush=True)
   summary=run(cfg,sink,emit)
  summary['request']=reference(root/base/'request.json');writer.publish_json(base+'/worker.json',summary);require(C.sources()==plan['sources'],'Scientific source changed during training')
 print('Full native MoE/dense trajectory completed;independent parent acceptance required',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=['quality'],required=True);p.add_argument('--slot',type=int,required=True);p.add_argument('--global-fd',type=int,required=True);p.add_argument('--run-fd',type=int,required=True);p.add_argument('--request-sha',required=True);worker(p.parse_args())
