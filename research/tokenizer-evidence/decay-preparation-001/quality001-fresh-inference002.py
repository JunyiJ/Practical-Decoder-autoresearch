"""Read-only inference for the three frozen, independently accepted fresh pairs."""
import argparse
import json
from pathlib import Path
import torch
from research.decay_quality_v2 import quality_execution as q, campaign_contract as c
from research.decay_v1 import inference as i
from research.block_norm_v1.fixture_dispatch import reference, read
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter


B=q.EVIDENCE
PLAN=B/'quality001-fresh-inference-plan.json'
parser=argparse.ArgumentParser()
parser.add_argument('mode',choices=['prepare','run'])
parser.add_argument('--pair',type=int,choices=range(3))
parser.add_argument('--plan-sha256')
parser.add_argument('--freeze-sha256',required=True)
args=parser.parse_args()
SHA=args.freeze_sha256
torch.set_num_threads(4)

def panel(writer):
    plan=q.bind(writer,SHA)
    c.require(reference(Path(__file__).absolute())==plan['final_sources'][Path(__file__).name],'Frozen inference runner changed')
    state=q.state(plan,SHA)
    c.require(len(state['records'])==6,'All six accepted endpoints required')
    return plan,state

def requests(state,index):
    rows=state['receipts'][2*index:2*index+2]
    return {r['request']['trial']['spec']['arm']:{role:r['request']['files'][role] for role in ('config','parent','h200')} for r in rows}

def verify(saved,writer):
    c.require(reference(Path(__file__).absolute())==saved['runner'],'Runner changed')
    plan,state=panel(writer)
    c.require(saved['sources']==plan['sources'] and saved['environment']==c.environment(),'Source/environment changed')
    c.require(saved['anchors']==[reference(q.anchor_path(n)) for n in range(6)],'Accepted panel changed')
    for index,row in enumerate(saved['pairs']):
        req=requests(state,index)
        c.require(row['requests']==req and row['identity']==i.identity(req,index),'Pair provenance changed')
    return state

with session_lock(GLOBAL_LOCK.parent) as global_fd,session_lock(q.ROOT) as campaign_fd:
    q.idle()
    writer=MetadataWriter(q.ROOT,(global_fd,campaign_fd))
    if args.mode=='prepare':
        c.require(not PLAN.exists(),'Existing plan cannot be replaced')
        with CpuPrerequisiteGuard(B/'quality001-fresh-inference-prepare-guard.json',q.ROOT,timeout_seconds=120,poll_seconds=.25):
            plan,state=panel(writer)
            saved=dict(status='frozen_readonly_three_fresh_pairs',runner=reference(Path(__file__).absolute()),
                freeze=dict(path=str(q.FREEZE),sha256=SHA),sources=plan['sources'],environment=c.environment(),
                anchors=[reference(q.anchor_path(n)) for n in range(6)],
                pairs=[dict(requests=requests(state,n),identity=i.identity(requests(state,n),n)) for n in range(3)],
                limits=dict(seconds_per_pair=600,cpu_threads=4,rss_bytes=4*1024**3,driver_bytes=6*1024**3,reserve_bytes=8*1024**3),
                training_allowed=False,deletion_allowed=False,quality_adoption_allowed=False,holdout_scored=False)
            q.exclusive(PLAN,saved)
        print(json.dumps(reference(PLAN)))
    else:
        c.require(args.pair is not None and args.plan_sha256,'Pair and frozen plan hash required')
        saved=read(dict(path=str(PLAN),sha256=args.plan_sha256))
        stem=f'quality001-fresh-inference-pair{args.pair}'
        guard_path=B/(stem+'-guard.json');result_path=B/(stem+'-result.json');review_path=B/(stem+'-review.json')
        c.require(not any(p.exists() for p in (guard_path,result_path,review_path)),'No retry of existing attempt')
        error=None
        try:
            with StrictMpsGuard(guard_path,q.ROOT):
                verify(saved,writer)
                req=saved['pairs'][args.pair]['requests']
                result=i.measure_pair(req,args.pair)
                review=i.review(result,req,args.pair)
                verify(saved,writer)
                q.exclusive(result_path,result)
            guard=read(reference(guard_path))
            c.require(guard['status']=='finished' and guard['reason'] is None,'Guard failed')
            value=dict(status='passed',pair_index=args.pair,plan=reference(PLAN),result=reference(result_path),guard=reference(guard_path),
                values_review=review,accepted_panel_unchanged=True,quality_adopted=False,training_launched=False,holdout_scored=False,deletion_performed=False)
        except Exception as exc:
            error=repr(exc)
            value=dict(status='failed',pair_index=args.pair,error=error,plan=reference(PLAN),no_retry=True)
        q.exclusive(review_path,value)
        print(json.dumps(value))
        if error:raise SystemExit(1)
