"""Independent arithmetic and provenance review of the complete frozen panel."""
import argparse
import json
import math
import shutil
from pathlib import Path
import numpy as np
import torch
from research.branch_scale_quality_v1 import quality_execution as q, campaign_contract as c, campaign_rules as r
from research.branch_scale_worker_v1 import inference as i
from research.block_norm_v1.fixture_dispatch import reference, read
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter


parser=argparse.ArgumentParser();parser.add_argument('--freeze-sha256',required=True);args=parser.parse_args();SHA=args.freeze_sha256
B=q.EVIDENCE
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g,session_lock(q.ROOT) as lock:
    q.idle()
    guardpath=B/'quality001-final-review-guard.json'
    c.require(not guardpath.exists() and not (B/'quality001-final-review.json').exists(),'No review replacement')
    with CpuPrerequisiteGuard(guardpath,q.ROOT,timeout_seconds=120,poll_seconds=.25):
        writer=MetadataWriter(q.ROOT,(g,lock));plan=q.bind(writer,SHA);state=q.state(plan,SHA)
        c.require(reference(Path(__file__).absolute())==plan['final_sources'][Path(__file__).name],'Frozen final review runner changed')
        c.require(len(state['records'])==6,'Complete accepted quality panel required')
        panel=[];losses=[];target=None;timings=[];refs=[]
        timing_plan=read(reference(B/'quality001-fresh-inference-plan.json'))
        c.require(timing_plan['sources']==plan['sources'] and timing_plan['environment']==c.environment(),'Inference source/environment differs')
        c.require(reference(Path(timing_plan['runner']['path']))==timing_plan['runner'],'Inference runner changed')
        c.require(timing_plan['anchors']==[reference(q.anchor_path(n)) for n in range(6)],'Inference panel differs')
        for index,seed in enumerate(plan['seeds']['fresh']):
            receipts=state['receipts'][2*index:2*index+2]
            byarm={x['request']['trial']['spec']['arm']:x for x in receipts}
            requests={arm:{role:byarm[arm]['request']['files'][role] for role in ('config','parent','h200')} for arm in r.ARMS}
            c.require(timing_plan['pairs'][index]['requests']==requests,'Timing request differs')
            values=[];ce=[]
            for arm in r.ARMS:
                receipt=byarm[arm];v=receipt['artifact_acceptance']['validation']
                c.require(receipt['request']['trial']['spec']['seed']==seed,'Seed pairing differs')
                c.require(target is None or target==v['target'],'Validation target alignment differs');target=v['target']
                values.append(v['block_loss_sums']);ce.append(v['ce'])
            losses.append(values)
            panel.append(dict(seed=seed,unit_ce=ce[0],fixed_half_ce=ce[1],delta=ce[1]-ce[0]))
            reviewref=reference(B/f'quality001-fresh-inference-pair{index}-review.json');review=read(reviewref)
            c.require(review['status']=='passed' and review['plan']==reference(B/'quality001-fresh-inference-plan.json'),'Inference failed or plan differs')
            guard=read(review['guard']);c.require(guard['status']=='finished' and guard['reason'] is None,'Inference guard failed')
            result=read(review['result']);checked=i.review(result,requests,index)
            c.require(checked==review['values_review'],'Saved inference arithmetic differs')
            timings.append(dict(pair_index=index,seed=seed,cells=checked['cells'],scope=checked['scope']))
            refs.append(reviewref)
        stats=r.paired_statistics(plan['seeds'],losses,target['block_counts'])
        # Separate implementation: explicit contiguous grouping and numpy's
        # linear quantile, checked against the frozen reduction/interpolation.
        a=np.array(losses,dtype=np.float64);counts=np.array(target['block_counts'])
        deltas=(a[:,1,:].sum(1)-a[:,0,:].sum(1))/sum(counts)
        c.require(np.max(np.abs(deltas-np.array(stats['seed_deltas'])))<1e-12,'Independent seed arithmetic differs')
        c.require(max(abs(x['delta']-d) for x,d in zip(panel,deltas))<1e-12,'Block totals disagree with accepted CE')
        independent={}
        for size in (2048,4096,8192):
            factor=size//2048
            totals=np.array([np.mean(a[:,1,j:j+factor].sum(1)-a[:,0,j:j+factor].sum(1)) for j in range(0,len(counts),factor)])
            weights=np.array([counts[j:j+factor].sum() for j in range(0,len(counts),factor)])
            rng=np.random.Generator(np.random.PCG64(plan['seeds']['bootstrap']))
            draw=[]
            for _ in range(2000):
                ix=rng.integers(0,len(weights),size=len(weights));draw.append(totals[ix].sum()/weights[ix].sum())
            bound=np.quantile(draw,[.025,.975,1-.05/3],method='linear').tolist()
            expected=stats['bounds'][str(size)]
            c.require(np.max(np.abs(np.array(bound)-np.array(expected['descriptive_95pct']+[expected['gate_upper']])))<1e-12,'Independent conditional bootstrap differs')
            independent[str(size)]=bound
        gate=r.quality_gate(stats)
        checks=dict(mean_improvement=stats['mean_delta']<-.02,every_seed_improves=all(x<0 for x in stats['seed_deltas']),simultaneous_text_bounds=all(x['gate_upper']<0 for x in stats['bounds'].values()))
        c.require(gate['strict_quality']==all(checks.values()),'Independent gate differs')
        q.state(plan,SHA)  # Recheck retained checkpoints after read-only review.
        result=dict(status='reviewed_gate_passed_pending_adoption' if gate['strict_quality'] else 'reviewed_not_adopted',candidate='fixed_half',incumbent='unit',
            runner=reference(Path(__file__).absolute()),freeze=dict(path=str(q.FREEZE),sha256=SHA),
            accepted_anchors=[reference(q.anchor_path(n)) for n in range(6)],selection=None,
            fresh_pairs=panel,statistics=stats,independent_bootstrap_quantiles=independent,gate_checks=checks,
            strict_quality=gate['strict_quality'],quality_adopted=False,decision=('Fixed half branch gain passed the frozen quality gate; complete a separate adoption review.' if gate['strict_quality'] else 'Retain independent decay0.00006 with the adopted D256 MLA/score0.5/SwiGLU672/SiLU/blockRMS/H200 recipe; retain unit branch gains; no fixed-half adoption.'),
            inference_reviews=refs,inference=timings,accepted_panel_unchanged=True,free_bytes=shutil.disk_usage(q.ROOT).free,
            scope='Three trained seeds on reused validation text; conditional text intervals are not population-seed or holdout confidence. Fixed common LR0.00125 and equal zero LR-search budgets compare standard recipes on the same D256 MLA backbone, not separately tuned optima. The complete serial attention and FFN branches use fixed gains1 versus0.5; schedule and integrated LR exposure are identical. No rescue, extension or pilot-loss selection. Inference is descriptive, two models resident, allocation/loading excluded.',
            training_launched=False,deletion_performed=False,holdout_scored=False)
    result['guard']=reference(guardpath)
    c.require(read(result['guard'])['status']=='finished','Final review guard failed')
    q.exclusive(B/'quality001-final-review.json',result)
    print(json.dumps(dict(status=result['status'],statistics=stats,gates=checks)))
