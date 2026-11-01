"""Independent arithmetic and provenance review of the complete frozen panel."""
import json
import math
import shutil
from pathlib import Path
import numpy as np
import torch
from research.ffn_quality_v1 import quality_execution as q, campaign_contract as c, campaign_rules as r
from research.ffn_structure_v1 import inference as i
from research.block_norm_v1.fixture_dispatch import reference, read
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter

SHA='4db95f1c36da55dc54e6d5b54bc4b99a2a6423c4b307d012a78c56b23c9bca39'
B=q.EVIDENCE
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g,session_lock(q.ROOT) as lock:
    q.idle()
    guardpath=B/'quality001-final-review-guard.json'
    c.require(not guardpath.exists() and not (B/'quality001-final-review.json').exists(),'No review replacement')
    with CpuPrerequisiteGuard(guardpath,q.ROOT,timeout_seconds=120,poll_seconds=.25):
        writer=MetadataWriter(q.ROOT,(g,lock));plan=q.bind(writer,SHA);state=q.state(plan,SHA)
        c.require(len(state['records'])==12,'Complete accepted quality panel required')
        panel=[];losses=[];target=None;timings=[];refs=[]
        timing_plan=read(reference(B/'quality001-fresh-inference-plan.json'))
        c.require(timing_plan['sources']==plan['sources'] and timing_plan['environment']==c.environment(),'Inference source/environment differs')
        c.require(reference(Path(timing_plan['runner']['path']))==timing_plan['runner'],'Inference runner changed')
        c.require(timing_plan['anchors']==[reference(q.anchor_path(n)) for n in range(12)],'Inference panel differs')
        for index,seed in enumerate(plan['seeds']['fresh']):
            receipts=state['receipts'][6+2*index:8+2*index]
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
            panel.append(dict(seed=seed,swiglu_ce=ce[0],ungated_silu_ce=ce[1],delta=ce[1]-ce[0]))
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
        c.require(not gate['strict_quality'],'Passing candidate requires separate adoption review')
        q.state(plan,SHA)  # Recheck retained checkpoints after read-only review.
        result=dict(status='closed_not_adopted',candidate='ungated_silu',incumbent='swiglu',
            runner=reference(Path(__file__).absolute()),freeze=dict(path=str(q.FREEZE),sha256=SHA),
            accepted_anchors=[reference(q.anchor_path(n)) for n in range(12)],selection=reference(q.selection_path()),
            fresh_pairs=panel,statistics=stats,independent_bootstrap_quantiles=independent,gate_checks=checks,
            strict_quality=False,quality_adopted=False,decision='Retain adopted SwiGLU672, SiLU, RMSNorm and H200; no ungated FFN adoption.',
            inference_reviews=refs,inference=timings,accepted_panel_unchanged=True,free_bytes=shutil.disk_usage(q.ROOT).free,
            scope='Three trained seeds on reused validation text; conditional text intervals are not population-seed or holdout confidence. Equal FFN matrix budget comparison jointly changes hidden width and multiplicative factorization; not an isolated gating effect. Inference is descriptive, two models resident, allocation/loading excluded.',
            training_launched=False,deletion_performed=False,holdout_scored=False)
    result['guard']=reference(guardpath)
    c.require(read(result['guard'])['status']=='finished','Final review guard failed')
    q.exclusive(B/'quality001-final-review.json',result)
    print(json.dumps(dict(status=result['status'],statistics=stats,gates=checks)))
