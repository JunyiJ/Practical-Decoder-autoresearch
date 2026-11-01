"""Independent complete-panel provenance, inference and paired statistical review."""
import json,shutil
from statistics import median
from pathlib import Path
import numpy as np
import torch
from research.ema_horizon_v1 import contract as c,artifacts as a,execution as e
from research.block_norm_v1 import campaign_rules as rules,validation as v
from research.block_norm_v1.validation_data import load as validation_data
from research.block_norm_v1.fixture_dispatch import reference,read
from research.block_norm_v1.inference import sample
from src.research.tokenizer_wide_hybrid_timing import _correctness,token_digest
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter

B=e.EVIDENCE;SHA='a58f1c2c7d8b1c10c28acb4d89c1ecc535c6f3e457d46052e75116c433b64bbf'
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g,session_lock(e.ROOT) as lock:
    e.idle();guardpath=B/'quality001-final-review-guard.json'
    assert not guardpath.exists() and not (B/'quality001-final-review.json').exists()
    with CpuPrerequisiteGuard(guardpath,e.ROOT,timeout_seconds=120,poll_seconds=.25):
        plan=e.bind(MetadataWriter(e.ROOT,(g,lock)),'quality',SHA)
        assert plan['finite_design']['input_arm_order']==[200,100]
        assert sorted(p.name for p in (e.ROOT/'evidence/quality').iterdir())==['00','01','02']
        retained=[];previous=[];panel=[];losses=[];target=None;timings=[];reviewrefs=[]
        for stage,slot in [('pilots',0),('quality',0),('quality',1),('quality',2)]:
            anchorref=reference(e.anchor_path(stage,slot));anchor=read(anchorref);accepted=read(anchor['acceptance'])
            assert accepted['stage']==stage and accepted['slot']==slot and accepted['artifact_acceptance']['accepted']
            assert accepted['previous']==previous
            for ref in anchor['files']:
                assert e.ref_file(ref['path'])==ref
                if ref['path'].endswith('.pt'):retained.append(dict(ref,bytes=Path(ref['path']).stat().st_size))
            for guard in [read(anchor['parent_guard']),read(reference(e.ROOT/f'control/worker-{stage}-{slot:02d}.json'))]:
                assert guard['status']=='finished' and guard['reason'] is None
            process=read(accepted['process']);assert process['status']=='completed' and process['returncode']==0 and process['error'] is None and not process['log_truncated']
            previous.append(anchorref)
            if stage=='pilots':continue
            assert accepted['freeze']==reference(e.freeze_path('quality'))
            request=read(accepted['request'])['acceptance_request'];cfg=a.verify_request(request)
            assert request['config']==plan['configs'][slot] and cfg.training.seed==plan['seeds']['fresh'][slot]
            data=validation_data(cfg);values=[];ce=[]
            for half in (200,100):
                report=accepted['artifact_acceptance']['validation'][str(half)]
                v.validate_report(report,a.binding(request,half),data)
                assert target is None or report['target']==target
                target=report['target'];values.append(report['block_loss_sums']);ce.append(report['ce'])
            losses.append(values);panel.append(dict(seed=cfg.training.seed,h200_ce=ce[0],h100_ce=ce[1],delta=ce[1]-ce[0]))
            reviewref=reference(B/f'quality001-slot{slot:02d}-review.json');review=read(reviewref)
            assert review['status']=='accepted_with_fresh_inference' and review['acceptance']==anchorref
            guard=read(review['guard']);assert guard['status']=='finished' and guard['reason'] is None
            timing=read(review['inference']);launch=read(timing['plan'])
            assert launch['freeze']==reference(e.freeze_path('quality')) and launch['acceptance']==anchorref
            assert launch['protocol']==plan['fresh_inference'] and reference(launch['runner']['path'])==launch['runner']
            assert reference(plan['fresh_inference']['source']['path'])==plan['fresh_inference']['source']
            assert timing['status']=='passed' and timing['pair_index']==slot and timing['request_sha256']==c.inherited.fingerprint(request)
            assert timing['tokens_sha256']==token_digest(data.train_data[:96].tolist())
            assert len(timing['correctness'])==4
            for row,(batch,half) in zip(timing['correctness'],[(b,h) for b in (1,8) for h in (200,100)]):
                assert row['batch']==batch and row['half_life']==half
                _correctness({k:row[k] for k in ('passed','max_absolute_error')})
            for field,count in [('warmups',2),('records',8)]:
                expected=[(b,h,r,p) for b in (1,8) for r in range(count)
                          for p,h in enumerate((200,100) if (slot+r)%2==0 else (100,200))]
                assert len(timing[field])==len(expected)
                for row,(batch,half,repeat,position) in zip(timing[field],expected):
                    assert (row['batch'],row['half_life'],row['repeat'],row['position'])==(batch,half,repeat,position)
                    keys=('prefill_seconds','decode_seconds','state_bytes','observed_mps_current_bytes','observed_mps_driver_bytes')
                    checked=sample({k:row[k] for k in keys},batch)
                    assert row['decode_tokens_per_second']==checked['decode_tokens_per_second']
            cells=[]
            for batch in (1,8):
                for half in (200,100):
                    rows=[x for x in timing['records'] if x['batch']==batch and x['half_life']==half]
                    cells.append(dict(batch=batch,half_life=half,state_bytes=196608*batch,
                        prefill_median_seconds=median(x['prefill_seconds'] for x in rows),
                        decode_median_seconds=median(x['decode_seconds'] for x in rows)))
            assert cells==timing['cells'];timings.append(dict(seed=cfg.training.seed,cells=cells));reviewrefs.append(reviewref)
        assert len(retained)==12 and len(panel)==3 and target['block_counts']==[2048]*16+[1510]
        stats=rules.paired_statistics(plan['seeds'],losses,target['block_counts'])
        array=np.array(losses,dtype=np.float64);counts=np.array(target['block_counts'])
        ds=(array[:,1].sum(1)-array[:,0].sum(1))/sum(counts)
        assert np.max(np.abs(ds-np.array(stats['seed_deltas'])))<1e-12
        assert max(abs(p['delta']-d) for p,d in zip(panel,ds))<1e-12
        independent={}
        for size in (2048,4096,8192):
            factor=size//2048
            totals=np.array([np.mean(array[:,1,j:j+factor].sum(1)-array[:,0,j:j+factor].sum(1)) for j in range(0,len(counts),factor)])
            weights=np.array([counts[j:j+factor].sum() for j in range(0,len(counts),factor)])
            rng=np.random.Generator(np.random.PCG64(plan['seeds']['bootstrap']));draw=[]
            for _ in range(2000):
                ix=rng.integers(0,len(weights),size=len(weights));draw.append(totals[ix].sum()/weights[ix].sum())
            bounds=np.quantile(draw,[.025,.975,1-.05/3],method='linear').tolist();expected=stats['bounds'][str(size)]
            assert np.max(np.abs(np.array(bounds)-np.array(expected['descriptive_95pct']+[expected['gate_upper']])))<1e-12
            independent[str(size)]=bounds
        gate=rules.quality_gate(stats)
        checks=dict(mean_improvement=stats['mean_delta']<-.02,every_seed_improves=all(x<0 for x in stats['seed_deltas']),
            simultaneous_text_bounds=all(x['gate_upper']<0 for x in stats['bounds'].values()))
        assert gate['strict_quality']==all(checks.values())
        for ref in retained:assert e.ref_file(ref['path'])=={k:ref[k] for k in ('path','sha256')}
        assert c.source_inventory()==plan['sources']
        aggregate=[]
        for batch in (1,8):
            for half in (200,100):
                cells=[x for t in timings for x in t['cells'] if x['batch']==batch and x['half_life']==half]
                aggregate.append(dict(batch=batch,half_life=half,state_bytes=196608*batch,
                    median_of_prefill_medians=median(x['prefill_median_seconds'] for x in cells),
                    median_of_decode_medians=median(x['decode_median_seconds'] for x in cells)))
        result=dict(status='eligible_pending_adoption_review' if gate['strict_quality'] else 'closed_not_adopted',
            candidate='H100',incumbent='H200',runner=reference(__file__),freeze=reference(e.freeze_path('quality')),
            accepted_anchors=previous,fresh_pairs=panel,statistics=stats,independent_bootstrap_quantiles=independent,
            gate_checks=checks,strict_quality=gate['strict_quality'],quality_adopted=False,
            decision='Separate adoption review required.' if gate['strict_quality'] else 'Retain H200 with RMSNorm and SiLU; no candidate or control promotion.',
            inference_reviews=reviewrefs,inference=timings,inference_aggregate=aggregate,
            retained_weights=retained,free_bytes=shutil.disk_usage(e.ROOT).free,
            scope='Same online trajectory per H100/H200 pair; three trained seeds and reused validation text. Conditional-text intervals are not population-seed or holdout confidence. Two-model resident inference timing is descriptive; allocation/loading excluded.',
            holdout_scored=False,deletion_performed=False,training_launched=False)
    result['guard']=reference(guardpath);assert read(result['guard'])['status']=='finished'
    e.exclusive(B/'quality001-final-review.json',result)
    if not gate['strict_quality']:
        with CpuPrerequisiteGuard(B/'quality001-closure-guard.json',e.ROOT,timeout_seconds=120,poll_seconds=.25):
            e.exclusive(e.ROOT/'STOP',dict(reason='finite_panel_closed_not_adopted',review=reference(B/'quality001-final-review.json')))
        e.exclusive(B/'quality001-closure.json',dict(status='closed_not_adopted',review=reference(B/'quality001-final-review.json'),
            stop=reference(e.ROOT/'STOP'),guard=reference(B/'quality001-closure-guard.json'),retained_weights=12,holdout_scored=False))
    print(json.dumps(dict(status=result['status'],statistics=stats,gates=checks,inference=aggregate)))
