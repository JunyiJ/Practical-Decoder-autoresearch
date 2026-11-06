"""Read-only closure of the completed structural study, never a numerical replay."""
import hashlib,json,math,shutil,statistics
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,source_inventory
from research.body_allocation_worker_v1.execution import checked,sources
from research.body_allocation_campaign_v1.pilot import campaign_sources
from research.body_allocation_quality_v1.contract import quality_sources
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
B=Path(__file__).resolve().parent;RUN=ROOT/'research_runs/tokenizer-body-allocation-quality-v1-001';OUT=B/'quality001-closure001.json'
def artifact(ref):
    p=Path(ref['path']);s=p.stat();require(p.is_file() and not p.is_symlink() and s.st_nlink==1 and s.st_size==ref['size'],'Unique original retained payload required')
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    t=p.stat();require(h==ref['sha256'] and (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns),'Original checkpoint identity changed')
    return ref
def publish(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
with session_lock(GLOBAL_LOCK.parent):
    idle();require(not OUT.exists(),'Closed evidence reviewer cannot rerun')
    with CpuPrerequisiteGuard(B/'quality001-closure001-guard.json',ROOT,timeout_seconds=300,poll_seconds=.25):
        plan=checked(reference(B/'quality001-plan.json'));seal=checked(reference(B/'quality001-freeze.json'));confirm=checked(reference(B/'quality001-confirmation-plan.json'))
        report=checked(reference(B/'quality001-final-review.json'));stop=checked(reference(RUN/'STOP'));selection=checked(reference(B/'quality001-selection.json'))
        require(stop['review']==reference(B/'quality001-final-review.json') and stop['never_rerun_training_inference_review'] is True,'Terminal original closure required')
        require(plan['sources']==sources() and plan['quality_sources']==quality_sources() and plan['pilot_sources']==campaign_sources(),'Frozen sources changed')
        require(report['status']=='complete_fresh_panel_reviewed' and report['statistics']['all_strict_gates_pass'] is False and report['adopted'] is False,'Complete non-adoption review required')
        require(checked(reference(B/'quality001-final-review-guard.json'))['status']=='finished','Original complete final guard required')
        require(report['selection']==reference(B/'quality001-selection.json') and report['confirmation']==reference(B/'quality001-confirmation-plan.json'),'Original selection/fresh binding differs')
        rows=[];protected=[];ces={a:[] for a in ('incumbent','shallow','deeper')};seconds={a:[] for a in ces}
        retained_slots={v['selected_slot'] for v in selection['audits'].values()}|set(range(9,18))
        retirement=checked(reference(B/'quality001-loser-retirement001/completed.json'));manifest=checked(retirement['manifest'])
        for slot in range(18):
            anchor=checked(reference(B/f'quality001-accepted-{slot:02d}.json'));a=checked(anchor['accepted']);w=checked(a['worker'])
            expected=plan['specs'][slot] if slot<9 else confirm['specs'][slot-9]
            require(w['config']==expected['config'] and a['arm']==expected['arm'] and w['request']==a['request'],'Original accepted configuration differs')
            accepted=a['acceptance'];require(accepted['status']=='accepted' and accepted['step']==1200 and accepted['targets']==1228800 and accepted['ema_snapshots']==41,'Complete trajectory required')
            process=checked(a['process']);require(process['status']=='completed' and process['returncode']==0,'Clean worker termination required')
            for name in ('worker_guard','parent_guard'):
                g=checked(a[name]);require(g['status']=='finished' and g['reason'] is None and g['limits']['rss_limit_bytes']==4*1024**3
                    and g['limits']['mps_driver_limit_bytes']==6*1024**3 and g['limits']['reserve_bytes']==8*1024**3 and g['limits']['timeout_seconds']==7200,'Original guards differ')
            for name in ('parent','h200'):
                ref=w[name]
                if slot in retained_slots:protected.append(artifact(ref))
                else:require(not Path(ref['path']).exists() and any(r['path']==ref['path'] and r['sha256']==ref['sha256'] and r['size']==ref['size'] for r in manifest['delete']),'Retired losing payload lacks exact permanent manifest')
            rows.append(dict(slot=slot,arm=a['arm'],seed=w['config']['training']['seed'],lr=w['config']['training']['learning_rate'],accepted=anchor['accepted'],worker=a['worker'],process=a['process'],ce=accepted['full_validation']['ce'],training_seconds=w['training_seconds'],process_seconds=process['elapsed_seconds'],clipped_updates=accepted['clipped_updates'],retained_weights=slot in retained_slots))
            if slot>=9:ces[a['arm']].append(accepted['full_validation']['ce']);seconds[a['arm']].append(w['training_seconds'])
        require(len(protected)==24 and len({r['path'] for r in protected})==24,'Three selected and nine fresh RAW/H200 pairs protected')
        inference=checked(report['inference']);require(inference['all_fresh_triplets'] and len(inference['rows'])==9 and inference['selection'] is False,'Complete fresh inference evidence required')
        latency={}
        for arm in ('shallow','deeper'):
            latency[arm]={}
            for batch in (1,8):
                ratios={'full128':[],'decode32':[]}
                for seed in plan['seeds']['fresh']:
                    group={r['arm']:r for r in inference['rows'] if r['seed']==seed}
                    a=next(r for r in group[arm]['measurements'] if r['batch']==batch);c=next(r for r in group['incumbent']['measurements'] if r['batch']==batch)
                    require(len(a['full128_seconds'])==len(a['decode32_after_prefill96_seconds'])==5,'Full timing coverage required')
                    ratios['full128'].append(a['median_full128_seconds']/c['median_full128_seconds']);ratios['decode32'].append(a['median_decode32_seconds']/c['median_decode32_seconds'])
                latency[arm][str(batch)]={k:statistics.median(v) for k,v in ratios.items()}
        result=dict(status='CLOSED_NOT_ADOPTED',source=reference(__file__),original_final_review=reference(B/'quality001-final-review.json'),original_stop=reference(RUN/'STOP'),
            sources=plan['sources'],quality_sources=plan['quality_sources'],pilot_sources=plan['pilot_sources'],all_939_bindings_unchanged=True,
            decision='Retain incumbent4x256H672 learned-scale/bias adopted epsilon1e-6 recipe. Deeper fails both original practical-quality comparisons; no shallow fallback or cost-only adoption.',
            selection=selection['choices'],fresh_h200_ce=ces,mean_h200_ce={a:statistics.mean(v) for a,v in ces.items()},statistics=report['statistics'],
            fresh_mean_training_seconds={a:statistics.mean(v) for a,v in seconds.items()},median_of_paired_seed_latency_ratios=latency,
            inference_scope='Single resident model, setup/load/cache allocation excluded; descriptive local FP32 MPS timings, not a quality or adoption gate.',
            trajectories=rows,scientific_trajectories=18,fresh_trajectories=9,all_original_guards_passed=True,protected_original_payloads=protected,
            protected_weight_bytes=sum(r['size'] for r in protected),old_failures_and_retirement_receipts_immutable=True,
            fresh_and_bootstrap_seeds_all_used=True,holdout_scored=False,worker_active=False,free_bytes=shutil.disk_usage(ROOT).free,
            next_action='Continue evidence-grounded model improvement. Analyze accepted curves/cost and train-only context availability, review relevant primary papers and contrary old horizon/table results; rank qualified next tests, no automatic beta1 or frozen new panel yet.')
        publish(OUT,result)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','mean_h200_ce','fresh_mean_training_seconds','median_of_paired_seed_latency_ratios','protected_weight_bytes','free_bytes')}))
