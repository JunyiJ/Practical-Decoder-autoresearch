"""Closed worker evidence/hash/arithmetic review, zero training or scoring replay."""
import json,math,os,shutil
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_mla_gate_worker_v1.scoring import validate_report
from research.tokenizer_mla_gate_worker_v1.execution import idle
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock

E=Path(__file__).resolve().parent;R=ROOT/'research_runs/tokenizer-mla-gate-pilot-v1-001'
OUT=E/'gate-pilot-review001.json';GUARD=E/'gate-pilot-review001-guard.json';MIRROR=E/'gate-pilot001-metadata-mirror001'
def read(p):return json.loads(Path(p).read_text())
def checked(r):require(reference(r['path'])==r,'Original evidence reference changed');return read(r['path'])
def publish(p,v):
    raw=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<4*1024**2,'Independent review metadata cap required')
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

with session_lock(GLOBAL_LOCK.parent):
    idle();require(not OUT.exists() and not GUARD.exists() and not MIRROR.exists(),'Fresh closed review only;no retry')
    with CpuPrerequisiteGuard(GUARD,ROOT,timeout_seconds=120):
        review=read(E/'gate-pilot001-review.json');plan=checked(review['plan']);stop=read(R/'STOP');prior=checked(plan['independent_worker_review']);worker_plan=checked(prior['plan']);source=checked(worker_plan['source_review'])
        require(review['status']=='all_three_gate_resource_pilots_independently_accepted' and stop['status']=='three_arm_resource_pilot_complete' and stop['review']==reference(E/'gate-pilot001-review.json') and stop['never_rerun'] and stop['all_training_acceptance_commands_closed'],'Complete terminal three-arm worker panel required')
        require(source['status']=='passed_finite_gate_worker_and_native_score_contracts' and source['sources']==plan['sources'] and source['prepared_builder']==worker_plan['runner'] and worker_plan['runner']==reference(E/'gate-worker001-build-run001.py') and plan['runner']==reference(E/'gate-pilot001-build-run001.py') and prior['status']=='all_three_gate_workers_closed_independently_reviewed','Actual source/builder qualification required')
        source_guard=read(E/'gate-worker-source001-guard.json');source_stop=read(E/'STOP-gate-worker-source001')
        require(source_guard['status']=='finished' and source_guard['reason'] is None and source_guard['elapsed_seconds']<120 and source_guard['observation']['peak_rss_bytes']<4*1024**3 and source_stop['status']=='complete_gate_worker_source_contracts' and source_stop['never_rerun'],'Successful original CPU source gate and terminal required')
        pins=plan['sources'];require(len(pins)==1074 and all(pins.get(p)==h for p,h in read(E/'gate-resume002/plan.json')['sources'].items()),'All1074 sources/preserved1068 predecessors required')
        for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Immutable bound source changed '+p)
        require(plan['root']==str(R) and plan['compare_cpu'] is True and plan['seconds']==600 and not plan['scientific_quality'] and not review['scientific_quality'] and not review['new_scientific_seed'],'Own prerequisite scope required')
        arms=['none','static_half','sigmoid_elementwise_zero'];hashes=set();weight_rows=[];maximum_ce_error=0.;guards=[]
        require(len(review['rows'])==3 and len(plan['specs'])==3,'Complete three-arm panel required')
        for slot,(arm,row,spec) in enumerate(zip(arms,review['rows'],plan['specs'])):
            accepted=read(R/f'evidence/pilots/{slot:02d}/accepted.json');require(accepted==row and row['slot']==slot and row['arm']==spec['arm']==arm,'Original acceptance/config order differs')
            rq=checked(row['request']);worker=checked(row['worker']);process=checked(row['process']);a=row['acceptance']
            require(rq['plan']==review['plan'] and rq['spec']==spec and worker['request']==row['request'] and worker['config']==spec['config'],'Exact worker request/config ancestry required')
            require(process['status']=='completed' and process['returncode']==0 and process['error'] is None and process['log_truncated'] is False and process['timeout_seconds']==600 and process['required_free_bytes']>=8*1024**3 and process['observation']['free_bytes']>=8*1024**3 and process['observation']['output_bytes']<=process['output_limit_bytes'] and process['observation']['log_bytes']<=process['log_limit_bytes'],'Complete bounded exited process required')
            for k in ('worker_guard','parent_guard'):
                g=checked(row[k]);require(g['status']=='finished' and g['reason'] is None and g['elapsed_seconds']<600 and g['limits']['timeout_seconds']==600 and g['observation']['peak_rss_bytes']<4*1024**3 and g['observation']['peak_observed_mps_driver_bytes']<6*1024**3 and g['observation']['free_bytes']>=8*1024**3,'Original successful worker/parent guards required');guards.append(g)
            count=92 if arm=='sigmoid_elementwise_zero' else 88
            require(a['status']=='accepted' and a['complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked'] and a['step']==20 and a['targets']==20480 and a['ema_snapshots']==8 and a['parameter_tensors']==count and len(a['parameter_gradient_steps'])==count and all(type(v) is int and v==20 for v in a['parameter_gradient_steps'].values()) and not a['holdout_scored'],'Complete actual RAW/H200 state and exposure acceptance required')
            pair=0
            for name in ('parent','h200'):
                ref=worker[name];p=Path(ref['path']);s=p.stat();require(p==R/f'evidence/pilots/{slot:02d}/checkpoints/{name}.pt' and not p.is_symlink() and s.st_nlink==1 and s.st_size==ref['size'] and reference(p)['sha256']==ref['sha256'] and ref['sha256'] not in hashes,'Exact unique original retained checkpoint required')
                hashes.add(ref['sha256']);pair+=s.st_size;weight_rows.append(dict(**ref,inode=s.st_ino,device=s.st_dev))
            require(pair<=spec['weight_cap'],'Complete retained pair exceeds original cap')
            for name in ('raw','h200'):
                r=a['full_validation'][name];mps=r['mps'];cpu=r['cpu'];target=mps['target'];require(cpu['target']==target and target['targets']==34278 and target['target_bytes']==111536 and target['target_ids_sha256']=='dad00eaab7c2076736432b83eb9f1cc404c87aa5e4820b5f7abebf384c7fcdaf','Full original native target required')
                validate_report(mps,target);validate_report(cpu,target);delta=abs(cpu['ce']-mps['ce']);require(delta==r['cpu_mps_ce_difference'] and delta<=3e-5,'Original full native CPU/MPS tolerance required');maximum_ce_error=max(maximum_ce_error,delta)
            for device in ('cpu','mps'):
                cache=a['cache'][device];require([r['batch'] for r in cache]==[1,8] and all(r['tokens']==128 and r['cache_bytes_per_item']==196608 and all(type(r[k]) is float and math.isfinite(r[k]) and r[k]>=0 for k in ('maximum_logit_difference','reset_maximum_logit_difference')) for r in cache),'Both full-serving/cache/reset geometries required')
        retained=sum(r['size'] for r in weight_rows);require(len(hashes)==6 and retained==review['retained_weight_bytes'],'All six pilot originals retained exactly once')
        require(len(review['projections'])==3,'Complete actual three-arm time projections required')
        for row,projection in zip(review['rows'],review['projections']):
            summary=checked(row['worker']);process=checked(row['process']);parent=checked(row['parent_guard'])
            expected=2*(summary['training_seconds']/20*1200+max(0.,process['elapsed_seconds']-summary['training_seconds'])+parent['elapsed_seconds'])
            require(projection['arm']==row['arm'] and projection['conservative1200_seconds']==expected and math.isfinite(expected) and expected<7200,'Original conservative horizon projection arithmetic must pass')
        for ref in prior['weights']:
            p=Path(ref['path']);st=p.stat();require(not p.is_symlink() and st.st_nlink==1 and st.st_size==ref['size'] and st.st_ino==ref['inode'] and st.st_dev==ref['device'] and reference(p)['sha256']==ref['sha256'] and ref['sha256'] not in hashes,'All six worker originals must survive pilot qualification');hashes.add(ref['sha256'])
        require(len(hashes)==12 and prior['retained_weight_bytes']==plan['prerequisites']['worker_retained_bytes'],'All twelve prerequisite originals retained, no duplicate credit')
        # Mirror only closed metadata; never copy or load numerical payloads.
        require(not list(R.rglob('*.tmp')),'Unfinished output cannot qualify');MIRROR.mkdir();mirrors=[]
        for p in sorted(R.rglob('*')):
            if not p.is_file() or p.name.endswith('.lock') or p.name=='.lock':continue
            require(not p.is_symlink(),'Symlink forbidden in closed evidence')
            if 'checkpoints' in p.relative_to(R).parts:continue
            raw=p.read_bytes();require(len(raw)<=4*1024**2,'Closed metadata file cap required');target=MIRROR/p.relative_to(R);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(raw)
            require(reference(target)['sha256']==reference(p)['sha256'],'Metadata mirror differs');mirrors.append(dict(original=reference(p),mirror=reference(target),bytes=len(raw)))
        design=checked(plan['design']);floor=max(8*1024**3,design['complete_retained_budget']['full_floor']-retained-prior['retained_weight_bytes']);target=floor+2*1024**3;free=shutil.disk_usage(ROOT).free;require(free>=target,'Full remaining retained requirement plus2GiB required')
        out=dict(status='all_three_gate_resource_pilots_closed_independently_reviewed',source=reference(__file__),source_bindings=1074,source_guard=reference(E/'gate-worker-source001-guard.json'),source_terminal=reference(E/'STOP-gate-worker-source001'),plan=review['plan'],original_review=reference(E/'gate-pilot001-review.json'),terminal=reference(R/'STOP'),weights=weight_rows,retained_weight_bytes=retained,worker_retained_weight_bytes=prior['retained_weight_bytes'],all_prerequisite_retained_weight_bytes=retained+prior['retained_weight_bytes'],projections=review['projections'],maximum_cpu_mps_ce_difference=maximum_ce_error,peak_rss_bytes=max(g['observation']['peak_rss_bytes'] for g in guards),peak_mps_driver_bytes=max(g['observation']['peak_observed_mps_driver_bytes'] for g in guards),metadata_mirrors=mirrors,metadata_mirror_bytes=sum(x['bytes'] for x in mirrors),weights_copied=0,models_or_scoring_replayed=False,remaining_full_floor=floor,remaining_target=target,free_bytes=free,scientific_quality=False,scientific_seeds_assigned=False,adopted=False)
        publish(OUT,out);print(json.dumps(dict(status=out['status'],retained_weight_bytes=retained,metadata_files=len(mirrors),maximum_cpu_mps_ce_difference=maximum_ce_error)),flush=True)
