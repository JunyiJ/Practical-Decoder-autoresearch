"""Independent terminal metadata/source review; no model, gradient or proof replay."""
import hashlib,json,os,shutil
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.tokenizer_moe_worker_v1.execution import idle

E=Path(__file__).resolve().parent;R=E/'gate-resume002';OUT=E/'resume-review001.json';GUARD=E/'resume-review001-guard.json'

def read(p):return json.loads(Path(p).read_text())

with session_lock(GLOBAL_LOCK.parent):
    idle();require(not OUT.exists() and not GUARD.exists(),'Attempted/closed independent review cannot rerun')
    with CpuPrerequisiteGuard(GUARD,ROOT,timeout_seconds=120):
        result=read(R/'result.json');plan=read(R/'plan.json');guard=read(R/'guard.json');stop=read(R/'STOP');failure=read(E/'gate-resume-failure001.json')
        require(result['status']=='all_three_native_gate_real_repeat_and_serialized_resume_passed' and result['plan']==reference(R/'plan.json') and stop==dict(status='complete_own_gate_real_data_resume_proof',result=reference(R/'result.json'),never_rerun=True),'Actual complete corrected proof and terminal required')
        require(guard['status']=='finished' and guard['reason'] is None and guard['elapsed_seconds']<600 and guard['observation']['peak_rss_bytes']<4*1024**3 and guard['observation']['peak_observed_mps_driver_bytes']<6*1024**3 and guard['observation']['free_bytes']>=8*1024**3,'All original resource guards required')
        require(plan['original_failed_proof']==reference(E/'gate-resume-failure001.json'),'Corrected harness must explicitly bind closed failed original')
        for k in ('source','plan','guard','terminal'):require(reference(failure[k]['path'])==failure[k],'Original failure source/evidence changed')
        failed_guard=read(failure['guard']['path']);failed_stop=read(failure['terminal']['path'])
        require(failure['status']=='closed_harness_nonpersistent_buffer_lookup_failure' and failure['exception']=="KeyError('transformer.h.0.attn.causal_mask')" and failed_guard['status']=='fixture_exception' and failed_stop['no_retry'] and not (E/'gate-resume001/result.json').exists(),'Original failed proof must remain failed without partial acceptance')
        pins=plan['sources'];require(len(pins)==result['source_bindings']==1068,'Full corrected proof binding count required')
        old=read(E/'gate-model001/plan.json')['sources'];require(len(old)==1062 and all(pins.get(p)==h for p,h in old.items()),'Preserve all generated gate and original scientific sources')
        for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Bound source changed: '+p)
        roles=read(plan['roles']['path']);require(reference(plan['roles']['path'])==plan['roles'] and roles['no_prior_matches'] and not roles['scientific_seeds'] and roles['roles']==result['roles_used']==dict(initializer=600523,private_train=600524,private_evaluation=600525),'Actual new role audit and usage required')
        counts={'none':(28492416,88),'static_half':(28492416,88),'sigmoid_elementwise_zero':(28754560,92)}
        require([r['arm'] for r in result['reports']]==list(counts),'Complete three-arm report required')
        for row in result['reports']:
            arm=row['arm'];e=row['endpoint'];require(row==read(R/('arm-'+arm+'.json')),'Original per-arm receipt differs')
            require((row['parameters'],row['learned_tensors'])==counts[arm] and row['fresh_repeat_exact'] and row['serialized_2_plus_2_exact'] and 0<row['serialized_bytes']<=plan['buffer_caps'][arm],'Exact generated inventory/repeat/serialized resume required')
            require(row['raw_transactional_rejections']==26 and row['h200_transactional_rejections']==4 and e['snapshots']==3 and e['clipping_updates']==4 and e['learned_parameter_tensors']==counts[arm][1] and len(e['per_parameter_updates'])==counts[arm][1] and all(type(v) is int and v==4 for v in e['per_parameter_updates'].values()),'Complete rejection/exposure/EMA arithmetic required')
            require(len(e['streams'])==4 and e['streams']==result['reports'][0]['endpoint']['streams'] and all(set(s)=={'train_input','train_targets','eval_input','eval_targets'} for s in e['streams']),'Exact paired train/evaluation replay and shared streams required')
            require(all(len(e[k])==64 for k in ('raw_model_optimizer_ema_history_sampler_cpu_mps_rng','h200','h200_logits')),'Complete state hash receipts required')
        require(result['training_target_evaluations']==result['sampled_validation_target_evaluations']==3*3*4*8*128 and result['same_inputs_across_arms'] and result['global_cpu_mps_rng_preserved'] and not result['full_validation_scored'] and not result['holdout_encoded_or_scored'] and not result['quality_result'] and not result['adopted'] and result['weight_files_written']==0 and not list(R.glob('*.pt*')),'Prerequisite-only scope and complete target accounting required')
        require(sum(p.stat().st_size for p in R.iterdir() if p.is_file())<=plan['metadata_cap']==2*1024**2,'Closed metadata cap required')
        design=read(E/'quality-design001.json');require(reference(E/'quality-design001.json')==plan['design'] and shutil.disk_usage(ROOT).free>=design['complete_retained_budget']['target'],'Full prospective retained-output target remains covered')
        out=dict(status='own_gate_real_data_resume_independently_reviewed',source=reference(__file__),source_bindings=1068,original_failed_proof=reference(E/'gate-resume-failure001.json'),plan=reference(R/'plan.json'),result=reference(R/'result.json'),guard=reference(R/'guard.json'),terminal=reference(R/'STOP'),arms=list(counts),raw_transactional_rejections=3*26,h200_transactional_rejections=3*4,inference_only_resume_rejections=3*3,training_target_evaluations=36864,sampled_validation_target_evaluations=36864,source_invariant=True,model_or_gradient_or_proof_replayed=False,quality_panel_frozen=False,scientific_seeds_assigned=False,quality_result=False,adopted=False)
        raw=(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<2*1024**2,'Independent metadata cap required')
        with OUT.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        print(json.dumps(dict(status=out['status'],source_bindings=1068)),flush=True)
