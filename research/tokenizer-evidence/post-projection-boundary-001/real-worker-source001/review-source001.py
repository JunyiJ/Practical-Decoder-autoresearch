"""One source-only review: AST/compile/scalar bounds, never imports worker modules."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def publish(name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    require(sum(p.stat().st_size for p in HERE.iterdir() if p.is_file())+len(raw)<=2*1024**2,'Source review metadata cap')
    with (HERE/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    require(sha(HERE/'plan.json')==arg.plan_sha,'Exact source-only plan required')
    require(set(p.name for p in HERE.iterdir())=={'plan.json','review-source001.py'},'One pristine source-only review required')
    plan=json.loads((HERE/'plan.json').read_text());require(plan['scope']=='SOURCE_ONLY_NO_WORKER_IMPORT','Source-only scope')
    publish('attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha,executed_source_sha256=sha(__file__)))
    try:
        with CpuPrerequisiteGuard(HERE/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
            require(shutil.disk_usage(ROOT).free>=plan['source_review_target_bytes'],'Source review capacity')
            for p,h in plan['sources'].items():require(sha(ROOT/p)==h,'Pinned source changed: '+p)
            require(len(plan['sources'])==1198,'1191 accepted predecessors plus6 implementation files and this reviewer')
            trees={};texts={}
            for relative in plan['new_implementation_files']:
                p=ROOT/relative;text=p.read_text();tree=ast.parse(text,filename=str(p));compile(tree,str(p),'exec')
                texts[p.name]=text;trees[p.name]=tree
                imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
                require(not any('readout_projection' in n for n in imports),'No foreign projection worker/model/state imports')
            train=texts['training.py'];validation=texts['validation.py'];execution=texts['execution.py'];runner=texts['relu-squared-worker001.py']
            checks={
                'exact_4_20_only':'cfg.training.max_iters in (4,20)' in train,
                'prerequisite_lr':'cfg.training.learning_rate==.00125' in train,
                'explicit_global_roles':all('cfg.rng_identity.'+n in train for n in ('global_cpu','global_mps')),
                'complete_gradient_exposure':'active==[n for n,_ in model.named_parameters()]' in train,
                'counted_bounded_serialization':'C.FILE_CAPS[payload[\'kind\']]' in train and 'ref[\'size\']==counter.size' in train,
                'own_hash_checked_reads':validation.count('C.read_payload(')==2,
                'caller_validates_physical_pairs':validation.index('S.validate_report(cpu_report')<validation.index('S.publish_pair_then_require(')
                    and validation.index('S.validate_report(mps_report')<validation.index('S.publish_pair_then_require('),
                'own_pair_durable_before_acceptance':validation.index('S.publish_pair_then_require(')<validation.index('reports[name]=dict('),
                'new_41_87_cache_partition':'start_pos=41' in validation and 'prefill=41,decode=87' in validation,
                'fixed_cache_tolerances':'rtol=3e-5,atol=3e-5' in validation and 'size==196608' in validation,
                'no_quality_worker_stage':"choices=['pilots']" in execution,
                'future_harnesses_required':all(n in execution for n in ('relu-squared-worker-source001.py','relu-squared-worker-seed-census001.py','len(pins)==1200')),
                'all_four_before_twenty':runner.index("if phase == 'pilot':")<runner.index('specs=specs(pins, phase, roles[\'roles\'])')
                    and 'all_three_ffn_four_update_workers_independently_accepted' in runner,
                'full_twelve_role_identity':"len(set(roles.values()))==12" in runner and 'ungated_seed(roles[prefix+' in runner,
                'no_retry_stop':"never_retry=True" in runner,
            }
            require(all(checks.values()),'Static invariant failed: '+repr([n for n,v in checks.items() if not v]))
            # Literal scalar bounds only. No Torch import, generator, tensors or serialization.
            learned=28492416*4;rope=65536;model=learned+rope;moments=216929456;ema=learned
            raw=learned+rope+moments+ema;h200=model
            raw_upper=raw+16*1024**2;h200_upper=h200+8*1024**2
            require(raw_upper<512*1024**2 and h200_upper<128*1024**2,'Prospective payload caps')
            pair=640*1024**2;root_meta=64*1024**2;external_meta=96*1024**2;entry=2*1024**2;reserve=8*1024**3
            floor=6*pair+2*root_meta+external_meta+entry+reserve;target=floor+2*1024**3
            require(target==plan['prospective_chain_target_bytes'],'Full prospective retained-output accounting differs')
            # Serial phases, not sum of mutually exclusive resident phases. Cover
            # parent RAW read, live/saved EMA, CPU weight copies, staging/deepclone,
            # temporary full log probabilities and allocator/runtime margin.
            working=dict(parent_read=512*1024**2,live_model=model,live_ema=ema,saved_ema=ema,
                live_moments=moments,staged_moments=moments,loader_deepclone=moments,
                cpu_model=model,raw_and_h200_cpu_weights=2*model,
                three_B8_T128_V50257_FP32_arrays=3*8*128*50257*4,
                runtime_allocator_margin=768*1024**2)
            require(sum(working.values())<4*1024**3,'Conservative serial working bytes exceed RSS gate')
            require('torch' not in sys.modules,'Source-only review imported Torch')
            publish('scalar-accounting.json',dict(raw_tensor_subtotal=raw,h200_tensor_subtotal=h200,raw_upper_bytes=raw_upper,
                h200_upper_bytes=h200_upper,raw_cap=512*1024**2,h200_cap=128*1024**2,pair_cap=pair,
                all12_output_cap=6*pair,serialization_tmp_policy='one allocation within each pair cap; failed partial stops chain',
                metadata_per_root=root_meta,external_metadata=external_meta,entry=entry,reserve=reserve,
                prospective_chain_floor_bytes=floor,prospective_chain_target_bytes=target,
                prospective_working_components=working,prospective_working_bytes=sum(working.values()),
                mps_forecast_bytes=int(3.5*1024**3),worker_forecast_seconds=180.,parent_forecast_seconds=240.,
                complete_six_run_forecast_seconds=2640.,measured_archives_or_runtime=False))
            publish('review.json',dict(status='closed_accepted_source_only_real_ffn_prerequisite_worker_preparation',
                sources=plan['sources'],static_checks=checks,new_implementation_bytes=sum((ROOT/p).stat().st_size for p in plan['new_implementation_files']),
                no_worker_import_or_numerical_execution=True,no_roles_assigned=True,no_census_or_source_gate_or_launch_admission=True,
                runtime_paths_untested=True,torch_imported=False,free_bytes=shutil.disk_usage(ROOT).free,
                next_action='Implement/pin/review NEW twelve-role census and finite config/serializer source gate; exact sources/cases/predicates/resources before any execution. No worker/model/corpus or quality launch admitted by SOURCE review.'))
        publish('STOP',dict(status='complete_source_only_real_ffn_worker_preparation',no_replay=True,no_retry=True))
    except BaseException as exc:
        if not (HERE/'STOP').exists():publish('STOP',dict(status='closed_failed_source_only_review',error=repr(exc),no_retry=True))
        raise

if __name__=='__main__':main()
