"""Distinct structural-AST own FFN metadata review after closed string-check defect; AST/scalar metadata, no Torch/NumPy."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
CAP=2*1024**2

def require(value,message):
    if not value:raise ValueError(message)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def publish(name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())+len(raw)<=CAP,'Source metadata cap')
    with (HERE/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def integer(node,values):
    if isinstance(node,ast.Constant) and type(node.value) is int:return node.value
    if isinstance(node,ast.Name) and node.id in values:return values[node.id]
    if isinstance(node,ast.BinOp):
        a,b=integer(node.left,values),integer(node.right,values)
        if isinstance(node.op,ast.Add):return a+b
        if isinstance(node.op,ast.Mult):return a*b
        if isinstance(node.op,ast.Pow):return a**b
    raise ValueError('Unsupported scalar accounting AST')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    require(sha(HERE/'plan.json')==arg.plan_sha and set(p.name for p in HERE.iterdir())=={'review-source002.py','plan.json','source-review.json'},'One pristine exact source review only')
    plan=json.loads((HERE/'plan.json').read_text());publish('attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha,source_sha256=sha(__file__)))
    try:
        with CpuPrerequisiteGuard(HERE/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
            require(shutil.disk_usage(ROOT).free>=plan['source_only_target_bytes'],'Source preparation reserve/headroom')
            require(len(plan['sources'])==1213,'1201predecessors+tenimplementations+both saved metadata reviewers')
            for p,h in plan['sources'].items():require(sha(ROOT/p)==h,'Pinned source differs: '+p)
            texts={};trees={}
            for p in plan['implementation_files']:
                file=ROOT/p;raw=file.read_text();tree=ast.parse(raw,filename=str(file));compile(tree,str(file),'exec');texts[file.name]=raw;trees[file.name]=tree
                imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
                require(not any('readout_projection' in name for name in imports),'Foreign projection model/state/worker import')
            require(len(texts)==10,'All ten new scientific modules')
            t=texts['training.py'];v=texts['validation.py'];c=texts['contract.py'];x=texts['execution.py'];b=texts['benchmark.py'];s=texts['statistics.py'];f=texts['final_review.py']
            checks=dict(
                full1200_only="cfg.training.max_iters==1200" in t,
                exact41_ema=any(ast.dump(node)==ast.dump(ast.parse("averaging_contract(cfg.training)==dict(kind='spaced_tail_ema',total_steps=1200,start_step=800,interval=10,half_lives=[200])",mode='eval').body) for node in ast.walk(trees['training.py'])) and any(ast.dump(node)==ast.dump(ast.parse("raw['ema']['snapshots']==summary['ema_snapshots']==41",mode='eval').body) for node in ast.walk(trees['validation.py'])),
                full_exposure="active==[n for n,_ in model.named_parameters()]" in t,
                explicit_global_roles=all('cfg.rng_identity.'+n in t for n in ('global_cpu','global_mps')),
                own_full_binding='ffn_full_quality_execution_v1' in t and 'Q.sources()' in t,
                no_prerequisite_masquerade="'relu_squared_execution' not in plain(cfg)" in t,
                own_pair_saved_before_acceptance=v.index('S.validate_report(cpu_report')<v.index('S.publish_pair_then_require(')<v.index('reports[name]=dict('),
                own_archive_reads=v.count('C.read_payload(')==2 and 'read_payload(dict(kind=H200' in b,
                prospective25_roles='len(set(values))==25' in c and all(name in c for name in ('initializer','private_ungated','global_cpu','global_mps')),
                fresh_bound_to_initializer="w['config']['initializer_seed']==seed" in x and "seed=cfg.initializer_seed" in b,
                all9_serving_preflight=b.index('len(accepted)==9')<b.index('torch.mps.get_rng_state()'),
                mps_rng_finally='finally:torch.mps.set_rng_state(saved_mps_rng)' in b,
                prespecified_new_cache='prefill=57,decode=71' in v and 'range(83,128)' in b,
                target_policy_identity='6dcd972d01877136396b4a7c1bd11be02097623b7becbd77c58c469ec2c060d1' in s and 'relu_squared_resume_v1.scoring' in s,
                strict_both_controls="both_mean_ce_below_minus_point02" in s and "every_paired_seed_delta_negative" in s and "all_six_simultaneous_conditional_text_upper_bounds_negative" in s,
                independent_window_quantile="window_loss_sums" in f and 'ordered=sorted(samples[a])' in f and '1-.05/6' in f,
                complete_source_bridge='len(pins)==1211' in c and "len(review['sources'])==1201" in c,
                numpy_runtime_pinned='numpy_version=str(np.__version__)' in c,
                no_quality_seal_or_roles_assigned=plan['no_roles_assigned_or_gate_freeze_launch_admitted'] is True,
                guards_unchanged="(600,7200)" in c and 'timeout_seconds=120' in v,
            )
            require(all(checks.values()),'Static invariant failed: '+repr([k for k,z in checks.items() if not z]))
            values={}
            for node in trees['contract.py'].body:
                if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('RESERVE','ENTRY','PREREQUISITE_WEIGHTS','CONSUMER_SCRATCH','FLOOR','TARGET'):
                    values[node.targets[0].id]=integer(node.value,values)
            require(values['FLOOR']==23490199552 and values['TARGET']==25637683200,'Complete new output/metadata/prerequisite/consumer/reserve accounting differs')
            require(1+(1200-800)//10==41 and 1200*8*128==1228800 and 18*1228800==22118400,'Full horizon scalar scope differs')
            require(3*256*672==2*256*1008==516096,'Matched matrix budget differs')
            require('torch' not in sys.modules and 'numpy' not in sys.modules,'Source review imported numerical runtime')
            require(all(sha(ROOT/p)==digest for p,digest in plan['unchanged_implementation_sources'].items()),'Byte-identical implementations required;no source rescue')
            publish('review.json',dict(prior_failed_review=plan['prior_failed_review'],unchanged_implementations=True,status='closed_accepted_source_only_full_horizon_ffn_design_and_namespace',sources=plan['sources'],runtime_source_bindings=1211,
                static_checks=checks,implementation_bytes=sum((ROOT/p).stat().st_size for p in plan['implementation_files']),scalar_accounting=values,
                no_torch_numpy_worker_import=True,no_models_optimizer_corpus_checkpoints_rng_scoring_bootstrap_serialization_calls=True,
                no_roles_assigned_or_quality_budget_freeze_or_launch_admitted=True,all_new_runtime_paths_untested=True,free_bytes=shutil.disk_usage(ROOT).free,
                next_action='Prepare NEW full-horizon generated actual-state/config/selection/final-statistics/resource proof and role-census sources/cases/plans;no old fixture replay/stepcount masquerade. Review complete prerequisites/allconsumers before execution.'))
        publish('STOP',dict(status='complete_source_only_full_horizon_ffn_design_namespace_preparation',no_replay=True,no_retry=True))
    except BaseException as exc:
        if not (HERE/'STOP').exists():publish('STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
        raise

if __name__=='__main__':main()
