"""Finite worker routes/scorer rejection and arithmetic gate; no training/full scores."""
import copy,importlib.util,json,gc,math
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference,environment
from research.gpt2_prefix_data_v1.data import config,load,architecture
from research.gpt2_prefix_model_v2.model import GPT
from research.gpt2_prefix_worker_v1.execution import sources
from research.gpt2_prefix_worker_v1.training import run
from research.gpt2_prefix_worker_v1.scoring import target_contract,validate_report
from research.tokenizer_common_bytes_v1.metric import loss_summary
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent

def reject(fn):
    try:fn()
    except (ValueError,RuntimeError,KeyError,TypeError):return
    raise AssertionError('Forgery or disallowed route accepted')
def publish(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
with session_lock(GLOBAL_LOCK.parent):
    torch.set_num_threads(4);idle();env=environment();require(not(E/'worker-source-review001-plan.json').exists(),'New source gate only')
    pins=sources();require(len(pins)==993,'Exact current source count required');proof=json.loads((E/'resume001.json').read_text());require(proof['status']=='passed_exact_native_and8K_real_repeat_and_serialized_resume' and all(pins[n]==h for n,h in proof['sources'].items()),'Actual transport proof/source bridge required')
    refs={n:reference(E/n) for n in ['component002.json','data001.json','resume001.json']};builder=E/'worker001-build-run001.py'
    publish(E/'worker-source-review001-plan.json',dict(source=reference(__file__),prepared_builder=reference(builder),sources=pins,prerequisites=refs,scientific_quality=False,fixture_seed=600425,training=False,full_validation_scoring=False,weight_files=0))
    try:
        with StrictMpsGuard(E/'worker-source-review001-guard.json',ROOT),torch.random.fork_rng(devices=[]):
            rows=[]
            for kind in ['native','prefix8192']:
                for steps in [4,20]:
                    for lr in [.000625,.00125,.0025]:require(architecture(config(kind,steps=steps,seed=600425,lr=lr))['vocab_size']==(50257 if kind=='native' else 8192),'Finite genuine config route differs')
                reject(lambda:run(config(kind,steps=1200,seed=600425),None,None))
                cfg=config(kind,seed=600425);data=load(cfg,'cpu');torch.manual_seed(600425);m=GPT(plain(cfg.prefix_model)).to('mps');reject(lambda:target_contract(m,data,cfg));m.eval();target=target_contract(m,data,cfg)
                forged=copy.copy(data);forged.common_byte_plan=dict(data.common_byte_plan,target_sha256='0'*64);reject(lambda:target_contract(m,forged,cfg))
                values=[.25+(i%19)*.07 for i in range(target['target_tokens'])];summary=loss_summary(data.common_byte_plan,values)
                report=dict(summary,target=target,window_loss_sums=[math.fsum(values[i:i+128]) for i in range(0,len(values),128)],holdout_scored=False)
                validate_report(report,target,data.common_byte_plan)
                cases=[]
                bad=copy.deepcopy(report);bad['window_loss_sums']=bad['window_loss_sums'][:-1];cases.append(bad)
                bad=copy.deepcopy(report);bad['bits_per_byte']+=.01;cases.append(bad)
                bad=copy.deepcopy(report);bad['target_sha256']='0'*64;cases.append(bad)
                bad=copy.deepcopy(report);bad['target_tokens']-=1;cases.append(bad)
                bad=copy.deepcopy(report);bad['raw_byte_blocks']['4096']['byte_counts'][0]-=1;cases.append(bad)
                bad=copy.deepcopy(report);bad['raw_byte_blocks']['8192']['loss_sums'][0]=float('nan');cases.append(bad)
                bad=copy.deepcopy(report);bad['raw_byte_blocks']['16384']['loss_sums'][0]+=.1;cases.append(bad)
                bad=copy.deepcopy(report);bad['holdout_scored']=True;cases.append(bad)
                for bad in cases:reject(lambda bad=bad:validate_report(bad,target,data.common_byte_plan))
                rows.append(dict(kind=kind,finite_prerequisite_configs=6,full1200route_rejected_before_data_or_model=True,train_mode_and_mutated_target_rejected=True,score_report_forgeries_rejected=len(cases),target_tokens=target['target_tokens'],target_bytes=target['target_bytes']))
                del m,data,forged;gc.collect();torch.mps.empty_cache()
            spec=importlib.util.spec_from_file_location('prepared_prefix_worker_builder',builder);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);specs=module.specs()
            require(len(specs)==2 and [s['arm'] for s in specs]==['native','prefix8192'] and [s['weight_cap'] for s in specs]==[560*1024**2,160*1024**2] and all(s['config']['training']['max_iters']==4 and 'model' not in s['config'] for s in specs),'Prepared exact pair scope differs')
            require(not module.RUN.exists() and not module.PLAN.exists(),'Generated pair already attempted')
            require(sources()==pins,'Worker source changed during review')
        publish(E/'worker-source-review001.json',dict(status='passed_genuine_worker_source_and_common_byte_contracts',source=reference(__file__),prepared_builder=reference(builder),sources=pins,prerequisites=refs,reports=rows,environment=env,total_source_bindings=993,
            numerical_quality=False,training=False,full_validation_scoring=False,scientific_seed_used=False,holdout_encoded_or_scored=False,weight_files=0,next_action='Execute prepared new finite worker builder ONCE only if actual roots/plans absent and full pair budget/strict environment pass.'))
        print(json.dumps(dict(status='passed_genuine_worker_source_and_common_byte_contracts',reports=rows,total_sources=993)),flush=True)
    except Exception as exc:publish(E/'STOP-worker-source-review001',dict(error=repr(exc),closed=True,no_retry=True));raise
