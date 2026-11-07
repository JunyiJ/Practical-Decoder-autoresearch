"""One complete three-endpoint,32-draw fixed-weight diagnostic; no implicit retry."""
import gc,json,os,math,shutil
import torch
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_moe_resume_v1.contract import runtime as environment
from research.tokenizer_moe_worker_v1.execution import idle
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.tokenizer_gradient_noise_v1.component import Moments
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from . import runtime as C
from .data import load_train
RUN=ROOT/'research_runs/tokenizer-gradient-noise-diagnostic-v1-001';META=16*1024**2;FLOOR=8*1024**3+META+1267904512+1458176;TARGET=FLOOR+2*1024**3

def publish(p,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();used=sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file())
    require(len(raw)+used<=META,'Complete16MiB metadata output cap required')
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def main():
    torch.set_num_threads(4)
    with session_lock(GLOBAL_LOCK.parent):
        idle();env=environment();require(not RUN.exists(),'Closed/attempted diagnostic cannot rerun')
        pins=C.sources();proof=C.checked(reference(C.E/'gradient-model001.json'));guard=C.checked(reference(C.E/'gradient-model001-guard.json'));roles=C.checked(reference(C.E/'fixture-role-census001.json'))
        require(proof['status']=='own_fixed_weight_gradient_integration_cpu_mps_and_actual_payloads_passed' and proof['sources']==pins and guard['status']=='finished' and guard['reason'] is None,'Actual CPU/MPS/data/model/RAW/resource integration required')
        require(roles['no_prior_matches'] is True and roles['probe_train']==C.TRAIN_SEED and roles['probe_initializer']==C.INITIALIZER_SEED,'Original private unused fixture roles required')
        bound=C.bindings();free=shutil.disk_usage(ROOT).free;require(free>=FLOOR+1458176,'Full3endpoint metadata/reserve/live entry budget required');RUN.mkdir()
        publish(RUN/'plan.json',dict(kind='fixed_weight_raw_gradient_diagnostic_v1',sources=pins,integration=reference(C.E/'gradient-model001.json'),environment=env,bindings=[dict(slot=v['slot'],worker=v['worker_ref'],raw=v['worker']['parent']) for v in bound],draws_per_endpoint=32,targets_per_draw=1024,total_target_evaluations=98304,private_train_seed=C.TRAIN_SEED,initializer_seed=C.INITIALIZER_SEED,validation_encoded=False,holdout_encoded=False,optimizer_steps=0,weights_written=0,full_floor=FLOOR,entry_wake=FLOOR+1458176,target=TARGET,free=free))
        try:
            rows=[];mps_before=torch.mps.get_rng_state().clone();cpu_before=torch.get_rng_state().clone()
            with StrictMpsGuard(RUN/'guard.json',ROOT):
                for item in bound:
                    w=item['worker'];view=load_train(w,'mps',C.TRAIN_SEED);model=C.load_dense(w,'mps',C.INITIALIZER_SEED);initial=C.state_hash(model);moments=Moments();draws=[]
                    for i in range(32):
                        x,y=view.next_train();g,ce=C.gradients(model,x,y);moments.update(g);norms=C.gradient_norms(g)
                        draws.append(dict(draw=i+1,targets=1024,ce=ce,input_sha256=C.tensor_hash(x),target_sha256=C.tensor_hash(y),gradient_squared_norms=norms))
                        del g,x,y;model.zero_grad(set_to_none=True);gc.collect()
                    report=moments.report();require(C.state_hash(model)==initial,'Fixed model weights/buffers mutated')
                    for n,stat in report['parameter_statistics'].items():
                        alternate=(math.fsum(d['gradient_squared_norms'][n] for d in draws)-32*stat['mean_gradient_squared_norm'])/31
                        require(math.isclose(alternate,stat['trace_sample_covariance'],rel_tol=1e-8,abs_tol=1e-10),'Independent norm/covariance identity differs')
                    replay=torch.Generator().manual_seed(C.TRAIN_SEED)
                    for _ in range(32):torch.randint(267688-512,(2,),generator=replay)
                    require(torch.equal(replay.get_state(),view.generators['train'].get_state()),'Private paired512 sampler replay differs')
                    row=dict(slot=item['slot'],worker=item['worker_ref'],raw=w['parent'],source_training_seed=w['config']['training']['seed'],probe_seed=C.TRAIN_SEED,training_only_identity=view.identity,draws=draws,statistics=report,groups=C.grouped(report),model_unchanged=True,private_sampler_state_sha256=C.tensor_hash(view.generators['train'].get_state()),optimizer_steps=0)
                    publish(RUN/f'endpoint-{item["slot"]:02d}.json',row);rows.append(row)
                    del model,moments,view,report,row;gc.collect();torch.mps.empty_cache();C.verify_artifact(w['parent'])
                    require(C.sources()==pins,'Immutable sources changed during diagnostic')
                require(torch.equal(cpu_before,torch.get_rng_state()) and torch.equal(mps_before,torch.mps.get_rng_state()),'GlobalCPU/MPS RNG changed')
                require(all([(d['input_sha256'],d['target_sha256']) for d in r['draws']]==[(d['input_sha256'],d['target_sha256']) for d in rows[0]['draws']] for r in rows),'Different probe training stream across fixed endpoints')
            publish(RUN/'result.json',dict(status='complete_training_free_three_endpoint_gradient_diagnostic',endpoints=[reference(RUN/f'endpoint-{s:02d}.json') for s in C.SLOTS],sources=pins,total_target_evaluations=98304,optimizer_steps=0,weight_files_written=0,validation_encoded=False,holdout_encoded=False,global_cpu_mps_rng_unchanged=True,quality_result=False,adopted=False,scope='Raw unpreconditioned gradient moments at3fixedRAWendpoints underindependentB8paired512draws;notwholetraining,optimalbatch,Muon/Adamupdatenoise orquality/generalizationprediction.'))
            publish(RUN/'STOP',dict(status='complete_training_free_gradient_diagnostic',result=reference(RUN/'result.json'),never_rerun=True));print(json.dumps(dict(status='complete_training_free_three_endpoint_gradient_diagnostic',endpoints=3)),flush=True)
        except BaseException as exc:
            if not(RUN/'STOP').exists():publish(RUN/'STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
            raise
if __name__=='__main__':main()
