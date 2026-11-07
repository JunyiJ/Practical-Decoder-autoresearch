"""Complete four independent cohorts at each of three fixed RAW controls, once."""
import gc,json,shutil
import torch
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_moe_resume_v1.contract import runtime as environment
from research.tokenizer_moe_worker_v1.execution import idle
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.tokenizer_gradient_probe_v1.data import load_train
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from . import contract as K
C=K.C

def main():
    torch.set_num_threads(4)
    with session_lock(GLOBAL_LOCK.parent):
        idle();env=environment();require(not K.RUN.exists(),'Closed/attempted cohort diagnostic cannot rerun')
        pins=K.sources();prerequisites=K.prerequisites()
        proof=C.checked(reference(C.E/'cohort-contracts001.json'));guard=C.checked(reference(C.E/'cohort-contracts001-guard.json'))
        require(proof['status']=='generated_cohort_combined_stream_and_transactional_contracts_passed' and proof['sources']==pins and guard['status']=='finished' and guard['reason'] is None,'New complete generated contracts required')
        bound=C.bindings();free=shutil.disk_usage(ROOT).free
        require(free>=K.FLOOR+1458176,'Complete retained metadata/reserve/external/entry forecast required')
        forecast=dict(metadata_cap=K.META,projected_metadata_bytes=5718764,mean_trees_bytes=455878656,prior_peak_rss=2122596352,projected_peak_rss=2122596352+227939328+512*1024**2,rss_limit=4*1024**3,mps_limit=6*1024**3,conservative_seconds=438.844586,timeout_seconds=600)
        require(forecast['projected_peak_rss']<forecast['rss_limit'] and forecast['conservative_seconds']<600 and forecast['projected_metadata_bytes']<K.META,'Full live forecast must fit unchanged guards')
        K.RUN.mkdir()
        K.publish(K.RUN/'plan.json',dict(kind='fixed_weight_independent_gradient_cohorts_v1',sources=pins,prerequisites=prerequisites,contracts=reference(C.E/'cohort-contracts001.json'),environment=env,bindings=[dict(slot=v['slot'],worker=v['worker_ref'],raw=v['worker']['parent']) for v in bound],cohort_seeds=K.SEEDS,initializer=K.INITIALIZER,cohorts_per_endpoint=4,draws_per_cohort=32,combined_draws_per_endpoint=128,total_target_evaluations=393216,old_probe_excluded=True,validation_encoded=False,holdout_encoded=False,optimizer_steps=0,weights_written=0,full_floor=K.FLOOR,entry_wake=K.FLOOR+1458176,target=K.TARGET,free=free,forecast=forecast))
        try:
            streams=[];mps_before=torch.mps.get_rng_state().clone();cpu_before=torch.get_rng_state().clone()
            with StrictMpsGuard(K.RUN/'guard.json',ROOT):
                for item in bound:
                    w=item['worker'];model=K.load_dense(w,'mps',K.INITIALIZER);initial=C.state_hash(model);moments=K.Cohorts();draws=[];cohort_rows=[]
                    for index,seed in enumerate(K.SEEDS):
                        view=load_train(w,'mps',seed);training_identity=view.identity;cohort_draws=[]
                        for draw in range(1,33):
                            x,y=view.next_train();g,ce=C.gradients(model,x,y)
                            report=moments.update(index,seed,draw,g)
                            entry=dict(cohort=index,seed=seed,draw=draw,targets=1024,ce=ce,input_sha256=C.tensor_hash(x),target_sha256=C.tensor_hash(y),gradient_squared_norms=C.gradient_norms(g))
                            cohort_draws.append(entry);draws.append(entry)
                            del g,x,y;model.zero_grad(set_to_none=True);gc.collect()
                        K.check_norm_identity(report,cohort_draws)
                        replay=torch.Generator().manual_seed(seed)
                        for _ in range(32):torch.randint(267688-512,(2,),generator=replay)
                        require(torch.equal(replay.get_state(),view.generators['train'].get_state()),'Exact independent private sampler replay differs')
                        cohort_rows.append(dict(cohort=index,seed=seed,statistics=report,groups=C.grouped(report),private_sampler_state_sha256=C.tensor_hash(view.generators['train'].get_state())))
                        del view,report,cohort_draws
                    reports=moments.report();K.check_norm_identity(reports['combined'],draws)
                    final=C.state_hash(model);require(final==initial,'Fixed model weights/buffers mutated')
                    stream=[(d['input_sha256'],d['target_sha256']) for d in draws];streams.append(stream)
                    require(len(set(tuple(stream[i*32:(i+1)*32]) for i in range(4)))==4,'Four independent cohort streams required')
                    row=dict(slot=item['slot'],worker=item['worker_ref'],raw=w['parent'],source_training_seed=w['config']['training']['seed'],training_only_identity=training_identity,draws=draws,cohorts=cohort_rows,combined=reports['combined'],combined_groups=C.grouped(reports['combined']),model_before=initial,model_after=final,model_unchanged=True,optimizer_steps=0,old_probe_excluded=True)
                    K.publish(K.RUN/f'endpoint-{item["slot"]:02d}.json',row)
                    print(json.dumps(dict(slot=item['slot'],complete_cohorts=4,complete_draws=128)),flush=True)
                    del model,moments,draws,cohort_rows,reports,row;gc.collect();torch.mps.empty_cache()
                    C.verify_artifact(w['parent']);require(K.sources()==pins,'Immutable sources changed during measurement')
                require(torch.equal(cpu_before,torch.get_rng_state()) and torch.equal(mps_before,torch.mps.get_rng_state()),'Global CPU/MPS RNG changed')
                require(streams[0]==streams[1]==streams[2],'Same cohort streams across all fixed models required')
            K.publish(K.RUN/'result.json',dict(status='complete_training_free_three_endpoint_four_cohort_diagnostic',endpoints=[reference(K.RUN/f'endpoint-{s:02d}.json') for s in C.SLOTS],sources=pins,total_target_evaluations=393216,cohort_seeds=K.SEEDS,combined_draws_per_endpoint=128,optimizer_steps=0,weight_files_written=0,validation_encoded=False,holdout_encoded=False,old_probe_excluded=True,global_cpu_mps_rng_unchanged=True,quality_result=False,adopted=False,scope='Raw unpreconditioned endpoint moments in independent paired512 B8 draw units;biased ratios,not IIDtokens/optimalbatch/MuonAdamupdatenoise/qualityprediction.'))
            K.publish(K.RUN/'STOP',dict(status='complete_training_free_gradient_cohorts',result=reference(K.RUN/'result.json'),never_rerun=True))
            print(json.dumps(dict(status='complete_training_free_three_endpoint_four_cohort_diagnostic')),flush=True)
        except BaseException as exc:
            if not(K.RUN/'STOP').exists():K.publish(K.RUN/'STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
            raise
if __name__=='__main__':main()
