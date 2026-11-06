"""Close actual prerequisite/resource evidence;replay data exposure,never model updates/scores."""
import gc,hashlib,json,math,shutil
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.gpt2_prefix_worker_v1.execution import sources,checked
from research.gpt2_prefix_worker_v1.validation import verify_artifact
from research.gpt2_prefix_data_v1.data import load
from research.gpt2_prefix_tokenizer_v1.codec import Codec
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent

def publish(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def exposure(worker):
    cfg=OmegaConf.create(worker['config']);data=load(cfg,'cpu');codec=Codec.from_descriptor(worker['data']['tokenizer'])
    lengths=torch.tensor([len(codec._engine.decode_single_token_bytes(i)) for i in range(data.vocab_size)],dtype=torch.int64)
    total=0;updates=[]
    for i in range(cfg.training.max_iters):
        x,y=data.get_batch('train');n=int(lengths[y].sum());total+=n;updates.append(n)
        if i%cfg.training.eval_every==0 or i==cfg.training.max_iters-1:
            for _ in range(cfg.training.eval_iters):data.get_batch('val')
    raw=torch.load(worker['parent']['path'],map_location='cpu',weights_only=True)
    require(all(torch.equal(g.get_state(),raw['sampler_state'][name]) for name,g in data.generators.items()),'Actual private sampler replay differs')
    del raw;gc.collect()
    return dict(training_target_tokens=1024*cfg.training.max_iters,sampled_training_target_bytes=total,per_update_target_bytes=updates,mean_sampled_target_bytes_per_token=total/(1024*cfg.training.max_iters),
        actual_saved_samplers_match=True,scope='Repeated sampled target-token UTF8 byte lengths,not unique corpus coverage or literal epochs. Target-byte context/exposure differs across tokenizers;no equal-byte/FLOP/time claim.')
with session_lock(GLOBAL_LOCK.parent):
    torch.set_num_threads(4);idle();env=environment();require(not(E/'prerequisite-close001-plan.json').exists(),'New one-shot closure only')
    refs={kind:reference(E/f'{kind}001-review.json') for kind in ['worker','pilot']};reviews={n:checked(v) for n,v in refs.items()};pins=sources()
    require(reviews['worker']['status']=='two_gpt2_prefix_generated_workers_independently_accepted' and reviews['pilot']['status']=='two_prefix_resource_pilots_independently_accepted' and reviews['pilot']['all_original7200_projections_pass'],'Complete actual accepted prerequisite/pilot pairs with passing projections required')
    roots={n:Path(checked(d['plan'])['root']) for n,d in reviews.items()}
    for n,r in roots.items():require(checked(reference(r/'STOP'))['status']==('prerequisite_pair_complete' if n=='worker' else 'resource_pair_complete'),'Original root not terminally complete')
    publish(E/'prerequisite-close001-plan.json',dict(source=reference(__file__),review_refs=refs,sources=pins,no_training_or_model_rescoring=True,no_deletion_or_weight_copies=True))
    try:
        with CpuPrerequisiteGuard(E/'prerequisite-close001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
            reports=[];weights=[];mirrors=[];destination=E/'complete-metadata-mirror001';require(not destination.exists(),'No repeat metadata mirror');destination.mkdir()
            for kind,review in reviews.items():
                plan=checked(review['plan']);require(plan['sources']==pins and len(review['rows'])==2,'Original plan source/scope differs')
                for row in review['rows']:
                    for key in ['worker_guard','parent_guard']:
                        g=checked(row[key]);require(g['status']=='finished' and g['reason'] is None,'Original guard failed')
                    process=checked(row['process']);require(process['status']=='completed' and process['returncode']==0,'Original process incomplete')
                    worker=checked(row['worker']);require(row['acceptance']['status']=='accepted' and row['acceptance']['complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked'],'Original endpoint not independently accepted')
                    for n in ['parent','h200']:verify_artifact(worker[n]);weights.append(worker[n])
                    reports.append(dict(kind=kind,arm=row['arm'],worker=row['worker'],updates=worker['step'],snapshots=worker['ema_snapshots'],actual_byte_exposure=exposure(worker),worker_guard=row['worker_guard'],parent_guard=row['parent_guard']))
                root=roots[kind]
                for src in sorted(root.rglob('*')):
                    if not src.is_file() or src.is_symlink() or 'checkpoints' in src.relative_to(root).parts or src.name.endswith('.lock') or src.name=='.lock':continue
                    require(src.name=='STOP' or src.suffix in ['.json','.jsonl','.log'],'Unexpected non-payload artifact requires explicit review')
                    data=src.read_bytes();require(len(data)<=16*1024**2,'Bounded closed metadata file required');h=hashlib.sha256(data).hexdigest();dst=destination/kind/src.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True)
                    with dst.open('xb') as stream:stream.write(data)
                    require(hashlib.sha256(dst.read_bytes()).hexdigest()==h and hashlib.sha256(src.read_bytes()).hexdigest()==h,'Lossless metadata mirror differs');mirrors.append(dict(original=str(src),mirror=str(dst.relative_to(ROOT)),sha256=h,bytes=len(data)))
            require(len(weights)==8 and len({w['path'] for w in weights})==8,'Exactly eight distinct current prerequisite/pilot weight files required')
            for w in weights:verify_artifact(w)
            require(sources()==pins,'Current proven source changed')
        publish(E/'complete-metadata-mirror001.json',dict(status='complete_original_metadata_mirrored_losslessly',files=mirrors,file_count=len(mirrors),weight_copies=0,original_paths_and_bytes_unchanged=True))
        result=dict(status='actual_native_and8K_workers_pilots_and_byte_exposure_closed',source=reference(__file__),review_refs=refs,sources=pins,total_source_bindings=993,reports=reports,protected_current_payloads=weights,protected_weight_bytes=sum(w['size'] for w in weights),complete_metadata_mirror=reference(E/'complete-metadata-mirror001.json'),
            resource_projections=reviews['pilot']['projections'],environment=env,free_bytes=shutil.disk_usage(ROOT).free,numerical_worker_active=False,training=False,model_rescoring=False,holdout_encoded_or_scored=False,new_scientific_seed=False,weight_deletion_or_copies=0,
            next_action='Use actual measured cost/exposure plus existing evidence/papers to finalize finite native/8K common-byte quality design,independent statistics and all-output budget,then numerical source contracts/seed census before a NEW scientific freeze. No quality selection/adoption from prerequisite scores;do not rerun closed commands or retire tensors without a new exact downstream closure audit.')
        publish(E/'prerequisite-close001.json',result);print(json.dumps(dict(status=result['status'],protected_weight_bytes=result['protected_weight_bytes'],metadata_files=len(mirrors),exposures=[dict(kind=r['kind'],arm=r['arm'],**r['actual_byte_exposure']) for r in reports])),flush=True)
    except Exception as exc:publish(E/'STOP-prerequisite-close001',dict(error=repr(exc),closed=True,no_retry=True));raise
