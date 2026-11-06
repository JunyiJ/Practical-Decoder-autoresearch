"""New finite generated-worker pair; no quality training, imports, retry or deletion."""
import json,shutil,gc
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference,environment
from research.gpt2_prefix_data_v1.data import config
from research.gpt2_prefix_worker_v1.execution import sources,checked
from research.gpt2_prefix_worker_v1.validation import verify_artifact
from research.gpt2_prefix_worker_v1.dispatch import launch
from research.final_ln_scale_quality_v3.support import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
E=Path(__file__).resolve().parent;RUN=ROOT/'research_runs/tokenizer-gpt2-prefix-worker-v1-001';PLAN=E/'worker001-plan.json'
CAPS=[560*1024**2,160*1024**2];METADATA=16*1024**2;RESERVE=8*1024**3;EXTERNAL=1267904512;ENTRY=1458176;FLOOR=sum(CAPS)+METADATA+RESERVE+EXTERNAL+ENTRY

def specs():
    result=[]
    for i,kind in enumerate(['native','prefix8192']):
        cfg=config(kind,steps=4,seed=600426+i);cfg.training.eval_seed=600428;directory=RUN/f'evidence/pilots/{i:02d}'
        cfg.training.checkpoint_dir=str(directory/'checkpoints');cfg.training.metrics_file=str(directory/'metrics.jsonl');cfg.training.summary_file=str(directory/'worker.json')
        result.append(dict(stage='pilots',slot=i,arm=kind,config=plain(cfg),weight_cap=CAPS[i]))
    return result

def publish(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def main():
    torch.set_num_threads(4)
    with session_lock(GLOBAL_LOCK.parent) as g:
        idle();env=environment();require(not RUN.exists() and not PLAN.exists(),'Unattempted new generated pair only')
        pins=sources();proof=checked(reference(E/'worker-source-review001.json'))
        require(proof['status']=='passed_genuine_worker_source_and_common_byte_contracts' and proof['sources']==pins and proof['prepared_builder']==reference(__file__),'Actual source/builder gate changed')
        refs={n:reference(E/n) for n in ['component002.json','data001.json','resume001.json']}
        require(refs==proof['prerequisites'],'Actual prerequisite references changed')
        for n in ['component002','data001','resume001']:require(checked(reference(E/(n+'-guard.json')))['status']=='finished','Original prerequisite guard incomplete')
        require(shutil.disk_usage(ROOT).free>=FLOOR+ENTRY,'Complete finite pair/serialization/metadata/reserve/headroom plus measured entry margin required')
        plan=dict(kind='gpt2_prefix_generated_prerequisite_pair_v1',root=str(RUN),runner=reference(__file__),sources=pins,environment=env,specs=specs(),seconds=600,compare_cpu=True,
            prerequisites=refs,source_review=reference(E/'worker-source-review001.json'),scientific_quality=False,new_scientific_seed=False,fixture_seeds=[600426,600427],fixture_evaluation_seed=600428,
            full_pair_floor=FLOOR,first_wake=FLOOR+ENTRY,cleanup_target=FLOOR+2*1024**3,weight_caps=CAPS,metadata_cap=METADATA,reserve=RESERVE,external_headroom=EXTERNAL,entry_headroom=ENTRY,
            capacity_policy='Subtract only actual independently accepted retained weight bytes once before the next unstarted slot. No conditional retirement/pilot/hardlink credit. Atomic streaming temporary is within cap,not duplicate disk copy.',
            scoring='RAW and H200 each on CPU/MPS over identical111475rawbytes,codec-specific token targets. New baseline,not old nativeCE. H200cache/resetB1/8 on both devices.',no_retirement=True)
        RUN.mkdir();publish(PLAN,plan)
        try:
            with session_lock(RUN) as r:
                writer=MetadataWriter(RUN,(g,r));rows=[];used=0
                for spec in plan['specs']:
                    require(shutil.disk_usage(ROOT).free>=max(RESERVE,FLOOR-used)+ENTRY,'Complete live remaining capacity required')
                    row=launch(writer,reference(PLAN),spec,max(RESERVE,FLOOR-used));rows.append(row);worker=checked(row['worker'])
                    for n in ['parent','h200']:verify_artifact(worker[n])
                    used+=sum(worker[n]['size'] for n in ['parent','h200']);publish(E/f'worker001-accepted-{spec["slot"]:02d}.json',dict(accepted=reference(RUN/f'evidence/pilots/{spec["slot"]:02d}/accepted.json'),scientific_quality=False))
                    gc.collect();torch.mps.empty_cache()
                require(sources()==pins,'Worker sources changed')
                review=dict(status='two_gpt2_prefix_generated_workers_independently_accepted',plan=reference(PLAN),rows=rows,retained_weight_bytes=used,scientific_quality=False,new_scientific_seed=False,holdout_encoded_or_scored=False,
                    next_action='Actual worker dependencies/common-byte scoring acceptance permit finite20update resource pilot integration;then new finite scientific design/common-byte inference/statistics and complete retained-output budget. Do not retire generated files without new exact closure audit.')
                publish(E/'worker001-review.json',review);publish(RUN/'STOP',dict(status='prerequisite_pair_complete',all_training_commands_closed=True,never_rerun=True));print(json.dumps(dict(status=review['status'],retained_weight_bytes=used)),flush=True)
        except Exception as exc:
            if not(RUN/'STOP').exists():publish(RUN/'STOP',dict(error=repr(exc),closed=True,no_retry=True))
            raise
if __name__=='__main__':main()
