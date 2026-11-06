"""Prepared finite two-arm real-worker prerequisite. NEVER invoke below its wake budget."""
import gc,json,shutil
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference,environment
from research.embedding_input_dropout_resume_v1.contract import config
from research.embedding_input_dropout_worker_v1.execution import sources,checked
from research.embedding_input_dropout_worker_v1.dispatch import launch
from research.final_ln_scale_quality_v3.support import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
E=Path(__file__).resolve().parent;RUN=ROOT/'research_runs/tokenizer-embedding-input-dropout-worker-v1-001';PLAN=E/'worker001-plan.json'
PAIR_CAP=587202560;FLOOR=11050479616;ENTRY=1458176;WAKE=FLOOR+ENTRY;TARGET=FLOOR+2*1024**3

def publish(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def main():
    torch.set_num_threads(4)
    with session_lock(GLOBAL_LOCK.parent) as g:
        idle();environment();require(not RUN.exists() and not PLAN.exists(),'New unattempted prerequisite only')
        # No root,freeze,request/config/worker/seed consumption below full pair budget.
        require(shutil.disk_usage(ROOT).free>=WAKE,'Complete worker-pair budget plus measured entry margin required')
        pins=sources();prereq={}
        for name in ('correctness001-cpu.json','correctness001-mps.json','resume001.json','resume-metadata001.json','worker-source-review001.json'):
            ref=reference(E/name);row=checked(ref);require(row['status'].startswith('passed'),'Passed actual prerequisite required')
            require(all(pins[k]==v for k,v in row['sources'].items()),'Exact predecessor plus declared additive worker sources required');prereq[name]=ref
        require(checked(prereq['worker-source-review001.json'])['prepared_builder']==reference(__file__),'Prepared builder changed')
        specs=[]
        for slot,prob in enumerate((0.,.1)):
            cfg=config(prob,steps=4,seed=600406,lr=.00125);cfg.training.eval_seed=600407;directory=RUN/f'evidence/pilots/{slot:02d}'
            cfg.training.checkpoint_dir=str(directory/'checkpoints');cfg.training.metrics_file=str(directory/'metrics.jsonl');cfg.training.summary_file=str(directory/'worker.json')
            specs.append(dict(slot=slot,stage='pilots',arm='input_zero' if prob==0. else 'input_feature_p10',config=plain(cfg),weight_cap=PAIR_CAP))
        plan=dict(kind='embedding_input_dropout_four_update_worker_pair_v1',root=str(RUN),runner=reference(__file__),sources=pins,environment=environment(),specs=specs,seconds=600,compare_cpu=True,
            required_free=FLOOR,wake_free=WAKE,cleanup_target=TARGET,prerequisites=prereq,scientific_quality=False,fixture_seeds=[600406,600407],
            budget='Two original incumbent560MiB RAW/H200 caps +16MiBmetadata +8GiBreserve +1267904512B measured external headroom +1458176B measured entry headroom. Atomic streaming .tmp counts within per-slot cap;no second disk copy. Actual accepted weights only credited once. Additional1458176B launch margin protects first staging check.',
            no_pilot_or_future_retirement_credit=True)
        RUN.mkdir();publish(PLAN,plan)
        try:
            with session_lock(RUN) as r:
                writer=MetadataWriter(RUN,(g,r));used=0;rows=[]
                for spec in specs:
                    required=max(8*1024**3,FLOOR-used)
                    require(shutil.disk_usage(ROOT).free>=required+ENTRY,'Complete live remaining pair budget and entry margin required')
                    row=launch(writer,reference(PLAN),spec,required);worker=checked(row['worker']);used+=sum(worker[n]['size'] for n in ('parent','h200'))
                    publish(E/f'worker001-accepted-{spec["slot"]:02d}.json',dict(accepted=reference(RUN/f'evidence/pilots/{spec["slot"]:02d}/accepted.json'),scientific_quality=False));rows.append(row)
                require(sources()==pins,'Frozen worker sources changed')
                result=dict(status='two_generated_workers_independently_accepted',plan=reference(PLAN),rows=rows,retained_weight_bytes=used,weight_retirement_permitted=False,
                    quality_result=False,scientific_seed=False,next_action='Audit both actual accepted artifacts/inference and finite20update pilot integration;retain generated payloads until exact downstream closure. No quality freeze yet.')
                publish(E/'worker001-review.json',result);publish(RUN/'STOP',dict(status='prerequisite_pair_complete',all_training_commands_closed=True));print(json.dumps(dict(status=result['status'],retained_weight_bytes=used)))
        except Exception as exc:
            if not(RUN/'STOP').exists():publish(RUN/'STOP',dict(error=repr(exc),closed=True,no_retry=True))
            raise
if __name__=='__main__':main()
