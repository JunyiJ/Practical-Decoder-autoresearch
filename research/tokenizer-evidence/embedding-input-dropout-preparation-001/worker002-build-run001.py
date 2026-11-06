"""Separate execution: import the already trained control;launch only unstarted candidate."""
import copy,json,shutil
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,canonical
from research.body_allocation_model_v1.evidence import reference,environment
from research.embedding_input_dropout_worker_v2.execution import sources,checked
from research.embedding_input_dropout_worker_v2.validation import verify_artifact
from research.embedding_input_dropout_worker_v2.dispatch import launch
from research.final_ln_scale_quality_v3.support import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
E=Path(__file__).resolve().parent;RUN=ROOT/'research_runs/tokenizer-embedding-input-dropout-worker-v1-002';PLAN=E/'worker002-plan.json';FLOOR=11050479616;ENTRY=1458176

def publish(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def main():
    torch.set_num_threads(4)
    with session_lock(GLOBAL_LOCK.parent) as g:
        idle();environment();require(not RUN.exists() and not PLAN.exists(),'New unattempted execution context required')
        proofref=reference(E/'worker002-import001.json');proof=checked(proofref);pins=sources()
        require(proof['status']=='passed_cross_root_generated_endpoint_reaudit_and_scoring_adapter' and proof['sources']==pins,'Exact parent-only adapter and source bridge required')
        ready=checked(reference(E/'worker002-ready001.json'));require(ready['sources']==pins and ready['prepared_builder']==reference(__file__),'Prepared runner/source review changed')
        row0=proof['imported_endpoint'];w0=checked(row0['worker']);oldplan=checked(proof['original_plan']);oldstop=checked(proof['original_failed_stop'])
        require('AttributeError' in oldstop['error'] and not(ROOT/'research_runs/tokenizer-embedding-input-dropout-worker-v1-001/evidence/pilots/01').exists(),'Original stopped execution scope changed')
        refs=[w0[n] for n in ('parent','h200')];require(len({v['path'] for v in refs})==2,'Duplicate imported weight credit')
        for v in refs:verify_artifact(v)
        used=sum(v['size'] for v in refs);required=max(8*1024**3,FLOOR-used)
        require(shutil.disk_usage(ROOT).free>=required+ENTRY,'Complete actual remaining pair budget plus measured entry margin required')
        spec1=copy.deepcopy(oldplan['specs'][1]);directory=RUN/'evidence/pilots/01';t=spec1['config']['training']
        t.update(checkpoint_dir=str(directory/'checkpoints'),metrics_file=str(directory/'metrics.jsonl'),summary_file=str(directory/'worker.json'))
        def numerical(cfg):
            d=copy.deepcopy(cfg)
            for n in ('checkpoint_dir','metrics_file','summary_file','resume_from'):d['training'].pop(n,None)
            return d
        require(numerical(spec1['config'])==numerical(oldplan['specs'][1]['config']) and spec1['slot']==1 and spec1['weight_cap']==587202560,'Only unstarted candidate paths may differ')
        plan=dict(kind='embedding_input_dropout_same_prerequisite_pair_continuation_v2',root=str(RUN),runner=reference(__file__),sources=pins,environment=environment(),specs=[oldplan['specs'][0],spec1],seconds=600,compare_cpu=True,
            required_original_pair_free=FLOOR,remaining_required_free=required,remaining_entry_wake=required+ENTRY,cleanup_target=required+2*1024**3,
            original_plan=proof['original_plan'],original_failed_stop=proof['original_failed_stop'],imported_control=proofref,imported_actual_weight_bytes=used,
            numerical_training_source=reference(ROOT/'research/embedding_input_dropout_worker_v1/training.py'),new_training_updates_for_import=0,weight_copies=0,scientific_quality=False,
            fixture_seeds=[600406,600407],no_future_retirement_credit=True)
        RUN.mkdir();publish(PLAN,plan)
        try:
            with session_lock(RUN) as r:
                writer=MetadataWriter(RUN,(g,r));writer.publish_json('evidence/pilots/00/accepted.json',row0)
                writer.publish_json('control/imported-00.json',dict(proof=proofref,worker=row0['worker'],original_stop=proof['original_failed_stop'],zero_training_updates=True,weight_copies=0))
                row1=launch(writer,reference(PLAN),spec1,required);w1=checked(row1['worker']);used+=sum(w1[n]['size'] for n in ('parent','h200'))
                publish(E/'worker002-accepted-01.json',dict(accepted=reference(RUN/'evidence/pilots/01/accepted.json'),scientific_quality=False))
                require(sources()==pins and reference(proof['original_failed_stop']['path'])==proof['original_failed_stop'],'Source or original STOP changed')
                publish(E/'worker002-review.json',dict(status='two_generated_workers_independently_accepted',plan=reference(PLAN),rows=[row0,row1],retained_weight_bytes=used,imported_zero_updates=True,weight_copies=0,
                    scientific_quality=False,new_scientific_seed=False,next_action='Re-audit actual accepted endpoints,close finite20-update pilot integration,then NEW exact generated-payload retirement audit;all selected/fresh/original evidence protected.'))
                publish(RUN/'STOP',dict(status='prerequisite_pair_complete',all_training_commands_closed=True));print(json.dumps(dict(status='two_generated_workers_independently_accepted',retained_weight_bytes=used)))
        except Exception as exc:
            if not(RUN/'STOP').exists():publish(RUN/'STOP',dict(error=repr(exc),closed=True,no_retry=True))
            raise
if __name__=='__main__':main()
