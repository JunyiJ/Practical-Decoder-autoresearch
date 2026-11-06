"""Completed-body metadata review and exact train-only context-availability census."""
import bisect,hashlib,json,math,re,statistics
from pathlib import Path
import numpy as np
import tiktoken
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.body_allocation_worker_v1.execution import checked
from research.final_ln_scale_model_v1.runtime.models.output_bias import digest
from research.final_ln_scale_quality_v3.support import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
B=Path(__file__).resolve().parent;E=ROOT/'research/tokenizer-evidence/body-allocation-preparation-001';OUT=B/'analysis001.json'
with session_lock(GLOBAL_LOCK.parent):
    idle();require(not OUT.exists(),'Closed diagnostic cannot rerun')
    with CpuPrerequisiteGuard(B/'analysis001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
        closure=checked(reference(E/'quality001-closure001.json'));require(closure['status']=='CLOSED_NOT_ADOPTED','Complete original panel required')
        curves=[];data=None
        for row in closure['trajectories']:
            if row['slot']<9:continue
            worker=checked(row['worker']);values=worker['metrics'];require([v['step'] for v in values]==[1,301,601,901,1200],'Exact saved diagnostics required')
            require(all(math.isfinite(v['validation_ce']) for v in values),'Finite curves required')
            if data is None:data=worker['data']
            require(worker['data']==data,'Data identity differs across fresh endpoints')
            curves.append(dict(arm=row['arm'],seed=row['seed'],worker=row['worker'],steps=[v['step'] for v in values],
                sampled_validation_ce=[v['validation_ce'] for v in values],sampled_train_ce=[v['train_ce'] for v in values],
                last299_validation_change=values[-1]['validation_ce']-values[-2]['validation_ce'],
                last599_validation_change=values[-1]['validation_ce']-values[-3]['validation_ce'],
                every_saved_validation_point_improves=all(b['validation_ce']<a['validation_ce'] for a,b in zip(values,values[1:])),
                clipped_updates=row['clipped_updates'],training_seconds=row['training_seconds']))
        grouped={}
        for arm in ('incumbent','shallow','deeper'):
            rows=[v for v in curves if v['arm']==arm];require(len(rows)==3,'Three fresh seeds required')
            grouped[arm]=dict(mean_last299_validation_change=statistics.mean(v['last299_validation_change'] for v in rows),
                mean_last599_validation_change=statistics.mean(v['last599_validation_change'] for v in rows),
                curves_monotone_at_all_saved_points=sum(v['every_saved_validation_point_improves'] for v in rows),
                clipped_updates=[v['clipped_updates'] for v in rows],
                interpretation='Sampled online diagnostics, not checkpoint selectors or proof of an optimal duration; train/validation batches differ.')
        path=Path(data['path']);require(hashlib.sha256(path.read_bytes()).hexdigest()==data['corpus_sha256'],'Original corpus differs')
        text=path.read_text();train=text[:data['raw_train_end']];del text
        require(len(train)==892315,'Original train-only boundary required')
        enc=tiktoken.get_encoding('gpt2');ids=enc.encode(train,disallowed_special=());require(len(ids)==267688 and digest(ids)==data['train_ids_sha256'],'Original train ID stream differs')
        decoded,offsets=enc.decode_with_offsets(ids);require(decoded==train and len(offsets)==len(ids),'Exact character/token offset mapping required')
        labels=list(re.finditer(r"(?m)^[A-Za-z][A-Za-z .',-]{0,48}:[ \t]*$",train))
        # Complete marker must precede the predicted token; token containing its
        # first character must remain in context. This is availability, not use.
        events=[]
        for m in labels:
            start=max(0,bisect.bisect_right(offsets,m.start())-1);end=bisect.bisect_left(offsets,m.end())
            if end<len(ids):events.append((end,start))
        latest=np.full(len(ids),-1,dtype=np.int64)
        for i,(end,start) in enumerate(events):
            stop=events[i+1][0] if i+1<len(events) else len(ids);latest[end:stop]=start
        count=len(ids)-512;starts=np.arange(count,dtype=np.int64);retained128=retained256=rescued=0
        for offset in range(512):
            marker=latest[starts+offset+1];a=marker>=starts+(offset//128)*128;b=marker>=starts+(offset//256)*256
            retained128+=int(np.count_nonzero(a));retained256+=int(np.count_nonzero(b));rescued+=int(np.count_nonzero(b & ~a))
        total=count*512
        context=dict(train_tokens=len(ids),unique_train_token_ids=len(set(ids)),vocabulary=50257,complete_colon_label_markers=len(events),
            exact_uniform512_start_positions=count,target_start_offset_combinations=total,
            marker_available_context128=retained128/total,marker_available_context256=retained256/total,
            marker_lost_at128_retained_at256=rescued/total,
            counted_inputs='All valid starts of the ORIGINAL512-token sampler and all512 target offsets, deterministic exhaustive marginal census; both independent block draws have the same marginal.',
            counterfactual='Repartition the same512-token input/target stream into4x128 versus2x256 chunks. This measures label availability only, not a trained256 model or validation gain.',
            limitations=['Regex labels are a syntactic proxy, not semantic speaker annotation.','More visible context is not proof the model uses it.','Random-window target exposure is not literal whole-corpus epochs.','No val/test tokenization or holdout scoring.'])
        historical={n:checked(reference(ROOT/n)) for n in ['research/tokenizer-evidence/horizon-001/final-review.json','research/tokenizer-evidence/table-001/final-review.json']}
        result=dict(status='completed_existing_data_diagnostic',source=reference(__file__),closure=reference(E/'quality001-closure001.json'),
            curves=curves,curve_summary=grouped,train_context_census=context,
            sampled_target_exposure=dict(targets_per_trajectory=1228800,train_tokens=267688,ratio=1228800/267688,not_literal_epochs=True),
            contrary_evidence=dict(old_gqa_horizon=reference(ROOT/'research/tokenizer-evidence/horizon-001/final-review.json'),
                old_gqa_mean_delta1200_minus3000=historical['research/tokenizer-evidence/horizon-001/final-review.json']['mean_delta'],
                old_scaled_table_sharing=reference(ROOT/'research/tokenizer-evidence/table-001/final-review.json'),
                old_scaled_table_sharing_decision=historical['research/tokenizer-evidence/table-001/final-review.json']['decision']),
            no_model_construction=True,no_checkpoint_tensor_loading=True,no_training=True,no_new_scientific_seed=True,no_new_panel_admitted=True,holdout_scored=False,
            next_question='Rank a controlled context-training/conditioning test against horizon/optimization and inference-only MLA absorption, using primary literature, original negative results and full feasibility; do not infer novelty from vocabulary-table size alone.')
        with OUT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(status=result['status'],curve_summary=grouped,train_context_census=context)))
