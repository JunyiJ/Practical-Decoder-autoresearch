"""Independent closed-metadata arithmetic/failure review;no backend/optimizer replay."""
import itertools,json,shutil
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.tokenizer_moe_worker_v1.execution import idle
E=Path(__file__).resolve().parent;R=ROOT/'research_runs/tokenizer-optimizer-precision-localization-v1-001'
def checked(r):
    p=Path(r['path']);require(reference(p)=={k:r[k] for k in ('path','sha256')},'Immutable evidence changed');return json.loads(p.read_text())
def flat(x):
    if isinstance(x,list):return [a for y in x for a in flat(y)]
    return [x]
def error(a,b):
    x=flat(a);y=flat(b);require(len(x)==len(y),'Complete matching primitive output required');return max(abs(v-w) for v,w in zip(x,y))
def publish(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
with session_lock(GLOBAL_LOCK.parent):
    idle();require(not(E/'optimizer-precision-review001.json').exists() and not(E/'optimizer-precision-review001-guard.json').exists(),'Fresh terminal metadata review only')
    with CpuPrerequisiteGuard(E/'optimizer-precision-review001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
        terminal=reference(R/'STOP');stop=checked(terminal);result=checked(stop['result']);plan=checked(result['source_plan']);guard=checked(reference(R/'guard.json'));failure=checked(plan['failure']);failed_stop=checked(failure['terminal']);failed_guard=checked(failure['guard']);failed_root=Path(failure['root'])
        require(stop['status']==result['status']=='complete_distinct_primitive_precision_localization' and stop['never_rerun'] and guard['status']=='finished' and guard['reason'] is None,'Complete bounded distinct localization required')
        require(failure['status']=='CLOSED_FAILED' and failed_stop['status']=='closed_failed' and failed_guard['status']=='fixture_exception' and not(failed_root/'result.json').exists(),'Original transport must remain failed and unaccepted')
        require(result['transport_acceptance'] is False and result['tolerances_changed'] is False and result['old_optimizer_proof_or_cases_replayed'] is False and result['optimizer_api_calls']==0 and result['actual_models_or_raw_loaded']==0 and result['weight_files_written']==0,'No transport rescue/replay/model work allowed')
        require(len(plan['sources'])==result['source_bindings']==1054 and result['global_cpu_mps_rng_unchanged'] and result['inputs_unchanged'],'Complete sources and unchanged caller/RNG required')
        wanted=set(itertools.product([(2,4),(3,4)],['torch.float32','torch.bfloat16']));require({(tuple(r['shape']),r['dtype']) for r in result['rows']}==wanted and len(result['rows'])==4,'Complete distinct primitive Cartesian product required')
        summaries=[];max_review_error=0.
        for row in result['rows']:
            a=row['devices']['cpu'];b=row['devices']['mps'];expected={k:error(a[k],b[k]) for k in ('matmul','norm','normalized')};expected['addmm']=[error(x['value'],y['value']) for x,y in zip(a['addmm'],b['addmm'])]
            require(expected==row['cpu_mps_max_abs'],'Independent CPU/MPS primitive difference differs')
            for device in ('cpu','mps'):
                for addition in row['devices'][device]['addmm']:
                    require(len(addition['scalar_candidates'])==4,'All four prospective scalar models required')
                    for name,values in addition['scalar_candidates'].items():
                        err=abs(error(addition['value'],values)-addition['candidate_max_abs_error'][name]);max_review_error=max(max_review_error,err);require(err<1e-12,'Independent scalar-candidate difference differs')
            summaries.append(dict(shape=row['shape'],dtype=row['dtype'],cpu_mps=expected))
        trace_summary=[]
        for trace in result['traces']:
            require(len(trace['stages'])==18,'Complete 18-stage trace required')
            errors=[error(x['cpu'],x['mps']) for x in trace['stages']]
            require(errors==[x['max_abs'] for x in trace['stages']] and max(errors)==trace['maximum_stage_difference'],'Independent stage difference differs')
            first=next((x['stage'] for x,e in zip(trace['stages'],errors) if e>0),None);require(first==trace['first_exact_difference'],'Independent first divergence differs')
            trace_summary.append(dict(shape=trace['shape'],first=first,maximum=max(errors)))
        vector_errors={k:error(result['vector_control']['cpu'][k],result['vector_control']['mps'][k]) for k in result['vector_control']['cpu']};require(vector_errors==result['vector_cpu_mps_max_abs'],'Independent vector control differs')
        for p,h in plan['sources'].items():require(reference(ROOT/p)['sha256']==h,'Immutable source changed '+p)
        mirror=E/'mps-precision-metadata-mirror001';mirror.mkdir();files=[]
        for label,root,names in [('transport-failed',failed_root,['plan.json','guard.json','STOP']),('localization',R,['plan.json','guard.json','result.json','STOP'])]:
            dest=mirror/label;dest.mkdir()
            for name in names:
                p=root/name;before=reference(p);q=dest/name;shutil.copyfile(p,q);require(reference(p)==before and reference(q)['sha256']==before['sha256'],'Closed metadata mirror changed');files.append(dict(source=before,destination=reference(q),size=p.stat().st_size))
        require(reference(R/'STOP')==terminal and reference(Path(failure['terminal']['path']))==failure['terminal'],'Original terminal changed')
        publish(mirror/'completion.json',dict(files=files,total_bytes=sum(v['size'] for v in files),weight_copies=0))
    publish(E/'optimizer-precision-review001.json',dict(status='failed_transport_and_distinct_primitive_metadata_independently_reviewed',source=reference(Path(__file__)),terminal=terminal,result=stop['result'],source_plan=result['source_plan'],guard=reference(R/'guard.json'),failure=plan['failure'],metadata_mirror=reference(mirror/'completion.json'),primitive_summary=summaries,trace_summary=trace_summary,vector_errors=vector_errors,max_review_error=max_review_error,source_bindings=1054,backend_calls=0,optimizer_steps=0,models_loaded=0,original_transport_still_closed_failed=True,transport_acceptance=False,quality_result=False,adopted=False))
    print(json.dumps(dict(status='reviewed',source_bindings=1054,max_review_error=max_review_error,trace_summary=trace_summary)),flush=True)
