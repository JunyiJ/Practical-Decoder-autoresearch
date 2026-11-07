"""Distinct primitive localization after closed transport failure;no optimizer APIs."""
import copy,json,math,os,shutil,struct
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_moe_resume_v1.contract import runtime as environment
from research.tokenizer_moe_worker_v1.execution import idle
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
E=Path(__file__).resolve().parent;R=ROOT/'research_runs/tokenizer-optimizer-precision-localization-v1-001';META=2*1024**2

def checked(r):
    p=Path(r['path']);require(reference(p)=={k:r[k] for k in ('path','sha256')},'Immutable reference changed');return json.loads(p.read_text())
def serialized(v,used):
    require(type(used) is int and used>=0,'Nonnegative exact output usage required');raw=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)+used<=META,'Complete2MiB metadata cap required');return raw
def publish(p,v):
    raw=serialized(v,sum(p.stat().st_size for p in R.glob('*') if p.is_file()))
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def f32(v):return struct.unpack('<f',struct.pack('<f',float(v)))[0]
def bf(v):
    n=struct.unpack('<I',struct.pack('<f',float(v)))[0];n=(n+0x7fff+((n>>16)&1))&0xffff0000;return struct.unpack('<f',struct.pack('<I',n))[0]
def snap(t):
    value=t.detach().cpu().float();require(bool(torch.isfinite(value).all()),'Finite complete primitive output required');return value.tolist()
def diff(a,b):return float((a.detach().cpu().float()-b.detach().cpu().float()).abs().max())
def models(bias,left,right,alpha,beta,dtype):
    q=bf if dtype==torch.bfloat16 else f32;B=snap(bias);L=snap(left);Q=snap(right);out={k:[] for k in ('fused_float_coefficients','rounded_product_float_coefficients','staged_float_coefficients','staged_input_dtype_coefficients')}
    for i,row in enumerate(L):
        rows={k:[] for k in out}
        for j in range(len(Q[0])):
            dot=math.fsum(row[k]*Q[k][j] for k in range(len(row)));mm=q(dot);z=B[i][j]
            rows['fused_float_coefficients'].append(q(f32(alpha)*dot+f32(beta)*z))
            rows['rounded_product_float_coefficients'].append(q(f32(alpha)*mm+f32(beta)*z))
            rows['staged_float_coefficients'].append(q(q(f32(alpha)*mm)+q(f32(beta)*z)))
            rows['staged_input_dtype_coefficients'].append(q(q(q(alpha)*mm)+q(q(beta)*z)))
        for k in out:out[k].append(rows[k])
    return out
def trace(matrix,device):
    x=matrix.to(device=device,dtype=torch.bfloat16,copy=True);out=[]
    def save(name,t):out.append(dict(stage=name,value=snap(t)))
    save('cast',x);norm=x.norm().clamp(min=1e-7);save('norm',norm);x=x/norm;save('normalized',x)
    for i in range(5):
        a=x@x.T;save(f'iteration{i+1}/gram',a)
        b=torch.addmm(a,a,a,beta=-4.775,alpha=2.0315);save(f'iteration{i+1}/polynomial',b)
        x=torch.addmm(x,b,x,beta=3.4445);save(f'iteration{i+1}/matrix',x)
    return out

with session_lock(GLOBAL_LOCK.parent):
    idle();torch.set_num_threads(4);env=environment();require(not R.exists(),'Fresh distinct primitive localization only;no retry')
    design_ref=reference(E/'optimizer-precision-design001.json');design=checked(design_ref);failure=checked(design['failure']);stop=checked(failure['terminal']);failed_plan=checked(failure['source_plan']);failed_guard=checked(failure['guard'])
    require(failure['status']=='CLOSED_FAILED' and stop['status']=='closed_failed' and stop['no_retry'] and 'differs on mps' in stop['error'] and not(Path(failure['root'])/'result.json').exists(),'Original transport failure must remain closed and unaccepted')
    require(failed_guard['status']=='fixture_exception' and failure['tolerances_unchanged'],'Preserved actual numerical failure required')
    pins=dict(failed_plan['sources']);require(len(pins)==1053,'All1053 current bindings preserved')
    for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Immutable predecessor changed '+p)
    pins[str(Path(__file__).resolve().relative_to(ROOT))]=reference(Path(__file__))['sha256'];require(len(pins)==1054,'Only one new primitive localization script required')
    for r in [*failed_plan['libraries'],failed_plan['actual_inherited_decay_source']]:require(reference(r['path'])==r,'Installed runtime source changed')
    cost=design['cost'];free=shutil.disk_usage(ROOT).free
    require(free>=cost['target_bytes'] and cost['conservative_rss_bytes']<cost['rss_limit_bytes'] and cost['conservative_mps_bytes']<cost['mps_limit_bytes'] and cost['conservative_seconds']<cost['timeout_seconds'],'Complete LIVE storage and resource forecast required')
    require(torch.version.git_version==design['primary_source']['git_version'],'Exact retrieved primary implementation revision required')
    R.mkdir();publish(R/'plan.json',dict(kind='distinct_generated_primitive_precision_localization_v1',sources=pins,design=design_ref,failure=design['failure'],original_failed_terminal=failure['terminal'],environment=env,torch_git_version=torch.version.git_version,backend_flags={k:os.environ.get(k) for k in ('PYTORCH_MPS_PREFER_METAL','PYTORCH_MPS_FAST_MATH','PYTORCH_ENABLE_MPS_FALLBACK')},shapes=[[2,4],[3,4]],input_formula='x[i]=((i*7)%19-9)/17;bias[i]=((i*5)%13-6)/23',dtypes=['float32','bfloat16'],coefficient_pairs=[[2.0315,-4.775],[1.,3.4445]],precision_models=['fused_float_coefficients','rounded_product_float_coefficients','staged_float_coefficients','staged_input_dtype_coefficients'],optimizer_api_calls=0,original_proof_or_cases_replayed=False,agreement_gate=False,full_floor=cost['full_floor_bytes'],target=cost['target_bytes'],free_bytes=free,cost=cost,metadata_cap=META))
    try:
        with StrictMpsGuard(R/'guard.json',ROOT):
            mps_rng=torch.mps.get_rng_state().clone();cpu_rng=torch.get_rng_state().clone();rows=[];traces=[]
            for shape in ((2,4),(3,4)):
                original=torch.tensor([((i*7)%19-9)/17 for i in range(math.prod(shape))],dtype=torch.float32).reshape(shape);unchanged=original.clone();n=shape[0]
                for dtype in (torch.float32,torch.bfloat16):
                    left=original.to(dtype);right=left.T;bias=torch.tensor([((i*5)%13-6)/23 for i in range(n*n)],dtype=torch.float32).reshape(n,n).to(dtype)
                    device_results={};row=dict(shape=shape,dtype=str(dtype),left=snap(left),right=snap(right),bias=snap(bias),devices={})
                    for device in ('cpu','mps'):
                        x=left.to(device=device,copy=True);y=right.to(device=device,copy=True);b=bias.to(device=device,copy=True);z=x@y;normal=x.norm();normalized=x/normal.clamp(min=1e-7);adds=[]
                        for alpha,beta in ((2.0315,-4.775),(1.,3.4445)):
                            value=torch.addmm(b,x,y,alpha=alpha,beta=beta);candidates=models(bias,left,right,alpha,beta,dtype);v=snap(value)
                            errors={k:float((torch.tensor(v)-torch.tensor(candidate)).abs().max()) for k,candidate in candidates.items()}
                            adds.append(dict(alpha=alpha,beta=beta,value=v,scalar_candidates=candidates,candidate_max_abs_error=errors))
                        data=dict(matmul=snap(z),norm=snap(normal),normalized=snap(normalized),addmm=adds);row['devices'][device]=data;device_results[device]=data
                    def host(v):return torch.tensor(v,dtype=torch.float32)
                    a=device_results['cpu'];b=device_results['mps'];row['cpu_mps_max_abs']=dict(matmul=diff(host(a['matmul']),host(b['matmul'])),norm=diff(host(a['norm']),host(b['norm'])),normalized=diff(host(a['normalized']),host(b['normalized'])),addmm=[diff(host(v['value']),host(w['value'])) for v,w in zip(a['addmm'],b['addmm'])]);rows.append(row)
                a=trace(original,'cpu');b=trace(original,'mps');require(len(a)==len(b)==18 and [s['stage'] for s in a]==[s['stage'] for s in b],'Complete distinct matrix-stage trace required')
                stages=[dict(stage=u['stage'],cpu=u['value'],mps=v['value'],max_abs=diff(torch.tensor(u['value']),torch.tensor(v['value']))) for u,v in zip(a,b)]
                traces.append(dict(shape=shape,stages=stages,first_exact_difference=next((s['stage'] for s in stages if s['max_abs']>0),None),maximum_stage_difference=max(s['max_abs'] for s in stages)))
                require(torch.equal(original,unchanged),'Generated input mutated')
            vectors={}
            for device in ('cpu','mps'):
                g=torch.tensor([(i*3-7)/29 for i in range(5)],dtype=torch.float32,device=device);m=torch.tensor([(i-2)/37 for i in range(5)],dtype=torch.float32,device=device);v=torch.tensor([(i+1)/41 for i in range(5)],dtype=torch.float32,device=device)
                first=m.lerp(g,.1);second=v.clone().mul_(.999).addcmul_(g,g,value=.001);quotient=first/(second.sqrt()+1e-6)
                vectors[device]=dict(first=snap(first),second=snap(second),quotient=snap(quotient),norm=snap(g.norm()))
            vector_errors={k:diff(torch.tensor(vectors['cpu'][k]),torch.tensor(vectors['mps'][k])) for k in vectors['cpu']}
            rejected=0
            for value,used in [({},META),({'x':float('nan')},0),({},-1)]:
                try:serialized(value,used)
                except ValueError:rejected+=1
                else:raise AssertionError('Invalid primitive metadata budget accepted')
            torch.mps.synchronize();require(torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(mps_rng,torch.mps.get_rng_state()),'Global RNG changed')
            for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Source changed during primitive probe')
            require(reference(Path(failure['terminal']['path']))==failure['terminal'],'Closed failed transport STOP changed')
        result=dict(status='complete_distinct_primitive_precision_localization',source_plan=reference(R/'plan.json'),source_bindings=1054,rows=rows,traces=traces,vector_control=vectors,vector_cpu_mps_max_abs=vector_errors,output_budget_rejections=rejected,global_cpu_mps_rng_unchanged=True,inputs_unchanged=True,optimizer_api_calls=0,old_optimizer_proof_or_cases_replayed=False,actual_models_or_raw_loaded=0,corpus_loaded=False,weight_files_written=0,validation_encoded=False,holdout_encoded=False,original_transport_still_closed_failed=True,transport_acceptance=False,tolerances_changed=False,quality_result=False,adopted=False,scope='Distinct primitive arithmetic observations only. Candidate error comparisons are descriptive,not a rescued optimizer transport acceptance.')
        publish(R/'result.json',result);publish(R/'STOP',dict(status='complete_distinct_primitive_precision_localization',result=reference(R/'result.json'),never_rerun=True));print(json.dumps(dict(status=result['status'],rows=[dict(shape=r['shape'],dtype=r['dtype'],cpu_mps=r['cpu_mps_max_abs'],candidate_errors={d:[a['candidate_max_abs_error'] for a in r['devices'][d]['addmm']] for d in ('cpu','mps')}) for r in rows],traces=[{k:v for k,v in t.items() if k!='stages'} for t in traces],vector_errors=vector_errors)),flush=True)
    except BaseException as exc:
        if not(R/'STOP').exists():publish(R/'STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
        raise
