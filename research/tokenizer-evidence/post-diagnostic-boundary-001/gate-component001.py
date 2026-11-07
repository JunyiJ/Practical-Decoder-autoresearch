"""Independent scalar output/gradient oracle and native-shape CPU/MPS gate proof."""
import copy,json,math,os,shutil
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_moe_resume_v1.contract import runtime as environment
from research.tokenizer_moe_worker_v1.execution import idle
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.tokenizer_mla_gate_v1.component import ElementwiseOutputGate,KINDS
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
E=Path(__file__).resolve().parent;R=E/'gate-component001';META=2*1024**2
FLOOR=8*1024**3+META+1267904512+1458176;TARGET=FLOOR+2*1024**3
TOL=dict(cpu64_atol=1e-12,cpu64_rtol=1e-11,fp32_atol=2e-6,fp32_rtol=2e-5)

def checked(r):
    p=Path(r['path']);require(reference(p)=={k:r[k] for k in ('path','sha256')},'Immutable reference differs');return json.loads(p.read_text())
def serialized(v,used):
    require(type(used) is int and used>=0,'Exact nonnegative output usage required');raw=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(used+len(raw)<=META,'Complete2MiB metadata cap required');return raw
def publish(p,v):
    raw=serialized(v,sum(v.stat().st_size for v in R.glob('*') if v.is_file()))
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def same(a,b):
    if type(a) in (torch.Tensor,torch.nn.Parameter):
        x=a.detach().cpu();y=b.detach().cpu();return x.shape==y.shape and x.dtype==y.dtype and bool(((x==y)|(torch.isnan(x)&torch.isnan(y))).all())
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if type(a) in (list,tuple):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b
def sigmoid(z):
    if z>=0:return 1/(1+math.exp(-z))
    v=math.exp(z);return v/(1+v)
def oracle(x,y,w,up):
    # Independent scalar calculus;no torch linear/matmul/sigmoid/autograd reference.
    X=x.detach().cpu().double().tolist();Y=y.detach().cpu().double().tolist();W=w.detach().cpu().double().tolist();U=up.detach().cpu().double().tolist()
    b,t,c=x.shape;h=y.shape[1];d=y.shape[3]
    out=[[[[0.]*d for _ in range(t)] for _ in range(h)] for _ in range(b)];gy=copy.deepcopy(out);gx=[[[0.]*c for _ in range(t)] for _ in range(b)];gw=[[0.]*c for _ in range(c)]
    for i in range(b):
        for j in range(t):
            for k in range(h):
                for a in range(d):
                    row=k*d+a;g=sigmoid(math.fsum(W[row][z]*X[i][j][z] for z in range(c)));out[i][k][j][a]=Y[i][k][j][a]*g;gy[i][k][j][a]=U[i][k][j][a]*g;q=U[i][k][j][a]*Y[i][k][j][a]*g*(1-g)
                    for z in range(c):gx[i][j][z]+=q*W[row][z];gw[row][z]+=q*X[i][j][z]
    return [torch.tensor(v,dtype=torch.float64) for v in (out,gx,gy,gw)]
def values(shape,offset=0,scale=1.):return ((torch.arange(math.prod(shape),dtype=torch.float64)+offset)%13-6).reshape(shape)*scale/17
def getgrad(t):return torch.zeros_like(t) if t.grad is None else t.grad

with session_lock(GLOBAL_LOCK.parent):
    idle();torch.set_num_threads(4);env=environment();require(not R.exists(),'Fresh standalone gate component proof only;never retry')
    prior_ref=reference(E.parent/'post-moe-boundary-001/continuation-completed005.json');prior=checked(prior_ref);prior_plan=checked(prior['source_plan']);close=checked(prior['branch_close']);terminal=checked(prior['primitive_terminal']);review=checked(prior['review'])
    require(prior['current_bindings']==1054 and close['status']=='gradient_and_optimizer_response_diagnostic_branch_closed_inconclusive_for_batch_selection' and terminal['status']=='complete_distinct_primitive_precision_localization' and review['status']=='failed_transport_and_distinct_primitive_metadata_independently_reviewed','Actual closed predecessor boundary required')
    direction_ref=reference(E/'direction001.json');direction=checked(direction_ref);design_ref=reference(E/'quality-design001.json');design=checked(design_ref)
    require(direction['status']=='native_mla_elementwise_gate_selected_for_prerequisites_only' and design['status']=='prospective_not_frozen_prerequisites_only' and design['finite_quality_design']['new_seed_roles'].startswith('UNASSIGNED'),'New prospective quality direction without frozen seeds/panel required')
    for ref in direction['local_evidence']:checked_path=reference(ref['path']);require(checked_path==ref,'Original local evidence changed')
    pins=dict(prior_plan['sources']);require(len(pins)==1054,'All1054 immutable predecessors required')
    for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Immutable predecessor changed '+p)
    for p in [*sorted((ROOT/'research/tokenizer_mla_gate_v1').glob('*.py')),Path(__file__).resolve()]:pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
    require(len(pins)==1057,'Exact two-file component plus new scalar proof required')
    free=shutil.disk_usage(ROOT).free;require(free>=TARGET and free>=design['complete_retained_budget']['target'],'Component and complete prospective retained-output targets must fit actual capacity');R.mkdir()
    publish(R/'plan.json',dict(kind='native_mla_elementwise_gate_generated_component_proof_v1',sources=pins,predecessor_closure=prior_ref,direction=direction_ref,prospective_quality_design=design_ref,environment=env,tolerances=TOL,small_oracle_geometry=dict(batch=2,heads=2,tokens=3,head_dim=3,width=6),weight_cases=['zero','ordinary','saturated'],backends=['CPUFP64','CPUFP32','MPSFP32'],native_shape_cases=dict(batch=[1,8],tokens=128,width=256,heads=8,head_dim=32,zero_weight=True),no_rng_draws=True,no_corpus_or_language_model=True,optimizer_steps=0,tensor_files=0,full_floor=FLOOR,target=TARGET,free_bytes=free,metadata_cap=META))
    try:
        with StrictMpsGuard(R/'guard.json',ROOT):
            mps_rng=torch.mps.get_rng_state().clone();cpu_rng=torch.get_rng_state().clone();rows=[];controls=[];native=[];max_error=0.;rejected=0
            for device,dtype in [('cpu',torch.float64),('cpu',torch.float32),('mps',torch.float32)]:
                atol=TOL['cpu64_atol' if dtype==torch.float64 else 'fp32_atol'];rtol=TOL['cpu64_rtol' if dtype==torch.float64 else 'fp32_rtol']
                for kind in ('zero','ordinary','saturated'):
                    m=ElementwiseOutputGate('sigmoid_elementwise_zero',6,2,3,device=device,dtype=dtype)
                    with torch.no_grad():m.weight.copy_((torch.zeros(6,6,dtype=torch.float64) if kind=='zero' else values((6,6),3,.7 if kind=='ordinary' else 50.)).to(device=device,dtype=dtype))
                    x=values((2,3,6),1,.8).to(device=device,dtype=dtype).requires_grad_();y=values((2,2,3,3),2,.6).to(device=device,dtype=dtype).requires_grad_();up=values(y.shape,4,.4).to(device=device,dtype=dtype);before=copy.deepcopy([x.detach(),y.detach(),m.state_dict()]);expected=oracle(x,y,m.weight,up)
                    out=m(x,y);(out*up).sum().backward();actual=[out,x.grad,y.grad,m.weight.grad];errors=[]
                    for a,b in zip(actual,expected):
                        err=float((a.detach().cpu().double()-b).abs().max());max_error=max(max_error,err);errors.append(err);require(torch.allclose(a.detach().cpu().double(),b,atol=atol,rtol=rtol),'Independent scalar output/direct-gradient oracle differs')
                    require(same([x.detach(),y.detach(),m.state_dict()],before),'Gate forward/backward changed inputs or learned values')
                    full=out.detach();parts=torch.cat([m(x[:,i:i+1].detach(),y[:,:,i:i+1].detach()) for i in range(3)],dim=2);require(torch.allclose(full,parts,atol=atol,rtol=rtol),'Gate is not query-local under token partition')
                    if kind=='zero':require(torch.equal(out,y*.5) and bool((m.weight.grad!=0).any()),'Zero gate must exactly match static-half and receive gradient')
                    # Gate precedes output projection;projection bias remains unscaled.
                    if kind=='zero':
                        z=out.transpose(1,2).reshape(2,3,6);u=y.transpose(1,2).reshape(2,3,6);bias=torch.tensor([.25,-.5],device=device,dtype=dtype);projection=values((2,6),2,.5).to(device=device,dtype=dtype);post=torch.nn.functional.linear(z,projection,bias);reference_post=torch.nn.functional.linear(u*.5,projection,bias);require(torch.equal(post,reference_post),'Fixed-half must not halve output bias')
                    rows.append(dict(device=device,dtype=str(dtype),weight_case=kind,output_x_y_w_maxerrors=errors,query_local_partition=True,inputs_and_parameter_values_unchanged=True))
                for kind in ('none','static_half'):
                    m=ElementwiseOutputGate(kind,6,2,3,device=device,dtype=dtype);x=values((2,3,6)).to(device=device,dtype=dtype).requires_grad_();y=values((2,2,3,3),2).to(device=device,dtype=dtype).requires_grad_();up=values(y.shape,1).to(device=device,dtype=dtype);out=m(x,y);(out*up).sum().backward();factor=1. if kind=='none' else .5
                    require(not list(m.parameters()) and torch.equal(out,y*factor) and torch.equal(y.grad,up*factor) and x.grad is None,'Exact parameter-free controls required')
                    if kind=='none':require(out.data_ptr()==y.data_ptr(),'Ungated path must preserve identity arithmetic')
                    controls.append(dict(device=device,dtype=str(dtype),kind=kind,exact=True))
            # Native geometry atzero-init: independent FP64 contraction for weight gradients.
            for device in ('cpu','mps'):
                for b in (1,8):
                    m=ElementwiseOutputGate('sigmoid_elementwise_zero',256,8,32,device=device);x=values((b,128,256),1,.1).float().to(device).requires_grad_();y=values((b,8,128,32),2,.1).float().to(device).requires_grad_();up=values(y.shape,3,.1).float().to(device);out=m(x,y);(out*up).sum().backward()
                    expected=.25*(y.detach().cpu().double()*up.detach().cpu().double()).transpose(1,2).reshape(-1,256).T@x.detach().cpu().double().reshape(-1,256);err=float((m.weight.grad.detach().cpu().double()-expected).abs().max());max_error=max(max_error,err)
                    require(torch.equal(out,y*.5) and torch.equal(y.grad,up*.5) and bool((x.grad==0).all()) and torch.allclose(m.weight.grad.cpu().double(),expected,atol=TOL['fp32_atol'],rtol=TOL['fp32_rtol']) and m.weight.numel()==65536,'Complete native-shape zero-gate gradient/control identity differs')
                    native.append(dict(device=device,batch=b,tokens=128,gate_parameters=m.weight.numel(),four_layer_added_parameters=4*m.weight.numel(),weight_gradient_maxerror=err))
            base=ElementwiseOutputGate('sigmoid_elementwise_zero',6,2,3);base.weight.grad=torch.ones_like(base.weight);x=values((2,3,6)).float();y=values((2,2,3,3)).float();invalid=[]
            def altered(fn):
                m=copy.deepcopy(base);a=x.clone();b=y.clone();m,a,b=fn(m,a,b);invalid.append((m,a,b))
            for fn in [lambda m,a,b:(m,None,b),lambda m,a,b:(m,a[0],b),lambda m,a,b:(m,a,b[0]),lambda m,a,b:(m,a[:1],b),lambda m,a,b:(m,a,b[:,:,:2]),lambda m,a,b:(m,a[:,:,:5],b),lambda m,a,b:(m,a.double(),b),lambda m,a,b:(m,a,b.long()),lambda m,a,b:(m,a.to('mps'),b),lambda m,a,b:(m,a,torch.full_like(b,float('nan'))),lambda m,a,b:(m,torch.full_like(a,float('inf')),b)]:altered(fn)
            m=copy.deepcopy(base);m.kind='static_half';invalid.append((m,x,y));m=copy.deepcopy(base);m.weight=torch.nn.Parameter(torch.zeros(5,6));invalid.append((m,x,y));m=copy.deepcopy(base);m.weight.data[0,0]=float('nan');invalid.append((m,x,y))
            for m,a,b in invalid:
                before=copy.deepcopy([m.state_dict(),m.weight.grad,a,b])
                try:m(a,b)
                except (ValueError,RuntimeError,TypeError):pass
                else:raise AssertionError('Malformed gate accepted')
                require(same([m.state_dict(),m.weight.grad,a,b],before),'Rejected input mutated existing parameter/gradient/caller');rejected+=1
            for args in [('unknown',6,2,3),('none',False,2,3),('none',6,0,3),('none',5,2,3),('none',512,16,32)]:
                try:ElementwiseOutputGate(*args)
                except ValueError:rejected+=1
                else:raise AssertionError('Foreign gate policy accepted')
            for value,used in [({},META),({'x':float('nan')},0),({},-1)]:
                try:serialized(value,used)
                except ValueError:rejected+=1
                else:raise AssertionError('Invalid component metadata cap accepted')
            require(len(rows)==9 and len(controls)==6 and len(native)==4 and rejected==22,'Complete generated component contract panel required')
            torch.mps.synchronize();require(torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(mps_rng,torch.mps.get_rng_state()),'Global CPU/MPS RNG changed')
            for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Frozen source changed during gate proof')
        result=dict(status='elementwise_gate_scalar_output_gradient_and_native_cpu_mps_contracts_passed',source_plan=reference(R/'plan.json'),source_bindings=1057,small_oracle_cases=rows,parameter_free_control_cases=controls,native_shape_cases=native,max_absolute_error=max_error,rejections_before_mutation=rejected,global_cpu_mps_rng_unchanged=True,zero_initialization_rng_draws=0,corpus_loaded=False,language_models_constructed=0,optimizer_steps=0,tensor_files=0,scientific_seeds_assigned=False,quality_result=False,adopted=False,scope='Standalone query-local gate math only;full MLA/model/optimizer/checkpoint/cache/data/quality integration remains unproved.')
        publish(R/'result.json',result);publish(R/'STOP',dict(status='complete_standalone_elementwise_gate_component_proof',result=reference(R/'result.json'),never_rerun=True));print(json.dumps({k:v for k,v in result.items() if k not in ('small_oracle_cases','parameter_free_control_cases','native_shape_cases','source_plan')}),flush=True)
    except BaseException as exc:
        if not(R/'STOP').exists():publish(R/'STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
        raise
