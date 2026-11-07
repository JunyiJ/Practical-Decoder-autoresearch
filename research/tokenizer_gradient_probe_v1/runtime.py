"""Own accepted-RAW weight consumer and complete untransformed gradient extraction."""
import gc,hashlib,json
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_moe_model_v1.model import GPT,load_weights
from research.tokenizer_moe_model_v1.contract import signature
from research.tokenizer_moe_worker_v1.validation import verify_artifact
E=ROOT/'research/tokenizer-evidence/post-moe-boundary-001';OLD=E.parent/'post-prefix-boundary-001'
TRAIN_SEED=600492;INITIALIZER_SEED=600493;SLOTS=(9,14,16)

def checked(ref):
    require(reference(ref['path'])=={k:ref[k] for k in ('path','sha256')},'Immutable reference changed');return json.loads(Path(ref['path']).read_text())
def sources():
    result=checked(reference(E/'gradient-component001.json'));plan=checked(reference(E/'gradient-component001-plan.json'));guard=checked(reference(E/'gradient-component001-guard.json'))
    require(result['status']=='generated_gradient_moment_oracle_and_transactional_contracts_passed' and guard['status']=='finished' and guard['reason'] is None and result['sources']==plan['sources'],'Actual original gradient-moment proof required')
    pins=dict(plan['sources']);require(len(pins)==1038,'Exact1038 predecessor bindings required')
    for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Immutable predecessor changed '+p)
    for p in sorted(Path(__file__).parent.glob('*.py')):pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
    p=E/'gradient-model001.py';pins[str(p.relative_to(ROOT))]=reference(p)['sha256'];require(len(pins)==1043,'Exact own four-file runtime plus proof required');return pins

def bindings():
    close=checked(reference(OLD/'quality001-closure001.json'));require(close['status']=='CLOSED_NOT_ADOPTED' and close['accepted_endpoints']==18,'Actual closed complete MoE panel required')
    stop=checked(close['terminal']);require(stop['status']=='complete_finite_scientific_panel','Original terminal STOP required');out=[]
    for slot in SLOTS:
        row=checked(close['accepted_references'][slot]);w=checked(row['worker']);rq=checked(row['request'])
        require(row['slot']==slot and row['arm']=='dense' and row['acceptance']['status']=='accepted' and row['acceptance']['complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked'],'Original independently accepted dense endpoint required')
        require(w['step']==1200 and w['kind']=='tokenizer_moe_quality_worker_v1' and w['config']==rq['spec']['config'] and w['request']==row['request'] and w['architecture']==signature(w['config']['moe_model']) and w['architecture']['arm']=='dense','Exact own complete RAW configuration required')
        expected=next(v for v in close['protected_payloads'] if v['slot']==slot and v['kind']=='parent');require(all(w['parent'][k]==expected[k] for k in ('path','sha256','size')),'Exact original RAW binding differs')
        verify_artifact(w['parent']);out.append(dict(slot=slot,worker_ref=row['worker'],worker=w))
    return out

def raw_weights(raw,worker):
    keys={'kind','architecture','cfg','step','model_state','optimizer_state','ema','training_identity','clipping_history','sampler_state','cpu_rng','mps_rng'}
    require(type(raw) is dict and set(raw)==keys and raw['kind']=='raw_tokenizer_moe_v1' and raw['architecture']==worker['architecture'] and raw['cfg']==worker['config'] and raw['step']==worker['step']==1200,'Only exact accepted own RAW weights may be extracted;no resume')
    require(raw['architecture']['arm']=='dense' and 'model' not in raw['cfg'],'Genuine dense own policy required')
    return dict(kind='tokenizer_moe_weights_v1',architecture=raw['architecture'],model_state=raw['model_state'])

def load_dense(worker,device,initializer_seed):
    require(type(initializer_seed) is int and initializer_seed in (INITIALIZER_SEED,600494),'Declared private diagnostic initializer required')
    verify_artifact(worker['parent']);raw=torch.load(worker['parent']['path'],map_location='cpu',weights_only=True);weights=raw_weights(raw,worker)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(initializer_seed);model=GPT(plain(worker['config']['moe_model']));load_weights(model,weights)
    del weights,raw;gc.collect();model=model.to(device).train();return model

def tensor_hash(t):
    v=t.detach().cpu().contiguous();return hashlib.sha256(v.numpy().tobytes()).hexdigest()
def state_hash(model):return {n:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=tensor_hash(v)) for n,v in model.state_dict().items()}

def gradients(model,x,y):
    require(type(model) is GPT and model.signature['arm']=='dense' and model.training,'Genuine training-mode dense model required');model.validate()
    device=next(model.parameters()).device
    require(type(x) is torch.Tensor and type(y) is torch.Tensor and x.ndim==2 and 1<=x.shape[0]<=8 and 1<=x.shape[1]<=128 and x.shape==y.shape and x.dtype==y.dtype==torch.int64 and x.device==y.device==device,'Exact device-local native input/target matrices required')
    require(bool((x>=0).all()) and bool((x<50257).all()) and bool((y>=0).all()) and bool((y<50257).all()),'Native IDs required before gradient mutation')
    model.zero_grad(set_to_none=True);logits,loss=model(x,y,include_aux_loss=False,include_training_objective=False)
    require(bool(torch.isfinite(loss)),'Finite CE-only objective required');loss.backward();out={}
    for name,p in model.named_parameters():
        require(p.grad is not None and p.grad.shape==p.shape and p.grad.dtype==torch.float32 and bool(torch.isfinite(p.grad).all()),'Complete finite learned gradient inventory required: '+name)
        out[name]=p.grad.detach().to('cpu',copy=True)
    require(len(out)==88,'All88 dense learned gradients required');value=float(loss.detach());del logits,loss;return out,value

def gradient_norms(grads):return {n:float(g.double().square().sum()) for n,g in grads.items()}

def grouped(report):
    import math
    out={}
    for group in ('input','readout','body'):
        values=[v for n,v in report['parameter_statistics'].items() if ('input' if n.startswith('transformer.wte.') else 'readout' if n.startswith('lm_head.') else 'body')==group]
        norm=math.fsum(v['mean_gradient_squared_norm'] for v in values);cov=math.fsum(v['trace_sample_covariance'] for v in values);signal=norm-cov/report['draws']
        out[group]=dict(parameters=len(values),mean_gradient_squared_norm=norm,trace_sample_covariance=cov,signal_squared_unbiased=signal,ratio_in_independent_B8_draw_units=(cov/signal if signal>0 else None))
    return out
