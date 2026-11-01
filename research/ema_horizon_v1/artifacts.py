"""Bounded dual-EMA checkpoint acceptance; unchanged scorer and runtime math."""
import copy,gc,json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from . import contract as c
from research.block_norm_v1 import artifact_acceptance as inherited_audit, validation as scoring
from research.block_norm_v1.validation_data import load as validation_data
from research.block_norm_v1.runtime.models.gpt import GPT
from research.block_norm_v1.runtime.models.block_norm import CHECKPOINT_KEY,contract,validate_model
from research.block_norm_v1.runtime.train.weight_averaging import averaging_contract
from research.block_norm_v1.runtime.utils.checkpoint import read_checkpoint,load_model_weights
from research.block_norm_v1.runtime.attention.cache import build_caches
from src.research.tokenizer_wide_hybrid_measurements import check_kernel
from src.research.tokenizer_wide_hybrid_timing import allocated_bytes,token_digest

require=c.require
ROLES={'config','raw','h100','h200','summary','metrics','export100','export200'}
SCOPE=c.KIND+'_artifact_request_v1'
def equal(a,b,message):require(c.inherited.canonical(a)==c.inherited.canonical(b),message)

def verify_request(request):
    require(type(request) is dict and set(request)=={'scope','sources','stage','slot','seeds','config','files'},'Exact dual-EMA request required')
    require(request['scope']==SCOPE,'Dual-EMA scope required')
    equal(request['sources'],c.source_inventory(),'Source inventory changed')
    cfg=request['config'];base=Path(cfg['training']['checkpoint_dir']).parent
    c.validate_config(cfg,base,request['stage'],request['slot'],request['seeds'])
    files=request['files'];require(type(files) is dict and set(files)==ROLES,'Complete dual-EMA bundle required')
    require(len({x['path'] for x in files.values()})==len(ROLES),'Distinct artifact paths required')
    for role,binding in files.items():
        c.check_artifact(role,binding)
        path=Path(binding['path']);require(path.is_relative_to(base),'Artifact outside bound slot')
        if role in ('raw','h100','h200'):
            require(path.parent==base/'checkpoints' and path.suffix=='.pt','Checkpoint path differs')
            if role!='raw':require(path.name==role+'.pt','Export basename differs')
        else:
            expected={'config':'config.json','summary':'training-summary.json','metrics':'metrics.jsonl',
                      'export100':'export100.json','export200':'export200.json'}[role]
            require(path==base/expected,'Metadata path differs')
    equal(json.loads(Path(files['config']['path']).read_text()),cfg,'Saved configuration differs')
    return OmegaConf.create(cfg)

def binding(request,half):
    require(type(half) is int and half in c.HALF_LIVES,'Declared half-life required')
    f=request['files']
    return dict(config=f['config'],raw=f['raw'],averaged=f[f'h{half}'],half_life=half)

def validate_endpoint(saved,request,half):
    """Metadata before constructing/loading the full model; no H200 relabeling."""
    binding(request,half)
    equal(saved['cfg'],request['config'],'Export configuration differs')
    equal(saved['parent_checkpoint'],request['files']['raw'],'Raw parent differs')
    require('optimizer_state' not in saved and 'training_state' not in saved,'Export must be inference-only')
    cfg=OmegaConf.create(request['config']);spec=averaging_contract(cfg.training)
    require(spec['half_lives']==[100,200],'Both declared averages required')
    meta=saved['averaged_weights'];n=cfg.training.max_iters;count=1+(n-spec['start_step'])//spec['interval']
    require(type(saved['step']) is int and saved['step']==n,'Final endpoint required')
    require(saved['model_math_version']==2 and saved['parameter_layout_version']==2,'Layout/math version differs')
    require(type(meta) is dict and set(meta)=={'spec','half_life','snapshots','completed_steps','parameter_names','purpose'},'Exact export metadata required')
    equal(meta['spec'],spec,'Averaging contract differs')
    require(type(meta['half_life']) is int and meta['half_life']==half and meta['snapshots']==count
        and meta['completed_steps']==n and meta['purpose']=='inference_only_not_optimizer_resume','Half-life/progress/purpose differs')
    names=meta['parameter_names'];require(type(names) is list and len(names)==len(set(names)),'Unique parameter names required')

def load_endpoint(request,half,device='mps'):
    cfg=verify_request(request);saved=read_checkpoint(request['files'][f'h{half}']['path'])
    validate_endpoint(saved,request,half)
    model_contract=contract(cfg.model);data=validation_data(cfg,saved)
    require(set(data.splits)=={'train','val'},'Closed holdout required')
    with torch.random.fork_rng(devices=[]):
        model=GPT(cfg.model).to(device).eval();load_model_weights(model,saved)
    validate_model(model,model_contract)
    require(saved['averaged_weights']['parameter_names']==[n for n,_ in model.named_parameters()],'Constructed parameter names differ')
    del saved
    return model,data,cfg

def evaluate(request):
    result={}
    for half in c.HALF_LIVES:
        model,data,cfg=load_endpoint(request,half)
        try:result[str(half)]=dict(binding=binding(request,half),**scoring.score(model,data))
        finally:del model,data,cfg;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
    return result

def audit_export(raw,avg,request,half,export,data):
    validate_endpoint(avg,request,half)
    cfg=OmegaConf.create(request['config']);spec=averaging_contract(cfg.training)
    ema=raw['training_state']['weight_averaging_state']
    equal(ema['spec'],spec,'Raw EMA spec differs');require(set(ema['averages'])=={'100','200'},'Both raw EMA arrays required')
    equal(avg[CHECKPOINT_KEY],raw[CHECKPOINT_KEY],'Model signatures differ')
    equal(avg['tokenizer_contract'],data.tokenizer_contract,'Tokenizer differs')
    equal(avg['averaged_weights']['parameter_names'],ema['parameter_names'],'EMA names differ')
    values=ema['averages'][str(half)];require(set(values)==set(ema['parameter_names']),'EMA tensor inventory differs')
    require(raw['model_state'].keys()==avg['model_state'].keys(),'Export tensor inventory differs')
    for name,tensor in avg['model_state'].items():
        parent=values[name] if name in values else raw['model_state'][name]
        require(tensor.dtype==parent.dtype and tensor.shape==parent.shape and torch.equal(tensor,parent),'Export differs from raw EMA/buffer')
        require(bool(torch.isfinite(tensor).all()),'Nonfinite export')
    equal(export,dict(checkpoint=request['files'][f'h{half}']['path'],checkpoint_sha256=request['files'][f'h{half}']['sha256'],
        parent_checkpoint=avg['parent_checkpoint'],averaged_weights=avg['averaged_weights']),'Export receipt differs')

def cache_audit(request):
    model,data,cfg=load_endpoint(request,200);ids=data.train_data[:96].tolist();rows=[]
    try:
        for role in ('raw','h100','h200'):
            saved=read_checkpoint(request['files'][role]['path']);load_model_weights(model,saved);del saved
            for batch in (1,8):
                fixed=torch.tensor(ids,dtype=torch.int64,device='mps').unsqueeze(0).repeat(batch,1)
                factory=lambda:build_caches(cfg,'mps',batch_size=batch,dtype=torch.float32)
                check=check_kernel(model,fixed,factory);cache=factory()
                size=allocated_bytes(b for layer in cache for b in layer.buffers());require(size==196608*batch,'Cache size differs')
                rows.append(dict(endpoint=role,batch=batch,state_bytes=size,**check));del cache,fixed
        return dict(training96_sha256=token_digest(ids),rows=rows)
    finally:del model,data,cfg;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()

def accept(request,reported):
    """Independently recompute both validation reports and resume/export audits."""
    cfg=verify_request(request);files=request['files']
    require(type(reported) is dict and set(reported)=={'100','200'},'Both validation reports required')
    raw=read_checkpoint(files['raw']['path']);avg200=read_checkpoint(files['h200']['path'])
    model,data,cfg=load_endpoint(request,200);bundle=None
    try:
        # Reuse H200 tensor/optimizer/schedule audit arithmetic only AFTER this
        # new contract's bounded validation. Never invoke or weaken the old scope.
        bundle=dict(cfg=cfg,raw=raw,avg=avg200,summary=json.loads(Path(files['summary']['path']).read_text()),
            metrics=[json.loads(x) for x in Path(files['metrics']['path']).read_text().splitlines()],
            export=json.loads(Path(files['export200']['path']).read_text()))
        mapping=dict(config=request['config'],files=dict(parent=files['raw'],h200=files['h200']))
        audit=inherited_audit.audit_payloads(mapping,bundle,data)
        load_model_weights(model,raw,for_resume=True)
        reviews={}
        for half in c.HALF_LIVES:
            avg=avg200 if half==200 else read_checkpoint(files['h100']['path'])
            export=json.loads(Path(files[f'export{half}']['path']).read_text())
            audit_export(raw,avg,request,half,export,data);load_model_weights(model,avg)
            scoring.validate_report(reported[str(half)],binding(request,half),data)
            recomputed=dict(binding=binding(request,half),**scoring.score(model,data))
            equal(reported[str(half)],recomputed,'Parent validation recomputation differs')
            reviews[str(half)]=recomputed
            del avg
        verify_request(request)
    finally:del model,data,cfg,raw,avg200,bundle;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
    return dict(kind=c.KIND+'_parent_acceptance_v1',request_sha256=c.inherited.fingerprint(request),
        source_sha256=c.inherited.fingerprint(request['sources']),files=copy.deepcopy(files),audit=audit,
        validation=reviews,cache=cache_audit(request),accepted=True,launch_permitted=False,quality_adopted=False)
