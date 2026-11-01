"""Parent-side audit of existing v2 output bundles; no launcher or cleanup."""
import copy,json,math
from pathlib import Path
import torch
from omegaconf import OmegaConf
from . import campaign_contract as c
from research.ffn_norm_v1 import validation as v
from research.ffn_norm_v1.runtime.train.train import _training_contract
from research.ffn_norm_v1.runtime.train.lr_schedule import schedule_contract,multiplier
from research.ffn_norm_v1.runtime.train.weight_averaging import averaging_contract
from research.ffn_norm_v1.runtime.utils.checkpoint import read_checkpoint,load_model_weights
from research.ffn_norm_v1.runtime.models.block_norm import CHECKPOINT_KEY

require=c.require
KINDS={'config','parent','h200','summary','metrics','export'}
SCOPE='warmup_real_artifact_acceptance_v1'
TRIAL_SCOPE='finite_warmup_quality_v1_trial_acceptance_v1'


def verify_request(request):
    require(type(request) is dict and request.get('scope') in (SCOPE,TRIAL_SCOPE),'Declared acceptance scope required')
    require(set(request)=={'scope','sources','config','files'}|({'trial'} if request['scope']==TRIAL_SCOPE else set()),'Exact acceptance request required')
    c.verify_sources(request['sources']);require(type(request['files']) is dict and set(request['files'])==KINDS,'Complete output bundle required')
    for kind,binding in request['files'].items():
        require(type(binding) is dict and set(binding)=={'path','sha256'},'Exact path/SHA binding required')
        maximum=512*1024**2 if kind in ('parent','h200') else 2*1024**2
        require(c.file_sha(Path(binding['path']),root=c.ROOT,max_bytes=maximum)==binding['sha256'],'Bundle artifact hash differs')
    cfg=json.loads(Path(request['files']['config']['path']).read_text())
    require(c.canonical(cfg)==c.canonical(request['config']),'Requested configuration differs')
    require(type(cfg['training']['max_iters']) is int and 2<=cfg['training']['max_iters']<=1200,'Bounded final horizon required')
    require(cfg['training']['device']=='mps' and cfg['training']['strict_device'] is True,'Strict MPS fixture required')
    require(cfg['training']['keep_best_checkpoint'] is False and cfg['training']['checkpoint_every']==0,'Final-only outputs required')
    from research.ffn_norm_v1.runtime.models.block_norm import contract, FFN_NORM_POLICY
    require(contract(cfg['model'])['policy']==FFN_NORM_POLICY,'Product normalization model policy required')
    require(cfg['model']['ffn_hidden_norm']=='none','Adopted no-product-norm model required')
    require(schedule_contract(OmegaConf.create(cfg).training)['kind'] in ('linear_cooldown','warmup_linear_cooldown'),'Declared cooldown/warmup schedule required')
    if request['scope']==TRIAL_SCOPE:
        trial=request['trial'];require(type(trial) is dict and set(trial)=={'spec','seeds'},'Exact finite trial binding required')
        c.validate_trial_config(cfg,Path(cfg['training']['checkpoint_dir']).parent,trial['spec'],trial['seeds'])
    return OmegaConf.create(cfg)


def read_bundle(request):
    cfg=verify_request(request);files=request['files']
    return dict(cfg=cfg,raw=read_checkpoint(files['parent']['path']),avg=read_checkpoint(files['h200']['path']),
                summary=json.loads(Path(files['summary']['path']).read_text()),
                metrics=[json.loads(s) for s in Path(files['metrics']['path']).read_text().splitlines()],
                export=json.loads(Path(files['export']['path']).read_text()))


def equal(a,b,message):require(c.canonical(a)==c.canonical(b),message)


def audit_metric_rows(cfg,metrics):
    """The frozen trainer logs evaluation steps, not every optimizer update."""
    t=cfg.training;n=t.max_iters
    steps=sorted(set(range(0,n,t.eval_every))|{n-1})
    require(type(metrics) is list and len(metrics)==len(steps),'Every scheduled evaluation metric required')
    schedule=schedule_contract(t)
    for i,row in zip(steps,metrics):
        require(type(row['step']) is int and row['step']==i and row['tokens_processed']==(i+1)*t.batch_size*cfg.model.block_size,'Metric update/token sequence differs')
        require(row['device']=='mps','Metric backend differs')
        require(row['base_learning_rate']==t.learning_rate and row['lr_multiplier']==multiplier(schedule,i)
                and row['scheduled_learning_rate']==t.learning_rate*multiplier(schedule,i),'Metric schedule differs')
        require(type(row['train_ce_loss']) in (int,float) and math.isfinite(row['train_ce_loss']) and row['train_ce_loss']>=0,'Finite training loss required')


def audit_payloads(request,bundle,data):
    cfg=bundle['cfg'];raw=bundle['raw'];avg=bundle['avg'];summary=bundle['summary'];metrics=bundle['metrics'];export=bundle['export']
    plain=OmegaConf.to_container(cfg,resolve=True);t=cfg.training;n=t.max_iters
    equal(plain,request['config'],'Bundle configuration differs')
    equal(raw['cfg'],plain,'Raw configuration differs');equal(avg['cfg'],plain,'H200 configuration differs')
    require(type(raw['step']) is int and raw['step']==n and type(avg['step']) is int and avg['step']==n,'Final checkpoint step differs')
    require('averaged_weights' not in raw and 'optimizer_state' in raw and 'training_state' in raw,'Online parent required')
    require('optimizer_state' not in avg and 'training_state' not in avg,'H200 must be inference-only')
    require(raw['model_math_version']==avg['model_math_version']==2 and raw['parameter_layout_version']==avg['parameter_layout_version']==2,'Layout/math version differs')
    equal(raw[CHECKPOINT_KEY],avg[CHECKPOINT_KEY],'Normalization signature differs')
    state=raw['training_state'];equal(state['contract'],_training_contract(cfg,'mps',data),'Training contract differs')
    equal(avg['tokenizer_contract'],data.tokenizer_contract,'Tokenizer differs')
    equal(avg['parent_checkpoint'],request['files']['parent'],'H200 parent differs')
    equal(export,dict(checkpoint=request['files']['h200']['path'],checkpoint_sha256=request['files']['h200']['sha256'],
        parent_checkpoint=avg['parent_checkpoint'],averaged_weights=avg['averaged_weights']),'Export receipt differs')
    require(summary['completed_steps']==summary['max_iters']==n and summary['stop_reason']=='max_iters' and summary['device']=='mps','Incomplete training summary')
    require(summary['final_checkpoint']==request['files']['parent']['path'] and summary['best_checkpoint'] is None,'Summary checkpoint path differs')
    equal(summary['metrics'],metrics,'Summary/log metrics differ');equal(state['metrics'],metrics,'Checkpoint/log metrics differ')
    audit_metric_rows(cfg,metrics)
    averaging=averaging_contract(t)
    equal(summary['final_metric'],metrics[-1],'Final metric differs');equal(summary['best_metric'],state['best_metric'],'Best metric differs')
    require(summary['training_time_sec']==state['training_seconds'] and summary['training_time_sec']>=0,'Training time differs')
    require(summary['eval_every']==t.eval_every and summary['eval_iters']==t.eval_iters,'Evaluation settings differ')
    ema=state['weight_averaging_state'];count=1+(n-averaging['start_step'])//averaging['interval']
    equal(ema['spec'],averaging,'EMA specification differs')
    require(ema['completed_steps']==n and ema['snapshots']==count,'EMA progress differs')
    equal(summary['weight_averaging'],dict(spec=averaging,completed_steps=n,snapshots=count),'Summary EMA differs')
    equal(avg['averaged_weights'],dict(spec=averaging,half_life=200,snapshots=count,completed_steps=n,
          parameter_names=ema['parameter_names'],purpose='inference_only_not_optimizer_resume'),'H200 metadata differs')
    parameters=ema['averages']['200'];require(set(parameters)==set(ema['parameter_names']),'EMA parameter inventory differs')
    require(raw['model_state'].keys()==avg['model_state'].keys(),'H200 state inventory differs')
    for name,tensor in avg['model_state'].items():
        original=parameters[name] if name in parameters else raw['model_state'][name]
        require(tensor.dtype==original.dtype and tensor.shape==original.shape and torch.equal(tensor,original),'H200 tensor differs from parent EMA/buffer')
        require(bool(torch.isfinite(tensor).all()) and bool(torch.isfinite(raw['model_state'][name]).all()),'Nonfinite model tensor')
    return dict(completed_steps=n,tokens_processed=metrics[-1]['tokens_processed'],ema_parameters=len(parameters),ema_snapshots=count)


def accept(request,reported_validation):
    """Recompute validation rather than trusting a child's claimed score."""
    bundle=read_bundle(request);binding={k:request['files'][k] for k in ('config','parent','h200')}
    model,data,cfg=v.load(binding,'mps')
    try:
        audit=audit_payloads(request,bundle,data)
        # Actual optimizer/RNG/EMA shape preflight; no optimizer step is taken.
        load_model_weights(model,bundle['raw'],for_resume=True)
        load_model_weights(model,bundle['avg'])
        v.validate_report(reported_validation,binding,data)
        recomputed=dict(binding=copy.deepcopy(binding),**v.score(model,data))
        equal(reported_validation,recomputed,'Parent recomputation differs from claimed validation')
        verify_request(request)
        return dict(kind='warmup_real_artifact_acceptance_v1',scope=request['scope'],request_sha256=c.fingerprint(request),
                    source_sha256=c.fingerprint(request['sources']),files=copy.deepcopy(request['files']),audit=audit,
                    validation=recomputed,accepted=True,launch_permitted=False,quality_adopted=False)
    finally:del model,data,cfg,bundle
