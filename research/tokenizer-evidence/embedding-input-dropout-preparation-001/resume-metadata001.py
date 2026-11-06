"""New adversarial metadata proof at step0; no optimizer update or old fixture replay."""
import copy,gc,hashlib,json
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_model_v1.optimizer import build_optimizer
from research.embedding_input_dropout_v1.model import GPT
from research.embedding_input_dropout_resume_v1.contract import config,sources as inherited_sources
from research.embedding_input_dropout_resume_v1.checkpoint import raw_payload,validate_resume as inherited_validate
from research.embedding_input_dropout_resume_guard_v1.checkpoint import validate_resume
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;torch.set_num_threads(4)
def sources():
    d=inherited_sources()
    for p in sorted((ROOT/'research/embedding_input_dropout_resume_guard_v1').glob('*.py')):d[str(p.relative_to(ROOT))]=reference(p)['sha256']
    return d
def digest(values):
    out=[]
    for t in values:
        t=t.detach().cpu().contiguous();out.append([str(t.dtype),list(t.shape),hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()])
    return out
def rejected(fn):
    try:fn()
    except (ValueError,TypeError):return
    raise AssertionError('Malformed resume metadata accepted')
with session_lock(GLOBAL_LOCK.parent):
    idle();environment();require(not(E/'resume-metadata001.json').exists() and not(E/'resume-metadata001-guard.json').exists(),'Closed metadata proof cannot retry');pinned=sources()
    with StrictMpsGuard(E/'resume-metadata001-guard.json',ROOT),torch.random.fork_rng(devices=[]):
        cfg=config(.1);torch.manual_seed(600405);torch.mps.manual_seed(600405);m=GPT(plain(cfg.input_embedding_regularization)).to('mps');o=build_optimizer(m,cfg);e=TailWeightAverages(m,cfg.training)
        sampler={n:torch.Generator().manual_seed(i).get_state() for i,n in enumerate(('train','val'))}
        payload=raw_payload(m,o,e,cfg,0,[],sampler);require(validate_resume(m,o,e,payload,cfg)==0,'Valid metadata rejected')
        before=digest([*m.state_dict().values(),torch.get_rng_state(),torch.mps.get_rng_state()]);checks=[]
        # Demonstrate the inherited layout gap without invoking an invalid resume.
        short=dict(payload,mps_rng=payload['mps_rng'][:1]);old_accepts=inherited_validate(m,o,e,short,cfg)==0
        require(old_accepts,'Expected inherited prevalidation gap not present;review changed source')
        for field in ('cpu_rng','mps_rng'):
            for kind in ('short','float','rank2'):
                value=payload[field];bad_value=value[:1] if kind=='short' else value.float() if kind=='float' else value.unsqueeze(0)
                bad=dict(payload,**{field:bad_value});rejected(lambda:validate_resume(m,o,e,bad,cfg));checks.append(field+'_'+kind)
        for name in ('train','val'):
            bad=dict(payload,sampler_state=dict(sampler,**{name:sampler[name][:1]}));rejected(lambda:validate_resume(m,o,e,bad,cfg));checks.append(name+'_sampler_short')
        badcfg=copy.deepcopy(payload['cfg']);badcfg['input_embedding_regularization']['probability']=0.
        bad=dict(payload,cfg=badcfg);rejected(lambda:validate_resume(m,o,e,bad,cfg));checks.append('copied_probability_forgery')
        badcfg=copy.deepcopy(payload['cfg']);badcfg.pop('input_embedding_regularization');bad=dict(payload,cfg=badcfg);rejected(lambda:validate_resume(m,o,e,bad,cfg));checks.append('missing_copied_policy')
        require(digest([*m.state_dict().values(),torch.get_rng_state(),torch.mps.get_rng_state()])==before,'Metadata proof changed model or global RNG')
        require(not any(child.state for child in o.optimizers) and e.completed_steps==0 and e.snapshots==0,'Metadata proof changed optimizer/EMA')
        del m,o,e,payload,bad,badcfg;gc.collect();torch.mps.empty_cache();require(sources()==pinned,'Sources changed')
    result=dict(status='passed_additive_resume_metadata_preflight',source=reference(__file__),sources=pinned,
        actual_zero_step_payload=True,optimizer_updates=0,original_valid_resume001_unchanged=True,
        inherited_short_mps_rng_accepted_by_validation_only=old_accepts,new_rejected_cases=checks,
        rejected_before_model_optimizer_ema_or_live_rng_mutation=True,new_scientific_seed=False,guard=reference(E/'resume-metadata001-guard.json'))
    with (E/'resume-metadata001.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','source','guard')}))
