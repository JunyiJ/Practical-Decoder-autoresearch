"""Actual full-model/optimizer/cache identity and CPU/MPS numerical prerequisite."""
import copy,datetime,gc,hashlib,json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.mla_score_quality_v1 import campaign_contract as old
from research.mla_score_quality_v1.support import idle
from research.ffn_structure_v1 import pilot_contract as pilot
from research.mla_score_v1.runtime.models.gpt import GPT as LegacyGPT
from research.mla_score_v1.runtime.train.optimizers import build_optimizer as legacy_optimizer_factory
from research.ffn_norm_v1.runtime.models.gpt import GPT
from research.ffn_norm_v1.runtime.models.block_norm import FFN_NORM_POLICY,contract,checkpoint_signature
from research.ffn_norm_v1.runtime.models.block_norm import MLA_SCORE_POLICY
from research.ffn_norm_v1.runtime.models.tokenizer_architecture import validate_architecture_model
from research.ffn_norm_v1.runtime.train.optimizers import build_optimizer
from research.ffn_norm_v1.runtime.attention.cache import build_caches
from research.ffn_norm_v1.runtime.utils.checkpoint import save_checkpoint,load_model_weights
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
ROOT=old.ROOT;B=Path(__file__).resolve().parent

def reference(p):return dict(path=str(Path(p).relative_to(ROOT)),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def sources():return {str(p.relative_to(ROOT)):reference(p)['sha256'] for p in sorted((ROOT/'research/ffn_norm_v1').rglob('*.py'))}
def compare(a,b,rtol=3e-4,atol=3e-5):
    a=a.detach().cpu();b=b.detach().cpu();torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
    return float((a-b).abs().max())
def reject(fn):
    try:fn()
    except (ValueError,TypeError) as exc:return str(exc)
    raise AssertionError('Invalid semantics accepted')
def groups(opt,model):
    names={id(p):n for n,p in model.named_parameters()}
    all_parameters=[p for g in opt.param_groups for p in g['params']]
    assert len({id(p) for p in all_parameters})==len(all_parameters)==len(names)
    assert {id(p) for p in all_parameters}==set(names)
    return {type(child).__name__:dict(parameters=sum(p.numel() for g in child.param_groups for p in g['params']),
        tensors=sum(len(g['params']) for g in child.param_groups),groups=[dict(names=[names[id(p)] for p in g['params']],**{k:v for k,v in g.items() if k!='params'}) for g in child.param_groups]) for child in opt.optimizers}
def state_equal(a,b):return a.keys()==b.keys() and all(torch.equal(a[n],b[n]) for n in a)

torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    idle();assert not (B/'model001.json').exists()
    plan=json.loads((B/'model001-plan.json').read_text())
    for n,h in plan['sources'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h
    frozen=old.source_inventory();assert frozen==plan['frozen_sources']
    current=sources();snapshot=B/'model001-source';snapshot.mkdir()
    for n in current:
        dest=snapshot/Path(n).relative_to('research/ffn_norm_v1');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/n).read_bytes())
    with StrictMpsGuard(B/'model001-guard.json',ROOT):
        plan=json.loads((pilot.EVIDENCE/'pilot001-plan.json').read_text());cfg=OmegaConf.create(pilot.config(0,plan['seeds']))
        cfg.model.tokenizer_architecture_policy=MLA_SCORE_POLICY
        torch.manual_seed(90829);legacy=LegacyGPT(cfg.model);legacy_optimizer=legacy_optimizer_factory(legacy,cfg.training)
        baseline_groups=groups(legacy_optimizer,legacy);legacy_decay=legacy_optimizer.decay
        control_cfg=copy.deepcopy(cfg);control_cfg.model.tokenizer_architecture_policy=FFN_NORM_POLICY
        control_cfg.model.ffn_hidden_norm='none';control_cfg.model.ffn_hidden_norm_policy='mla_swiglu_product_rms_672_v1'
        torch.manual_seed(90829);control=GPT(control_cfg.model)
        assert state_equal(legacy.state_dict(),control.state_dict())
        tokens=(torch.arange(18).reshape(2,9)%5+23).long();targets=(tokens+1)%50257
        with torch.no_grad():assert torch.equal(legacy(tokens)[0],control(tokens)[0])
        control_weights={n:p.clone() for n,p in control.state_dict().items()}
        control_signature=checkpoint_signature(control)
        del legacy,legacy_optimizer,control;gc.collect()
        records=[]
        for kind in ('none','rms'):
            cfg=copy.deepcopy(control_cfg);cfg.model.ffn_hidden_norm=kind
            torch.manual_seed(90829);cpu=GPT(cfg.model)
            assert all(torch.equal(cpu.state_dict()[n],v) for n,v in control_weights.items())
            added=set(cpu.state_dict())-set(control_weights)
            assert added==({f'transformer.h.{i}.mlp.hidden_norm.weight' for i in range(4)} if kind=='rms' else set())
            assert all(torch.equal(cpu.state_dict()[n],torch.ones(672)) for n in added)
            assert all(type(b.attn).__module__.startswith('research.ffn_norm_v1.') for b in cpu.transformer.h)
            assert sum(p.numel() for p in cpu.parameters())==28492416+(2688 if kind=='rms' else 0)
            optimizer=build_optimizer(cpu,cfg.training);optimizer_groups=groups(optimizer,cpu)
            filtered=copy.deepcopy(optimizer_groups)
            for name,panel in filtered.items():
                panel['groups']=[g for g in panel['groups'] if g['names'][0] not in added]
                panel['parameters']-=2688 if kind=='rms' and name=='AdamW' else 0
                panel['tensors']-=4 if kind=='rms' and name=='AdamW' else 0
            assert filtered==baseline_groups
            gains=[g for panel in optimizer_groups.values() for g in panel['groups'] if g['names'][0] in added]
            assert len(gains)==len(added) and all(g['lr']==cfg.training.learning_rate and g['width_lr_scale']==1. and g['weight_decay']==0. for g in gains)
            assert optimizer.decay==legacy_decay
            gpu=copy.deepcopy(cpu).to('mps')
            a,la=cpu(tokens,targets);b,lb=gpu(tokens.to('mps'),targets.to('mps'))
            row=dict(kind=kind,total_parameters=sum(p.numel() for p in cpu.parameters()),gain_groups=gains,independent_decay=optimizer.decay,logits_max_abs=compare(a,b),loss_abs=compare(la,lb,1e-5,1e-4))
            la.backward();lb.backward()
            row['gradient_max_abs']=max(compare(p.grad,q.grad,3e-3,3e-5) for p,q in zip(cpu.parameters(),gpu.parameters()))
            optimizer.step();assert all(bool(torch.isfinite(p).all()) for p in cpu.parameters())
            row['optimizer_step_finite']=True
            cpu.zero_grad(set_to_none=True);gpu.zero_grad(set_to_none=True)
            saved={}
            save_checkpoint(B/'not-written.pt',cpu,optimizer=optimizer,step=1,cfg=cfg,payload_writer=lambda p,payload:saved.update(payload))
            load_model_weights(cpu,saved)
            other=copy.deepcopy(saved);other['block_normalization_signature']=copy.deepcopy(saved['block_normalization_signature'])
            other['block_normalization_signature']['architecture']['ffn_product_normalization']['kind']='rms' if kind=='none' else 'none'
            row['cross_arm_checkpoint_rejected']=reject(lambda:load_model_weights(cpu,other))
            bad=copy.copy(saved);bad['cfg']=copy.deepcopy(saved['cfg']);bad['cfg']['model']['ffn_hidden_norm']='rms' if kind=='none' else 'none'
            row['changed_saved_config_rejected']=reject(lambda:load_model_weights(cpu,bad))
            if kind=='rms':
                cpu.transformer.h[0].mlp.hidden_norm.eps=1e-6
                row['mutated_epsilon_rejected']=reject(lambda:validate_architecture_model(cpu))
                cpu.transformer.h[0].mlp.hidden_norm.eps=1e-5
            else:
                cpu.transformer.h[0].mlp.hidden_norm=torch.nn.RMSNorm(672,eps=1e-5)
                row['mutated_control_rejected']=reject(lambda:validate_architecture_model(cpu))
                cpu.transformer.h[0].mlp.hidden_norm=None
            validate_architecture_model(cpu)
            with torch.no_grad():
                x=tokens.to('mps');y=x+31;full=gpu(x)[0];full_y=gpu(y)[0]
                ca=build_caches(cfg,'mps',batch_size=2,dtype=torch.float32);cb=build_caches(cfg,'mps',batch_size=2,dtype=torch.float32)
                parts=[];other=[];pos=0
                for count in (3,1,5):
                    parts.append(gpu(x[:,pos:pos+count],cache=ca,start_pos=pos)[0]);other.append(gpu(y[:,pos:pos+count],cache=cb,start_pos=pos)[0]);pos+=count
                row['cache_max_abs']=compare(torch.cat(parts,1),full);row['interleaved_max_abs']=compare(torch.cat(other,1),full_y)
                changed=x.clone();changed[:,4:]+=3;row['causality_max_abs']=compare(gpu(changed)[0][:,:4],full[:,:4],0,0)
                for cache in ca:cache.reset()
                row['reset_max_abs']=compare(gpu(x,cache=ca,start_pos=0)[0],full)
                wrong=copy.deepcopy(cfg);wrong.model.ffn_hidden_norm='rms' if kind=='none' else 'none'
                foreign=build_caches(wrong,'mps',batch_size=2,dtype=torch.float32)
                row['cross_arm_cache_rejected']=reject(lambda:gpu(x,cache=foreign,start_pos=0))
                assert all(cache.position==0 and all(torch.count_nonzero(v)==0 for v in cache.buffers()) for cache in foreign)
                row['cache_bytes']=sum(v.numel()*v.element_size() for cache in ca for v in cache.buffers())
            records.append(row)
            del cpu,gpu,optimizer,a,b,la,lb,saved,bad,other,ca,cb,foreign;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
        invalid=[]
        for key,value in [('ffn_hidden_norm','layer'),('ffn_hidden_norm',True),('ffn_hidden_norm_policy','other'),('attention_logit_scale',1.),('ffn','silu'),('norm_type','layernorm'),('dropout',.1),('ffn_norm_eps',1e-6)]:
            bad=copy.deepcopy(control_cfg.model);setattr(bad,key,value)
            invalid.append(dict(field=key,value=value,error=reject(lambda:contract(bad))))
        legacy_bad=copy.deepcopy(control_cfg.model);legacy_bad.tokenizer_architecture_policy=MLA_SCORE_POLICY
        invalid.append(dict(field='legacy_policy',error=reject(lambda:LegacyGPT(legacy_bad))))
        assert old.source_inventory()==frozen and sources()==current
        result=dict(status='full_model_product_norm_cpu_mps_prerequisite_passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),runner=reference(__file__),frozen_sources=frozen,new_sources=current,rows=records,
            control_initialization_and_logits_bitwise_equal=True,shared_candidate_initialization_bitwise_equal=True,total_parameters_by_arm=dict(none=28492416,rms=28495104),optimizer_groups=baseline_groups,
            invalid_contracts_rejected=invalid,holdout_read=False,corpus_read=False,checkpoint_io=False,scientific_training=False,scientific_seeds_reserved=False,
            checkpoint_scope='Real save/load payload adapters in memory only. Serialized raw/H200 and optimizer/RNG/EMA resume remain required.',
            remaining=['serialized raw/H200 and exact optimizer/RNG/EMA resume','real worker and parent acceptance, validation and inference','actual storage projection and cleanup','seed census, pilot freeze/pilots and quality freeze'])
    result['guard']=reference(B/'model001-guard.json');assert json.loads((B/'model001-guard.json').read_text())['status']=='finished'
    with (B/'model001.json').open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','rows','total_parameters_by_arm')}))
