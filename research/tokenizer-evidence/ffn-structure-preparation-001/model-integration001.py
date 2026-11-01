"""Bounded full-model CPU prerequisite, synthetic tokens only, no checkpoint IO."""
import copy, datetime, hashlib, json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.ema_horizon_v1 import contract as c, execution as e
from research.block_norm_v1.fixture_dispatch import reference, read
from research.block_norm_v1.runtime.models.gpt import GPT as LegacyGPT
from research.ffn_structure_v1.runtime.models.gpt import GPT
from research.ffn_structure_v1.runtime.models.block_norm import FFN_POLICY, contract, checkpoint_signature, validate_checkpoint
from research.ffn_structure_v1.runtime.models.tokenizer_architecture import validate_architecture_model
from research.ffn_structure_v1.runtime.train.optimizers import build_optimizer
from research.ffn_structure_v1.runtime.attention.cache import KVCache
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
B=Path(__file__).resolve().parent
SOURCE=c.ROOT/'research/ffn_structure_v1'
SNAP=B/'model-integration001-source'
def sources():return {str(p.relative_to(c.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE.rglob('*.py'))}
def rejected(fn):
    try:fn()
    except (ValueError,TypeError) as exc:return str(exc)
    raise AssertionError('Invalid semantics accepted')
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    e.idle()
    assert not SNAP.exists();SNAP.mkdir()
    original=c.source_inventory();current=sources()
    for name in current:
        q=SNAP/Path(name).relative_to(SOURCE.relative_to(c.ROOT));q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((c.ROOT/name).read_bytes())
    with CpuPrerequisiteGuard(B/'model-integration001-guard.json',c.ROOT,timeout_seconds=120,poll_seconds=.25):
        seeds=dict(pilot=376951,screen=376952,fresh=[376961,376962,376963],evaluation=376953,bootstrap=376954)
        cfg=OmegaConf.create(c.config(c.ROOT/'research_runs/tokenizer-ema-horizon-v1-ffn-component/evidence/pilots/00','pilot',0,seeds))
        torch.manual_seed(376955);legacy=LegacyGPT(cfg.model)
        control_cfg=copy.deepcopy(cfg);control_cfg.model.tokenizer_architecture_policy=FFN_POLICY
        torch.manual_seed(376955);control=GPT(control_cfg.model)
        assert legacy.state_dict().keys()==control.state_dict().keys()
        assert all(torch.equal(v,control.state_dict()[n]) for n,v in legacy.state_dict().items())
        tokens=(torch.arange(32).reshape(2,16)*13 % 50257).long();targets=(tokens+1)%50257
        with torch.no_grad():assert torch.equal(legacy(tokens)[0],control(tokens)[0])
        del legacy
        candidate_cfg=copy.deepcopy(control_cfg);candidate_cfg.model.ffn='silu';candidate_cfg.model.ffn_width_policy='ungated_silu_63_16'
        torch.manual_seed(376955);model=GPT(candidate_cfg.model)
        assert sum(p.numel() for p in model.parameters())==28492416
        assert sum(p.numel() for n,p in model.named_parameters() if '.mlp.' in n)==2064384
        optimizer=build_optimizer(model,candidate_cfg.training)
        all_params=[p for g in optimizer.param_groups for p in g['params']]
        assert len({id(p) for p in all_params})==len(all_params)==len(list(model.parameters()))
        assert {id(p) for p in all_params}=={id(p) for p in model.parameters()}
        groups={type(child).__name__:dict(parameters=sum(p.numel() for g in child.param_groups for p in g['params']),tensors=sum(len(g['params']) for g in child.param_groups)) for child in optimizer.optimizers}
        assert groups=={'Muon':dict(parameters=2752512,tensors=40),'AdamW':dict(parameters=25739904,tensors=44)}
        assert all(g['width_lr_scale']==.5 and g['adjust_lr_fn']=='match_rms_adamw' for g in optimizer.optimizers[0].param_groups)
        logits,loss=model(tokens,targets);assert bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss))
        loss.backward()
        gradients={n:float(p.grad.norm()) for n,p in model.named_parameters() if '.mlp.' in n}
        assert len(gradients)==8 and all(v>0 for v in gradients.values())
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        before=model.transformer.h[0].mlp.linear_up.weight.detach().clone();optimizer.step()
        assert not torch.equal(before,model.transformer.h[0].mlp.linear_up.weight)
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters());optimizer.zero_grad()
        # Full-model chunked inference and causality on synthetic IDs, independently of corpus.
        model.eval();caches=[KVCache(candidate_cfg,torch.device('cpu'),batch_size=2,dtype=torch.float32,layer_index=i) for i in range(4)]
        with torch.no_grad():
            full=model(tokens)[0];chunks=[];start=0
            for count in (1,4,11):
                chunks.append(model(tokens[:,start:start+count],cache=caches,start_pos=start)[0]);start+=count
            cached=torch.cat(chunks,dim=1);max_error=float((full-cached).abs().max())
            torch.testing.assert_close(full,cached,atol=2e-5,rtol=2e-5)
            changed=tokens.clone();changed[:,8:]=(changed[:,8:]+123)%50257
            causal_error=float((full[:,:8]-model(changed)[0][:,:8]).abs().max());assert causal_error==0.
        bad_cache=[KVCache(control_cfg,torch.device('cpu'),batch_size=2,dtype=torch.float32,layer_index=i) for i in range(4)]
        cache_rejection=rejected(lambda:model(tokens[:,:1],cache=bad_cache,start_pos=0));assert all(x.position==0 for x in bad_cache)
        saved=dict(cfg=OmegaConf.to_container(candidate_cfg,resolve=True),model_state=model.state_dict(),block_normalization_signature=checkpoint_signature(model))
        validate_checkpoint(model,saved)
        bad=copy.copy(saved);bad['block_normalization_signature']=checkpoint_signature(control)
        signature_rejection=rejected(lambda:validate_checkpoint(model,bad))
        badcfg=copy.deepcopy(candidate_cfg.model);badcfg.tokenizer_architecture_policy=cfg.model.tokenizer_architecture_policy
        legacy_rejection=rejected(lambda:LegacyGPT(badcfg))
        model.transformer.h[0].mlp.activation=torch.nn.GELU()
        module_rejection=rejected(lambda:validate_architecture_model(model))
        model.transformer.h[0].mlp.activation=torch.nn.SiLU();validate_architecture_model(model)
        assert c.source_inventory()==original and sources()==current
        result=dict(status='full_model_cpu_prerequisite_passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),runner=reference(__file__),
            frozen_sources=original,new_sources=current,source_snapshot=str(SNAP.relative_to(c.ROOT)),
            control_initialization_and_logits_bitwise_equal=True,total_parameters=28492416,ffn_parameters=2064384,optimizer_groups=groups,
            loss_before_one_synthetic_step=float(loss.detach()),ffn_gradient_norms=gradients,synthetic_optimizer_step_passed=True,
            cache_chunk_max_abs_error=max_error,causality_max_abs_error=causal_error,
            rejected=dict(cross_arm_cache=cache_rejection,cross_arm_checkpoint_signature=signature_rejection,legacy_candidate=legacy_rejection,changed_activation=module_rejection),
            checkpoint_scope='In-memory signature/tensor inventory only; no file serialization or resume claim.',
            initialization='Ordinary bias-free Linear fan-in draws; control matches legacy bitwise. Candidate changes RNG draw shapes; paired seeds do not imply shared initial attention/readout values.',
            scientific_seeds_reserved=False,scientific_training_launched=False,corpus_read=False,holdout_read=False,checkpoint_io=False,
            remaining=['MPS numerical and cache parity','serialized raw/H200 and optimizer resume','worker/parent validation and inference integration','actual serialization/transient projection','scientific seed census and finite pilot/quality freezes'])
    result['guard']=reference(B/'model-integration001-guard.json');assert read(result['guard'])['status']=='finished'
    e.exclusive(B/'model-integration001.json',result)
    print(json.dumps({k:result[k] for k in ('status','total_parameters','optimizer_groups','cache_chunk_max_abs_error','causality_max_abs_error')}),flush=True)
