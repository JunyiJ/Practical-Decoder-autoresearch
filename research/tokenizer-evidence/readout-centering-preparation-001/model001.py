"""Full-model readout equations, groups, numerical and readout payload checks."""
import copy, gc, hashlib, json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.gradient_clip_model_v1.clipping import clip_gradients
from research.readout_centering_model_v1.integration import identity, build_optimizer, load_model_weights
from research.readout_centering_model_v1.runtime.models.gpt import GPT
from research.readout_centering_model_v1.runtime.models.block_norm import READOUT_POLICY
from research.readout_centering_model_v1.runtime.attention.cache import build_caches
from research.readout_centering_model_v1.runtime.utils.checkpoint import save_checkpoint
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.gradient_clip_quality_v2.quality_execution import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock

B = Path(__file__).resolve().parent
ROOT = B.parents[2]
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def save(p, value):
    with p.open('x') as f: json.dump(value, f, indent=2); f.write('\n')
def verify(plan):
    for n, h in plan['sources'].items(): assert sha(ROOT/n) == h, n
def state_digest(model):
    h = hashlib.sha256()
    for n, p in model.state_dict().items():
        h.update(n.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def groups(opt, model):
    names = {id(p):n for n,p in model.named_parameters()}
    actual = [p for g in opt.param_groups for p in g['params']]
    assert len(actual) == len(names) == len({id(p) for p in actual})
    assert {id(p) for p in actual} == set(names)
    return [dict(child=type(child).__name__, groups=[dict(names=[names[id(p)] for p in g['params']], **{k:v for k,v in g.items() if k!='params'}) for g in child.param_groups]) for child in opt.optimizers]
def close(a,b,rtol=3e-4,atol=3e-5):
    a=a.detach().cpu(); b=b.detach().cpu()
    torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
    return float((a-b).abs().max())
def rejected(fn):
    try: fn()
    except (ValueError,TypeError) as exc: return str(exc)
    raise AssertionError('Invalid readout identity accepted')

plan=json.loads((B/'model001-plan.json').read_text());verify(plan)
assert not any((B/n).exists() for n in ('model001.json','model001-guard.json','model001-failure.json'))
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    idle()
    try:
        with StrictMpsGuard(B/'model001-guard.json',ROOT):
            base=OmegaConf.create(json.loads((ROOT/plan['config_path']).read_text()))
            base.model.tokenizer_architecture_policy=READOUT_POLICY
            base.training.checkpoint_dir=str(B/'never-written')
            base.training.metrics_file=str(B/'never-written-metrics.jsonl')
            base.training.summary_file=str(B/'never-written-summary.json')
            base.model.residual_gain_width_policy='mla4_fixed_complete_branch_gain_v1'
            base.model.residual_gain_kind='unit'
            base.model.readout_norm_width_policy='mla4_readout_centering_affine_v1'
            base.model.readout_norm_epsilon=1e-5
            initial=None; baseline_groups=None; initial_rng=None; initial_logits=None; rows=[]
            tokens=(torch.arange(18).reshape(2,9)%5+23).long();targets=tokens+1
            for kind in ('layer_affine','rms_affine_bias'):
                threshold=1.
                cfg=copy.deepcopy(base);cfg.model.readout_norm_kind=kind;cfg.training.weight_decay=.00006;cfg.training.grad_clip=1.;contract=identity(cfg)
                torch.manual_seed(91461);cpu=GPT(cfg.model);opt=build_optimizer(cpu,cfg)
                digest=state_digest(cpu);rng=torch.get_rng_state().clone();gg=groups(opt,cpu)
                assert sum(p.numel() for p in cpu.parameters())==28492416
                assert len(list(cpu.parameters()))==88
                with torch.no_grad(): logits=cpu(tokens)[0].clone()
                if initial is None:
                    initial=digest;initial_rng=rng;initial_logits=logits;baseline_groups=gg
                    from research.branch_scale_model_v1.runtime.models.gpt import GPT as Incumbent
                    from research.branch_scale_model_v1.runtime.models.block_norm import BRANCH_POLICY, contract as old_contract
                    old=copy.deepcopy(cfg.model);old.tokenizer_architecture_policy=BRANCH_POLICY
                    for field in ('readout_norm_kind','readout_norm_width_policy','readout_norm_epsilon'):del old[field]
                    torch.manual_seed(91461);incumbent=Incumbent(old)
                    assert state_digest(incumbent)==digest and torch.equal(torch.get_rng_state(),rng)
                    with torch.no_grad():assert torch.equal(incumbent(tokens)[0],logits)
                    del incumbent
                    rejected(lambda:old_contract(cfg.model))
                    from research.block_norm_v1.runtime.models.readout_norm import readout_norm_contract as old_readout
                    rejected(lambda:old_readout(cfg.model))
                else:
                    assert digest==initial and torch.equal(rng,initial_rng) and gg==baseline_groups
                    assert float((logits-initial_logits).abs().max())>1e-4
                # Check actual full-model readout input against explicit equations.
                captured=[]
                handle=cpu.transformer.ln_f.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
                with torch.no_grad():
                    observed=cpu(tokens)[0];h=captured[0];norm=cpu.transformer.ln_f
                    z=h-h.mean(-1,keepdim=True) if kind=='layer_affine' else h
                    expected=(z/(z.square().mean(-1,keepdim=True)+1e-5).sqrt())*norm.weight+norm.bias
                    expected=cpu.lm_head(expected)/cpu.width_mult
                    readout_error=close(observed,expected,3e-5,3e-5)
                handle.remove();del captured,h,z,norm,expected,observed
                # Independent explicit complete-branch equation on each actual MLA block.
                with torch.no_grad():
                    hidden=torch.sin(torch.arange(2*9*256).reshape(2,9,256).float()/97)
                    for block in cpu.transformer.h:
                        attn,_=block.attn(block.ln1(hidden),freq_cos=block.freq_cos,freq_sin=block.freq_sin)
                        gain=1.
                        u=hidden+gain*attn; reference=u+gain*block.mlp(block.ln2(u))
                        actual,_=block(hidden)
                        close(actual,reference,0,0)
                        hidden=reference
                gpu=copy.deepcopy(cpu).to('mps');gopt=build_optimizer(gpu,cfg)
                assert groups(gopt,gpu)==gg
                a,loss=cpu(tokens,targets);b,gloss=gpu(tokens.to('mps'),targets.to('mps'))
                row=dict(kind=kind,identity=contract,initial_state_sha256=digest,logits_max_abs=close(a,b),loss_abs=close(loss,gloss,1e-5,1e-4))
                loss.backward();gloss.backward()
                row['gradient_max_abs']=max(close(p.grad,g.grad,3e-3,3e-5) for p,g in zip(cpu.parameters(),gpu.parameters()))
                row['explicit_readout_max_abs']=readout_error
                for model in (cpu,gpu):clip_gradients(model.parameters(),1.)
                opt.step();gopt.step();torch.mps.synchronize()
                assert all(bool(torch.isfinite(p).all()) for m in (cpu,gpu) for p in m.parameters())
                cpu.zero_grad(set_to_none=True);gpu.zero_grad(set_to_none=True)
                payload={}
                save_checkpoint(B/'not-written.pt',cpu,opt,step=1,cfg=cfg,payload_writer=lambda path,value:payload.update(value))
                load_model_weights(cpu,payload,cfg)
                before=state_digest(cpu);other=copy.deepcopy(cfg);other.model.readout_norm_kind='rms_affine_bias' if kind=='layer_affine' else 'layer_affine'
                row['cross_arm_rejected']=rejected(lambda:load_model_weights(cpu,payload,other))
                assert state_digest(cpu)==before
                tampered=dict(payload);tampered['optimizer_state']=dict(payload['optimizer_state']);tampered['optimizer_state']['independent_decay']=0.
                row['optimizer_decay_tamper_rejected']=rejected(lambda:load_model_weights(cpu,tampered,cfg,for_resume=True))
                assert state_digest(cpu)==before
                from research.readout_centering_model_v1.runtime.models.tokenizer_architecture import validate_architecture_model
                original=cpu.transformer.ln_f.eps
                cpu.transformer.ln_f.eps=1e-4
                row['mutated_epsilon_rejected']=rejected(lambda:validate_architecture_model(cpu))
                cpu.transformer.ln_f.eps=original
                original_module=cpu.transformer.ln_f
                cpu.transformer.ln_f=torch.nn.RMSNorm(256,eps=1e-5)
                row['bias_free_norm_rejected']=rejected(lambda:validate_architecture_model(cpu))
                cpu.transformer.ln_f=original_module
                validate_architecture_model(cpu)
                with torch.no_grad():
                    x=tokens.to('mps');full=gpu(x)[0];cache=build_caches(cfg,'mps',batch_size=2,dtype=torch.float32)
                    parts=[];pos=0
                    for count in (3,1,5):
                        parts.append(gpu(x[:,pos:pos+count],cache=cache,start_pos=pos)[0]);pos+=count
                    foreign=build_caches(other,'mps',batch_size=2,dtype=torch.float32)
                    row['foreign_readout_cache_rejected']=rejected(lambda:gpu(x,cache=foreign,start_pos=0))
                    assert all(cell.position==0 for cell in foreign)
                    del foreign
                    row['cache_max_abs']=close(torch.cat(parts,1),full)
                    for cell in cache:cell.reset()
                    row['reset_max_abs']=close(gpu(x,cache=cache,start_pos=0)[0],full)
                    changed=x.clone();changed[:,4:]+=3
                    row['causality_max_abs']=close(gpu(changed)[0][:,:4],full[:,:4],0,0)
                    row['cache_bytes']=sum(v.untyped_storage().nbytes() for cell in cache for v in cell.buffers())
                    assert row['cache_bytes']==2*196608
                rows.append(row)
                del cpu,gpu,opt,gopt,payload,tampered,a,b,loss,gloss,logits,full,cache,parts
                gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
            invalid=[]
            base.model.readout_norm_kind='layer_affine'
            for field,value in [('readout_norm_kind','rms'),('readout_norm_epsilon',1e-4),('readout_norm_epsilon',True),('readout_norm_width_policy','mha4_readout_norm'),('residual_gain_kind','fixed_half')]:
                bad=copy.deepcopy(base);bad.model[field]=value
                invalid.append(dict(field=field,value=repr(value),error=rejected(lambda:identity(bad))))
            verify(plan)
            result=dict(status='passed',rows=rows,invalid_readout_policies_rejected=invalid,optimizer_groups=baseline_groups,shared_initialization_rng_exact=True, layernorm_incumbent_logits_exact=True, candidate_function_distinct=True,
                        parameter_count=28492416,ema_tensor_count=88,sources_unchanged=len(plan['sources']),corpus_read=False,scientific_training=False,
                        checkpoint_io=False,scope='Real CPU/MPS one-update model and in-memory raw payload only. Serialized raw/EMA/RNG resume, final-horizon H200 and actual worker/parent validation still required.')
        save(B/'model001.json',result);print(json.dumps(dict(status='passed',rows=rows)))
    except BaseException as exc:
        save(B/'model001-failure.json',dict(status='failed_closed',error=repr(exc),no_retry=True))
        raise
