"""Full-model separate-RMS residual topology, groups, numerical and payload checks."""
import copy, gc, hashlib, json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.parallel_residual_v1.integration import identity, build_optimizer, load_model_weights
from research.parallel_residual_v1.runtime.models.gpt import GPT
from research.parallel_residual_v1.runtime.models.block_norm import PARALLEL_POLICY
from research.parallel_residual_v1.runtime.attention.cache import build_caches
from research.parallel_residual_v1.runtime.utils.checkpoint import save_checkpoint
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.local_mix_quality_v3.support import idle
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
    raise AssertionError('Invalid topology identity accepted')

plan=json.loads((B/'model001-plan.json').read_text());verify(plan)
assert not any((B/n).exists() for n in ('model001.json','model001-guard.json','model001-failure.json'))
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    idle()
    try:
        with StrictMpsGuard(B/'model001-guard.json',ROOT):
            base=OmegaConf.create(json.loads((ROOT/plan['config_path']).read_text()))
            base.model.tokenizer_architecture_policy=PARALLEL_POLICY
            del base.model.ffn_input_mixing
            base.training.checkpoint_dir=str(B/'never-written')
            base.training.metrics_file=str(B/'never-written-metrics.jsonl')
            base.training.summary_file=str(B/'never-written-summary.json')
            base.model.residual_topology_width_policy='mla4_separate_rms_residual_v1'
            base.model.residual_topology_kind='serial'
            initial=None; baseline_groups=None; initial_rng=None; initial_logits=None; rows=[]
            tokens=(torch.arange(18).reshape(2,9)%5+23).long();targets=tokens+1
            for kind in ('serial','parallel_separate_norms'):
                cfg=copy.deepcopy(base);cfg.training.weight_decay=.00006;cfg.model.residual_topology_kind=kind;contract=identity(cfg)
                torch.manual_seed(91061);cpu=GPT(cfg.model);opt=build_optimizer(cpu,cfg)
                digest=state_digest(cpu);rng=torch.get_rng_state().clone();gg=groups(opt,cpu)
                assert sum(p.numel() for p in cpu.parameters())==28492416
                assert len(list(cpu.parameters()))==88
                with torch.no_grad(): logits=cpu(tokens)[0].clone()
                if initial is None:
                    initial=digest;initial_rng=rng;initial_logits=logits;baseline_groups=gg
                    from research.local_mix_v1.runtime.models.gpt import GPT as Incumbent
                    from research.local_mix_v1.runtime.models.block_norm import GROUPED_POLICY
                    old=copy.deepcopy(cfg.model);old.tokenizer_architecture_policy=GROUPED_POLICY
                    del old.residual_topology_kind;del old.residual_topology_width_policy
                    torch.manual_seed(91061);incumbent=Incumbent(old)
                    assert state_digest(incumbent)==digest and torch.equal(torch.get_rng_state(),rng)
                    with torch.no_grad():assert torch.equal(incumbent(tokens)[0],logits)
                    del incumbent
                    from research.local_mix_v1.runtime.models.block_norm import contract as old_contract
                    rejected(lambda:old_contract(cfg.model))
                else:
                    assert digest==initial and torch.equal(rng,initial_rng)
                    assert not torch.equal(logits,initial_logits) and gg==baseline_groups
                gpu=copy.deepcopy(cpu).to('mps');gopt=build_optimizer(gpu,cfg)
                assert groups(gopt,gpu)==gg
                a,loss=cpu(tokens,targets);b,gloss=gpu(tokens.to('mps'),targets.to('mps'))
                row=dict(kind=kind,identity=contract,initial_state_sha256=digest,logits_max_abs=close(a,b),loss_abs=close(loss,gloss,1e-5,1e-4))
                loss.backward();gloss.backward()
                row['gradient_max_abs']=max(close(p.grad,g.grad,3e-3,3e-5) for p,g in zip(cpu.parameters(),gpu.parameters()))
                torch.nn.utils.clip_grad_norm_(cpu.parameters(),1.,error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(gpu.parameters(),1.,error_if_nonfinite=True)
                opt.step();gopt.step();torch.mps.synchronize()
                assert all(bool(torch.isfinite(p).all()) for m in (cpu,gpu) for p in m.parameters())
                cpu.zero_grad(set_to_none=True);gpu.zero_grad(set_to_none=True)
                payload={}
                save_checkpoint(B/'not-written.pt',cpu,opt,step=1,cfg=cfg,payload_writer=lambda path,value:payload.update(value))
                load_model_weights(cpu,payload,cfg)
                before=state_digest(cpu);other=copy.deepcopy(cfg);other.model.residual_topology_kind='serial' if kind!='serial' else 'parallel_separate_norms'
                row['cross_arm_rejected']=rejected(lambda:load_model_weights(cpu,payload,other))
                assert state_digest(cpu)==before
                tampered=dict(payload);tampered['optimizer_state']=dict(payload['optimizer_state']);tampered['optimizer_state']['independent_decay']=0.
                row['optimizer_decay_tamper_rejected']=rejected(lambda:load_model_weights(cpu,tampered,cfg,for_resume=True))
                assert state_digest(cpu)==before
                from research.parallel_residual_v1.runtime.models.tokenizer_architecture import validate_architecture_model
                original=cpu.transformer.h[0].residual_topology_kind
                cpu.transformer.h[0].residual_topology_kind=other.model.residual_topology_kind
                row['mutated_block_rejected']=rejected(lambda:validate_architecture_model(cpu))
                cpu.transformer.h[0].residual_topology_kind=original
                validate_architecture_model(cpu)
                with torch.no_grad():
                    x=tokens.to('mps');full=gpu(x)[0];cache=build_caches(cfg,'mps',batch_size=2,dtype=torch.float32)
                    parts=[];pos=0
                    for count in (3,1,5):
                        parts.append(gpu(x[:,pos:pos+count],cache=cache,start_pos=pos)[0]);pos+=count
                    foreign=build_caches(other,'mps',batch_size=2,dtype=torch.float32)
                    row['foreign_cache_rejected']=rejected(lambda:gpu(x,cache=foreign,start_pos=0))
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
            for value in (True,'shared','parallel'):
                bad=copy.deepcopy(base);bad.model.residual_topology_kind=value
                invalid.append(dict(value=value,error=rejected(lambda:identity(bad))))
            verify(plan)
            result=dict(status='passed',rows=rows,invalid_topology_rejected=invalid,optimizer_groups=baseline_groups,shared_initialization_rng_exact=True, serial_incumbent_logits_exact=True, topology_logits_differ=True,
                        parameter_count=28492416,ema_tensor_count=88,sources_unchanged=len(plan['sources']),corpus_read=False,scientific_training=False,
                        checkpoint_io=False,scope='Real CPU/MPS one-update model and in-memory raw payload only. Serialized raw/EMA/RNG resume, final-horizon H200 and actual worker/parent validation still required.')
        save(B/'model001.json',result);print(json.dumps(dict(status='passed',rows=rows)))
    except BaseException as exc:
        save(B/'model001-failure.json',dict(status='failed_closed',error=repr(exc),no_retry=True))
        raise
