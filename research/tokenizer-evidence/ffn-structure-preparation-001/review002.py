"""Read-only direction/capacity review plus standalone ungated-SiLU oracle."""
import datetime,hashlib,json,math,shutil,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from research.ema_horizon_v1 import contract as c,execution as e
from research.block_norm_v1.fixture_dispatch import reference,read
from research.block_norm_v1.runtime.models.gpt import GPT
from research.block_norm_v1.runtime.models.block_norm import contract,validate_model
from research.block_norm_v1.runtime.train.optimizers import build_optimizer
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

B=c.ROOT/'research/tokenizer-evidence/ffn-structure-preparation-001'
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    e.idle();assert not (B/'review002.json').exists()
    with CpuPrerequisiteGuard(B/'review002-guard.json',c.ROOT,timeout_seconds=120,poll_seconds=.25):
        sources=c.source_inventory();prior=read(reference(e.EVIDENCE/'quality001-final-review.json'))
        assert prior['status']=='closed_not_adopted' and (e.ROOT/'STOP').is_file()
        seeds=dict(pilot=376901,screen=376902,fresh=[376911,376912,376913],evaluation=376903,bootstrap=376904)
        cfg=c.config(c.ROOT/'research_runs/tokenizer-ema-horizon-v1-shape-only/evidence/pilots/00','pilot',0,seeds)
        from omegaconf import OmegaConf
        cfg=OmegaConf.create(cfg)
        with torch.device('cpu'):
            model=GPT(cfg.model);validate_model(model,contract(cfg.model));optimizer=build_optimizer(model,cfg.training)
            parameters=sum(p.numel() for p in model.parameters());ffn=sum(p.numel() for n,p in model.named_parameters() if '.mlp.' in n)
            buffers=sum(t.numel()*t.element_size() for n,t in model.named_buffers() if n in model.state_dict())
            groups={}
            for child in optimizer.optimizers:
                groups[type(child).__name__]=dict(parameters=sum(p.numel() for g in child.param_groups for p in g['params']),
                    tensors=sum(len(g['params']) for g in child.param_groups))
            candidate=nn.Sequential(nn.Linear(256,1008,bias=False),nn.SiLU(),nn.Linear(1008,256,bias=False))
            candidate_count=sum(p.numel() for p in candidate.parameters())
        assert parameters==28492416 and ffn==4*516096 and candidate_count==516096
        assert 3*256*672==2*256*1008 and buffers==65536
        hidden=groups['Muon']['parameters'];adam=groups['AdamW']['parameters'];adam_tensors=groups['AdamW']['tensors']
        optbytes=4*hidden+8*adam+4*adam_tensors
        payload=12*parameters+optbytes+2*buffers
        allowance=math.ceil((24*parameters+2*buffers+2*1024**2)*1.15)
        reserve=8*1024**3;metadata=256*1024**2
        phases=dict(pilots=2*allowance,screen=6*allowance,fresh_with_two_selected_pairs=8*allowance)
        required=reserve+metadata+max(phases.values());free=shutil.disk_usage(c.ROOT).free
        # Separate NumPy analytic derivative oracle: not an autograd copy of the module.
        errors=[]
        for dim,width,scale in [(8,31,.1),(8,31,3.),(256,1008,.1)]:
            rng=np.random.Generator(np.random.PCG64(376905+dim+int(scale*10)))
            x=rng.normal(size=(11,dim))*scale;up=rng.normal(size=(width,dim))/math.sqrt(dim)
            down=rng.normal(size=(dim,width))/math.sqrt(width);grad=rng.normal(size=(11,dim))
            module=nn.Sequential(nn.Linear(dim,width,bias=False),nn.SiLU(),nn.Linear(width,dim,bias=False)).double()
            with torch.no_grad():module[0].weight.copy_(torch.from_numpy(up));module[2].weight.copy_(torch.from_numpy(down))
            tx=torch.tensor(x,requires_grad=True);y=module(tx);y.backward(torch.from_numpy(grad))
            z=x@up.T;sigma=1/(1+np.exp(-z));activation=z*sigma
            expected=activation@down.T;dz=(grad@down)*(sigma+z*sigma*(1-sigma))
            comparisons=dict(output=(y.detach().numpy(),expected),input_grad=(tx.grad.numpy(),dz@up),
                up_grad=(module[0].weight.grad.numpy(),dz.T@x),down_grad=(module[2].weight.grad.numpy(),grad.T@activation))
            row=dict(dim=dim,hidden=width,input_scale=scale)
            for name,(actual,wanted) in comparisons.items():
                assert np.allclose(actual,wanted,rtol=1e-10,atol=1e-10),name
                row[name+'_max_abs_error']=float(np.max(np.abs(actual-wanted)))
            errors.append(row)
        # Confirm current full-model policy still rejects this unsupported design.
        bad=OmegaConf.create(OmegaConf.to_container(cfg.model,resolve=True));bad.ffn='silu';bad.ffn_width_policy='ungated_silu_63_16'
        try:
            with torch.device('cpu'):GPT(bad)
        except (ValueError,TypeError) as exc:rejection=str(exc)
        else:raise AssertionError('Existing policy unexpectedly accepted new FFN')
        refs=['research/RESIDUAL_GAIN_RESULTS.md','research/PARALLEL_BLOCK_RESULTS.md','research/TOKENIZER_MOE_PREPARATION.md',
            'research/TOKENIZER_ARCHITECTURE_ROADMAP.md','research/TOKENIZER_CHECKPOINT_RETENTION.md',
            'research/tokenizer-evidence/moe-preparation-001/ffn-storage-projection.json',
            'research/tokenizer-evidence/ema-horizon-preparation-001/quality001-final-review.json']
        assert c.source_inventory()==sources
        result=dict(status='ungated_silu_direction_selected_capacity_blocked',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),runner=reference(__file__),
            references={p:hashlib.sha256((c.ROOT/p).read_bytes()).hexdigest() for p in refs},sources=sources,
            hypothesis='Can ungated SiLU with hidden1008 improve over SwiGLU hidden672 at identical matrix-parameter budget on the adopted width256 MLA backbone?',
            equations=dict(control='Wd(silu(Wg x) * Wu x)',candidate='Wd(silu(Wu x))'),
            shapes=dict(control=[[672,256],[672,256],[256,672]],candidate=[[1008,256],[256,1008]],
                bias=False,ffn_parameters_per_layer=516096,layers=4,total_parameters=parameters),
            interpretation='Removing the multiplicative gate reallocates matrices to a1.5x wider hidden representation. SiLU stays fixed. This jointly changes factorization/hidden width and optimizer geometry; not an isolated gating mechanism or equal wall-time claim.',
            alternatives=dict(residual_scaling='Deferred: historical static0.1 and learned0.1 worsen every seed at widths64/256; no new MLA-specific instability or benefit established.',
                moe='Deferred: previous full three-arm projection18,782,075,912B exceeds current capacity; retain wider-dense diagnostic control.',
                ema='Closed: small H100 gain misses frozen practical threshold; no new half-life sweep.',
                gate_nonlinearity='Closed SiLU versus exact-GELU gate study; this proposal retains SiLU and changes FFN structure.'),
            proposed_panel=dict(pilots=2,screen_rates=[.000625,.00125,.0025],screen_runs=6,fresh_pairs=3,
                quality_runs_total=12,steps=1200,context=128,batch=8,h200_start=800,h200_interval=10,
                selection='Each arm minimum H200 full-validation CE; lowerLR tie break; no boundary grid extension.',
                adoption='Candidate mean paired H200 CE delta below-.02, each freshseed below0, all simultaneous text uppers below0 for2048/4096/8192;2000 paired PCG64 draws. No cost-only route.',
                seeds_reserved=False,launch_frozen=False),
            capacity=dict(baseline_optimizer_groups=groups,candidate_muon_tensor_count=groups['Muon']['tensors']-4,
                optimizer_tensor_bytes=optbytes,persistent_buffer_bytes=buffers,raw_h200_payload_bytes=payload,
                per_trajectory_allowance_bytes=allowance,phase_allowance_bytes=phases,reserve_bytes=reserve,metadata_bytes=metadata,
                required_free_bytes=required,observed_free_bytes=free,shortfall_bytes=max(0,required-free),
                selected_plus_fresh_payload_lower_bound=reserve+8*payload,
                six_fresh_only_payload_lower_bound=reserve+6*payload,cleanup_credit_bytes=0,
                assumptions='Prospective successful-pilot and losing-screen deletion only after complete audits. Keep2selected and6fresh pairs. Existing protected files already consume observed storage. Exact new model/serialization/transient audit must revalidate before freeze.'),
            component_oracle=dict(status='passed',dtype='float64',device='cpu',cases=errors,scope='Standalone two-linear-layer SiLU values and analytic gradients only; no full model, MPS, checkpoint/resume or quality claim.'),
            old_full_policy_rejection=rejection,full_candidate_validated=False,
            required_integration=['New narrow FFN policy outside399 frozen sources; preserve all old errors/signatures.',
                'Exact module/optimizer layout and shape-dependent Muon scaling, model/checkpoint/resume and H200 contracts.',
                'Actual full-model forward/gradient/MPS/cache parity and serialization resource audits; do not spend full fixture disk before capacity fit.',
                'Complete retention-aware high-water budget, fresh seed census, finite plan and guarded serial pilots before quality.'],
            training_launched=False,checkpoint_io=False,holdout_read=False,deletion_performed=False)
    result['guard']=reference(B/'review002-guard.json');assert read(result['guard'])['status']=='finished'
    e.exclusive(B/'review002.json',result)
    print(json.dumps(dict(status=result['status'],capacity=result['capacity'],oracle=result['component_oracle'])))
