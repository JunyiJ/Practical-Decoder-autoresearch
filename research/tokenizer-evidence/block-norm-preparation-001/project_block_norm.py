"""Guarded read-only meta-shape feasibility; no proposed full-model validation."""
from datetime import datetime,timezone
import hashlib,json,math,shutil,inspect
from pathlib import Path
import torch
from omegaconf import OmegaConf
from src.research.protocol import ROOT,digest,file_digest
from src.research.runner import source_fingerprint
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
from src.models.gpt import GPT
from src.models.tokenizer_architecture import validate_architecture_model
from src.train.optimizers import build_optimizer

base=ROOT/'research/tokenizer-evidence/block-norm-preparation-001'
out=base/'shape-storage-review.json';guard=base/'shape-storage-guard.json'
assert not out.exists() and not guard.exists(),'No evidence overwrite'
recipe=ROOT/'research/tokenizer-evidence/mla-001/adopted-width256-mla-recipe.json'
with session_lock(GLOBAL_LOCK.parent):
    idle()
    with CpuPrerequisiteGuard(guard,ROOT,timeout_seconds=120):
        torch.set_num_threads(1);source_before=source_fingerprint()
        cfg=OmegaConf.create(json.loads(recipe.read_text())['config'])
        with torch.device('meta'):
            model=GPT(cfg.model);validate_architecture_model(model);optimizer=build_optimizer(model,cfg.training)
            params=dict(model.named_parameters());total=sum(p.numel() for p in params.values())
            assert total==28492416 and len(model.transformer.h)==4
            norms=[]
            for i,block in enumerate(model.transformer.h):
                for name in ('ln1','ln2'):
                    norm=getattr(block,name)
                    assert isinstance(norm,torch.nn.RMSNorm) and norm.eps is None
                    assert norm.normalized_shape==(256,) and sum(p.numel() for p in norm.parameters())==256
                    norms.append(f'transformer.h.{i}.{name}')
            epsilon=torch.finfo(torch.float32).eps
            candidate=torch.nn.LayerNorm(256,eps=epsilon,elementwise_affine=True,bias=True)
            default_layer=torch.nn.LayerNorm(256)
            assert candidate.eps==epsilon and default_layer.eps==1e-5
            extra=len(norms)*(sum(p.numel() for p in candidate.parameters())-256)
            assert extra==2048
            groups={type(child).__name__:dict(parameters=sum(p.numel() for g in child.param_groups for p in g['params']),
                tensors=sum(len(g['params']) for g in child.param_groups)) for child in optimizer.optimizers}
            assert set(groups)=={'Muon','AdamW'} and sum(v['parameters'] for v in groups.values())==total
            assert groups['Muon']['parameters']==2752512 and groups['AdamW']['parameters']==25739904
            buffers=sum(t.numel()*t.element_size() for n,t in model.named_buffers() if n in model.state_dict())
        arms={}
        for name,added,added_tensors in [('rms_incumbent',0,0),('layer_matched_epsilon',extra,8)]:
            count=total+added;muon=groups['Muon']['parameters'];adam=groups['AdamW']['parameters']+added
            steps=groups['AdamW']['tensors']+added_tensors
            opt=4*muon+8*adam+4*steps
            payload=12*count+opt+2*buffers
            allowance=math.ceil((24*count+2*buffers+2*1024**2)*1.15)
            arms[name]=dict(parameters=count,muon_parameters=muon,adam_parameters=adam,adam_tensors=steps,
                optimizer_tensor_bytes=opt,raw_h200_payload_bytes=payload,per_trajectory_allowance_bytes=allowance)
        # Independent delta algebra: model/raw EMA/export=12 bytes/newparam;
        # two Adam moments=8 bytes/newparam plus8 scalar steps at4 bytes each.
        assert arms['layer_matched_epsilon']['raw_h200_payload_bytes']-arms['rms_incumbent']['raw_h200_payload_bytes']==20*2048+4*8
        reserve=8*1024**3;metadata=256*1024**2;per=sum(a['per_trajectory_allowance_bytes'] for a in arms.values())
        phases=dict(pilots=per,lr_screen=3*per,fresh_confirmation=4*per)
        required=reserve+metadata+max(phases.values())
        minimum=reserve+4*sum(a['raw_h200_payload_bytes'] for a in arms.values())
        free=shutil.disk_usage(ROOT).free
        refs=['research/tokenizer-evidence/mla-001/adopted-width256-mla-recipe.json',
            'research/FFN_HIDDEN_NORM_RESULTS.md','research/QKNORM_QUALITY_RESULTS.md','research/RESIDUAL_GAIN_RESULTS.md',
            'src/models/blocks.py','src/models/tokenizer_architecture.py','src/models/gpt.py','src/train/optimizers.py',
            'src/utils/checkpoint.py','src/train/weight_averaging.py',
            'research/tokenizer-evidence/moe-preparation-001/ffn-storage-projection.json']
        normalization_file=Path(inspect.getfile(torch.nn.RMSNorm))
        result=dict(status='prospective_shape_accounting_not_launch_authorization',observed_at=datetime.now(timezone.utc).isoformat(),
            source_sha256=digest(source_before),source_refs={n:file_digest(ROOT/n) for n in refs},helper_sha256=file_digest(Path(__file__)),
            torch_version=torch.__version__,torch_norm_module_sha256=file_digest(normalization_file),
            baseline_validated_on_meta=True,candidate_full_model_validated=False,block_normalizer_paths=norms,
            epsilon=dict(incumbent_attribute=None,fp32_effective=epsilon,default_layer=default_layer.eps,
                candidate_explicit=epsilon,final_layer_norm_unchanged=True),
            baseline_optimizer_groups=groups,persistent_buffer_bytes=buffers,arms=arms,
            panel=dict(pilots=2,screens=6,fresh=6,selected_pairs=2,fresh_pairs=6,
                phase_allowance_bytes=phases,reserve_bytes=reserve,metadata_bytes=metadata,
                required_free_bytes=required,observed_free_bytes=free,shortfall_bytes=max(0,required-free),
                final_payload_plus_reserve_bytes=minimum,payload_only_shortfall_bytes=max(0,minimum-free),
                fits_projection=False if free<required else True),
            limitations=['Only original full model is validated. Candidate uses independently counted LayerNorm components; no unsupported full policy constructed.',
                'Payload excludes headers/RNG and transient audit overlap; it is a lower bound, not a launch budget.',
                'All optimizer moments charged; pilot/losing-screen cleanup credited only at audited boundaries.',
                'Existing selected/fresh/failure evidence remains protected; no deletion credit.',
                'Default-RMS epsilon inference is for explicit FP32 only; full forward/gradient/cache parity and dtype rejection remain prerequisites.'],
            training_launched=False,checkpoints_loaded=False,holdout_scored=False,deletion_performed=False)
        assert source_fingerprint()==source_before;idle()
    result['guard']=dict(path=str(guard.relative_to(ROOT)),sha256=file_digest(guard))
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'status':result['status'],'epsilon':result['epsilon'],'panel':result['panel']}))
