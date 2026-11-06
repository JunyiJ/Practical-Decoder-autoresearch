"""Bounded CPU inventory and hypothetical component arithmetic, never training.

The incumbent uses its real validators and optimizer. Candidate components do
not constitute supported full models. No validator is patched or disabled.
"""
import copy
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from research.final_ln_scale_model_v1.runtime.models.gpt import GPT
from research.final_ln_scale_model_v1.runtime.models.mlp import SwiGLU
from research.final_ln_scale_model_v1.runtime.models.block_norm import contract
from research.final_ln_scale_model_v1.runtime.attention.cache import build_caches
from research.final_ln_scale_model_v1.integration import build_optimizer
from research.final_ln_scale_quality_v3.campaign_contract import source_inventory
from research.final_ln_scale_quality_v3.support import idle
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock

ROOT = Path(__file__).resolve().parents[3]
B = Path(__file__).resolve().parent
BASE = ROOT / 'research/tokenizer-evidence/final-ln-scale-preparation-001'
CFG = BASE / 'quality002-slot03-evidence001/config.json'
OUT = B / 'geometry001.json'
GUARD = B / 'geometry001-guard.json'
V, T = 50257, 128


def ref(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def count(module):
    return sum(p.numel() for p in module.parameters())


def projection(layers, d, kind='dense'):
    """Leaf tensor shapes on meta; not a constructed or validated transformer."""
    assert d % 64 == 0 and d % 32 == 0
    rank, hidden = d // 4, 21 * d // 8
    with torch.device('meta'):
        attn = torch.nn.ModuleDict({name: torch.nn.Linear(a, b, bias=True)
            for name, a, b in [('dkv', d, rank), ('uk', rank, d), ('uv', rank, d),
                               ('dq', d, rank), ('uq', rank, d), ('qr', rank, d),
                               ('kr', d, 32), ('out', d, d)]})
        norms = torch.nn.ModuleList([torch.nn.RMSNorm(d), torch.nn.RMSNorm(d)])
        if kind == 'moe_top2of4':
            ffns = torch.nn.ModuleList([SwiGLU(d, hidden // 2, 0.) for _ in range(4)])
            router = torch.nn.Linear(d, 4, bias=False)
            router_params = count(router)
            active_ffn = 2 * count(ffns[0])
        else:
            ffns = torch.nn.ModuleList([SwiGLU(d, hidden * (2 if kind == 'wide_dense' else 1), 0.)])
            router_params = 0
            active_ffn = count(ffns)
        final = torch.nn.LayerNorm(d, eps=1e-5)
        embedding = torch.nn.Embedding(V, d)
        readout = torch.nn.Linear(d, V, bias=False)
    attention_weight = sum(p.numel() for p in attn.parameters() if p.ndim == 2)
    attention_bias = count(attn) - attention_weight
    assert attention_weight == 6 * d * rank + 32 * d + d * d
    assert attention_bias == 2 * rank + 5 * d + 32
    assert count(norms) == 2 * d and count(final) == 2 * d
    ffn_params = count(ffns)
    block = count(attn) + count(norms) + ffn_params + router_params
    tables = count(embedding) + count(readout)
    total = tables + count(final) + layers * block
    muon = layers * (attention_weight + ffn_params)
    adam = total - muon
    muon_groups = layers * (8 + 3 * len(ffns))
    adam_groups = layers * (8 + 2 + (1 if router_params else 0)) + 4
    # The counterfactual charges all expert momentum and moments, even if idle.
    optimizer = 4 * muon + 8 * adam + 4 * adam_groups
    buffers = layers * 2 * T * (32 // 2) * 4
    active_matmuls = layers * (attention_weight + active_ffn + router_params)
    linear_flops = 2 * (active_matmuls + V * d)
    attention_flops = 6 * T * d * layers
    return dict(layers=layers, width=d, heads=d // 32, latent_rank=rank,
        ffn_width=hidden, kind=kind, total_parameters=total,
        table_parameters=tables, nonembedding_parameters=total-tables,
        muon_parameters=muon, adam_parameters=adam,
        muon_groups=muon_groups, adam_groups=adam_groups,
        parameter_tensors=muon_groups+adam_groups,
        optimizer_state_payload_bytes=optimizer, persistent_buffer_bytes=buffers,
        raw_plus_h200_tensor_payload_bytes=12 * total + optimizer + 2 * buffers,
        model_plus_ema_parameter_bytes=8 * total,
        estimated_cache_bytes_per_item=4 * layers * T * (rank + 32),
        forward_dense_linear_flops_per_token=linear_flops,
        forward_attention_matmul_flops_per_token=attention_flops,
        forward_matmul_flops_per_token=linear_flops + attention_flops,
        inherited_hidden_lr_if_width_rule_kept=.00125 / math.sqrt(d / 64),
        inherited_readout_multiplier_if_width_rule_kept=64 / d,
        full_model_supported=(layers == 4 and d == 256 and kind == 'dense'),
        scope='Candidate leaf shapes and algebraic projection only; no model/optimizer/cache correctness or runtime fit claim.')


torch.set_num_threads(4)
assert not OUT.exists() and not GUARD.exists()
with session_lock(GLOBAL_LOCK.parent):
    idle()
    with CpuPrerequisiteGuard(GUARD, ROOT, timeout_seconds=120):
        frozen = json.loads((BASE / 'quality002-plan.json').read_text())['sources']
        assert source_inventory() == frozen and len(frozen) == 918
        cfg = OmegaConf.create(json.loads(CFG.read_text()))
        before = torch.get_rng_state().clone()
        # Real CPU storage is required: the existing anti-alias check correctly
        # rejects full meta construction where storage data_ptr values are zero.
        with torch.random.fork_rng(devices=[]):
            model = GPT(cfg.model)
            opt = build_optimizer(model, cfg)
            params = dict(model.named_parameters())
            actual = dict(total_parameters=sum(p.numel() for p in params.values()),
                parameter_tensors=len(params), groups={type(child).__name__:dict(
                    groups=len(child.param_groups),
                    parameters=sum(p.numel() for g in child.param_groups for p in g['params']),
                    lr=sorted({g['lr'] for g in child.param_groups})) for child in opt.optimizers})
            grouped = [p for child in opt.optimizers for g in child.param_groups for p in g['params']]
            assert len(grouped) == len({id(p) for p in grouped}) == len(params)
            assert {id(p) for p in grouped} == {id(p) for p in params.values()}
            assert all(not child.state for child in opt.optimizers)
            actual['persistent_buffer_bytes'] = sum(v.numel() * v.element_size()
                for n, v in model.state_dict().items() if n not in params and isinstance(v, torch.Tensor))
            actual['layer0_shapes'] = {n:list(p.shape) for n,p in params.items() if n.startswith('transformer.h.0.')}
            actual['cache_bytes_per_item'] = sum(v.numel()*v.element_size()
                for c in build_caches(cfg, 'cpu', batch_size=1, dtype=torch.float32) for v in c.buffers())
            source_files = {inspect.getsourcefile(x) for x in (GPT, SwiGLU, contract, build_optimizer, build_caches)}
            del grouped, params, opt, model
            gc.collect()
        assert torch.equal(before, torch.get_rng_state())
        baseline = projection(4, 256)
        assert actual['total_parameters'] == baseline['total_parameters'] == 28492416
        assert actual['parameter_tensors'] == baseline['parameter_tensors'] == 88
        assert actual['persistent_buffer_bytes'] == baseline['persistent_buffer_bytes']
        assert actual['cache_bytes_per_item'] == baseline['estimated_cache_bytes_per_item'] == 196608
        for child, key in [('Muon','muon'), ('AdamW','adam')]:
            assert actual['groups'][child]['groups'] == baseline[key+'_groups']
            assert actual['groups'][child]['parameters'] == baseline[key+'_parameters']
        rejections = {}
        for name, updates in [('seven_layers', dict(n_layers=7)), ('width192',dict(dim=192,n_heads=6)), ('moe',dict(mlp_type='moe'))]:
            changed = copy.deepcopy(cfg.model)
            for k,v in updates.items():changed[k]=v
            try:contract(changed)
            except ValueError as exc:rejections[name]=str(exc)
            else:raise AssertionError('Existing contract unexpectedly accepts '+name)
        shapes = [projection(l, d) for l in range(2,17) for d in range(128,385,64)]
        for row in shapes:
            row['body_parameter_ratio'] = row['nonembedding_parameters'] / baseline['nonembedding_parameters']
            row['total_parameter_ratio'] = row['total_parameters'] / baseline['total_parameters']
            row['cache_ratio'] = row['estimated_cache_bytes_per_item'] / baseline['estimated_cache_bytes_per_item']
            row['forward_matmul_ratio'] = row['forward_matmul_flops_per_token'] / baseline['forward_matmul_flops_per_token']
        shortlist = [x for x in shapes if abs(x['body_parameter_ratio'] - 1) <= .02]
        assert any(x['layers']==7 and x['width']==192 for x in shortlist)
        assert source_inventory() == frozen
        result = dict(status='cpu_geometry_and_component_projection_passed',source=ref(__file__),config=ref(CFG),
            actual_incumbent=actual, baseline=baseline, body_matched_shortlist=shortlist,
            enumeration=dict(depths=list(range(2,17)),widths=list(range(128,385,64)),body_tolerance=.02,
                scope='Deterministic feasibility domain, up to4x incumbent depth; not a scientific screening grid.'),
            dense_moe_projections=[projection(4,256,k) for k in ('dense','wide_dense','moe_top2of4')],
            rejected_by_original_contract=rejections,source_files=[ref(p) for p in sorted(source_files)],
            frozen_source_count=918,frozen_sources_unchanged=True,cpu_rng_restored=True,
            initialized_incumbent_counting_model=True,optimizer_updates=0,forward_passes=0,
            checkpoint_loads=0,checkpoint_writes=0,mps_used=False,scientific_seeds_assigned=False,
            full_candidate_models_constructed=False,full_panel_capacity_frozen=False,
            limitations=['Matmul FLOPs count multiply/add as2 and full dense causal attention; softmax,norm,activation,routing dispatch,backward,optimizer,embedding reduction and I/O excluded.',
                'Tensor-payload bytes omit serialization headers,RNG,clipping history and metadata; not a complete retained-output high-water or launch gate.',
                'Width changes vocabulary table capacity,latent rank,head count and readout/hidden width rules; this cannot isolate pure depth.',
                'No new depth scaling or LR transfer is justified by an inherited width formula.',
                'The original full-model meta attempt failed the normalizer storage anti-alias check; no validator was bypassed. CPU construction was used instead.'])
    result['guard']=ref(GUARD)
    assert json.loads(GUARD.read_text())['status']=='finished'
    raw=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode();assert len(raw)<512*1024
    with OUT.open('xb') as f:f.write(raw)
print(json.dumps(dict(status=result['status'],baseline=baseline,shortlist=shortlist,rejections=rejections)))
