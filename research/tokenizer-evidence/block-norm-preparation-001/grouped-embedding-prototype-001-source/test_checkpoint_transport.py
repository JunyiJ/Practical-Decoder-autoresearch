"""Fresh synthetic serialized artifacts; no corpus or trained weights accessed."""
import copy
import gc
import hashlib
import os
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from research.block_norm_v1.test_model_policy import config
from research.block_norm_v1.runtime.models.gpt import GPT
from research.block_norm_v1.runtime.models.block_norm import CHECKPOINT_KEY, PLACEMENTS, checkpoint_signature
from research.block_norm_v1.runtime.train.optimizers import build_optimizer
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages, export_averaged_checkpoint
from research.block_norm_v1.runtime.utils.checkpoint import save_checkpoint, read_checkpoint, load_model_weights, load_checkpoint


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def write_exclusive(path,payload):
    with path.open('xb') as stream:torch.save(payload,stream)
    assert path.stat().st_size < 300*1024**2


@pytest.fixture(scope='module',params=['rms_incumbent','layer_matched_epsilon'])
def artifact(request):
    root=Path(os.environ['BLOCK_NORM_FIXTURE_ROOT']);assert root.is_dir()
    before=torch.get_rng_state().clone()
    try:
        torch.manual_seed(372201)
        cfg=config(request.param);cfg.training.max_iters=2
        cfg.training.weight_averaging=dict(kind='spaced_tail_ema',total_steps=2,start_step=1,interval=1,half_lives=[200])
        model=GPT(cfg.model);optimizer=build_optimizer(model,cfg.training);tracker=TailWeightAverages(model,cfg.training)
        tracker.update(1)
        with torch.no_grad():
            for name in PLACEMENTS:
                module=model.get_submodule(name);module.weight.fill_(1.25)
                if getattr(module,'bias',None) is not None:module.bias.fill_(.5)
        tracker.update(2)
        raw_path=root/(request.param+'-raw.pt');h200_path=root/(request.param+'-h200.pt')
        state=dict(contract=dict(weight_averaging=tracker.spec),weight_averaging_state=tracker.state_dict())
        save_checkpoint(raw_path,model,optimizer,step=2,cfg=cfg,training_state=state,payload_writer=write_exclusive)
        del state
        parent_hash=digest(raw_path);rng=torch.get_rng_state().clone()
        export=export_averaged_checkpoint(raw_path,h200_path,200,payload_writer=write_exclusive)
        assert torch.equal(torch.get_rng_state(),rng) and digest(raw_path)==parent_hash
        raw=read_checkpoint(raw_path,map_location='cpu');h200=read_checkpoint(h200_path,map_location='cpu')
        yield dict(cfg=cfg,model=model,optimizer=optimizer,tracker=tracker,raw=raw,h200=h200,
                   raw_path=raw_path,h200_path=h200_path,parent_hash=parent_hash,export=export)
        assert digest(raw_path)==parent_hash
    finally:
        torch.set_rng_state(before)
        gc.collect()


def test_serialized_raw_and_h200_signature_and_load(artifact):
    a=artifact;model=a['model'];expected=checkpoint_signature(model)
    assert a['raw'][CHECKPOINT_KEY]==a['h200'][CHECKPOINT_KEY]==expected
    assert 'optimizer_state' in a['raw'] and 'optimizer_state' not in a['h200']
    assert a['h200']['averaged_weights']['purpose']=='inference_only_not_optimizer_resume'
    assert a['h200']['parent_checkpoint']['sha256']==a['parent_hash']
    loaded=load_checkpoint(a['raw_path'],model,a['optimizer'],map_location='cpu')
    assert all(torch.equal(p,model.state_dict()[n]) for n,p in loaded['model_state'].items())
    assert a['optimizer'].state_dict()==loaded['optimizer_state']  # empty state, group transport only
    load_model_weights(model,a['h200'])
    assert all(torch.equal(p,model.state_dict()[n]) for n,p in a['h200']['model_state'].items())


def test_ema_includes_all_gains_biases_and_resumes_accumulator(artifact,record_property):
    a=artifact;model=a['model'];names=[n for n,_ in model.named_parameters()]
    assert a['h200']['averaged_weights']['parameter_names']==names
    beta=2.**(-1/200);max_gain=max_bias=0.
    for name in PLACEMENTS:
        value=a['h200']['model_state'][name+'.weight']
        expected=1. + .25*(1.-beta)
        max_gain=max(max_gain,(value.double()-expected).abs().max().item())
        torch.testing.assert_close(value.double(),torch.full(value.shape,expected,dtype=torch.float64),atol=1e-7,rtol=0.)
        bias=a['h200']['model_state'].get(name+'.bias')
        if bias is not None:
            expected=.5*(1.-beta);max_bias=max(max_bias,(bias.double()-expected).abs().max().item())
            torch.testing.assert_close(bias.double(),torch.full(bias.shape,expected,dtype=torch.float64),atol=1e-9,rtol=0.)
    shadow=a['raw']['training_state']['weight_averaging_state']
    restored=TailWeightAverages(model,a['cfg'].training);restored.load_state_dict(shadow,2)
    panel=restored.averaged_parameters(200)
    assert set(panel)==set(names) and all(torch.equal(p,a['h200']['model_state'][n]) for n,p in panel.items())
    for name in names:
        if not any(name.startswith(n+'.') for n in PLACEMENTS):
            assert torch.equal(a['raw']['model_state'][name],a['h200']['model_state'][name])
    record_property('max_gain_error_fp64_oracle',max_gain);record_property('max_bias_error_fp64_oracle',max_bias)


@pytest.mark.parametrize('mutation',['missing_signature','epsilon','opposite_arm','wrong_version_type','extra_signature','bad_cfg','missing_bias_or_gain','wrong_dtype','nonfinite'])
def test_corrupted_payload_rejected_before_weights_change(artifact,mutation):
    a=artifact;model=a['model'];raw=a['raw'];payload=dict(raw)
    payload[CHECKPOINT_KEY]=copy.deepcopy(raw[CHECKPOINT_KEY])
    name=PLACEMENTS[0]+'.weight';before=model.state_dict()[name].clone()
    if mutation=='missing_signature':del payload[CHECKPOINT_KEY]
    elif mutation=='epsilon':payload[CHECKPOINT_KEY]['architecture']['normalization']['epsilon']=1e-5
    elif mutation=='opposite_arm':payload[CHECKPOINT_KEY]['architecture']['normalization']['kind']='layer_matched_epsilon' if model.block_norm_spec.kind=='rms_incumbent' else 'rms_incumbent'
    elif mutation=='wrong_version_type':payload[CHECKPOINT_KEY]['version']=True
    elif mutation=='extra_signature':payload[CHECKPOINT_KEY]['unreviewed']=1
    elif mutation=='bad_cfg':
        payload['cfg']=copy.deepcopy(raw['cfg']);payload['cfg']['model']['block_normalization']['epsilon']=1e-5
    else:
        payload['model_state']=raw['model_state'].copy()
        if mutation=='missing_bias_or_gain':del payload['model_state'][PLACEMENTS[0]+('.bias' if model.block_norm_spec.bias else '.weight')]
        elif mutation=='wrong_dtype':payload['model_state'][name]=payload['model_state'][name].half()
        else:payload['model_state'][name]=torch.full((256,),float('nan'))
    with pytest.raises(ValueError):load_model_weights(model,payload,for_resume=True)
    assert torch.equal(model.state_dict()[name],before)


def test_h200_cannot_resume_optimizer_and_bad_save_never_calls_writer(artifact):
    a=artifact;model=a['model']
    with pytest.raises(ValueError,match='Averaged'):load_model_weights(model,a['h200'],for_resume=True)
    called=[]
    def writer(*args):called.append(args)
    with pytest.raises(ValueError):save_checkpoint(a['raw_path'].with_name('forbidden.pt'),model,cfg=None,payload_writer=writer)
    cfg=OmegaConf.create(OmegaConf.to_container(a['cfg']));cfg.model.block_normalization.epsilon=1e-5
    with pytest.raises(ValueError):save_checkpoint(a['raw_path'].with_name('forbidden.pt'),model,cfg=cfg,payload_writer=writer)
    assert not called and not (a['raw_path'].parent/'forbidden.pt').exists()


def test_legacy_policy_omits_signature_and_rejects_new_payload(artifact):
    cfg=OmegaConf.create(OmegaConf.to_container(artifact['cfg']))
    cfg.model.tokenizer_architecture_policy='tokenizer_mla_quarter_rope_v1'
    cfg.model.norm_type='rmsnorm';del cfg.model.block_normalization
    model=GPT(cfg.model);captured=[]
    save_checkpoint(Path('not-written.pt'),model,cfg=cfg,payload_writer=lambda path,payload:captured.append(payload))
    assert len(captured)==1 and CHECKPOINT_KEY not in captured[0]
    load_model_weights(model,captured[0])
    with pytest.raises(ValueError,match='Legacy model'):load_model_weights(model,artifact['raw'])
