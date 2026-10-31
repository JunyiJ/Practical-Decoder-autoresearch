"""Actual80/10/10 boundary and readonly audit of the preserved RMS20 pilot."""
import copy,gc,json,math,os
from pathlib import Path
import pytest,torch,tiktoken
from omegaconf import OmegaConf
from research.block_norm_v1 import validation_data as vd,validation as v,campaign_contract as c,artifact_acceptance as a
from research.block_norm_v1.runtime.models.output_bias import digest
from research.block_norm_v1.runtime.attention.cache import build_caches
from research.block_norm_v1.runtime.utils.checkpoint import read_checkpoint,load_model_weights
from src.research.tokenizer_wide_hybrid_measurements import check_kernel
from src.research.tokenizer_wide_hybrid_timing import allocated_bytes,token_digest


def plan():return json.loads(Path(os.environ['BLOCK_NORM_PLAN_PATH']).read_text())


def test_actual_80_10_10_view_never_encodes_or_exposes_holdout(monkeypatch,record_property):
    cfg=OmegaConf.create(plan()['pilot_request']['config']);expected=c.bind_data();text=Path(cfg.data.path).read_text();enc=tiktoken.get_encoding('gpt2');encode=enc.encode;calls=[]
    def observed(value,*args,**kwargs):calls.append(value);return encode(value,*args,**kwargs)
    before=torch.get_rng_state().clone()
    with monkeypatch.context() as patch:
        patch.setattr(enc,'encode',observed);view=vd.load(cfg)
    assert calls==[text[:expected['raw_train_end']],text[expected['raw_train_end']:expected['raw_val_end']]]
    assert set(view.splits)=={'train','val'} and not any(hasattr(view,n) for n in ('test_data','data','generators','get_batch','split_texts'))
    assert torch.equal(before,torch.get_rng_state())
    assert digest(view.train_data.tolist())==expected['train_ids_sha256'] and digest(view.val_data.tolist())==expected['validation_ids_sha256']
    targets=v.target_contract(view)
    assert targets['targets']==34278 and targets['target_bytes']==111536 and targets['window_counts']==[128]*267+[102]
    assert targets['block_counts']==[2048]*16+[1510] and targets['target_ids_sha256']==expected['validation_target_ids_sha256']
    record_property('actual_boundary',dict(train_tokens=len(view.train_data),validation_tokens=len(view.val_data),targets=targets,encode_calls=2,holdout_encoded=False,holdout_exposed=False,rng_preserved=True))


@pytest.mark.parametrize('field',['tokenizer','vocab','prior','split','checkpoint'])
def test_wrong_readonly_data_contract_refused(field):
    cfg=OmegaConf.create(plan()['pilot_request']['config']);saved=None
    if field=='tokenizer':cfg.data.tokenizer='char'
    elif field=='vocab':cfg.model.vocab_size=100
    elif field=='prior':cfg.model.output_bias_prior.train_ids_sha256='0'*64
    elif field=='split':cfg.data.val_fraction=.2
    else:saved={'tokenizer_contract':{}}
    with pytest.raises(ValueError):vd.load(cfg,saved)


def test_preserved_rms_full_validation_oracle_artifact_and_cache_recovery(record_property):
    req=plan()['pilot_request'];binding={k:req['files'][k] for k in ('config','parent','h200')}
    cpu_rng=torch.get_rng_state().clone();mps_rng=torch.mps.get_rng_state().clone()
    report=v.evaluate(binding,'mps');accepted=a.accept(req,report)
    assert accepted['accepted'] and not accepted['launch_permitted'] and accepted['audit']==dict(completed_steps=20,tokens_processed=20480,ema_parameters=88,ema_snapshots=8)
    cpu,data,cfg=v.load(binding,'cpu');ids=data.val_data;windows=[]
    with torch.no_grad():
        for start in range(0,len(ids)-1,128):
            count=min(128,len(ids)-1-start);logits=cpu(ids[start:start+count].unsqueeze(0))[0][0].double();target=ids[start+1:start+count+1]
            losses=torch.logsumexp(logits,dim=-1)-logits[torch.arange(count),target];windows.append(float(losses.sum()))
    del cpu,logits,losses;gc.collect()
    errors=[abs(x-y)/n for x,y,n in zip(windows,report['window_loss_sums'],report['target']['window_counts'])]
    assert max(errors)<3e-5 and abs(math.fsum(windows)/34278-report['ce'])<3e-5
    model,data,cfg=v.load(binding,'mps');cache=[];fixed_ids=data.train_data[:96].tolist()
    for endpoint in ('parent','h200'):
        load_model_weights(model,read_checkpoint(binding[endpoint]['path']))
        for batch in (1,8):
            inputs=torch.tensor(fixed_ids,device='mps',dtype=torch.int64).unsqueeze(0).repeat(batch,1)
            factory=lambda:build_caches(cfg,'mps',batch_size=batch,dtype=torch.float32)
            row=check_kernel(model,inputs,factory);cache_state=factory();size=allocated_bytes(b for layer in cache_state for b in layer.buffers())
            assert size==196608*batch;cache.append(dict(endpoint=endpoint,batch=batch,state_bytes=size,**row));del cache_state,inputs
    del model,data;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(mps_rng,torch.mps.get_rng_state())
    a.verify_request(req)
    result=dict(status='saved_rms_outputs_audited_readonly',receipt=accepted,cache=cache,training96_sha256=token_digest(fixed_ids),
        independent_cpu_fp64_ce=math.fsum(windows)/34278,max_window_mean_error=max(errors),rng_preserved=True,
        original_worker_status='failed_preserved',original_process_reclassified=False,training_repeated=False,optimizer_updates=0,holdout_scored=False,
        limitation='Component evidence only; original pilot001 process remains failed/closed. Separate finite continuation and explicit carried-evidence acceptance required.')
    raw=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode();assert len(raw)<128*1024
    with Path(plan()['recovery_result_path']).open('xb') as stream:stream.write(raw)
    record_property('readonly_rms_recovery',result)
