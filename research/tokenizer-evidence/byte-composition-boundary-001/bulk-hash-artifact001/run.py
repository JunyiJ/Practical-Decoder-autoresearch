"""Verify new typed training consumers against accepted RAW and H200 artifacts."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def read(p):return json.loads(p.read_text())


def publish(name,value):
    with (OUT/name).open('x') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def run():
    assert not (OUT/'guard.json').exists()
    plan=read(OUT/'plan.json');assert sha(Path(__file__))==plan['runner_sha256']
    for r in plan['bindings']:
        p=ROOT/r['path'];assert p.stat().st_size==r['size'] and sha(p)==r['sha256']
    mapping=read(OUT/'source-map.json')
    for r in (mapping['original'],mapping['candidate']):
        p=ROOT/r['path'];assert p.stat().st_size==r['size'] and sha(p)==r['sha256']
    s=(ROOT/mapping['candidate']['path']).read_text()
    for a,c in reversed(mapping['exact_replacements']):s=s.replace(c,a)
    assert s==(ROOT/mapping['original']['path']).read_text()
    integration=read(OUT.parent/'bulk-hash-integration-source001/source-map.json')
    for pair in integration['pairs']:
        for r in (pair['original'],pair['candidate']):
            p=ROOT/r['path'];assert p.stat().st_size==r['size'] and sha(p)==r['sha256']
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from research.byte_input_quality_recovery_v1.guard import Guard
        with Guard(OUT/'guard.json',ROOT,timeout_seconds=90,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,
                   weight_roots=[ROOT/'research_runs/byte-input-quality-v1-001',ROOT/'research_runs/byte-input-quality-recovery-v1-001']) as guard:
            import torch
            from research import byte_input_checkpoint_consumer_v4 as consumer
            from research import byte_input_checkpoint_consumer_v1 as original
            from research import byte_input_training_v6 as training
            from research import byte_input_inference_v5 as inference
            from research.byte_input_model_v6 import build_optimizer
            torch.set_num_threads(4);assert torch.backends.mps.is_available();guard.torch=torch
            before_cpu=torch.get_rng_state().clone();before_mps=torch.mps.get_rng_state().clone()
            recipe=read(OUT.parent/'optimizer-state-adoption001/recipe.json');loader=recipe['loader']
            model,identity,receipt=consumer.load(plan['implementation_record'],loader['plan_record'],loader['acceptance_record'],'mps')
            prepared=original.prepare(loader['plan_record'],loader['acceptance_record'])
            raw=training.read_bound(prepared['parent']);saved_h200=inference.read_bound(prepared['h200'])
            def tree(value):
                if isinstance(value,torch.Tensor):
                    t=value.detach().cpu()
                    return dict(dtype=str(t.dtype),shape=list(t.shape),sha256=hashlib.sha256(t.numpy().tobytes()).hexdigest())
                if type(value) is dict:return {k:tree(v) for k,v in value.items()}
                if type(value) is tuple:return dict(tuple=[tree(v) for v in value])
                if type(value) is list:return [tree(v) for v in value]
                assert value is None or type(value) in (str,int,float,bool)
                return value
            optimizer=build_optimizer(model,identity['recipe']['learning_rate'])
            context=training.Context(model,optimizer,training.load_view(identity['roles'],'mps'),1200,
                                     identity['recipe']['learning_rate'],identity['roles'],identity['binding'])
            try:
                step=training.restore_pristine(context,raw);assert step==1200 and context.averages.snapshots==41
                captured=training.capture(context);training.validate(context,captured)
                expected_raw=tree(raw);assert tree(captured)==expected_raw;del captured
                exported=inference.export(context,prepared['parent'])
                expected_h200=tree(saved_h200);assert tree(exported)==expected_h200
                loaded=inference.load(prepared['h200'],identity,prepared['parent'],prepared['codebook'],'mps')
                assert {n:tree(p) for n,p in loaded.ordered_parameters()}==expected_h200['parameters']
                assert type(loaded) is type(model)
                publish('raw-fingerprint.json',expected_raw);publish('h200-fingerprint.json',expected_h200)
            finally:
                torch.set_rng_state(before_cpu);torch.mps.set_rng_state(before_mps)
            assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_mps,torch.mps.get_rng_state())
            result=dict(completed=True,raw_restore_and_capture_exact=True,h200_export_exact=True,h200_load_exact=True,
                        original_model_class_preserved=True,restored_steps=1200,averaging_snapshots=41,
                        training_updates=0,quality_evaluations=0,new_weight_files=0,caller_rng_restored=True,
                        plan_sha256=sha(OUT/'plan.json'),runner_sha256=sha(Path(__file__)),
                        raw_receipt=prepared['parent'],h200_receipt=prepared['h200'])
        assert read(OUT/'guard.json')['status']=='finished'
        result['guard_sha256']=sha(OUT/'guard.json');publish('result.json',result)
        publish('STOP',dict(status='complete_exact_saved_artifact_interoperation',no_replay=True))
    print(json.dumps(result))


if __name__=='__main__':
    try:run()
    except Exception:
        publish('failure.json',dict(traceback=traceback.format_exc(),no_automatic_replay=True));raise
