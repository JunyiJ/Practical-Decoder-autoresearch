"""Real live verifier and worker composition; model loading/MPS/frozen inputs mocked."""
import copy
import json
import os
from pathlib import Path

import pytest

from src.research import tokenizer_wide_hybrid_inference_entry_v2 as entry
from src.research import tokenizer_wide_hybrid_inference_outputs_v2 as outputs
from src.research import tokenizer_wide_hybrid_inference_preparation_v2 as prep
from src.research.tokenizer_wide_hybrid_inference import PROJECTION
from test_tokenizer_wide_hybrid_metadata_writer import writer


def setup(writer,monkeypatch):
    root=writer.root/'evidence';selection={f:dict(lr=.00125,divisor=4) for f in outputs.FAMILIES}
    tokenizer={'synthetic':True};ids=list(range(96));token_sha=outputs.token_digest(ids)
    def write(name,value):
        path=writer.publish_json('evidence/'+name,value)
        return dict(path=name,sha256=outputs.file_digest(path))
    plan=write('meta/plan.json',dict(tokenizer=tokenizer,training_token_sha256=token_sha))
    projection=write('meta/projection.json',{'synthetic':True})
    rank=write('meta/rank-complete.json',{'synthetic':True})
    sha='a'*64;triples={s:{} for s in outputs.SEEDS};requests={};metadata={}
    rows=json.loads(PROJECTION.read_text())['records']
    for slot,spec in enumerate(outputs.confirmation_specs(selection),15):
        seed=spec['seed'];family=spec['family'];trial=root/f'quality/{slot:02d}'
        cfg=dict(synthetic_family=family,synthetic_seed=seed)
        rq=dict(spec=spec,config=cfg,result_path=str(trial/'result.json'))
        rq_ref=write(f'quality/{slot:02d}/request.json',rq);requests[str(slot)]=[rq];weights={}
        for role in ('raw','h200'):
            path=trial/'checkpoints'/f'{role}.pt';path.parent.mkdir(exist_ok=True);path.write_bytes(f'synthetic {slot} {role}'.encode())
            weights[role]=dict(checkpoint=str(path),checkpoint_sha256=outputs.file_digest(path))
        rs=write(f'quality/{slot:02d}/result.json',dict(status='completed',holdout_consumed=False,request_sha256=rq_ref['sha256'],**weights))
        triples[seed][family]=dict(trial=str(trial),request=rq,request_sha256=rq_ref['sha256'],result_sha256=rs['sha256'],training_contract=dict(tokenizer=tokenizer))
        if seed==3511:
            row=next(r for r in rows if r['width']==512 and r['family']==family and r['bottleneck_divisor']==4)
            metadata[family]=dict(checkpoint=weights['h200']['checkpoint'],sha256=weights['h200']['checkpoint_sha256'],
                seed=seed,family=family,divisor=4,lr=.00125,config_sha256=outputs.digest(cfg),tokenizer_sha256=outputs.digest(tokenizer),
                parameters=row['parameters'],parameter_storage_bytes=row['parameter_bytes'])
    review=dict(selection=selection,completed_records_sha256=sha,checkpoint_pairs_sha256=sha,stages_sha256=sha)
    review['evidence_inventory_sha256']=outputs.digest(prep.gate.history.acceptance._inventory(root))
    completion=write('meta/quality-dispatch-23.json',dict(review=review))
    inventory=prep.gate.history.acceptance._inventory(root)
    frozen=dict(files={str(root/plan['path']):plan['sha256']},requests=dict(quality=requests))
    bound=dict(plan_sha256=plan['sha256'],completion_ref=completion,selection=selection,tokenizer=tokenizer,
        token_sha256=token_sha,triples=triples,records_sha256=sha,checkpoint_pairs_sha256=sha,retention_sha256=sha,
        inventory_sha256=outputs.digest(inventory),controls_sha256=outputs.digest({}),
        contract=dict(outputs.CONTRACT,version=2,tokenizer_sha256=outputs.digest(tokenizer),token_sha256=token_sha))
    request=write(prep.REQUEST,prep._payload(bound,projection,rank,777))
    writer.publish_json(prep.GUARD,dict(kind=prep.KIND,status='finished',pid=777,reason=None,limits=prep.LIMITS,
        elapsed_seconds=1.,observation=dict(peak_rss_bytes=100000,free_bytes=64*1024**3)))
    files={str(root/n):v['sha256'] for n,v in inventory.items()}
    files.update({str(root/request['path']):request['sha256'],str(writer.root/prep.GUARD):outputs.file_digest(writer.root/prep.GUARD)})
    ready=write('inference/3511/ready.json',dict(kind=outputs.READY_KIND,seed=3511,parent_pid=os.getppid(),request_ref=request,
        bindings_sha256=outputs.digest(triples[3511]),protected_files_sha256=outputs.digest(files),contract=bound['contract']))
    argv=entry.worker_argv(writer,3511,ready);launch=entry._launch(argv,os.getppid())
    write('inference/3511/process/launch.json',launch)
    with writer.open_append('evidence/inference/3511/process/worker.log') as stream:stream.write(b'synthetic log\n')
    state=dict(launch,status='running',pid=os.getpid(),elapsed_seconds=.1,
        observation=dict(free_bytes=64*1024**3,output_bytes=1000,log_bytes=14))
    writer.publish_json('control/process-inference-3511.json',state)
    monkeypatch.setattr(entry,'bind_campaign',lambda *a:copy.deepcopy(frozen))
    monkeypatch.setattr(prep.gate,'_contract',lambda c:dict(tokenizer=tokenizer))
    monkeypatch.setattr(entry.os,'getpgid',lambda p:p)
    monkeypatch.setattr(outputs,'start_memory_guard',lambda:None)
    monkeypatch.setattr(outputs.torch,'set_num_threads',lambda n:None)
    monkeypatch.setattr(outputs,'load_selected_exports',lambda *a:dict(models={},configs={},token_ids=ids,metadata=metadata))
    monkeypatch.setattr(outputs,'mps_callbacks',lambda *a:(lambda f,b:dict(passed=True,max_absolute_error=0.),
        lambda f,b:dict(prefill_seconds=.1,decode_seconds=.2,state_bytes=100*b,
            observed_mps_current_bytes=1000,observed_mps_driver_bytes=2000)))
    writer._test_triples=triples
    return ready,argv


@pytest.mark.parametrize('live_change',['none','append','startup'])
def test_real_entry_composes_all_worker_boundaries(writer,monkeypatch,live_change):
    ready,argv=setup(writer,monkeypatch);seen=[];verify=entry.verify_live_entry
    def inspect(*args):
        proof=verify(*args);seen.append(args[-1]);return proof
    monkeypatch.setattr(entry,'verify_live_entry',inspect)
    control=writer.root/'control/process-inference-3511.json'
    if live_change=='append':
        original=entry._immutable
        def append(*args):
            proof=original(*args)
            # Simulate independent parent polling/log publication during hashing.
            with writer._serialized():
                with (writer.root/'evidence/inference/3511/process/worker.log').open('ab') as out:out.write(b'next log line\n')
            state=json.loads(control.read_text());state['elapsed_seconds']+=.1
            state['observation']['log_bytes']+=14
            writer.publish_json(str(control.relative_to(writer.root)),state,replace=True)
            return proof
        monkeypatch.setattr(entry,'_immutable',append)
    if live_change=='startup':
        state=json.loads(control.read_text());state.update(status='preflight',pid=None)
        writer.publish_json(str(control.relative_to(writer.root)),state,replace=True)
        original=entry._live
        def running(*args):
            value=original(*args)
            if value[0]:
                state.update(status='running',pid=os.getpid())
                writer.publish_json(str(control.relative_to(writer.root)),state,replace=True)
            return value
        monkeypatch.setattr(entry,'_live',running)
    ref=entry.run_worker(writer,3511,ready,argv)
    assert seen==['entry','loaded','measured','saved']
    assert outputs._document(writer.root/'evidence',ref)['quality_history_reaudited'] is False
    assert not (writer.root/'evidence/inference/3512').exists()


@pytest.mark.parametrize('bad',['parent','argv','fd','pgid','pid','limits','observation','terminal','future','temporary',
    'original','frozen','triples','ready','worker_guard','late_stop','log_rewrite'])
def test_real_entry_rejects_transport_and_binding_corruption(writer,monkeypatch,bad):
    ready,argv=setup(writer,monkeypatch);root=writer.root/'evidence'
    def change(path,fn):
        value=json.loads(path.read_text());fn(value);path.write_text(json.dumps(value))
    if bad=='parent':monkeypatch.setattr(entry.os,'getppid',lambda:1)
    elif bad=='argv':argv=argv+['--unexpected']
    elif bad=='fd':argv[argv.index('--campaign-fd')+1]=str(writer.lock_fds[0])
    elif bad=='pgid':monkeypatch.setattr(entry.os,'getpgid',lambda p:p+1)
    elif bad in ('pid','limits','observation'):
        def alter(v):
            if bad=='pid':v['pid']+=1
            elif bad=='limits':v['timeout_seconds']=601
            else:v['observation']['free_bytes']=1
        change(writer.root/'control/process-inference-3511.json',alter)
    elif bad in ('terminal','future','temporary'):
        name={'terminal':'inference/3511/process/process.json','future':'inference/3512/ready.json','temporary':'inference/3511/result.json.tmp'}[bad]
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    elif bad=='original':(root/'quality/15/checkpoints/raw.pt').write_bytes(b'changed')
    elif bad=='frozen':monkeypatch.setattr(entry,'bind_campaign',lambda *a:(_ for _ in ()).throw(ValueError('frozen source differs')))
    elif bad=='triples':monkeypatch.setattr(prep.gate,'_contract',lambda c:dict(tokenizer={'different':True}))
    elif bad=='ready':
        change(root/ready['path'],lambda v:v.update(parent_pid=1));ready['sha256']=outputs.file_digest(root/ready['path']);argv=entry.worker_argv(writer,3511,ready)
    if bad in ('worker_guard','late_stop','log_rewrite'):
        original=entry._immutable
        def late(*a):
            proof=original(*a)
            if bad=='worker_guard':change(writer.root/'control/inference-worker-3511.json',lambda v:v.update(status='finished'))
            elif bad=='late_stop':(writer.root/'STOP').touch()
            else:(root/'inference/3511/process/worker.log').write_bytes(b'X'*14)
            return proof
        monkeypatch.setattr(entry,'_immutable',late)
    with pytest.raises((ValueError,RuntimeError)):
        entry.run_worker(writer,3511,ready,argv)
    if bad=='temporary':
        assert (root/'inference/3511/result.json.tmp').read_text()=='{}'
        assert not (writer.root/'control/inference-worker-3511.json').exists()
    else:assert (root/'inference/3511/worker-failure.json').exists()
    assert not (root/'inference/3511/worker-outputs.json').exists()


@pytest.mark.parametrize('seed',[True,3511.,3512,3513])
def test_unintegrated_seed_rejected_without_writes(writer,seed):
    before=outputs.snapshot(writer.root)
    with pytest.raises(ValueError):entry.run_worker(writer,seed,{},[])
    assert outputs.snapshot(writer.root)==before
