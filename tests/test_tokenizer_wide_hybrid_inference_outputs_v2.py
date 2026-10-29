"""Worker-body integration only; live entry, loading and MPS are mocked."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_inference_outputs_v2 as outputs
from src.research.tokenizer_wide_hybrid_inference import PROJECTION
from src.research.tokenizer_wide_hybrid_rules import confirmation_specs
from test_tokenizer_wide_hybrid_metadata_writer import writer


def fixture(writer,monkeypatch,seed=3511):
    root=writer.root/'evidence';selection={f:dict(lr=.00125,divisor=4) for f in outputs.FAMILIES}
    tokenizer={'synthetic':True};ids=list(range(96));token_sha=outputs.token_digest(ids);events=[]
    contract=dict(outputs.CONTRACT,version=2,tokenizer_sha256=outputs.digest(tokenizer),token_sha256=token_sha)
    def write(name,value):
        path=writer.publish_json('evidence/'+name,value)
        return dict(path=name,sha256=outputs.file_digest(path))
    request=write(outputs.REQUEST,dict(seed_order=list(outputs.SEEDS),contract=contract,selection=selection,
        tokenizer_sha256=outputs.digest(tokenizer),token_sha256=token_sha))
    files={str(root/request['path']):request['sha256']};bindings={};metadata={}
    projection=json.loads(PROJECTION.read_text())['records']
    for slot,spec in enumerate(confirmation_specs(selection),15):
        if spec['seed']!=seed:continue
        family=spec['family'];trial=root/f'quality/{slot:02d}';cfg=dict(synthetic_family=family,synthetic_seed=seed)
        rq=dict(spec=spec,config=cfg,result_path=str(trial/'result.json'));rq_ref=write(f'quality/{slot:02d}/request.json',rq)
        checkpoints={}
        for role in ('raw','h200'):
            path=trial/'checkpoints'/f'{role}.pt';path.parent.mkdir(exist_ok=True);path.write_bytes(f'synthetic {seed} {family} {role}'.encode())
            checkpoints[role]=dict(checkpoint=str(path),checkpoint_sha256=outputs.file_digest(path));files[str(path)]=outputs.file_digest(path)
        rs=write(f'quality/{slot:02d}/result.json',dict(status='completed',holdout_consumed=False,request_sha256=rq_ref['sha256'],**checkpoints))
        files.update({str(root/ref['path']):ref['sha256'] for ref in (rq_ref,rs)})
        bindings[family]=dict(trial=str(trial),request=rq,request_sha256=rq_ref['sha256'],result_sha256=rs['sha256'],training_contract=dict(tokenizer=tokenizer))
        row=next(r for r in projection if r['width']==512 and r['family']==family and r['bottleneck_divisor']==4)
        metadata[family]=dict(checkpoint=checkpoints['h200']['checkpoint'],sha256=checkpoints['h200']['checkpoint_sha256'],
            seed=seed,family=family,divisor=4,lr=.00125,config_sha256=outputs.digest(cfg),tokenizer_sha256=outputs.digest(tokenizer),
            parameters=row['parameters'],parameter_storage_bytes=row['parameter_bytes'])
    bound=dict(bindings=bindings,selection=selection,tokenizer=tokenizer,token_sha256=token_sha,files=files)
    ready=write(f'inference/{seed}/ready.json',dict(kind=outputs.READY_KIND,seed=seed,parent_pid=os.getppid(),request_ref=request,
        bindings_sha256=outputs.digest(bindings),protected_files_sha256=outputs.digest(files),contract=contract))
    entry=dict(kind=outputs.ENTRY_KIND,seed=seed,parent_pid=os.getppid(),request_ref=request,ready_ref=ready,binding=bound)
    def check(phase):
        events.append(phase);return copy.deepcopy(entry)
    monkeypatch.setattr(outputs,'start_memory_guard',lambda:events.append('memory_guard'))
    monkeypatch.setattr(outputs.torch,'set_num_threads',lambda n:events.append(('threads',n)))
    def load(*a):
        assert events==['entry','memory_guard',('threads',4)]
        return dict(models={},configs={},token_ids=list(ids),metadata=copy.deepcopy(metadata))
    monkeypatch.setattr(outputs,'load_selected_exports',load)
    def callbacks(*a):
        return lambda f,b:dict(passed=True,max_absolute_error=0.),lambda f,b:dict(prefill_seconds=.1,decode_seconds=.2,
            state_bytes=(outputs.FAMILIES.index(f)+1)*100*b,observed_mps_current_bytes=1000,observed_mps_driver_bytes=2000)
    monkeypatch.setattr(outputs,'mps_callbacks',callbacks)
    return check,entry,events


@pytest.mark.parametrize('seed',outputs.SEEDS)
def test_each_seed_body_uses_same_guarded_contract_and_never_advances(writer,monkeypatch,seed):
    check,entry,events=fixture(writer,monkeypatch,seed)
    before={p:outputs.file_digest(Path(p)) for p in entry['binding']['files']}
    ref=outputs.execute_bound_triple(writer,seed,entry_check=check)
    report=outputs._document(writer.root/'evidence',ref)
    assert events==['entry','memory_guard',('threads',4),'loaded','measured','saved']
    assert report['status']=='inference_outputs_complete_pending_parent_review'
    assert not report['adopted'] and not report['dispatch_authorized'] and not report['quality_history_reaudited']
    independent=outputs.review_bound_triple(writer,seed,entry,ref)
    assert len(independent['review']['cells'])==6
    assert all(independent[k] is False for k in ('adopted','dispatch_authorized','transport_verified','quality_history_reaudited','next_seed_authorized'))
    result=outputs._document(writer.root/'evidence',report['artifacts']['result'])
    review=outputs.review_result(result,seed,entry['binding']['selection'],entry['binding']['token_sha256'])
    assert len(review['cells'])==6
    guard=json.loads((writer.root/f'control/inference-worker-{seed}.json').read_text())
    assert guard['status']=='finished' and guard['limits']==dict(rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,timeout_seconds=600)
    assert all(outputs.file_digest(Path(p))==sha for p,sha in before.items())
    with pytest.raises(ValueError,match='cannot be retried'):outputs.execute_bound_triple(writer,seed,entry_check=check)
    assert not any((writer.root/f'evidence/inference/{s}/worker-attempt.json').exists() for s in outputs.SEEDS if s!=seed)


@pytest.mark.parametrize('bad',['entry','load','metadata','ids','cache','timing','driver','changed_files','changed_entry','stop','space','publication','after_saved'])
def test_failure_is_preserved_and_blocks_retry(writer,monkeypatch,bad):
    check,entry,events=fixture(writer,monkeypatch)
    original_load=outputs.load_selected_exports;original_callbacks=outputs.mps_callbacks
    def verifier(phase):
        result=check(phase)
        if bad=='entry':result['parent_pid']=1
        if phase=='loaded' and bad=='changed_entry':result['binding']['selection']['hybrid']['divisor']=8
        if phase=='loaded' and bad=='changed_files':Path(next(p for p in entry['binding']['files'] if p.endswith('.pt'))).write_bytes(b'changed')
        if phase=='measured' and bad=='stop':(writer.root/'STOP').touch()
        if phase=='measured' and bad=='space':monkeypatch.setattr(outputs,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3-1)))
        if phase=='saved' and bad=='after_saved':raise RuntimeError('synthetic saved-boundary failure')
        return result
    def load(*a):
        if bad=='load':raise RuntimeError('synthetic loading failure')
        value=original_load(*a)
        if bad=='metadata':value['metadata']['hybrid']['parameters']+=1
        if bad=='ids':value['token_ids'][0]=499
        return value
    def callbacks(*a):
        check,measure=original_callbacks(*a)
        if bad=='cache':check=lambda f,b:dict(passed=False,max_absolute_error=1.)
        if bad in ('timing','driver'):
            def changed(f,b):
                value=measure(f,b);value['decode_seconds' if bad=='timing' else 'observed_mps_driver_bytes']=0. if bad=='timing' else 6*1024**3
                return value
            return check,changed
        return check,measure
    monkeypatch.setattr(outputs,'load_selected_exports',load);monkeypatch.setattr(outputs,'mps_callbacks',callbacks)
    publish=writer.publish_json
    if bad=='publication':
        def fail(name,value,**kw):
            if name=='evidence/inference/3511/models.json':raise RuntimeError('synthetic publication failure')
            return publish(name,value,**kw)
        monkeypatch.setattr(writer,'publish_json',fail)
    with pytest.raises((ValueError,RuntimeError)):outputs.execute_bound_triple(writer,3511,entry_check=verifier)
    root=writer.root/'evidence/inference/3511'
    assert json.loads((root/'worker-failure.json').read_text())['retry_allowed'] is False
    assert not (root/'worker-outputs.json').exists()
    if bad in ('publication','after_saved'):assert (root/'result.json').exists()
    with pytest.raises(ValueError,match='cannot be retried'):outputs.execute_bound_triple(writer,3511,entry_check=check)


@pytest.mark.parametrize('seed',[True,3511.,'3511',3510,3514])
def test_exact_seed_required_before_writes(writer,seed):
    before=outputs.snapshot(writer.root)
    with pytest.raises(ValueError):outputs.execute_bound_triple(writer,seed,entry_check=lambda p:None)
    assert outputs.snapshot(writer.root)==before


def test_whole_panel_cap_applies_across_seeds(writer,monkeypatch):
    check,entry,events=fixture(writer,monkeypatch)
    p=writer.root/'evidence/inference/3513/process/worker.log';p.parent.mkdir(parents=True);p.write_bytes(b'x'*(outputs.LIMIT-outputs.TERMINAL_RESERVE))
    with pytest.raises(ValueError,match='panel output'):outputs.execute_bound_triple(writer,3511,entry_check=check)
    assert not events and not (writer.root/'evidence/inference/3511/worker-attempt.json').exists()



@pytest.mark.parametrize('bad',['models','timing','driver','guard','attempt','claim','failure','temporary'])
def test_parent_artifact_review_rejects_rehashed_corruption(writer,monkeypatch,bad):
    check,entry,events=fixture(writer,monkeypatch)
    ref=outputs.execute_bound_triple(writer,3511,entry_check=check)
    root=writer.root/'evidence';path=root/ref['path'];saved=json.loads(path.read_text())
    if bad in ('models','timing','driver'):
        name='models' if bad=='models' else 'result';target=root/saved['artifacts'][name]['path'];value=json.loads(target.read_text())
        if bad=='models':value['hybrid']['parameters']+=1
        elif bad=='timing':value['records'][0]['decode_tokens_per_second']+=1
        else:value['records'][0]['observed_mps_driver_bytes']=6*1024**3
        target.write_text(json.dumps(value));saved['artifacts'][name]['sha256']=outputs.file_digest(target)
    elif bad=='guard':
        target=writer.root/'control/inference-worker-3511.json';value=json.loads(target.read_text());value['limits']['timeout_seconds']=600.
        target.write_text(json.dumps(value))
    elif bad=='attempt':
        target=root/saved['attempt_ref']['path'];value=json.loads(target.read_text());value['parent_pid']+=1;target.write_text(json.dumps(value));saved['attempt_ref']['sha256']=outputs.file_digest(target)
    elif bad=='claim':saved['quality_history_reaudited']=True
    else:(root/'inference/3511'/('worker-failure.json' if bad=='failure' else 'partial.json.tmp')).write_text('{}')
    path.write_text(json.dumps(saved));ref['sha256']=outputs.file_digest(path)
    with pytest.raises(ValueError):outputs.review_bound_triple(writer,3511,entry,ref)
