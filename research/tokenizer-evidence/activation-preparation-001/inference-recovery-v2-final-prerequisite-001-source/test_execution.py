"""Real bounded metadata/guards with explicit GPU, source-audit and process doubles.

The actual OS pipe/flock transport was tested by the census prerequisite. This
suite never loads research models, corpus or checkpoints and never runs MPS.
"""
from contextlib import contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from research.activation_inference_recovery_v2 import context as c,worker as w,controller as e,final as f
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research.protocol import digest
from src.research.tokenizer_activation_session import reference,Session
from src.research.tokenizer_activation_config import config
from src.research.tokenizer_activation_rules import confirmation_specs
from src.research.tokenizer_activation_timing import CONTRACT,FRESH_SEEDS,slot,measure_pair
from tests.test_tokenizer_activation_timing import IDS,SHA,measurement


@pytest.fixture
def ready(tmp_path,monkeypatch):
    global_lock=tmp_path/'global.lock';global_lock.touch()
    root=tmp_path/'recovery';root.mkdir();(root/'.lock').touch()
    original=tmp_path/'original';original.mkdir()
    monkeypatch.setattr(cleanup,'GLOBAL_LOCK',global_lock)
    with global_lock.open('rb') as g,(root/'.lock').open('rb') as d:
        fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(d,fcntl.LOCK_EX|fcntl.LOCK_NB)
        writer=c.MetadataWriter(root,(g.fileno(),d.fileno()))
        writer.publish_json(c.PLAN_NAME,{'synthetic_plan':True})
        ctx=c.Recovery(writer,reference(root,c.PLAN_NAME))
        class Source:
            verify=Session.verify;document=Session.document
            def __init__(self):self.root=original;self.plan={'synthetic_original_plan':True}
        source=Source();selection={'silu':.00125,'gelu_exact':.00125};pairs={str(s):{} for s in FRESH_SEEDS}
        inputs=dict(tokenizer={'synthetic':True},training96_sha256=SHA)
        for i,spec in enumerate(confirmation_specs(selection),6):
            stem=f'evidence/quality/{i:02d}';path=original/stem/'request.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(spec=spec,config=config(original/stem,spec))))
            weights={}
            for role in ('raw','h200'):
                p=original/stem/'checkpoints'/f'{role}.pt';p.parent.mkdir(exist_ok=True);p.write_bytes(b'synthetic-not-a-model')
                weights[role]=reference(original,str(p.relative_to(original)))
            pairs[str(spec['seed'])][spec['family']]=dict(request=reference(original,str(path.relative_to(original))),
                acceptance={'synthetic':True},weights=weights)
        imported=dict(original_root=str(original),selection=selection,inputs=inputs,plan=source.plan,
            final_contract={'representative_export_seed':3611},failure_audit={'synthetic_original_failure':True},
            pair_bindings={s:dict(plan=source.plan,seed=int(s),selection=selection,quality_reviews=[],dispatches=[],
                inputs=inputs,bindings=pair,contract=CONTRACT) for s,pair in pairs.items()})
        carried=dict(kind='activation_carried_pair_v2',seed=3611,original_root=str(tmp_path/'closed-v1'),
            pair=dict(acceptance={'path':'evidence/inference/3511/accepted.json','sha256':'a'*64,'size':1},
                result=measure_pair(3611,selection,IDS,SHA,check=lambda f,b:dict(passed=True,max_absolute_error=0.),measure=measurement)),
            accepted_timing_pairs=1,remaining_seeds=[3612,3613],repeat_first_pair_allowed=False,dispatch_authorized=False)
        for mod in (c,e,w,f):monkeypatch.setattr(mod,'carried_import',lambda ctx_:copy.deepcopy(carried))
        def bind(ctx_,**kwargs):
            assert ctx_ is ctx;c.budget(ctx_,**kwargs);return copy.deepcopy(imported)
        for mod in (e,w,f):monkeypatch.setattr(mod,'bind',bind)
        depth=[0]
        @contextmanager
        def original_session(ctx_):
            assert ctx_ is ctx and depth[0]==0
            depth[0]+=1
            try:yield source
            finally:depth[0]-=1
        for mod in (e,f):monkeypatch.setattr(mod,'original_session',original_session)
        def verify(ctx_,source_,pins):
            assert ctx_ is ctx and source_ is source and pins==imported and depth[0]==1
            for pair in pairs.values():
                for refs in pair.values():
                    source.verify(refs['request'])
                    for ref in refs['weights'].values():source.verify(ref)
            return copy.deepcopy(imported)
        for mod in (e,w):monkeypatch.setattr(mod,'verify_original',verify)
        monkeypatch.setattr(e,'idle',lambda *a:None);monkeypatch.setattr(f,'idle',lambda *a:None)
        monkeypatch.setattr(e,'_group_exists',lambda pid:False)
        monkeypatch.setattr(w,'strict_mps',lambda:None);monkeypatch.setattr(w,'start_memory_guard',lambda:None)
        monkeypatch.setattr(w.torch.backends.mps,'is_available',lambda:False)
        budgets=[]
        def load(source_,b,recheck):
            assert source_ is source and depth[0]==1
            recheck();budgets.append('load')
            return dict(metadata=w.expected_models(source,b),token_ids=IDS)
        def callbacks(loaded,sha,recheck):
            assert sha==SHA
            def check(family,batch):recheck();budgets.append('check');return dict(passed=True,max_absolute_error=0.)
            def measure(family,batch):recheck();budgets.append('measure');return measurement(family,batch)
            return check,measure
        monkeypatch.setattr(w,'load_pair',load);monkeypatch.setattr(w,'callbacks',callbacks)
        launches=[]
        def process(writer_,stage,physical,argv,cwd,**kwargs):
            assert depth[0]==0,'Parent must release original root before child launch'
            seed=FRESH_SEEDS[[slot(s) for s in FRESH_SEEDS].index(physical)]
            request=reference(root,f'evidence/inference/{physical}/request.json')
            assert writer_ is writer and stage=='inference' and argv==e.argv_for(ctx,seed,request)
            parent=os.getpid();child=parent+100000;launches.append(seed);kwargs['parent_check']()
            with monkeypatch.context() as child_context:
                child_context.setattr(e.os,'getpid',lambda:child)
                child_context.setattr(e.os,'getppid',lambda:parent)
                child_context.setattr(e.os,'getpgid',lambda pid:child)
                e.worker(ctx,seed,request)
            result=dict(status='completed',returncode=0,error=None,pid=child,parent_pid=parent,argv=argv,
                capture_kind='bounded_pipe_v2',log_truncated=False,reserve_bytes=CONTRACT['reserve_bytes'],
                required_free_bytes=kwargs['required_free_bytes'],output_limit_bytes=kwargs['output_limit_bytes'],
                log_limit_bytes=kwargs['log_limit_bytes'],timeout_seconds=kwargs['timeout_seconds'],elapsed_seconds=.1,
                observation=dict(free_bytes=CONTRACT['reserve_bytes']+10000000,output_bytes=10000,log_bytes=0),
                synthetic_process_fixture=True)
            writer.publish_json(f'evidence/inference/{physical}/process/process.json',result)
            return result
        monkeypatch.setattr(e,'run_process',process)
        yield SimpleNamespace(ctx=ctx,source=source,imported=imported,launches=launches,budgets=budgets,
            original_session=original_session,process=process,depth=depth,carried=carried)


def test_complete_three_seed_chain_and_summary(ready):
    r=ready;refs=[c.carried_credit(r.ctx)]+[e.run_once(r.ctx) for _ in range(2)]
    assert r.launches==[3612,3613] and r.depth==[0]
    with r.original_session(r.ctx) as source:assert e.accepted_prefix(r.ctx,source,r.imported)==refs
    summary=e.run_once(r.ctx);value=r.ctx.document(summary)
    assert value['accepted']==refs and value['original_import_sha256']==digest(r.imported)
    assert value['adopted'] is False and len(r.launches)==2
    assert r.budgets.count('load')==2 and r.budgets.count('check')==8 and r.budgets.count('measure')==80
    with pytest.raises(ValueError,match='cannot repeat'):e.run_once(r.ctx)
    assert c.budget(r.ctx)['checkpoint_logical_bytes']==0


@pytest.mark.parametrize('name',['evidence/inference/3512/request.json','evidence/inference/3513/accepted.json',
    'control/recovery-prepare-inference-3512.json','control/process-inference-3513.json',
    'control/process-inference-3512.json.tmp'])
def test_existing_or_future_attempt_blocks_dispatch(ready,name):
    path=ready.ctx.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    with pytest.raises((ValueError,KeyError)):e.run_once(ready.ctx)
    assert not ready.launches


@pytest.mark.parametrize('failure',['load','cache','measurement','resource'])
def test_worker_failure_preserved_and_no_retry(ready,monkeypatch,failure):
    def fail(*args,**kwargs):raise RuntimeError('synthetic '+failure+' failure')
    if failure=='load':monkeypatch.setattr(w,'load_pair',fail)
    elif failure=='resource':monkeypatch.setattr(w,'start_memory_guard',fail)
    else:
        def callbacks(*args):
            good=lambda f,b:dict(passed=True,max_absolute_error=0.)
            return (fail,measurement) if failure=='cache' else (good,fail)
        monkeypatch.setattr(w,'callbacks',callbacks)
    with pytest.raises(RuntimeError,match='synthetic'):e.run_once(ready.ctx)
    stem=ready.ctx.root/'evidence/inference/3512'
    assert (stem/'worker-failure.json').exists() and (stem/'dispatch-failure.json').exists()
    assert not (stem/'accepted.json').exists()
    with pytest.raises(ValueError,match='cannot retry'):e.run_once(ready.ctx)
    assert ready.launches==[3612]


@pytest.mark.parametrize('case',['parent','fd','process_group'])
def test_child_entry_identity_gate(ready,monkeypatch,case):
    original=ready.process
    def process(*args,**kwargs):
        # Alter a sealed request before actual doubled child entry.
        path=ready.ctx.root/'evidence/inference/3512/request.json';rq=json.loads(path.read_text())
        if case=='parent':rq['parent_pid']+=10
        elif case=='fd':rq['lock_fds']=[9998,9999]
        else:
            # A child identity mutation is introduced inside original process
            # through the worker wrapper, after its process doubles are installed.
            worker=e.worker
            def changed(*a):
                with monkeypatch.context() as m:
                    m.setattr(e.os,'getpgid',lambda pid:pid+1)
                    return worker(*a)
            monkeypatch.setattr(e,'worker',changed)
        path.write_text(json.dumps(rq))
        return original(*args,**kwargs)
    # Canonical argv's request hash also rejects parent/FD tampering before child.
    monkeypatch.setattr(e,'run_process',process)
    with pytest.raises((ValueError,AssertionError)):e.run_once(ready.ctx)
    assert not (ready.ctx.root/'evidence/inference/3513/accepted.json').exists()


@pytest.mark.parametrize('case',['returncode','timeout','truncated','cleanup','argv','descendants'])
def test_parent_rejects_invalid_process_receipt(ready,monkeypatch,case):
    original=ready.process
    def process(*args,**kwargs):
        result=original(*args,**kwargs)
        if case=='returncode':result['returncode']=1
        elif case=='timeout':result['timeout_seconds']=601
        elif case=='truncated':result['log_truncated']=True
        elif case=='cleanup':result['cleanup_error']='synthetic'
        elif case=='argv':result['argv']=result['argv']+['unexpected']
        else:monkeypatch.setattr(e,'_group_exists',lambda pid:True)
        p=ready.ctx.root/'evidence/inference/3512/process/process.json';p.write_text(json.dumps(result))
        return result
    monkeypatch.setattr(e,'run_process',process)
    with pytest.raises(ValueError):e.run_once(ready.ctx)
    assert not (ready.ctx.root/'evidence/inference/3513/accepted.json').exists()
    assert (ready.ctx.root/'evidence/inference/3512/dispatch-failure.json').exists()


@pytest.mark.parametrize('name',['result.json','models.json','worker-outputs.json','attempt.json'])
def test_accepted_artifact_tampering_blocks_later_seed(ready,name):
    e.run_once(ready.ctx);p=ready.ctx.root/'evidence/inference/3512'/name;p.write_text('{}')
    with pytest.raises((ValueError,KeyError)):e.run_once(ready.ctx)
    assert ready.launches==[3612]


def test_original_checkpoint_tampering_blocks_next_seed(ready):
    e.run_once(ready.ctx)
    ref=ready.imported['pair_bindings']['3611']['bindings']['silu']['weights']['h200']
    (ready.source.root/ref['path']).write_bytes(b'changed-original')
    with pytest.raises(ValueError,match='hash/size'):e.run_once(ready.ctx)
    assert ready.launches==[3612]


@pytest.mark.parametrize('case',['stop','training','checkpoint','disk','panel'])
def test_recovery_budget_rejects_unsafe_state(ready,monkeypatch,case):
    root=ready.ctx.root
    if case=='stop':(root/'STOP').touch()
    elif case=='training':
        p=root/'evidence/quality/00/request.json';p.parent.mkdir(parents=True);p.write_text('{}')
    elif case=='checkpoint':
        p=root/'evidence/quality/00/checkpoints/raw.pt';p.parent.mkdir(parents=True);p.write_bytes(b'forbidden')
    elif case=='disk':monkeypatch.setattr(c.shutil,'disk_usage',lambda p:SimpleNamespace(free=CONTRACT['reserve_bytes']-1))
    else:monkeypatch.setitem(c.CONTRACT,'panel_limit_bytes',100)
    with pytest.raises(ValueError):c.budget(ready.ctx)


def test_capacity_counts_all_remaining_metadata_without_weight_credit(ready,monkeypatch):
    seen=c.budget(ready.ctx);needed=CONTRACT['reserve_bytes']+c.METADATA-seen['metadata_bytes']
    monkeypatch.setattr(c.shutil,'disk_usage',lambda p:SimpleNamespace(free=needed-1))
    c.budget(ready.ctx)
    with pytest.raises(ValueError,match='allocation'):c.budget(ready.ctx,capacity=True)


def test_unfrozen_separate_root_cannot_bind(ready,monkeypatch):
    monkeypatch.setattr(c,'RECOVERY_ROOT',ready.ctx.root)
    with pytest.raises(ValueError,match='frozen recovery plan'):c.bind(ready.ctx)


@pytest.mark.parametrize('name',['evidence/meta/recovery-final-failure.json','emergency/failure.json','control/broken.json.tmp'])
def test_any_root_failure_or_partial_blocks_dispatch(ready,name):
    path=ready.ctx.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    with pytest.raises(ValueError,match='cannot retry or advance'):e.run_once(ready.ctx)
    assert not ready.launches


def test_no_carried_slot_regeneration_or_wrong_remaining_request(ready):
    r=ready
    path=r.ctx.root/'evidence/inference/3511/request.json';path.parent.mkdir(parents=True);path.write_text('{}')
    with pytest.raises(ValueError,match='regenerated'):e.run_once(r.ctx)
    assert not r.launches


def test_partial_final_blocks_dispatch_without_failure_receipt(ready):
    ready.ctx.writer.publish_json('evidence/meta/recovery-final-attempt.json',{'started':True})
    with pytest.raises(ValueError,match='Final recovery phase'):e.run_once(ready.ctx)
    assert not ready.launches


@pytest.mark.parametrize('case',['allowed','prior_result','prior_guard','unknown_control'])
def test_actual_later_child_entry_replays_prior_during_monitor_updates(ready,case):
    import sys
    from src.research.tokenizer_wide_hybrid_process_v2 import run_process
    r=ready;e.run_once(r.ctx)
    r.ctx.writer.publish_json(c.IMPORT_NAME,r.imported);r.ctx.writer.publish_json(c.CARRIED_NAME,r.carried)
    prior=[c.carried_credit(r.ctx),reference(r.ctx.root,'evidence/inference/3512/accepted.json')]
    with c.PhaseGuard(r.ctx.writer,3613,'prepare'):pass
    stem='evidence/inference/3513';free=CONTRACT['reserve_bytes']+CONTRACT['output_limit_bytes']
    r.ctx.writer.publish_json(stem+'/request.json',dict(kind=c.KIND,plan=r.ctx.plan,seed=3613,
        binding=r.imported['pair_bindings']['3613'],original_import_sha256=digest(r.imported),
        carried_import_sha256=digest(r.carried),prior=prior,parent_pid=os.getpid(),lock_fds=list(r.ctx.writer.lock_fds),
        prepare_guard=c.guard_ref(r.ctx,3613,'prepare'),required_free_bytes=free))
    injected=[]
    def monitor():
        log=r.ctx.root/stem/'process/worker.log'
        if injected or not log.exists() or log.stat().st_size<8193:return
        if case=='prior_result':
            p=r.ctx.root/'evidence/inference/3512/result.json'
            data=p.read_bytes();changed=data.replace(b'3612',b'3613',1);assert len(data)==len(changed) and data!=changed;p.write_bytes(changed)
        elif case=='prior_guard':
            p=r.ctx.root/'control/recovery-worker-inference-3512.json';value=json.loads(p.read_text());value['pid']+=1;p.write_text(json.dumps(value))
        elif case=='unknown_control':r.ctx.writer.publish_json('control/unexpected.json',{})
        r.ctx.writer.publish_json('control/recovery-worker-inference-3513.json',{'synthetic_ack':True});injected.append(True)
    result=run_process(r.ctx.writer,'inference',3513,[sys.executable,'-m','research.activation_inference_recovery_v2.entry_probe',
        str(r.ctx.root),str(cleanup.GLOBAL_LOCK),*map(str,r.ctx.writer.lock_fds),case],
        Path(__file__).resolve().parents[2],required_free_bytes=free,output_limit_bytes=CONTRACT['output_limit_bytes'],
        log_limit_bytes=CONTRACT['log_limit_bytes'],timeout_seconds=30,poll_seconds=.01,parent_check=monitor)
    assert result['returncode']==0 and injected==[True]
    lines=(r.ctx.root/stem/'process/worker.log').read_text().splitlines()
    reports=[json.loads(line.removeprefix('ENTRY_RESULT ')) for line in lines if line.startswith('ENTRY_RESULT ')]
    assert len(reports)==1 and reports[0]['status']==('passed' if case=='allowed' else 'rejected_expected')
    assert not (r.ctx.root/stem/'attempt.json').exists() # producer replaced; no model work
