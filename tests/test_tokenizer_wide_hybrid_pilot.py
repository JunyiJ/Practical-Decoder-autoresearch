"""Inline mocked workers with real lock files; no model/corpus/MPS training."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_pilot as pilot
from src.research.tokenizer_wide_hybrid_pilot_config import config, specs
from src.research.protocol import digest, file_digest


def write(path,value):path.write_text(json.dumps(value))


@pytest.fixture
def setup(tmp_path,monkeypatch):
    (tmp_path/'research_state').mkdir();root=tmp_path/'run';root.mkdir()
    monkeypatch.setattr(pilot,'ROOT',tmp_path)
    monkeypatch.setattr(pilot.shutil,'disk_usage',lambda p:SimpleNamespace(free=64*1024**3))
    corpus=tmp_path/'corpus';corpus.write_text('Synthetic fixture only')
    projection=tmp_path/'projection';projection.write_text('Synthetic projection binding')
    monkeypatch.setattr(pilot,'PROJECTION',projection)
    plan_path=root/'plan.json';write(plan_path,{'synthetic':True});sha=file_digest(plan_path)
    token={'synthetic':True};token_sha='1'*64;calls=[];events=[];requests={}
    for i,spec in enumerate(specs()):
        trial=root/f"{i:02d}-pilot-{spec['family']}-s3491-d2"
        cfg=config(trial,spec['family']);cfg['data']['path']=str(corpus)
        requests[spec['family']]=dict(spec=spec,config=cfg,result_path=str(trial/'result.json'))
    frozen={str(p):file_digest(p) for p in (plan_path,corpus,projection)}
    def bind(path,expected,family):
        assert Path(path)==plan_path and expected==sha
        pilot.recheck(frozen)
        if (root/'STOP').exists():raise ValueError('STOP')
        return dict(request=requests[family],files=frozen,tokenizer=token,training_token_sha256=token_sha)
    monkeypatch.setattr(pilot,'bind_pilot',bind)
    monkeypatch.setattr(pilot,'_inputs',lambda cfg:({'tokenizer':token},token_sha))
    def metadata(trial,request,request_sha,result_sha,contract):
        result=json.loads((trial/'result.json').read_text())
        return dict(pilot_spec=request['spec'],request_sha256=request_sha,result_sha256=result_sha,
            checkpoint_hashes={k:result[k]['checkpoint_sha256'] for k in ('raw','h200')},
            metadata_reviewed=True,tensor_values_audited=False,selection_eligible=False,adopted=False)
    monkeypatch.setattr(pilot,'review_pilot',metadata)
    def train(path):
        request=json.loads(path.read_text());family=request['spec']['family'];calls.append(family)
        trial=path.parent;cp=trial/'checkpoints';cp.mkdir();result={}
        for kind in ('raw','h200'):
            p=cp/(kind+'.synthetic');p.write_text(family+kind)
            result[kind]=dict(checkpoint=str(p),checkpoint_sha256=file_digest(p))
        write(trial/'result.json',result)
    monkeypatch.setattr(pilot,'train_worker',train)
    def caches(binding,tokenizer,token_hash):
        trial=Path(binding['trial']);request=binding['request'];family=request['spec']['family']
        events.append(family);row=pilot._projection(family)
        md=metadata(trial,request,binding['request_sha256'],binding['result_sha256'],binding['training_contract'])
        files={str(trial/'request.json'):md['request_sha256'],str(trial/'result.json'):md['result_sha256']}
        endpoints={};result=json.loads((trial/'result.json').read_text())
        for kind in ('raw','h200'):
            p=result[kind]['checkpoint'];files[p]=md['checkpoint_hashes'][kind]
            endpoints[kind]=dict(checkpoint=p,sha256=files[p],parameters=row['parameters'],
                parameter_storage_bytes=row['parameter_bytes'],checks=[dict(batch=b,passed=True,
                max_absolute_error=0.,state_bytes=b*row['cache_bytes_batch1'],
                observed_mps_current_bytes=1000,observed_mps_driver_bytes=2000) for b in (1,8)])
        values=dict.fromkeys(('finite_model_ema_optimizer_export_values','exact_export_matches_stored_ema',
                              'persistent_buffers_unchanged'),True)
        values.update(dict.fromkeys(('optimizer_structure_audited','optimizer_resume_audited',
                                     'ema_recurrence_audited','adopted'),False))
        tensor=dict(kind='pilot',metadata=md,values=values,files=dict(files),adopted=False)
        files[str(corpus)]=frozen[str(corpus)];files[str(projection)]=frozen[str(projection)]
        return dict(status='pilot_cache_checks_complete_pending_resource_review',spec=request['spec'],
            token_sha256=token_hash,tokenizer_sha256=digest(tokenizer),contract=pilot.CACHE_CONTRACT,
            endpoints=endpoints,tensor_review=tensor,files=files,adopted=False)
    monkeypatch.setattr(pilot,'review_caches',caches)
    class Guard:
        def __init__(self,path,root,**limits):self.path=path;self.limits=limits
        def __enter__(self):return self
        def __exit__(self,kind,error,tb):
            write(self.path,dict(status='finished' if kind is None else 'fixture_exception',reason=None,
                limits=self.limits,elapsed_seconds=1.,observation=dict(peak_rss_bytes=1000,free_bytes=64*1024**3)))
    monkeypatch.setattr(pilot,'CpuPrerequisiteGuard',Guard)
    def dispatch(argv,cwd,output,fds,**limits):
        assert limits['timeout_seconds']==600 and limits['log_limit_bytes']==4*1024**2
        assert limits['watch_root']==output.parent and limits['required_free_bytes']>40*1024**3
        output.mkdir();(output/'worker.log').write_text('Synthetic inline worker')
        opts=dict(zip(argv[4::2],argv[5::2]))
        pilot.worker(Path(opts['--plan']),opts['--plan-sha256'],opts['--family'],
                     opts['--request-sha256'],fds)
        write(output/'process.json',{'status':'completed','returncode':0})
    monkeypatch.setattr(pilot,'run_process',dispatch)
    return root,calls,events,dispatch


def test_complete_mocked_panel_no_reentry(setup):
    root,calls,events,_=setup;state=pilot.run(root)
    assert calls==events==list(pilot.FAMILIES)
    assert state['status']=='pilots_complete_pending_resource_and_numerical_review'
    assert len(state['reviews'])==3 and state['active'] is None and not state['adopted']
    with pytest.raises(FileExistsError):pilot.run(root)
    assert calls==list(pilot.FAMILIES)


@pytest.mark.parametrize('at',[0,1,2])
def test_worker_failure_stops_later_pilots(setup,monkeypatch,at):
    root,calls,events,_=setup;original=pilot.train_worker
    def fail(path):
        if len(calls)==at:raise RuntimeError('synthetic training failure')
        original(path)
    monkeypatch.setattr(pilot,'train_worker',fail)
    with pytest.raises(RuntimeError,match='synthetic'):pilot.run(root)
    state=json.loads((root/'state.json').read_text())
    assert state['status']=='failed' and state['active']==pilot.FAMILIES[at]
    assert len(calls)==len(events)==at
    trials=list(root.glob('*/worker-state.json'))
    assert sum(json.loads(p.read_text())['status']=='failed' for p in trials)==1
    with pytest.raises(FileExistsError):pilot.run(root)


@pytest.mark.parametrize('bad',['guard','cache','tokens','geometry','ema','files','receipt'])
def test_saved_artifact_rejection(setup,monkeypatch,bad):
    root,calls,_,dispatch=setup
    def corrupt(*args,**kwargs):
        dispatch(*args,**kwargs);trial=args[2].parent
        if bad=='guard':
            p=trial/'guard.json';obj=json.loads(p.read_text());obj['observation']['peak_rss_bytes']=5*1024**3;write(p,obj)
        elif bad=='receipt':
            p=trial/'worker-state.json';obj=json.loads(p.read_text());obj['status']='failed';write(p,obj)
        else:
            p=trial/'audit.json';obj=json.loads(p.read_text())
            if bad=='cache':obj['endpoints']['raw']['checks'][0]['passed']=False
            elif bad=='tokens':obj['token_sha256']='0'*64
            elif bad=='geometry':obj['endpoints']['h200']['parameters']-=1
            elif bad=='ema':obj['tensor_review']['values']['exact_export_matches_stored_ema']=False
            elif bad=='files':obj['tensor_review']['files']={}
            write(p,obj)
            state_path=trial/'worker-state.json';state=json.loads(state_path.read_text())
            state['files'][str(p)]=file_digest(p);write(state_path,state)
    monkeypatch.setattr(pilot,'run_process',corrupt)
    with pytest.raises(ValueError):pilot.run(root)
    assert calls==['hybrid'] and json.loads((root/'state.json').read_text())['status']=='failed'


@pytest.mark.parametrize('at',[1,3])
def test_stop_prevents_next_or_final_completion(setup,monkeypatch,at):
    root,calls,_,dispatch=setup
    def stop(*args,**kwargs):
        dispatch(*args,**kwargs)
        if len(calls)==at:(root/'STOP').touch()
    monkeypatch.setattr(pilot,'run_process',stop)
    state=pilot.run(root)
    assert state['status']=='paused' and len(calls)==at


def test_changed_previous_evidence_prevents_third(setup,monkeypatch):
    root,calls,_,dispatch=setup
    def change(*args,**kwargs):
        dispatch(*args,**kwargs)
        if len(calls)==2:
            next(root.glob('00-*/checkpoints/raw.synthetic')).write_text('changed')
    monkeypatch.setattr(pilot,'run_process',change)
    with pytest.raises(ValueError,match='evidence changed'):pilot.run(root)
    assert calls==['hybrid','gated_mla']


def test_full_panel_space_before_state(setup,monkeypatch):
    root,calls,_,_=setup
    monkeypatch.setattr(pilot.shutil,'disk_usage',lambda p:SimpleNamespace(free=20*1024**3))
    with pytest.raises(ValueError,match='Complete retained panel'):pilot.run(root)
    assert not (root/'state.json').exists() and not calls


def test_missing_numerical_gate_before_state(setup,monkeypatch):
    root,calls,_,_=setup
    def reject(*args):raise ValueError('Incomplete numerical prerequisites')
    monkeypatch.setattr(pilot,'bind_pilot',reject)
    with pytest.raises(ValueError,match='prerequisites'):pilot.run(root)
    assert not (root/'state.json').exists() and not calls


def test_wrong_inherited_locks(setup):
    root,_,_,_=setup
    with pytest.raises(ValueError,match='Two inherited'):pilot._locks((1,2),root)
    wrong=root/'wrong';wrong.write_text('')
    (pilot.ROOT/'research_state/tokenizer-worker-lock').mkdir()
    (pilot.ROOT/'research_state/tokenizer-worker-lock/.lock').touch();(root/'.lock').touch()
    with wrong.open('rb') as a,(root/'.lock').open('rb') as b:
        with pytest.raises(ValueError,match='identity'):pilot._locks((a.fileno(),b.fileno()),root)


def test_global_lock_contention_prevents_state(setup):
    root,calls,_,_=setup;lock=pilot.ROOT/'research_state/tokenizer-worker-lock';lock.mkdir()
    with pilot.session_lock(lock):
        with pytest.raises(BlockingIOError):pilot.run(root)
    assert not (root/'state.json').exists() and not calls


def test_wrong_input_prevents_training(setup,monkeypatch):
    root,calls,_,_=setup
    monkeypatch.setattr(pilot,'_inputs',lambda cfg:({'tokenizer':{'synthetic':True}},'0'*64))
    with pytest.raises(ValueError,match='training input differs'):pilot.run(root)
    assert not calls
    assert json.loads((root/'state.json').read_text())['status']=='failed'


def test_metadata_cap_before_state(setup,monkeypatch):
    root,calls,_,_=setup
    monkeypatch.setattr(pilot,'snapshot',lambda path:dict(output_bytes=4*1024**2+1))
    with pytest.raises(ValueError,match='complete allocation'):pilot.run(root)
    assert not (root/'state.json').exists() and not calls
