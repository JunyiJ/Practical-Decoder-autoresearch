"""Production-shaped quality integration with mocked dispatch and score audit.

Every plan/artifact is synthetic. No model training, worker or real corpus load.
"""
import json
import fcntl
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.research import tokenizer_wide_hybrid_driver as d
from src.research import tokenizer_wide_hybrid_rules as rules
from src.research.protocol import digest, file_digest


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root=tmp_path; (root/'research_state').mkdir(); out=root/'run'; out.mkdir()
    monkeypatch.setattr(d, 'ROOT', root)
    monkeypatch.setattr(d, 'source_fingerprint', lambda: {'fixture.py': 'synthetic-source'})
    monkeypatch.setattr(d, 'runtime_identity', lambda: {'runtime': 'synthetic'})
    for name in ('PROTOCOL', 'PROJECTION', 'RECIPE'):
        path=root/name; path.write_text(name); monkeypatch.setattr(d, name, path)
    corpus=root/'corpus-fixture'; corpus.write_text('synthetic data identity; not training data')
    monkeypatch.setattr(d, 'CORPUS_SHA256', file_digest(corpus))
    def config(output, family, *args, **kwargs):
        return dict(data=dict(path=str(corpus)), family=family)
    monkeypatch.setattr(d, 'rank_config', config)
    monkeypatch.setattr(d, '_contract', lambda cfg: {'fixture': True})
    monkeypatch.setattr(d, 'allocations', lambda: {(f, b): 512*1024+b for f in rules.FAMILIES for b in (2,4,8)})
    monkeypatch.setattr(d.shutil, 'disk_usage', lambda path: SimpleNamespace(free=64*1024**3))
    plan=dict(status='frozen', kind='wide_hybrid_quality_v1', output=str(out), source=d.source_fingerprint(),
              runtime=d.runtime_identity(), protocol_sha256=file_digest(d.PROTOCOL),
              projection_sha256=file_digest(d.PROJECTION), reserve_bytes=d.RESERVE, job_seconds=7200,
              max_trials=24, log_limit_bytes=d.METADATA, prerequisites={}, files=[])
    paths=[corpus,d.RECIPE]
    for kind, checks in {'numerical':['all_declared_ranks','geometry','cache','optimizer','resume','ema'],
                         'pilots':['three_families','raw_h200','cache','resource_limits','full_panel_fit']}.items():
        artifact=root/(kind+'.log'); artifact.write_text('synthetic fixture')
        review=root/(kind+'.json')
        write(review,dict(kind=kind,status='passed',source_sha256=digest(plan['source']),
                         runtime_sha256=digest(plan['runtime']),corpus_sha256=file_digest(corpus),
                         checks=dict.fromkeys(checks,True),artifacts={str(artifact):file_digest(artifact)}))
        plan['prerequisites'][kind]=str(review); paths += [review,artifact]
    plan['files']=[dict(path=str(p),sha256=file_digest(p)) for p in paths]
    plan['max_output_bytes']=d.remaining_bytes([],None,None)+d.METADATA
    write(out/'plan.json',plan)
    calls=[]
    def dispatch(argv,cwd,process_dir,fds,**limits):
        assert argv[1:4]==['-m','src.research.tokenizer_study','worker']
        assert len(set(fds))==2 and limits['watch_root']==process_dir.parent
        assert limits['timeout_seconds']==7200 and limits['log_limit_bytes']==d.METADATA
        process_dir.mkdir(); req=json.loads(Path(argv[-1]).read_text()); spec=req['spec']
        state=json.loads((out/'state.json').read_text())
        assert state['status']=='running' and state['active']==spec
        if spec['stage']!='lr': assert (out/'lr-selection.json').exists()
        if spec['stage']=='confirm': assert (out/'rank-selection.json').exists()
        calls.append(spec)
        result={}
        for k in ('raw','h200'):
            path=process_dir.parent/(k+'.synthetic'); path.write_text('not a checkpoint')
            result[k]=dict(checkpoint=str(path),checkpoint_sha256=file_digest(path))
        write(process_dir.parent/'result.json',result)
    def review(trial,request,request_hash,result_hash,contract):
        result=json.loads((trial/'result.json').read_text())
        return dict(selection_row=dict(request['spec'],status='completed',ce=5.),
                    result_sha256=result_hash, checkpoint_hashes={k:result[k]['checkpoint_sha256'] for k in result})
    monkeypatch.setattr(d,'run_process',dispatch); monkeypatch.setattr(d,'review_trial',review)
    return out,plan,calls


def test_complete_integration_and_no_reentry(setup):
    out,plan,calls=setup; state=d.run(out)
    assert len(calls)==24 and len(state['reviews'])==24
    assert state['status']=='quality_completed_pending_tensor_and_serving_audits' and not state['adopted']
    with pytest.raises(FileExistsError): d.run(out)
    assert len(calls)==24


@pytest.mark.parametrize('bad', ['draft','source','runtime','reserve','time','trials','protocol','projection','budget','corpus'])
def test_preflight_fails_before_state_or_dispatch(setup,bad):
    out,plan,calls=setup
    key=dict(draft='status',source='source',runtime='runtime',reserve='reserve_bytes',time='job_seconds',
             trials='max_trials',protocol='protocol_sha256',projection='projection_sha256',budget='max_output_bytes')
    if bad=='corpus': plan['files']=plan['files'][1:]
    else: plan[key[bad]]=None
    write(out/'plan.json',plan)
    with pytest.raises(ValueError): d.run(out)
    assert not calls and not (out/'state.json').exists()


def test_incomplete_numerical_review_cannot_be_promoted_by_rehash(setup):
    out,plan,calls=setup; path=Path(plan['prerequisites']['numerical'])
    review=json.loads(path.read_text()); review['checks']['resume']=False; write(path,review)
    for item in plan['files']:
        if item['path']==str(path): item['sha256']=file_digest(path)
    write(out/'plan.json',plan)
    with pytest.raises(ValueError,match='incomplete'): d.run(out)
    assert not calls


def test_prerequisite_artifacts_must_be_frozen(setup):
    out,plan,calls=setup
    plan['files']=[i for i in plan['files'] if not i['path'].endswith('pilots.log')]
    write(out/'plan.json',plan)
    with pytest.raises(ValueError,match='artifacts'): d.run(out)
    assert not calls


def test_storage_blocks_before_first_trial(setup,monkeypatch):
    out,plan,calls=setup
    need=d.RESERVE+d.remaining_bytes([],None,None)+d.METADATA
    monkeypatch.setattr(d.shutil,'disk_usage',lambda path:SimpleNamespace(free=need-1))
    with pytest.raises(RuntimeError,match='does not fit'): d.run(out)
    assert not calls and json.loads((out/'state.json').read_text())['status']=='failed'


@pytest.mark.parametrize('failure',['worker','audit','prior_artifact','selection'])
def test_failure_halts_without_next_dispatch(setup,monkeypatch,failure):
    out,plan,calls=setup; original=d.run_process; audit=d.review_trial
    def dispatch(*args,**kwargs):
        original(*args,**kwargs)
        if failure=='worker': raise RuntimeError('synthetic worker failure')
        if failure=='prior_artifact' and len(calls)==2:
            prior=json.loads((out/'state.json').read_text())['reviews'][0]['files']
            Path(next(iter(prior))).write_text('changed')
        if failure=='selection' and len(calls)==10: (out/'lr-selection.json').write_text('{}')
    def review(*args,**kwargs):
        if failure=='audit': raise ValueError('synthetic audit failure')
        return audit(*args,**kwargs)
    monkeypatch.setattr(d,'run_process',dispatch); monkeypatch.setattr(d,'review_trial',review)
    with pytest.raises((ValueError,RuntimeError)): d.run(out)
    assert len(calls)==dict(worker=1,audit=1,prior_artifact=2,selection=10)[failure]
    state=json.loads((out/'state.json').read_text()); assert state['status']=='failed' and state['active']
    with pytest.raises(FileExistsError): d.run(out)


def test_stop_before_any_trial(setup):
    out,plan,calls=setup; (out/'STOP').write_text('fixture boundary stop')
    assert d.run(out)['status']=='paused' and not calls


def test_inherited_descriptor_retains_lock_after_parent_context(tmp_path):
    inherited=None
    try:
        with d.session_lock(tmp_path) as fd:
            inherited=os.dup(fd)  # Same open-file description as an inherited fd.
        with pytest.raises(BlockingIOError):
            with d.session_lock(tmp_path): pass
        os.close(inherited); inherited=None
        with d.session_lock(tmp_path): pass
    finally:
        if inherited is not None: os.close(inherited)
