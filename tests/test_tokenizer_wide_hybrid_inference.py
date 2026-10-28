"""Inline mocked worker dispatch/load/device guards; no model or MPS run."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.research import tokenizer_wide_hybrid_inference as i
from src.research.protocol import file_digest
from src.research.tokenizer_wide_hybrid_timing import token_digest


def write(p,v):p.write_text(json.dumps(v))


@pytest.fixture
def setup(tmp_path,monkeypatch):
    root=tmp_path;(root/'research_state').mkdir();study=root/'quality';study.mkdir()
    monkeypatch.setattr(i,'ROOT',root)
    monkeypatch.setattr(i.shutil,'disk_usage',lambda p:SimpleNamespace(free=64*1024**3))
    evidence=study/'evidence';evidence.write_text('synthetic reviewed provenance')
    evidence_hash=file_digest(evidence)
    request=root/'requested.json';write(request,{'quality_run':str(study)})
    selection={f:dict(lr=.00125,divisor=4) for f in i.FAMILIES};ids=list(range(96));sha=token_digest(ids)
    events=[];dispatch=[]
    def bind(req,seed):
        events.append(('gate',seed))
        if (study/'STOP').exists():raise ValueError('STOP')
        if file_digest(evidence)!=evidence_hash:raise ValueError('changed evidence')
        return dict(bindings={f:{'seed':seed} for f in i.FAMILIES},selection=selection,
            tokenizer={'synthetic':True},token_sha256=sha,files={str(evidence):evidence_hash})
    monkeypatch.setattr(i,'bind_inference',bind)
    class Guard:
        def __init__(self,path,root,**limits):
            self.path=path;self.limits=limits
            assert limits==dict(rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,timeout_seconds=600)
        def __enter__(self):events.append(('guard',None));return self
        def __exit__(self,kind,error,tb):
            write(self.path,dict(status='finished' if error is None else 'fixture_exception',reason=None,
                limits=self.limits,elapsed_seconds=1.,observation=dict(peak_rss_bytes=64*1024**2,free_bytes=64*1024**3)))
    monkeypatch.setattr(i,'CpuPrerequisiteGuard',Guard)
    monkeypatch.setattr(i,'start_memory_guard',lambda:events.append(('memory_guard',None)))
    def load(bindings,seed,*args):
        assert ('guard',None) in events and ('memory_guard',None) in events
        events.append(('load',seed))
        return dict(models={},configs={},token_ids=ids,metadata={'seed':seed})
    monkeypatch.setattr(i,'load_selected_exports',load)
    def models(meta,bound,seed):
        if meta!={'seed':seed}:raise ValueError('model metadata changed')
    monkeypatch.setattr(i,'_models',models)
    def callbacks(*args):
        def measure(f,b):
            return dict(prefill_seconds=1.,decode_seconds=2.,state_bytes=(i.FAMILIES.index(f)+1)*100*b,
                observed_mps_current_bytes=1000,observed_mps_driver_bytes=2000)
        return lambda f,b:dict(passed=True,max_absolute_error=0.),measure
    monkeypatch.setattr(i,'mps_callbacks',callbacks)
    def run_process(argv,cwd,output,fds,**limits):
        assert argv[1:4]==['-m','src.research.tokenizer_wide_hybrid_inference','worker']
        assert limits==dict(required_free_bytes=i.RESERVE+i.LIMIT,output_limit_bytes=i.LIMIT,
            log_limit_bytes=i.LIMIT,timeout_seconds=600,watch_root=study/'inference')
        assert len(set(fds))==2 and cwd==root
        output.mkdir();write(output/'process.json',{'status':'completed'});(output/'worker.log').write_text('synthetic')
        value=lambda k:argv[argv.index(k)+1]
        seed=int(value('--seed'));dispatch.append(seed)
        return i.worker(Path(value('--request')),value('--request-sha256'),seed,Path(value('--output')),
                        (int(value('--global-lock-fd')),int(value('--study-lock-fd'))))
    monkeypatch.setattr(i,'run_process',run_process)
    return request,study,events,dispatch


def test_complete_inline_integration_and_no_reentry(setup):
    request,study,events,dispatch=setup
    state=i.run(request)
    assert dispatch==list(i.SEEDS) and state['status']=='inference_complete_pending_final_review'
    assert len(state['reviews'])==3 and not state['adopted']
    assert all(len(r['review']['cells'])==6 for r in state['reviews'])
    summary=json.loads((study/'inference/summary.json').read_text())
    assert len(summary['cells'])==18 and not summary['adopted']
    with pytest.raises(FileExistsError):i.run(request)
    assert dispatch==list(i.SEEDS)


@pytest.mark.parametrize('fail_seed',i.SEEDS)
def test_worker_failure_preserved_and_no_later_seed(setup,monkeypatch,fail_seed):
    request,study,events,dispatch=setup;original=i.load_selected_exports
    def load(bindings,seed,*args):
        if seed==fail_seed:raise RuntimeError('synthetic load failure')
        return original(bindings,seed,*args)
    monkeypatch.setattr(i,'load_selected_exports',load)
    with pytest.raises(RuntimeError,match='synthetic'):i.run(request)
    assert dispatch==list(i.SEEDS[:i.SEEDS.index(fail_seed)+1])
    state=json.loads((study/'inference/state.json').read_text())
    child=json.loads((study/f'inference/seed-{fail_seed}/worker-state.json').read_text())
    assert state['status']==child['status']=='failed' and state['active']==fail_seed
    with pytest.raises(FileExistsError):i.run(request)


@pytest.mark.parametrize('bad',['guard','guard_peak','models','receipt','result'])
def test_parent_rejects_corrupt_worker_artifacts(setup,monkeypatch,bad):
    request,study,events,dispatch=setup;original=i.run_process
    def run(*args,**kwargs):
        original(*args,**kwargs);child=args[2].parent
        if bad in ('guard','guard_peak'):
            p=child/'guard.json';v=json.loads(p.read_text())
            if bad=='guard':v['status']='failed'
            else:v['observation']['peak_rss_bytes']=4*1024**3
            write(p,v)
        elif bad=='models':
            p=child/'models.json';write(p,{'seed':0})
            s=child/'worker-state.json';v=json.loads(s.read_text());v['files'][str(p)]=file_digest(p);write(s,v)
        elif bad=='receipt':
            p=child/'worker-state.json';v=json.loads(p.read_text());v['files'].pop(str(child/'models.json'));write(p,v)
        else:
            p=child/'result.json';v=json.loads(p.read_text());v['records'].pop();write(p,v)
            s=child/'worker-state.json';v=json.loads(s.read_text());v['files'][str(p)]=file_digest(p);write(s,v)
    monkeypatch.setattr(i,'run_process',run)
    with pytest.raises(ValueError):i.run(request)
    assert dispatch==[3511]


def test_completed_evidence_rechecked_before_next_seed(setup,monkeypatch):
    request,study,events,dispatch=setup;original=i.run_process
    def run(*args,**kwargs):
        original(*args,**kwargs)
        if len(dispatch)==2:(study/'inference/seed-3511/result.json').write_text('changed')
    monkeypatch.setattr(i,'run_process',run)
    with pytest.raises(ValueError,match='evidence changed'):i.run(request)
    assert dispatch==[3511,3512]


def test_boundary_stop_does_not_dispatch_next_seed(setup,monkeypatch):
    request,study,events,dispatch=setup;original=i.run_process
    def run(*args,**kwargs):original(*args,**kwargs);(study/'inference/STOP').touch()
    monkeypatch.setattr(i,'run_process',run)
    state=i.run(request)
    assert state['status']=='paused' and dispatch==[3511]


def test_insufficient_space_rejected_before_attempt(setup,monkeypatch):
    request,study,events,dispatch=setup
    monkeypatch.setattr(i.shutil,'disk_usage',lambda p:SimpleNamespace(free=i.RESERVE))
    with pytest.raises(RuntimeError,match='allocation'):i.run(request)
    assert not dispatch and not (study/'inference').exists()


def test_output_cap_rejects_after_worker(setup,monkeypatch):
    request,study,events,dispatch=setup;original=i.run_process
    def run(*args,**kwargs):
        original(*args,**kwargs);(study/'inference/oversized').write_bytes(b'x'*(i.LIMIT+1))
    monkeypatch.setattr(i,'run_process',run)
    with pytest.raises(RuntimeError,match='cap'):i.run(request)
    assert dispatch==[3511]


def test_inherited_locks_cannot_be_arbitrary_files(setup):
    request,study,events,dispatch=setup;p=study/'wrong';p.touch()
    with p.open('a+b') as a,p.open('a+b') as b:
        with pytest.raises((ValueError,FileNotFoundError)):i._locks((a.fileno(),b.fileno()),study)
    with pytest.raises(ValueError):i._locks((0,1),study)


def test_global_lock_conflict_prevents_attempt(setup):
    request,study,events,dispatch=setup
    directory=i.ROOT/'research_state/tokenizer-worker-lock';directory.mkdir()
    with i.session_lock(directory):
        with pytest.raises(BlockingIOError):i.run(request)
    assert not dispatch and not (study/'inference').exists()


def test_missing_prerequisites_prevent_attempt(setup,monkeypatch):
    request,study,events,dispatch=setup
    def missing(*args):raise ValueError('missing complete tensor review')
    monkeypatch.setattr(i,'bind_inference',missing)
    with pytest.raises(ValueError,match='tensor'):i.run(request)
    assert not dispatch and not (study/'inference').exists()


def test_stop_after_last_seed_prevents_summary(setup,monkeypatch):
    request,study,events,dispatch=setup;original=i.run_process
    def run(*args,**kwargs):
        original(*args,**kwargs)
        if len(dispatch)==3:(study/'inference/STOP').touch()
    monkeypatch.setattr(i,'run_process',run)
    assert i.run(request)['status']=='paused' and dispatch==list(i.SEEDS)
    assert not (study/'inference/summary.json').exists()
