"""Real serial CLI children, inherited FDs and owned-group failure cleanup.

Only frozen science inputs/model loading/MPS are synthetic, via a test-only
sitecustomize guarded to tiny /tmp fixtures. Actual process argv is canonical.
"""
import copy
import json
import os
from pathlib import Path
import signal

import pytest

from src.research import tokenizer_wide_hybrid_inference_dispatch_v2 as dispatch
from src.research import tokenizer_wide_hybrid_process_v2 as process_module
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_inference_entry_v2 import setup
from tokenizer_wide_hybrid_inference_subprocess_fixture import install


def prepared(writer,monkeypatch,tmp_path):
    ref,_=setup(writer,monkeypatch,prepared_only=True)
    install(writer.root,monkeypatch.setattr)
    support=tmp_path/'support';support.mkdir()
    (support/'sitecustomize.py').write_text('import os\nfrom tokenizer_wide_hybrid_inference_subprocess_fixture import install\ninstall(os.environ["TOKENIZER_INFERENCE_TEST_ROOT"])\n')
    tests=Path(__file__).resolve().parent
    monkeypatch.setenv('PYTHONPATH',os.pathsep.join(map(str,(support,tests,tests.parent))))
    monkeypatch.setenv('TOKENIZER_INFERENCE_TEST_ROOT',str(writer.root))
    return ref


def test_three_actual_cli_children_are_serial_and_independently_reviewed(writer,monkeypatch,tmp_path):
    request=prepared(writer,monkeypatch,tmp_path);original=dispatch.run_process;seen=[]
    before={str(p):p.read_bytes() for p in writer.root.rglob('*') if p.is_file()}
    def one(*args,**kw):
        assert all(not process_module._group_exists(pid) for pid in seen)
        result=original(*args,**kw);seen.append(result['pid']);return result
    monkeypatch.setattr(dispatch,'run_process',one)
    result=dispatch.run_panel(writer,request)
    assert len(seen)==len(set(seen))==3 and os.getpid() not in seen
    assert len(result['reviews'])==3 and not result['adopted']
    assert all(Path(n).read_bytes()==v for n,v in before.items())
    root=writer.root/'evidence'
    for seed,pid,ref in zip(dispatch.outputs.SEEDS,seen,result['reviews']):
        assert not process_module._group_exists(pid)
        ready=dispatch.acceptance._ref(root,f'inference/{seed}/ready.json')
        controller=dispatch.acceptance._ref(root,f'inference/{seed}/controller.json')
        value=dispatch.verify_controller(writer,seed,ready,ref,controller)
        assert all(g['pid']==os.getpid() and g['status']=='finished' for g in value['guards'].values())
        terminal=json.loads((root/f'inference/{seed}/process/process.json').read_text())
        assert terminal['pid']==pid and terminal['parent_pid']==os.getpid()
        assert terminal['argv']==dispatch.entry.worker_argv(writer,seed,ready)
        guard=json.loads((writer.root/f'control/inference-worker-{seed}.json').read_text())
        assert guard['pid']==pid and guard['status']=='finished'
    final=dispatch.acceptance.verify_boundary(writer,result['reviews'][-1])
    assert final['scope']['previous_refs']==result['reviews']
    assert final['quality_history_reaudited']
    with pytest.raises(ValueError,match='retried'):dispatch.run_panel(writer,request)


@pytest.mark.parametrize('reason',['stop','rss','timeout','interrupt','sigterm'])
def test_parent_failure_reaps_real_child_and_stops_panel(writer,monkeypatch,tmp_path,reason):
    request=prepared(writer,monkeypatch,tmp_path);original=dispatch._Monitor.check;raised=[];known=[]
    old_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    def check(self):
        path=writer.root/f'control/process-inference-{self.seed}.json'
        if self.phase=='child' and path.exists() and not raised:
            state=json.loads(path.read_text())
            if state['status']=='running':
                raised.append(reason);known.append(state['pid'])
                if reason=='stop':writer.publish_json('STOP',{'synthetic_stop':True})
                elif reason in ('rss','timeout'):
                    value=self.sample()
                    if reason=='rss':value['observation']['peak_rss_bytes']=4*1024**3
                    else:value['elapsed_seconds']=600
                    why=dispatch.violation(value['observation'],value['elapsed_seconds'],**self.limits)
                    raise RuntimeError('Synthetic parent '+str(why))
                elif reason=='interrupt':raise KeyboardInterrupt('Synthetic parent interrupt')
                else:os.kill(os.getpid(),signal.SIGTERM)
        return original(self)
    monkeypatch.setattr(dispatch._Monitor,'check',check)
    with pytest.raises((ValueError,RuntimeError,KeyboardInterrupt)):dispatch.run_panel(writer,request)
    assert raised==[reason] and len(known)==1 and not process_module._group_exists(known[0])
    assert old_handlers=={sig:signal.getsignal(sig) for sig in old_handlers}
    root=writer.root/'evidence/inference'
    assert not (root/'3511/review.json').exists() and not (root/'3512').exists()
    assert json.loads((root/'3511/process/process.json').read_text())['status']=='failed'
    assert (writer.root/'emergency/inference-controller-failure-3511.json').exists()
    if reason in ('stop','interrupt','sigterm'):assert (writer.root/'STOP').exists()
    with pytest.raises(ValueError):dispatch.run_panel(writer,request)


def test_second_real_child_failure_preserves_first_and_never_starts_third(writer,monkeypatch,tmp_path):
    request=prepared(writer,monkeypatch,tmp_path)
    monkeypatch.setenv('TOKENIZER_INFERENCE_TEST_FAIL_SEED','3512')
    with pytest.raises(RuntimeError,match='exited with code'):dispatch.run_panel(writer,request)
    root=writer.root/'evidence/inference'
    assert (root/'3511/controller.json').exists() and (root/'3511/review.json').exists()
    assert (root/'3512/worker-failure.json').exists() and not (root/'3512/review.json').exists()
    assert not (root/'3513').exists()
    for seed in (3511,3512):
        result=json.loads((root/f'{seed}/process/process.json').read_text())
        assert not process_module._group_exists(result['pid'])


@pytest.mark.parametrize('bad',['limit','pid','reason','extra','missing','status'])
def test_controller_guard_receipt_is_independently_checked(writer,monkeypatch,tmp_path,bad):
    # No extra child needed: review exact synthetic parent receipt semantics.
    root=writer.root/'evidence';parent=os.getpid();seed=3511
    def write(name,value):
        path=writer.publish_json('evidence/'+name,value)
        return dict(path=name,sha256=dispatch.file_digest(path))
    ready=write('inference/3511/ready.json',dict(parent_pid=parent))
    reviewed=write('inference/3511/review.json',dict(ready_ref=ready,kind=dispatch.acceptance.KIND,status='inference_seed_reviewed',seed=seed))
    guards={phase:dispatch._Monitor(writer,seed,phase).sample() for phase in ('prepare','child','review')}
    record=dict(kind=dispatch.KIND,status='completed',seed=seed,parent_pid=parent,ready_ref=ready,review_ref=reviewed,
        guards=guards,adopted=False,dispatch_authorized=False)
    if bad=='limit':guards['child']['limits']['timeout_seconds']=601
    elif bad=='pid':guards['review']['pid']+=1
    elif bad=='reason':guards['prepare']['reason']='failure'
    elif bad=='extra':record['unexpected']=True
    elif bad=='missing':guards.pop('review')
    else:guards['child']['status']='starting'
    ref=write('inference/3511/controller.json',record)
    with pytest.raises(ValueError):dispatch.verify_controller(writer,seed,ready,reviewed,ref)


@pytest.mark.parametrize('failure',['limit','observer'])
def test_idle_watchdog_hard_failure_has_no_live_descendants(writer,monkeypatch,tmp_path,failure):
    prepared(writer,monkeypatch,tmp_path)
    code=f'''import time
from src.research import tokenizer_wide_hybrid_inference_dispatch_v2 as d
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
w=MetadataWriter({str(writer.root)!r},{writer.lock_fds!r})
count=0
def sample(root):
 global count
 count+=1
 if count>1:
  if {failure!r}=='observer':raise RuntimeError('Synthetic observer error')
  return dict(peak_rss_bytes=4*1024**3,free_bytes=64*1024**3)
 return dict(peak_rss_bytes=1000,free_bytes=64*1024**3)
d.observation=sample
with d._IdleMonitor(w,3511,'prepare'):time.sleep(10)
'''
    import sys
    with pytest.raises(RuntimeError,match='exited with code'):
        process_module.run_process(writer,'inference',3511,[sys.executable,'-c',code],dispatch.ROOT,
            required_free_bytes=8*1024**3,output_limit_bytes=1024**2,log_limit_bytes=8192,timeout_seconds=5)
    result=json.loads((writer.root/'evidence/inference/3511/process/process.json').read_text())
    assert result['returncode']==(90 if failure=='limit' else 91)
    assert not process_module._group_exists(result['pid'])
    assert (writer.root/'emergency/inference-controller-failure-3511.json').exists()
    assert not (writer.root/'evidence/inference/3511/worker-attempt.json').exists()
