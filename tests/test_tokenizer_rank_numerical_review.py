"""Terminal-review schema tests with tiny synthetic files; no model or worker."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

PATH=Path(__file__).resolve().parents[1]/'research/tokenizer-evidence/wide-hybrid-preparation-001/review_rank_numerical_v2.py'
spec=importlib.util.spec_from_file_location('rank_numerical_review',PATH)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def panel(tmp_path,monkeypatch):
    root=tmp_path/'synthetic';root.mkdir();(root/'.lock').touch()
    write(root/'plan.json',dict(synthetic=True));sha=r.file_digest(root/'plan.json')
    write(root/'attempt.json',dict(synthetic=True))
    write(root/'completed.json',dict(plan_sha256=sha,status='all27_phases_complete_pending_independent_review'))
    plan=dict(source={'synthetic':True},runtime={'synthetic':True},configs={})
    for i,case in enumerate(r.p.CASES):
        part=root/f'{i:02d}/partial.pt';part.parent.mkdir();part.write_bytes(b'not a real checkpoint')
        partial_sha=r.file_digest(part)
        for j,phase in enumerate(r.p.PHASES):
            d=root/f'{i:02d}'/phase;d.mkdir();(d/'worker.log').write_bytes(b'')
            pid=1000+i*3+j;config_sha='c'*64;plan['configs'][f'{i:02d}/{phase}']=config_sha
            write(d/'process.json',dict(status='completed',returncode=0,error=None,pid=pid,
                elapsed_seconds=1.,timeout_seconds=600,required_free_bytes=(8+2*(9-i))*r.p.GiB,
                reserve_bytes=8*r.p.GiB,output_limit_bytes=2*r.p.GiB,log_limit_bytes=4*1024**2,
                observation=dict(free_bytes=64*r.p.GiB,output_bytes=10000,log_bytes=0),
                argv=[sys.executable,str(r.HERE/'run_rank_numerical_v2.py'),'worker','--root',str(root),
                    '--plan-sha256',sha,'--index',str(i),'--phase',phase,'--global-fd','3','--local-fd','4']))
            write(d/'guard.json',dict(status='finished',reason=None,pid=pid,elapsed_seconds=.5,
                limits=dict(rss_limit_bytes=4*r.p.GiB,reserve_bytes=8*r.p.GiB,timeout_seconds=600),
                observation=dict(peak_rss_bytes=10000,free_bytes=64*r.p.GiB)))
            hashes=dict.fromkeys(('model','optimizer','ema','rng','loader_rng'),'a'*64)
            if phase=='partial':hashes['model']='b'*64
            steps={'baseline':[1,2,3,4],'partial':[1],'resumed':[2,3,4]}[phase]
            write(d/'receipt.json',dict(case=case,phase=phase,plan_sha256=sha,config_sha256=config_sha,
                completed_steps=1 if phase=='partial' else 4,
                ema_observations=[dict(step=s,snapshots=s) for s in steps],
                ema_sample_max_absolute_error=0.,parameter_sample_positions='first_middle_last_each_parameter',
                exact_resume_equal=phase=='resumed',ema_load_and_resume_rejection_checked=phase=='resumed',
                test_loss_computed=False,quality_evidence=False,
                partial_checkpoint_sha256=None if phase=='baseline' else partial_sha,hashes=hashes))
    monkeypatch.setattr(r.p,'verify',lambda root_arg,sha_arg:plan)
    return root,sha,tmp_path/'report'


def test_all27_synthetic_phase_records_bind(panel):
    root,sha,out=panel;r.review(root,sha,out)
    report=json.loads((out/'meta/numerical.json').read_text())
    assert report['status']=='passed' and len(report['cases'])==9
    trace=json.loads((out/'meta/numerical-trace.json').read_text())
    assert len(trace['original_files'])==121 and len(trace['cases'])==9


@pytest.mark.parametrize('bad',['failed_guard','resume_hash','checkpoint_hash','extra_file','nonfinite_error','wrong_argv','missing_phase'])
def test_incomplete_or_changed_evidence_cannot_publish(panel,bad):
    root,sha,out=panel
    if bad=='extra_file':(root/'extra.json').write_text('{}')
    elif bad=='missing_phase':(root/'08/resumed/receipt.json').unlink()
    else:
        name='guard.json' if bad=='failed_guard' else 'process.json' if bad=='wrong_argv' else 'receipt.json'
        path=root/'08/resumed'/name;value=json.loads(path.read_text())
        if bad=='failed_guard':value['status']='failed'
        elif bad=='resume_hash':value['hashes']['model']='f'*64
        elif bad=='checkpoint_hash':value['partial_checkpoint_sha256']='f'*64
        elif bad=='nonfinite_error':value['ema_sample_max_absolute_error']=float('nan')
        elif bad=='wrong_argv':value['argv'][8]='7'
        write(path,value)
    with pytest.raises(AssertionError):r.review(root,sha,out)
    assert not out.exists()
