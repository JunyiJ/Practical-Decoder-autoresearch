"""Real two-child serial transport and rejection of corrupted completion evidence."""
import copy
import json
import os
from pathlib import Path
import pytest
from . import campaign_contract as c, fixture_dispatch as d
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter


@pytest.fixture(scope='module')
def run():
    plan=json.loads(Path(os.environ['BLOCK_NORM_PLAN_PATH']).read_text())
    root=Path(plan['dispatch_root']);root.mkdir()
    with session_lock(root) as campaign_fd:
        writer=MetadataWriter(root,(int(os.environ['BLOCK_NORM_GLOBAL_FD']),campaign_fd))
        rows=[]
        for slot,arm in enumerate(('rms_incumbent','layer_matched_epsilon')):
            receipt=d.dispatch_fixture(writer,slot,plan['requests'][arm])
            base=root/d.stem(slot)
            rows.append(dict(slot=slot,arm=arm,receipt=receipt,request=plan['requests'][arm],
                seal_ref=d.reference(base/'seal.json'),process=json.loads((base/'process/process.json').read_text()),
                result=json.loads((base/'worker.json').read_text()),guard=json.loads((base/'worker-guard.json').read_text())))
        yield writer,rows,plan


def test_real_children_serial_and_parent_acceptance(run,record_property):
    writer,rows,plan=run
    assert rows[0]['process']['pid']!=rows[1]['process']['pid']
    for row in rows:
        receipt=row['receipt'];accepted=receipt['artifact_acceptance']
        assert accepted['validation']==plan['validation_reports'][row['arm']]
        assert accepted['accepted'] and not accepted['launch_permitted'] and not accepted['quality_adopted']
        assert row['guard']['pid']==row['process']['pid'] and row['guard']['status']=='finished'
        assert accepted['audit']['completed_steps']==2
    record_property('real_worker_dispatch',rows)


@pytest.mark.parametrize('slot',[0,1])
def test_completed_attempt_cannot_be_dispatched_again(run,slot):
    writer,rows,_=run
    with pytest.raises(ValueError,match='cannot be dispatched again'):
        d.dispatch_fixture(writer,slot,rows[slot]['request'])


@pytest.mark.parametrize('case',['nonzero','wrong_argv','wrong_parent','truncated','wrong_limits','cleanup_error',
    'worker_pid','worker_request','worker_environment','worker_extra','worker_training','worker_holdout',
    'guard_failed','guard_pid','guard_limits','guard_rss','guard_driver'])
def test_corrupt_completion_rejected_before_model_acceptance(run,case):
    writer,rows,_=run;row=copy.deepcopy(rows[0]);p=row['process'];r=row['result'];g=row['guard']
    if case=='nonzero':p['returncode']=1
    elif case=='wrong_argv':p['argv']=['wrong']
    elif case=='wrong_parent':p['parent_pid']+=1
    elif case=='truncated':p['log_truncated']=True
    elif case=='wrong_limits':p['timeout_seconds']=601
    elif case=='cleanup_error':p['cleanup_error']='failed'
    elif case=='worker_pid':r['pid']+=1
    elif case=='worker_request':r['request']['sha256']='0'*64
    elif case=='worker_environment':r['environment']['torch_threads']=8
    elif case=='worker_extra':r['unexpected']=True
    elif case=='worker_training':r['optimizer_updates']=1
    elif case=='worker_holdout':r['holdout_scored']=True
    elif case=='guard_failed':g['status']='failed'
    elif case=='guard_pid':g['pid']+=1
    elif case=='guard_limits':g['limits']['mps_driver_limit_bytes']*=2
    elif case=='guard_rss':g['observation']['peak_rss_bytes']=4*1024**3
    else:g['observation']['peak_observed_mps_driver_bytes']=6*1024**3
    with pytest.raises(ValueError):
        d.verify_completion(writer,0,row['seal_ref'],rows[0]['process']['argv'],p,r,g)


def test_parent_cannot_impersonate_child_entry(run):
    writer,rows,_=run;row=rows[0]
    with pytest.raises(ValueError,match='Live parent identity'):
        d.verify_entry(writer,0,row['seal_ref'],row['process']['argv'])


def test_stop_prevents_acceptance_and_new_dispatch(run):
    writer,rows,_=run;row=rows[0]
    writer.publish_json('STOP',dict(reason='readonly fixture test complete; no further dispatch'))
    with pytest.raises(ValueError,match='STOP'):
        d.verify_completion(writer,0,row['seal_ref'],row['process']['argv'],row['process'],row['result'],row['guard'])
    with pytest.raises(ValueError,match='STOP'):
        d.dispatch_fixture(writer,0,row['request'])
