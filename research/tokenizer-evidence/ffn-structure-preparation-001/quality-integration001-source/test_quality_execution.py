"""Real file retention integration and actual retained checkpoint acceptance.

The disposal payloads are explicitly synthetic bytes, never scientific weights.
The retained pilot audit loads real trained weights without updates or mutation.
"""
import copy,json,os
from pathlib import Path
from types import SimpleNamespace
import pytest
from research.ffn_quality_v1 import campaign_contract as c,campaign_rules as rules,quality_execution as q,quality_retention as r,artifact_acceptance as a
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import session_lock


def plan():return read(reference(Path(os.environ['FFN_QUALITY_PLAN_PATH'])))


@pytest.mark.parametrize('slot',[-1,12,None,True])
def test_invalid_slot_refused_before_any_io(slot):
    with pytest.raises(ValueError):q.run_once(None,slot,'0'*64)


def test_actual_quality_configs_and_guard():
    seeds=plan()['seeds'];specs=rules.screen_specs(seeds)+rules.confirmation_specs(seeds,dict(zip(rules.ARMS,[.000625,.0025])))
    for i,spec in enumerate(specs):
        cfg=c.trial_config(q.ROOT/f'evidence/quality/{i:02d}',spec,seeds);c.validate_trial_config(cfg,q.ROOT/f'evidence/quality/{i:02d}',spec,seeds)
        assert cfg['training']['max_iters']==1200 and cfg['training']['eval_every']==300
        assert cfg['training']['weight_averaging']['start_step']==800 and cfg['training']['weight_averaging']['interval']==10
    guard=q.QualityGuard(Path('/tmp/unused-quality-guard'),c.ROOT)
    assert guard.limits==dict(rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,timeout_seconds=7200,mps_driver_limit_bytes=6*1024**3)


def test_final_sink_rejects_early_or_wrong_path():
    sink=object.__new__(q.Sink);sink.raw=None;sink.base='evidence/quality/00'
    for path,step in [(q.ROOT/sink.base/'checkpoints/final.pt',20),(q.ROOT/'elsewhere.pt',1200)]:
        with pytest.raises(ValueError):sink.checkpoint(path,dict(step=step))


def test_confirmation_requires_completed_retention():
    current=dict(records=[{}]*6,cleanup=None)
    with pytest.raises(ValueError):q.next_spec(dict(seeds=plan()['seeds']),current)


def test_real_file_retention_and_corruption_rejection(record_property):
    root=Path(plan()['disposable_root']);root.mkdir();seeds=plan()['seeds'];requests=[];records=[]
    for slot,spec in enumerate(rules.screen_specs(seeds)):
        base=root/f'evidence/quality/{slot:02d}';(base/'checkpoints').mkdir(parents=True)
        cfg=c.trial_config(base,spec,seeds);files={}
        for role in ('parent','h200'):
            path=base/'checkpoints'/f'{role}.pt';path.write_bytes(('DISPOSABLE_NOT_A_CHECKPOINT_'+str(slot)+role).encode());files[role]=reference(path)
        (base/'audit.json').write_text('{"synthetic":true}\n');requests.append(dict(config=cfg,files=files));records.append(dict(spec,status='completed',h200_ce=5.+slot/100))
    selection=rules.choose_rates(seeds,records);original=r.manifest(root,seeds,records,selection,requests)
    assert len(original['delete'])==8 and len(original['retain'])==4
    bad=copy.deepcopy(requests);bad[0]['files']['parent']=requests[2]['files']['parent']
    with pytest.raises(ValueError):r.manifest(root,seeds,records,selection,bad)
    with pytest.raises(ValueError):r.manifest(root,seeds,records[:5],selection,requests)
    protected=next(iter(original['retain']));data=Path(protected).read_bytes();Path(protected).write_bytes(b'corrupt')
    with pytest.raises(ValueError):r.manifest(root,seeds,records,selection,requests)
    Path(protected).write_bytes(data)
    context=dict(scope='disposable_retention_integration_only');destination=Path(plan()['disposable_retention'])
    with session_lock(root) as fd:
        writer=SimpleNamespace(root=root,lock_fds=(int(os.environ['FFN_QUALITY_GLOBAL_FD']),fd))
        ref=r.execute(writer,destination,seeds,records,selection,requests,context,lambda:context)
        assert r.verify(ref,context)==original['delete']
        with pytest.raises(ValueError):r.execute(writer,destination,seeds,records,selection,requests,context,lambda:context)
    assert r.inventory(root/'evidence')=={k:v for k,v in original['inventory'].items() if k not in original['delete']}
    removed=next(iter(original['delete']));Path(removed).write_bytes(b'reappeared')
    with pytest.raises(ValueError):r.verify(ref,context)
    Path(removed).unlink()  # Disposable test bytes only; restores already-audited final state.
    assert r.verify(ref,context)==original['delete']
    record_property('retention',dict(receipt=ref,deleted_files=8,protected_files=len(original['inventory'])-8,scientific_weights_deleted=False))


def test_current_sources_accept_real_pilot_pair(record_property):
    receipts={}
    for arm,req in plan()['requests'].items():
        accepted=a.accept(req,plan()['validation'][arm]);assert accepted['accepted'] and not accepted['launch_permitted'];receipts[arm]=accepted
    record_property('real_parent_acceptance',dict(arms=list(receipts),audits={k:v['audit'] for k,v in receipts.items()},optimizer_updates=0))
