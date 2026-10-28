"""Prospective table panel fairness, fresh gates, storage and serial dispatch."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.research import tokenizer_table_study as s


def row(spec, ce):
    return dict(spec, status='completed',h200=dict(mean_ce=ce,context=128,block_tokens=2048,
            block_target_counts=[2048]*4,block_loss_sums=[2048*ce]*4))


def test_independent_selection_both_controls_and_fresh_gates():
    screen=[row(x,5) for x in s.specs()]
    selection=s.choose(screen)
    assert selection==dict.fromkeys(s.FAMILIES,.00125)
    panel=s.specs(selection)
    assert len(panel)==15 and len({r['id'] for r in panel})==15
    assert [x['family'] for x in panel[6:]]==list(s.FAMILIES+s.FAMILIES[1:]+s.FAMILIES[:1]+s.FAMILIES[2:]+s.FAMILIES[:2])
    assert {x['seed'] for x in panel[6:]}==set(s.SEEDS)
    # Independent LR selection can retain different rates; no arm is eliminated.
    varied=copy.deepcopy(screen);varied[2]['h200']['mean_ce']=4
    assert s.choose(varied)['matched_untied']==.0025
    for d,strict,neutral in [(-.03,True,True),(.005,False,True),(.02,False,False)]:
        records=screen+[row(x,5+d*(x['family']=='scaled_tied')) for x in panel[6:]]
        result=s.conclusion(records,selection)
        assert result['tied_strict_quality_gate']==strict and result['tied_neutral_quality_gate']==neutral
        assert not result['adopted']
    # Tied beating legacy cannot hide failure against matched control.
    records=screen+[row(x,4.9 if x['family']=='matched_untied' else 4.95 if x['family']=='scaled_tied' else 5) for x in panel[6:]]
    result=s.conclusion(records,selection)
    assert not result['tied_neutral_quality_gate'] and result['matched_strict_quality_gate']
    for bad in (records[:-1],records+[records[0]]):
        with pytest.raises(ValueError):s.conclusion(bad,selection)
    records=screen+[row(x,5+(.02 if x['seed']==3311 else -.2)*(x['family']=='scaled_tied')) for x in panel[6:]]
    assert not s.conclusion(records,selection)['tied_neutral_quality_gate']
    bad=copy.deepcopy(screen);bad[0]['h200']['mean_ce']=float('nan')
    with pytest.raises(ValueError):s.choose(bad)


def test_arm_specific_remaining_budget():
    p=dict(reserve_bytes=8*1024**3,per_trial_budget_bytes=s.ALLOWANCE,max_output_bytes=5*sum(s.ALLOWANCE.values())+4*1024**2)
    need=p['reserve_bytes']+p['max_output_bytes']
    s.check_budget(p,[],0,need)
    with pytest.raises(RuntimeError):s.check_budget(p,[],0,need-1)
    records=[row(s.specs()[0],5)]
    s.check_budget(p,records,100,need-s.ALLOWANCE['legacy_untied'])
    with pytest.raises(ValueError):s.check_budget(p,records*2,0,need)


def test_serial_controller_freeze_and_complete_panel(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path);(tmp_path/'research_state').mkdir()
    monkeypatch.setattr(s,'verify_pilot',lambda:None)
    monkeypatch.setattr(s.subprocess,'check_output',lambda args,**kw:'' if 'status' in args else 'test')
    monkeypatch.setattr(s.shutil,'disk_usage',lambda p:SimpleNamespace(free=40*1024**3))
    monkeypatch.setattr(s.signal,'signal',lambda *args:None)
    root=tmp_path/'run';s.prepare(root);seen=[]
    class Process:
        pid=123
        def __init__(self,args,**kw):
            reqpath=Path(args[-1]);req=json.loads(reqpath.read_text());spec=req['spec'];seen.append(spec)
            assert len(kw['pass_fds'])==2 and kw['start_new_session']
            assert req['config']==s.config(reqpath.parent,spec['family'],spec['seed'],spec['lr'])
            if spec['stage']=='confirm':assert (root/'selection.json').is_file()
            r=row(spec,5 if spec['stage']=='screen' else 5-.03*(spec['family']=='scaled_tied'))
            ckpt=reqpath.parent/'fixture.pt';ckpt.write_bytes(b'test')
            r['h200'].update(checkpoint=str(ckpt),checkpoint_sha256=s.file_digest(ckpt))
            r.update(raw=dict(checkpoint=str(ckpt),checkpoint_sha256=s.file_digest(ckpt)),holdout_consumed=False,request_sha256=s.file_digest(reqpath))
            Path(req['result_path']).write_text(json.dumps(r))
        def wait(self,timeout):return 0
    monkeypatch.setattr(s.subprocess,'Popen',Process)
    result=s.run(root)
    assert len(seen)==15 and result['conclusion']['tied_strict_quality_gate']
    s.run(root);assert len(seen)==15
    # Timing parent uses the same locks and refuses a second attempt.
    from src.research import tokenizer_table_bench as b, tokenizer_study as base
    monkeypatch.setattr(base,'ROOT',tmp_path)
    timing_calls=[]
    class TimingProcess:
        pid=124;returncode=0
        def __init__(self,args,**kw):
            assert len(kw['pass_fds'])==2 and kw['start_new_session']
            timing_calls.append(args)
            metadata={k:dict(sha256=s.file_digest(Path(args[args.index('--'+k.replace('_','-'))+1]))) for k in b.KINDS}
            Path(args[-1]).write_text(json.dumps(dict(status='completed',metadata=metadata,records=[{}]*48,correctness=[{}]*6)))
        def poll(self):return 0
    monkeypatch.setattr(b.subprocess,'Popen',TimingProcess)
    b.run_review(root)
    assert len(timing_calls)==3
    with pytest.raises(FileExistsError):b.run_review(root)
    assert len(timing_calls)==3
    interrupted=tmp_path/'interrupted';s.prepare(interrupted)
    state=json.loads((interrupted/'state.json').read_text());state['status']='running';s.atomic_json(interrupted/'state.json',state)
    with pytest.raises(ValueError,match='Interrupted'):s.run(interrupted)
    changed=tmp_path/'changed';s.prepare(changed)
    monkeypatch.setattr(s,'source_fingerprint',lambda:{'changed':'yes'})
    with pytest.raises(ValueError,match='Frozen'):s.run(changed)


def test_three_arm_benchmark_order_and_storage(tmp_path,monkeypatch):
    import torch
    from src.research import tokenizer_table_bench as b
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    monkeypatch.setattr(b,'start_memory_guard',lambda:None)
    original_to=torch.Tensor.to
    def to(self,*a,**kw):return self if a and a[0]=='mps' else original_to(self,*a,**kw)
    monkeypatch.setattr(torch.Tensor,'to',to)
    def checkpoint(path):
        return dict(cfg=dict(model=dict(tokenizer_architecture_policy=b.POLICY,weight_tying_kind=path.stem,dim=256),training=dict(seed=3311),data=dict(path='unused')),averaged_weights=dict(snapshots=41),step=1200)
    monkeypatch.setattr(b,'read_checkpoint',checkpoint)
    monkeypatch.setattr(b,'checkpoint_loader',lambda *a:SimpleNamespace(train_data=torch.arange(96)))
    class Model:
        def __init__(self,cfg):self.family=cfg.weight_tying_kind;self.p=torch.ones(1)
        def to(self,*a):return self
        def eval(self):return self
        def parameters(self):return [self.p,self.p] # Deliberately aliased storage must count once.
    monkeypatch.setattr(b,'GPT',Model);monkeypatch.setattr(b,'load_model_weights',lambda *a:None)
    monkeypatch.setattr(b,'file_digest',lambda p:p.stem);monkeypatch.setattr(b,'check_cached',lambda *a:0.)
    calls=[]
    def timed(model,cfg,ids):
        calls.append(model.family);return dict(prefill_seconds=.01,decode_seconds=.1,kv_bytes=100)
    monkeypatch.setattr(b,'timed',timed)
    paths={k:tmp_path/(k+'.pt') for k in b.KINDS}
    r=b.benchmark(paths,tmp_path/'result.json')
    assert len(r['records'])==48 and len(r['correctness'])==6 and len(calls)==60
    assert [x['family'] for x in r['records'][:9]]==list(b.KINDS+b.KINDS[1:]+b.KINDS[:1]+b.KINDS[2:]+b.KINDS[:2])
    assert all(v['parameter_bytes']==4 for v in r['metadata'].values())
    with pytest.raises(ValueError):b.benchmark({},tmp_path/'wrong.json')
