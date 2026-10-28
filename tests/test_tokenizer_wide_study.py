"""Prospective width512 direct attention panel fairness, gates, storage and serial dispatch."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.research import tokenizer_wide_study as s


def row(spec, ce):
    return dict(spec, status='completed',h200=dict(mean_ce=ce,context=128,block_tokens=2048,
            block_target_counts=[2048]*4,block_loss_sums=[2048*ce]*4))


def test_independent_selection_complete_fresh_quality_gates():
    screen=[row(x,5) for x in s.specs()];selection=s.choose(screen)
    assert selection==dict.fromkeys(s.FAMILIES,.000625)
    panel=s.specs(selection)
    assert len(panel)==12 and len({r['id'] for r in panel})==12
    assert [x['family'] for x in panel[6:]]==['gqa','mla','mla','gqa','gqa','mla']
    assert {x['seed'] for x in panel[6:]}==set(s.SEEDS)
    varied=copy.deepcopy(screen);varied[3]['h200']['mean_ce']=4
    assert s.choose(varied)==dict(gqa=.000625,mla=.0025)
    for d,strict,neutral,gqa in [(-.03,True,True,False),(.005,False,True,False),(.03,False,False,True)]:
        records=screen+[row(x,5+d*(x['family']=='mla')) for x in panel[6:]]
        result=s.conclusion(records,selection)
        assert result['mla_strict_quality_gate']==strict and result['quality_eligible']==neutral
        assert result['gqa_strict_quality_gate']==gqa and not result['adopted']
    for bad in (records[:-1],records+[records[0]]):
        with pytest.raises(ValueError):s.conclusion(bad,selection)
    # Favorable mean cannot hide one failed seed.
    records=screen+[row(x,5+(.02 if x['seed']==3411 else -.2)*(x['family']=='mla')) for x in panel[6:]]
    assert not s.conclusion(records,selection)['quality_eligible']
    bad=copy.deepcopy(screen);bad[0]['h200']['mean_ce']=float('nan')
    with pytest.raises(ValueError):s.choose(bad)
    # Preserve tuned selection and reject an undeclared width/rate/seed.
    with pytest.raises(ValueError):s.config(Path('/tmp/none'),'gqa',3401,.005)
    with pytest.raises(ValueError):s.config(Path('/tmp/none'),'gqa',3301,.00125)


def test_arm_specific_remaining_budget():
    p=dict(reserve_bytes=8*1024**3,per_trial_budget_bytes=s.ALLOWANCE,max_output_bytes=6*sum(s.ALLOWANCE.values())+4*1024**2)
    need=p['reserve_bytes']+p['max_output_bytes']
    s.check_budget(p,[],0,need)
    with pytest.raises(RuntimeError):s.check_budget(p,[],0,need-1)
    records=[row(s.specs()[0],5)]
    s.check_budget(p,records,100,need-s.ALLOWANCE['gqa'])
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
            r=row(spec,5 if spec['stage']=='screen' else 5-.03*(spec['family']=='mla'))
            ckpt=reqpath.parent/'fixture.pt';ckpt.write_bytes(b'test')
            r['h200'].update(checkpoint=str(ckpt),checkpoint_sha256=s.file_digest(ckpt))
            r.update(raw=dict(checkpoint=str(ckpt),checkpoint_sha256=s.file_digest(ckpt)),holdout_consumed=False,request_sha256=s.file_digest(reqpath))
            Path(req['result_path']).write_text(json.dumps(r))
        def wait(self,timeout):return 0
    monkeypatch.setattr(s.subprocess,'Popen',Process)
    result=s.run(root)
    assert len(seen)==12 and result['conclusion']['mla_strict_quality_gate']
    s.run(root);assert len(seen)==12
    # Timing parent uses the same locks and refuses a second attempt.
    from src.research import tokenizer_wide_bench as b, tokenizer_study as base
    monkeypatch.setattr(base,'ROOT',tmp_path)
    timing_calls=[]
    class TimingProcess:
        pid=124;returncode=0
        def __init__(self,args,**kw):
            assert len(kw['pass_fds'])==2 and kw['start_new_session']
            timing_calls.append(args)
            metadata={k:dict(sha256=s.file_digest(Path(args[args.index('--'+k.replace('_','-'))+1]))) for k in b.KINDS}
            Path(args[-1]).write_text(json.dumps(dict(status='completed',metadata=metadata,records=[{}]*32,correctness=[{}]*4)))
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


def test_paired_benchmark_order_and_storage(tmp_path,monkeypatch):
    import torch
    from src.research import tokenizer_wide_bench as b
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    monkeypatch.setattr(b,'start_memory_guard',lambda:None)
    original_to=torch.Tensor.to
    def to(self,*a,**kw):return self if a and a[0]=='mps' else original_to(self,*a,**kw)
    monkeypatch.setattr(torch.Tensor,'to',to)
    def checkpoint(path):
        return dict(cfg=dict(model=dict(tokenizer_architecture_policy=b.POLICIES[path.stem],attn_type=path.stem,dim=512),training=dict(seed=3411),data=dict(path='unused')),averaged_weights=dict(snapshots=41),step=1200)
    monkeypatch.setattr(b,'read_checkpoint',checkpoint)
    monkeypatch.setattr(b,'checkpoint_loader',lambda *a:SimpleNamespace(train_data=torch.arange(96)))
    class Model:
        def __init__(self,cfg):self.family=cfg.attn_type;self.p=torch.ones(1)
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
    assert len(r['records'])==32 and len(r['correctness'])==4 and len(calls)==40
    assert [x['family'] for x in r['records'][:6]]==['gqa','mla','mla','gqa','gqa','mla']
    assert all(v['parameter_bytes']==4 for v in r['metadata'].values())
    with pytest.raises(ValueError):b.benchmark({},tmp_path/'wrong.json')


def test_quality_failure_halts_without_dispatching_another_trial(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'ROOT',tmp_path);(tmp_path/'research_state').mkdir()
    monkeypatch.setattr(s,'verify_pilot',lambda:None)
    monkeypatch.setattr(s.subprocess,'check_output',lambda args,**kw:'' if 'status' in args else 'test')
    monkeypatch.setattr(s.shutil,'disk_usage',lambda p:SimpleNamespace(free=40*1024**3))
    monkeypatch.setattr(s.signal,'signal',lambda *args:None)
    root=tmp_path/'failed';s.prepare(root);seen=[]
    class Failed:
        pid=123
        def __init__(self,args,**kw):seen.append(args)
        def wait(self,timeout):return 86
    monkeypatch.setattr(s.subprocess,'Popen',Failed)
    result=s.run(root)
    assert result['status']=='failed' and result['records'][0]['exit_code']==86
    assert len(seen)==1
    s.run(root);assert len(seen)==1
