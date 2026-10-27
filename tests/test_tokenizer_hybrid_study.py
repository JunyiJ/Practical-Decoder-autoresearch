"""Finite hybrid selection, confirmation gates, resource and worker control."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.research import tokenizer_hybrid_study as study
from src.research import tokenizer_hybrid_bench as bench


def row(s, ce):
    return dict(s, status='completed', h200=dict(mean_ce=ce, context=128, block_tokens=2048,
                block_target_counts=[2048]*4, block_loss_sums=[2048*ce]*4))


@pytest.mark.parametrize('control', ['gated_mla', 'kda'])
def test_selection_and_both_routes(control):
    screen = [row(s, 5+s['lr']+(s['family'] not in ('hybrid', control))*.1) for s in study.specs()]
    selection = study.choose(screen)
    assert selection == dict(control=control, rates={f:.00125 for f in study.FAMILIES})
    panel = study.specs(selection)
    assert len(panel) == 12 and len({s['id'] for s in panel}) == 12
    assert {s['seed'] for s in panel[6:]} == {3011, 3012, 3013}
    assert [s['family'] for s in panel[6:]] == ['hybrid',control,control,'hybrid','hybrid',control]
    for delta, strict, neutral in [(-.03, True, True), (.005, False, True), (.02, False, False)]:
        records = screen + [row(s, 5+(delta if s['family']=='hybrid' else 0)) for s in panel[6:]]
        result = study.conclusion(records, selection)
        assert result['strict_quality_gate'] == strict
        assert result['neutral_quality_gate'] == neutral
        assert result['memory_route_eligible'] == (neutral and control=='gated_mla')
        assert not result['adopted']
        with pytest.raises(ValueError): study.conclusion(records[:-1], selection)
        with pytest.raises(ValueError): study.conclusion(records+[records[0]], selection)
    with pytest.raises(ValueError): study.choose(screen[:-1])
    bad = copy.deepcopy(screen); bad[0]['seed'] = 3011
    with pytest.raises(ValueError): study.choose(bad)
    bad = copy.deepcopy(screen); bad[0]['h200']['mean_ce'] = float('nan')
    with pytest.raises(ValueError): study.choose(bad)
    with pytest.raises(ValueError): study.specs(dict(control='hybrid', rates=selection['rates']))
    # One bad seed cannot hide behind a favorable mean.
    records = screen + [row(s, 5+(.02 if s['seed']==3011 else -.2)*(s['family']=='hybrid')) for s in panel[6:]]
    assert not study.conclusion(records, selection)['neutral_quality_gate']


def test_budget():
    p = dict(reserve_bytes=8000, per_trial_budget_bytes=600, max_output_bytes=7300)
    study.check_budget(p, 0, 10, 15200)
    with pytest.raises(RuntimeError): study.check_budget(p, 0, 10, 15199)
    study.check_budget(p, 6, 3500, 11600)
    with pytest.raises(RuntimeError): study.check_budget(p, 6, 3800, 11600)
    with pytest.raises(RuntimeError): study.check_budget(p, 12, 7100, 7999)
    with pytest.raises(ValueError): study.check_budget(p, 13, 0, 20000)


@pytest.mark.parametrize('family', study.FAMILIES)
def test_fixed_recipe(tmp_path, family):
    cfg = study.config(tmp_path, family, 3011, .0025)
    assert tuple(cfg['model']['attention_layout']) == study.LAYOUT[family]
    assert cfg['model']['dim']==64 and cfg['model']['rope']==0
    t=cfg['training']
    assert t['max_iters']==1200 and t['lr_schedule']['start_step']==800
    assert t['weight_averaging']['start_step']==800 and t['weight_averaging']['interval']==10
    with pytest.raises(ValueError): study.config(tmp_path, 'mha', 3011, .0025)


def test_serial_controller_freeze_and_no_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    (tmp_path/'research_state').mkdir()
    resource=tmp_path/'resource.json'; resource.write_text(json.dumps(dict(status='passed',per_trial_budget_bytes=100000)))
    preflight=tmp_path/'preflight.json'; preflight.write_text('{}')
    monkeypatch.setattr(study,'RESOURCE',resource);monkeypatch.setattr(study,'PREFLIGHT',preflight)
    monkeypatch.setattr(study,'verify_pilot',lambda:None)
    monkeypatch.setattr(study.subprocess,'check_output',lambda args,**kw:'' if 'status' in args else 'test')
    monkeypatch.setattr(study.shutil,'disk_usage',lambda p:SimpleNamespace(free=20*1024**3))
    monkeypatch.setattr(study.signal,'signal',lambda *args:None)
    seen=[]
    root=tmp_path/'run'; study.prepare(root)
    class Process:
        pid=123
        def __init__(self,args,**kw):
            req=json.loads(Path(args[-1]).read_text());s=req['spec']
            assert req['config']==study.config(Path(args[-1]).parent,s['family'],s['seed'],s['lr'])
            if s['stage']=='confirm': assert (root/'selection.json').exists()
            seen.append(s)
            ce=5+s['lr'] if s['stage']=='screen' else 5-.03*(s['family']=='hybrid')
            Path(req['result_path']).write_text(json.dumps(row(s,ce)))
        def wait(self,timeout):return 0
    monkeypatch.setattr(study.subprocess,'Popen',Process)
    result=study.run(root)
    assert result['status']=='completed' and len(seen)==12 and result['conclusion']['strict_quality_gate']
    study.run(root); assert len(seen)==12
    stopped=tmp_path/'stopped';study.prepare(stopped);(stopped/'STOP').touch()
    assert study.run(stopped)['status']=='paused' and len(seen)==12
    changed=tmp_path/'changed';study.prepare(changed)
    monkeypatch.setattr(study,'source_fingerprint',lambda:{'changed':'yes'})
    with pytest.raises(ValueError,match='Frozen'):study.run(changed)

@pytest.mark.parametrize('candidate', ['gated_mla','kda'])
def test_benchmark_preserves_family_labels_and_alternates(tmp_path,monkeypatch,candidate):
    # Exercise the real benchmark orchestration on CPU stubs; GPU math has a separate MPS pilot.
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    monkeypatch.setattr(bench,'start_memory_guard',lambda:None)
    original_to=torch.Tensor.to
    def tensor_to(self,*args,**kwargs):
        if args and args[0]=='mps':return self
        return original_to(self,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',tensor_to)
    def checkpoint(path):
        return {'cfg':{'model':{'attn_type':'hybrid','dim':64,'attention_layout':study.LAYOUT[path.stem], 'test_family':path.stem},'data':{'path':'unused'}}}
    monkeypatch.setattr(bench,'read_checkpoint',checkpoint)
    monkeypatch.setattr(bench,'checkpoint_loader',lambda *args:SimpleNamespace(train_data=torch.arange(96)))
    class Model:
        def __init__(self,cfg):self.family=cfg.test_family
        def to(self,*args):return self
        def eval(self):return self
        def parameters(self):return [torch.ones(1)]
    monkeypatch.setattr(bench,'GPT',Model)
    monkeypatch.setattr(bench,'load_model_weights',lambda *args:None)
    monkeypatch.setattr(bench,'file_digest',lambda p:p.stem)
    monkeypatch.setattr(bench,'check_cached',lambda *args:0.)
    calls=[]
    def timed(model,cfg,ids):
        calls.append((len(ids),model.family))
        return {'prefill_seconds': .01,'decode_seconds': .1,'kv_bytes': 100}
    monkeypatch.setattr(bench,'timed',timed)
    result=bench.benchmark(tmp_path/f'{candidate}.pt',tmp_path/'hybrid.pt',tmp_path/'bench.json',candidate)
    assert set(result['metadata'])=={'hybrid',candidate}
    assert len(result['records'])==32 and len(result['correctness'])==4
    assert [r['family'] for r in result['records'][:4]]==[candidate,'hybrid','hybrid',candidate]
    assert len(calls)==40
    with pytest.raises(ValueError):bench.benchmark(Path('a'),Path('b'),Path('c'),'typo')

