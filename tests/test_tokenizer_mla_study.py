"""Finite panel, safety budgets, and explicit MLA benchmark family prerequisites."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.research import tokenizer_attention_bench as bench
from src.research import tokenizer_mla_study as study


def row(spec, ce):
    return dict(spec, status='completed', h200={'mean_ce': ce, 'context': 128, 'block_tokens': 2048,
                'block_target_counts': [2048]*4, 'block_loss_sums': [2048*ce]*4})


def test_finite_lr_selection_fresh_panel_and_neutrality():
    screen = [row(s, 5+s['lr']) for s in study.specs()]
    selection = study.choose(screen)
    assert selection == {'mha': .00125, 'mla': .00125}
    assert [(s['family'], s['lr']) for s in study.specs()] == [('mha', .00125), ('mha', .0025), ('mla', .0025), ('mla', .00125)]
    all_specs = study.specs(selection)
    assert len(all_specs) == 10
    fresh = [s for s in all_specs if s['stage'] == 'confirm']
    assert [(s['seed'], s['family']) for s in fresh] == [(2911,'mha'),(2911,'mla'),(2912,'mla'),(2912,'mha'),(2913,'mha'),(2913,'mla')]
    with pytest.raises(ValueError): study.choose(screen[:-1])
    with pytest.raises(ValueError): study.choose(screen+[screen[0]])
    wrong = copy.deepcopy(screen); wrong[0]['seed'] = 2911
    with pytest.raises(ValueError): study.choose(wrong)
    bad = copy.deepcopy(screen); bad[0]['status'] = 'failed'
    with pytest.raises(ValueError): study.choose(bad)
    records = screen+[row(s, 5+(.005 if s['family']=='mla' else 0)) for s in fresh]
    result = study.conclusion(records,selection)
    assert result['quality_eligible'] and not result['adopted']
    with pytest.raises(ValueError): study.conclusion(records[:-1],selection)
    with pytest.raises(ValueError): study.conclusion(records+[records[-1]],selection)
    rejected = screen+[row(s, 5+(.02 if s['family']=='mla' else 0)) for s in fresh]
    assert not study.conclusion(rejected,selection)['quality_eligible']
    with pytest.raises(ValueError): study.specs({'mha': .00125, 'mla': .005})


def test_storage_reserves_all_remaining_trials():
    plan = {'reserve_bytes': 8000, 'max_output_bytes': 7000, 'per_trial_budget_bytes': 600}
    study.check_budget(plan,0,10,14000)
    with pytest.raises(RuntimeError): study.check_budget(plan,0,10,13999)
    study.check_budget(plan,4,2200,11600)
    with pytest.raises(RuntimeError): study.check_budget(plan,4,3500,20000)
    with pytest.raises(RuntimeError): study.check_budget(plan,10,6000,7999)
    with pytest.raises(ValueError): study.check_budget(plan,11,0,20000)


@pytest.mark.parametrize('family', ['mha','mla'])
def test_exact_recipe_and_schedule_changes(tmp_path,family):
    base=json.loads(study.RECIPE.read_text())['config']
    cfg=study.config(tmp_path,family,2911,.0025)
    expected=copy.deepcopy(base)
    expected['model'].update(attn_type=family,depth_width_policy='depth4_unscaled',
                            attention_width_policy='mha_head32' if family=='mha' else 'mla_quarter_head32',
                            tokenizer_architecture_policy=study.POLICY if family=='mha' else study.MLA_POLICY)
    expected['training'].update(seed=2911,learning_rate=.0025,checkpoint_dir=str(tmp_path/'checkpoints'),
                               metrics_file=str(tmp_path/'metrics.jsonl'),summary_file=str(tmp_path/'training-summary.json'))
    assert cfg==expected
    assert cfg['training']['max_iters']==1200
    assert cfg['training']['lr_schedule']['start_step']==800
    assert cfg['training']['weight_averaging']['interval']==10


@pytest.mark.parametrize('candidate', ['gqa','mla'])
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
        return {'cfg':{'model':{'attn_type':path.stem,'dim':256},'data':{'path':'unused'}}}
    monkeypatch.setattr(bench,'read_checkpoint',checkpoint)
    monkeypatch.setattr(bench,'checkpoint_loader',lambda *args:SimpleNamespace(train_data=torch.arange(96)))
    class Model:
        def __init__(self,cfg):self.family=cfg.attn_type
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
    result=bench.benchmark(tmp_path/'mha.pt',tmp_path/f'{candidate}.pt',tmp_path/'bench.json',candidate)
    assert set(result['metadata'])=={'mha',candidate}
    assert len(result['records'])==32 and len(result['correctness'])==4
    assert [r['family'] for r in result['records'][:4]]==['mha',candidate,candidate,'mha']
    assert len(calls)==40
    with pytest.raises(ValueError):bench.benchmark(Path('a'),Path('b'),Path('c'),'typo')


def test_serial_controller_freezes_selection_and_never_restarts(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    (tmp_path/'research_state').mkdir()
    monkeypatch.setattr(study.subprocess,'check_output',lambda args,**kwargs: '' if 'status' in args else 'test-commit')
    monkeypatch.setattr(study.shutil,'disk_usage',lambda p:SimpleNamespace(free=30*1024**3))
    monkeypatch.setattr(study.signal,'signal',lambda *args:None)
    # Real pilot verification, recipe construction, hashing and source checks remain exercised.
    root=tmp_path/'run';study.prepare(root)
    seen=[]
    class Process:
        pid=123
        def __init__(self,args,**kwargs):
            request=Path(args[-1]);req=json.loads(request.read_text());spec=req['spec']
            assert req['config']==study.config(request.parent,spec['family'],spec['seed'],spec['lr'])
            if spec['stage']=='confirm':
                assert json.loads((root/'selection.json').read_text())=={'mha':.00125,'mla':.00125}
            seen.append(spec)
            ce=5+spec['lr'] if spec['stage']=='screen' else 5+(.005 if spec['family']=='mla' else 0)
            Path(req['result_path']).write_text(json.dumps(row(spec,ce)))
        def wait(self,timeout):return 0
    monkeypatch.setattr(study.subprocess,'Popen',Process)
    result=study.run(root)
    assert result['status']=='completed' and len(seen)==10
    assert result['conclusion']['quality_eligible'] and not result['conclusion']['adopted']
    study.run(root);assert len(seen)==10
    stopped=tmp_path/'stopped';study.prepare(stopped);(stopped/'STOP').touch()
    assert study.run(stopped)['status']=='paused' and len(seen)==10
