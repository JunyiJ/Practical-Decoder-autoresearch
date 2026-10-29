"""Small orchestration checks; no model or closed numerical fixture invoked."""
import importlib.util
import json
from pathlib import Path
import pytest
import torch

PATH=Path(__file__).resolve().parents[1]/'research/tokenizer-evidence/wide-hybrid-preparation-001/run_rank_numerical_v2.py'
spec=importlib.util.spec_from_file_location('rank_numerical_procedure',PATH)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def test_stream_hash_detects_values_shapes_dtype_and_nonfinite():
    x=torch.arange(600000,dtype=torch.float32)
    h=p.tree_hash({'state':x})
    assert h==p.tree_hash({'state':x.clone()})
    assert h!=p.tree_hash({'state':x.view(2,-1)})
    assert h!=p.tree_hash({'state':x.double()})
    x[-1]+=1;assert h!=p.tree_hash({'state':x})
    for bad in (float('nan'),float('inf')):
        with pytest.raises(AssertionError):p.tree_hash(torch.tensor([bad]))


def test_exact_configs_have_one_partial_and_one_resume(tmp_path):
    for i,case in enumerate(p.CASES):
        configs={phase:p.configuration(tmp_path,i,phase) for phase in p.PHASES}
        for phase,cfg in configs.items():
            assert cfg['model']['dim']==512 and cfg['model']['attention_bottleneck_divisor']==case['divisor']
            assert cfg['training']['max_iters']==4 and cfg['training']['batch_size']==8
            assert cfg['model']['block_size']==128 and cfg['training']['device']=='cpu'
        assert configs['partial']['training']['max_train_seconds']==1e-9
        assert configs['baseline']['training']['resume_from'] is None
        assert configs['resumed']['training']['resume_from']==str(tmp_path/f'{i:02d}/partial.pt')


def test_freeze_identity_and_stop_rejections(tmp_path):
    root=tmp_path/'panel';p.freeze(root);sha=p.file_digest(root/'plan.json')
    assert p.verify(root,sha)['max_phases']==27
    with pytest.raises(FileExistsError):p.freeze(root)
    with pytest.raises(AssertionError):p.verify(root,'0'*64)
    (root/'STOP').touch()
    with pytest.raises(AssertionError):p.verify(root,sha)


def test_serial_dispatch_stops_first_failed_guard_and_cannot_retry(tmp_path,monkeypatch):
    root=tmp_path/'panel';p.freeze(root);sha=p.file_digest(root/'plan.json')
    monkeypatch.setattr(p,'GLOBAL_LOCK',tmp_path/'global/.lock')
    calls=[]
    def fake(argv,cwd,output,fds,**kw):
        calls.append(output.name);output.mkdir()
        assert kw['timeout_seconds']==600 and kw['output_limit_bytes']==2*p.GiB
        assert kw['required_free_bytes']==26*p.GiB
        (output/'guard.json').write_text(json.dumps(dict(status='finished' if len(calls)==1 else 'failed')))
        (output/'receipt.json').write_text('{}')
    monkeypatch.setattr(p,'run_process',fake)
    with pytest.raises(AssertionError):p.run(root,sha)
    assert calls==['baseline','partial'] and (root/'STOP').is_file() and (root/'failure.json').is_file()
    with pytest.raises(AssertionError):p.run(root,sha)
    assert calls==['baseline','partial']


def test_prior_artifact_mutation_is_rejected(tmp_path):
    path=tmp_path/'receipt.json';path.write_text('{"passed":true}')
    bound={str(path):p.file_digest(path)};p.recheck_prefix(bound)
    path.write_text('{"passed":false}')
    with pytest.raises(AssertionError,match='Earlier numerical evidence changed'):p.recheck_prefix(bound)
