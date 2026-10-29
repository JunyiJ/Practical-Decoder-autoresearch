"""One completed synthetic prefix for all nine bindings; production audits run."""
import copy
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_inference_gate_v2 as gate
from src.research import tokenizer_wide_hybrid_first_confirmation_startup_v2 as first
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_final_confirmation_v2 import sequence


def test_all_triples_share_one_fully_audited_prefix(writer, monkeypatch):
    args = sequence(writer, monkeypatch)
    args.pop('slot')
    root = writer.root/'evidence'
    plan = gate._document(root, dict(path='meta/plan.json', sha256=args['plan_sha256']))
    monkeypatch.setattr(gate, 'bind_campaign', first.bind_campaign)
    monkeypatch.setattr(gate, 'shutil', SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    loads = []
    def contract(config):
        assert config['fixture_only'] is True
        loads.append(config)
        return dict(tokenizer=plan['tokenizer'])
    monkeypatch.setattr(gate, '_contract', contract)
    inventory = gate.history.acceptance._inventory(root)
    controls = gate.history.acceptance.readiness._controls(writer, 'not-a-control')
    result = gate.bind_inference(**args)
    assert len(loads) == 9
    assert result['contract'] == dict(gate.CONTRACT, version=2, tokenizer_sha256=gate.digest(plan['tokenizer']),
                                    token_sha256=plan['training_token_sha256'])
    assert set(result['triples']) == {3511,3512,3513}
    assert len(result['history']['records']) == 24 and len(result['history']['checkpoint_pairs']) == 12
    assert result['history']['audited_indices'] == [9,10,11,*range(15,24)]
    assert all(result[k] is False for k in ('adopted','dispatch_authorized','deletion_authorized','next_stage_authorized'))
    paths = set()
    for seed, triple in result['triples'].items():
        assert set(triple) == {'hybrid','gated_mla','kda'}
        for family, binding in triple.items():
            assert set(binding) == {'trial','request','request_sha256','result_sha256','training_contract'}
            assert binding['request']['spec']['seed'] == seed and binding['request']['spec']['family'] == family
            paths.add(binding['trial'])
    assert len(paths) == 9 and sum(p.endswith('.pt') for p in result['files']) == 24
    assert any('/.retention/rank/' in p for p in result['files'])
    assert not (root/'quality/00/checkpoints/h200.pt').exists()
    assert inventory == gate.history.acceptance._inventory(root)
    assert controls == gate.history.acceptance.readiness._controls(writer, 'not-a-control')
    # Adapter fault cases reuse this independently audited, unchanged prefix.
    # They test the loader boundary, not a replacement for production replay.
    frozen = first.bind_campaign(root/'meta/plan.json', args['plan_sha256'])
    for bad in ('seed','order','missing_choice','ambiguous_choice','pair','tokenizer'):
        report = copy.deepcopy(result['history']); choices = copy.deepcopy(frozen)
        tokenizer = plan['tokenizer']
        if bad == 'seed':report['records'][15]['seed'] = 3511.0
        elif bad == 'order':report['records'][15],report['records'][16] = report['records'][16],report['records'][15]
        elif bad == 'missing_choice':choices['requests']['quality']['15'] = []
        elif bad == 'ambiguous_choice':choices['requests']['quality']['15'] *= 2
        elif bad == 'pair':report['checkpoint_pairs']['quality/15']['h200'] = report['checkpoint_pairs']['quality/16']['h200']
        else:tokenizer = {'changed': True}
        with pytest.raises(ValueError):gate._triples(root, report, choices, inventory, tokenizer)
    assert inventory == gate.history.acceptance._inventory(root)
    assert controls == gate.history.acceptance.readiness._controls(writer, 'not-a-control')


@pytest.mark.parametrize('path', ['meta/quality-dispatch-22.json','meta/quality-dispatch-24.json','../quality-dispatch-23.json',''])
def test_only_complete_final_history_is_eligible(writer, path):
    before = gate.snapshot(writer.root)
    with pytest.raises(ValueError, match='Final slot23'):
        gate.bind_inference(writer, plan_sha256='a'*64, completion_ref={'path':path}, projection_ref={}, rank_completion_ref={})
    assert gate.snapshot(writer.root) == before
