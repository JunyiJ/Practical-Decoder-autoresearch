"""Small arithmetic integration; no campaign, checkpoint, training or timing worker."""
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'research/tokenizer-evidence/wide-hybrid-preparation-001/review_final_campaign_v2.py'
SPEC = importlib.util.spec_from_file_location('final_campaign_review', PATH)
review = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(review)
from test_tokenizer_wide_hybrid_timing_review import fixtures


def inputs(delta, *, memory=True):
    selection, sha, measurements = fixtures()
    if not memory:
        for result in measurements:
            for key in ('warmup_records','records'):
                for row in result[key]:
                    if row['family']=='hybrid': row['state_bytes']=100*row['batch']
    counts=[2048]*16+[1510]; results=[]
    for spec in review.rules.confirmation_specs(selection):
        ce = 5.+delta if spec['family']=='hybrid' else 5.
        results.append(dict(**spec,status='completed',holdout_consumed=False,
            raw=dict(mean_ce=100.),h200=dict(mean_ce=ce,context=128,
                block_target_counts=counts,block_loss_sums=[ce*n for n in counts],
                target_count=sum(counts),target_bytes=111536),wall_seconds=1.))
    return results,measurements,selection,sha


@pytest.mark.parametrize('delta,memory,eligible,routes',[
    (-.03,False,True,['strict_quality']),
    (.005,True,True,['neutral_quality_and_memory']),
    (.005,False,False,[]),
    (.02,True,False,[]),
    (-.03,True,True,['strict_quality','neutral_quality_and_memory'])])
def test_both_routes_use_correct_boolean_composition(delta,memory,eligible,routes):
    report=review.combine(*inputs(delta,memory=memory))
    assert report['adoption_eligible'] is eligible
    assert report['passing_routes']==routes
    assert not report['adopted'] and not report['holdout_consumed']
    assert len(report['costs']['cells'])==18
    # Synthetic timings deliberately make hybrid twice as slow as MLA.
    assert all(r['median_ratio']==2. for r in report['costs']['paired_latency_ratios']
               if r['control']=='gated_mla')


@pytest.mark.parametrize('bad',['missing','order','holdout','coverage','nan','one_seed','one_control','one_memory_cell'])
def test_missing_or_inconsistent_panels_cannot_pass(bad):
    args=list(inputs(.005)); results,measurements,selection,sha=args
    if bad=='missing':results.pop()
    elif bad=='order':results.reverse()
    elif bad=='holdout':results[0]['holdout_consumed']=True
    elif bad=='coverage':results[0]['h200']['target_bytes']-=1
    elif bad=='nan':results[0]['h200']['block_loss_sums'][0]=float('nan')
    elif bad in ('one_seed','one_control'):
        for r in results:
            if (bad=='one_seed' and r['seed']==3513 and r['family']=='hybrid'
                    or bad=='one_control' and r['family']=='kda'):
                ce=5.03 if bad=='one_seed' else 4.97
                r['h200']['mean_ce']=ce
                r['h200']['block_loss_sums']=[ce*n for n in r['h200']['block_target_counts']]
    else:
        for key in ('warmup_records','records'):
            for row in measurements[2][key]:
                if row['family']=='hybrid' and row['batch']==8:row['state_bytes']=91*8
    if bad in ('one_seed','one_control','one_memory_cell'):
        assert not review.combine(*args)['adoption_eligible']
    else:
        with pytest.raises(ValueError):review.combine(*args)


def test_wrapper_requires_final_boundary_and_frozen_procedure(monkeypatch,tmp_path):
    with pytest.raises(ValueError):review.review_completed_campaign(None,dict(path='inference/3512/review.json'))
    monkeypatch.setattr(review,'snapshot',lambda root: {})
    monkeypatch.setattr(review.acceptance,'verify_boundary',lambda *a:dict(
        quality_history_reaudited=True,bound=dict(files={}),scope=dict(state='accepted',
        previous_refs=[dict(path=f'inference/{s}/review.json') for s in review.rules.SEEDS])))
    writer=type('Writer',(),dict(root=tmp_path))()
    with pytest.raises(ValueError,match='not frozen'):
        review.review_completed_campaign(writer,dict(path='inference/3513/review.json'))


@pytest.mark.parametrize('bad',[None,'missing_controller','changed_result'])
def test_wrapper_consumes_bound_files_and_rechecks(monkeypatch,tmp_path,bad):
    # This isolates the new wrapper; production acceptance is already covered
    # by retained integration evidence. No real-history/provenance claim here.
    results,measurements,selection,sha=inputs(.005)
    root=tmp_path/'evidence';root.mkdir()
    files={str(PATH.resolve()):review.file_digest(PATH)}; evidence={};triples={s:{} for s in review.rules.SEEDS}
    def save(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        hashed=review.file_digest(path);files[str(path)]=hashed
        evidence[name]=dict(size=path.stat().st_size,sha256=hashed)
        return path,hashed
    for slot,result in enumerate(results,15):
        path,hashed=save(f'quality/{slot:02d}/result.json',result)
        triples[result['seed']][result['family']]=dict(trial=str(path.parent),result_sha256=hashed)
    for seed,result in zip(review.rules.SEEDS,measurements):
        save(f'inference/{seed}/result.json',result)
        save(f'inference/{seed}/controller.json',{'synthetic_unit_test':True})
    if bad=='missing_controller':evidence.pop('inference/3512/controller.json')
    if bad=='changed_result':(root/'quality/15/result.json').write_text('{}')
    bound=dict(files=files,triples=triples,selection=selection,token_sha256=sha,plan_sha256='0'*64)
    scope=dict(state='accepted',evidence=evidence,
        previous_refs=[dict(path=f'inference/{s}/review.json') for s in review.rules.SEEDS])
    calls=[]
    def verify(*args):
        calls.append(args[1]);return dict(bound=bound,scope=scope,quality_history_reaudited=True)
    monkeypatch.setattr(review.acceptance,'verify_boundary',verify)
    monkeypatch.setattr(review,'snapshot',lambda root: {})
    writer=type('Writer',(),dict(root=tmp_path))()
    ref=dict(path='inference/3513/review.json',sha256='0'*64)
    if bad:
        with pytest.raises(ValueError):review.review_completed_campaign(writer,ref)
    else:
        report=review.review_completed_campaign(writer,ref)
        assert report['status']=='complete_independent_quality_cost_review'
        assert report['adoption_eligible'] and not report['adopted']
        assert len(report['result_refs'])==9 and len(report['controller_refs'])==3
        assert len(json.dumps(report))<review.LIMIT
    assert calls==[ref]
