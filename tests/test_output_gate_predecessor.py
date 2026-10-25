"""Synthetic final-decision propagation; no actual checkpoint or scoring work."""
import json
import pytest
from src.research import qknorm_study as qk
from src.research.output_gate_predecessor import reviewed_qknorm
from src.research.protocol import digest, file_digest
from src.research.runner import atomic_json
from tests.test_qknorm_study import plan, complete
from tests.test_schedule_study import coverage


def predecessor(folder, mode='qkrms', adopt=True):
    p = plan(); p['git_commit'] = 'a'*40
    s = complete(p, standard_lr=.0025, eligible=mode=='qkrms')
    if mode == 'standard':
        assert s['selection']['winner'] == 'standard'
        for r in s['records'][12:]:
            ce = {'standard':1.97, 'qkrms':1.95, qk.REFERENCE:2.}[r['family']]
            r['averaged_outputs']['200']['coverage']['evaluations']['128'] = coverage(ce)
    else:
        assert s['selection']['winner'] == 'qkrms'
    if not adopt:
        for r in s['records'][12:]:
            if r['family'] == mode and r['width'] == 256 and r['seed'] == 2193:
                r['averaged_outputs']['200']['coverage']['evaluations']['128'] = coverage(2.1)
    report = qk.summarize(p, s)
    for name, value in [('plan',p), ('selection',s['selection']), ('report',report)]:
        atomic_json(folder/f'{name}.json', value)
    review = dict(status='verified', distinct_trajectories=len(s['records']), distinct_checkpoints=2*len(s['records']),
                  plan_sha256=digest(p), source_commit=p['git_commit'], source_sha256=p['source_sha256'],
                  report_sha256=file_digest(folder/'report.json'), selection_sha256=file_digest(folder/'selection.json'),
                  quality_decision=report['quality_decision'])
    atomic_json(folder/'review.json', review)
    return folder/'review.json'


@pytest.mark.parametrize('mode', ['standard','qkrms'])
@pytest.mark.parametrize('adopt', [False,True])
def test_only_final_supported_recipe_propagates(tmp_path, mode, adopt):
    path = predecessor(tmp_path, mode, adopt); r = reviewed_qknorm(path)
    assert r['qk_family'] == (mode if adopt else 'standard')
    assert r['learning_rate'] == (.0025 if adopt else .005)
    m = r['config']['model']; t = r['config']['training']
    assert (m.get('qk_norm') == 'rms_head32') == (adopt and mode=='qkrms')
    assert t['learning_rate'] == r['learning_rate'] and t['max_iters'] == 3000
    assert t['lr_schedule']['start_step'] == 2000 and t['weight_averaging']['half_lives'] == [200]


@pytest.mark.parametrize('field,value', [('status','partial_verified'), ('distinct_trajectories',12),
    ('distinct_checkpoints',24), ('plan_sha256','0'*64), ('source_commit','b'*40),
    ('source_sha256','0'*64), ('report_sha256','0'*64), ('selection_sha256','0'*64)])
def test_partial_or_changed_review_rejected(tmp_path, field, value):
    path = predecessor(tmp_path); v = json.loads(path.read_text()); v[field] = value; atomic_json(path,v)
    with pytest.raises(ValueError):reviewed_qknorm(path)


def test_rehashed_false_promotion_still_rejected_from_raw_comparisons(tmp_path):
    path = predecessor(tmp_path, adopt=False); rp = tmp_path/'report.json'; report = json.loads(rp.read_text())
    report['quality_decision']['qualifies_for_incumbent_replacement'] = True; atomic_json(rp,report)
    review = json.loads(path.read_text()); review.update(quality_decision=report['quality_decision'],report_sha256=file_digest(rp)); atomic_json(path,review)
    with pytest.raises(ValueError,match='complete evidence'):reviewed_qknorm(path)


def test_rehashed_removed_confirmation_rejected(tmp_path):
    path = predecessor(tmp_path); rp = tmp_path/'report.json'; report = json.loads(rp.read_text())
    report['records'].pop(); atomic_json(rp,report)
    review = json.loads(path.read_text()); review['report_sha256'] = file_digest(rp); atomic_json(path,review)
    with pytest.raises(ValueError):reviewed_qknorm(path)


def test_actual_local_failure_cannot_be_replaced_with_better_confirmation(tmp_path):
    p = plan(); p['git_commit'] = 'a'*40; s = complete(p, .005, eligible=False)
    assert s['selection']['winner'] == 'incumbent_unchanged'
    report = qk.summarize(p,s)
    for name,value in [('plan',p),('selection',s['selection']),('report',report)]:atomic_json(tmp_path/f'{name}.json',value)
    review = dict(status='verified',distinct_trajectories=24,distinct_checkpoints=48,
                  plan_sha256=digest(p),source_commit=p['git_commit'],source_sha256=p['source_sha256'],
                  report_sha256=file_digest(tmp_path/'report.json'),selection_sha256=file_digest(tmp_path/'selection.json'),quality_decision=report['quality_decision'])
    path = tmp_path/'review.json';atomic_json(path,review)
    resolved = reviewed_qknorm(path)
    assert resolved['qk_family']=='standard' and resolved['learning_rate']==.005
