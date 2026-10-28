"""Prospective inference prerequisite binding; read-only, no dispatch/freeze.

A passing review is accepted only when it names the exact complete quality
panel and its retained files. This does not itself perform tensor/EMA audits.
"""
from pathlib import Path

from .protocol import digest, file_digest
from .tokenizer_wide_hybrid_driver import verify_plan, _trial_id, _contract
from .tokenizer_wide_hybrid_scores import _json
from .tokenizer_wide_hybrid_simulation import next_action
from .tokenizer_wide_hybrid_rules import FAMILIES, SEEDS

CONTRACT = dict(version=1, device='mps', dtype='float32', context=128, batches=[1,8],
                prefill_tokens=32, decode_tokens=64, warmups=2, repeats=8,
                seeds=list(SEEDS), job_seconds=600, output_limit_bytes=4*1024**2,
                reserve_bytes=8*1024**3, rss_limit_bytes=4*1024**3,
                driver_limit_bytes=6*1024**3, cache_rtol=3e-4, cache_atol=3e-5,
                cache_reset_replay=True, family_cycle=list(FAMILIES), seed_offsets=[0,1,2])
TENSOR_CHECKS = {'finite_tensors', 'exact_ema', 'optimizer_state'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def bind_inference(request, seed):
    """Return selected bindings only after complete frozen-panel review checks.

    Request/hash must be durably frozen by the future parent. The quality plan's
    inference contract and token/tokenizer hashes must be frozen before training.
    A tensor review is created after quality and hash-bound by that parent.
    No such real plan or complete tensor review currently exists.
    """
    _require(type(seed) is int and seed in SEEDS, 'Undeclared inference seed')
    _require(set(request) == {'quality_run','plan_sha256','state_sha256','tensor_review',
                             'tensor_review_sha256','tokenizer','token_sha256'}, 'Incomplete inference request')
    root = Path(request['quality_run']).resolve()
    _require(not (root/'STOP').exists(), 'Quality STOP prevents inference')
    plan = _json(root/'plan.json', request['plan_sha256'])
    verify_plan(plan)
    _require(Path(plan['output']).resolve() == root, 'Quality output mismatch')
    expected_contract = dict(CONTRACT, tokenizer_sha256=digest(request['tokenizer']),
                             token_sha256=request['token_sha256'])
    _require(isinstance(request['tokenizer'],dict) and bool(request['tokenizer'])
             and isinstance(request['token_sha256'],str) and len(request['token_sha256']) == 64
             and all(c in '0123456789abcdef' for c in request['token_sha256'])
             and digest(plan.get('inference_contract')) == digest(expected_contract),
             'Inference workload or input identity was not frozen')
    state = _json(root/'state.json', request['state_sha256'])
    _require(state.get('status') == 'quality_completed_pending_tensor_and_serving_audits'
             and state.get('active') is None and state.get('adopted') is False
             and state.get('plan_sha256') == request['plan_sha256'], 'Quality state incomplete')
    _require(next_action(state['records'],state['rates'],state['selection'])['kind'] == 'complete'
             and len(state['records']) == 24 and len(state['reviews']) == 24, 'Incomplete quality panel')
    bound = {str(root/'plan.json'):request['plan_sha256'],str(root/'state.json'):request['state_sha256']}
    for key,name in (('rates','lr-selection.json'),('selection','rank-selection.json')):
        path=root/name
        _require(digest(_json(path,file_digest(path))) == digest(state[key]), 'Persisted selection changed')
        bound[str(path)]=file_digest(path)
    selected, expected_audits = {}, []
    for index,(record,receipt) in enumerate(zip(state['records'],state['reviews'])):
        spec={k:record[k] for k in ('stage','family','seed','lr','divisor','width','steps')}
        trial=root/_trial_id(index,spec)
        _require(digest(receipt['review']['selection_row']) == digest(record)
                 and receipt['review'].get('metadata_reviewed') is True, 'Score receipt differs from quality row')
        files=receipt['files']
        rq=trial/'request.json';rs=trial/'result.json'
        _require(str(rq) in files and str(rs) in files, 'Missing request/result receipt binding')
        saved_request=_json(rq,files[str(rq)]);result=_json(rs,files[str(rs)])
        _require(digest(saved_request['spec']) == digest(spec)
                 and result.get('request_sha256') == files[str(rq)]
                 and result.get('status') == 'completed' and result.get('holdout_consumed') is False,
                 'Trial request/result identity mismatch')
        expected_files={str(rq):files[str(rq)],str(rs):files[str(rs)]}
        for kind in ('raw','h200'):
            path=Path(result[kind]['checkpoint']).resolve()
            _require(path.is_relative_to((trial/'checkpoints').resolve()), 'Checkpoint outside its trial')
            sha=result[kind]['checkpoint_sha256']
            _require(receipt['review']['checkpoint_hashes'][kind] == sha
                     and file_digest(path) == sha, 'Checkpoint no longer matches reviewed receipt')
            expected_files[str(path)]=sha
        _require(digest(files) == digest(expected_files), 'Incomplete/extra retained-file receipt')
        _require(len(expected_files)==4, 'Raw/export aliases are not distinct')
        bound.update(expected_files)
        expected_audits.append(dict(trial=str(trial),raw_sha256=result['raw']['checkpoint_sha256'],
            h200_sha256=result['h200']['checkpoint_sha256'],checks=dict.fromkeys(sorted(TENSOR_CHECKS),True)))
        if spec['stage']=='confirm' and spec['seed']==seed:
            selected[spec['family']]=dict(trial=str(trial),request=saved_request,
                request_sha256=files[str(rq)],result_sha256=files[str(rs)])
    review_path=Path(request['tensor_review']).resolve()
    audit=_json(review_path,request['tensor_review_sha256'])
    _require(audit.get('kind')=='wide_hybrid_tensor_audit_v1' and audit.get('status')=='passed'
             and audit.get('plan_sha256')==request['plan_sha256']
             and audit.get('state_sha256')==request['state_sha256']
             and audit.get('source_sha256')==digest(plan['source'])
             and audit.get('runtime_sha256')==digest(plan['runtime'])
             and digest(audit.get('records'))==digest(expected_audits), 'Tensor audit incomplete or from different panel')
    traces=audit.get('artifacts')
    _require(isinstance(traces,dict) and bool(traces), 'Tensor audit lacks underlying evidence')
    for path,sha in traces.items():
        resolved=str(Path(path).resolve())
        _require(resolved != str(review_path) and file_digest(Path(resolved))==sha,
                 'Tensor audit trace missing or changed')
        _require(resolved not in bound or bound[resolved]==sha,'Conflicting audit trace binding')
        bound[resolved]=sha
    bound[str(review_path)]=request['tensor_review_sha256']
    _require(set(selected)==set(FAMILIES), 'Missing fresh inference family')
    # Rebuild contracts only after all cheap provenance/audit checks. This reads
    # the corpus through the standard loader but evaluates no validation/test CE.
    for binding in selected.values():
        binding['training_contract']=_contract(binding['request']['config'])
        _require(digest(binding['training_contract'].get('tokenizer'))==digest(request['tokenizer']),
                 'Current tokenizer differs from frozen inference input')
    verify_plan(plan)
    _require(not (root/'STOP').exists(), 'Quality STOP prevents inference')
    recheck(bound)
    return dict(bindings=selected,selection=state['selection'],tokenizer=request['tokenizer'],
                token_sha256=request['token_sha256'],files=bound,contract=expected_contract)


def recheck(files):
    for path,sha in files.items():
        if file_digest(Path(path)) != sha:
            raise ValueError('Bound inference evidence changed')
