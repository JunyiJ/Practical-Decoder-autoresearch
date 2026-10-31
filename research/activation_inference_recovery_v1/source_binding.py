"""Read-only import of completed activation001 training, never a dispatcher.

The caller holds the existing shared worker lock and original campaign lock and
supplies an outer CPU/worker watchdog. No original entry, numerical evaluator,
checkpoint loader, deletion routine or inference producer is called here.
"""
import json
from pathlib import Path
import shutil
import time

from src.research.protocol import ROOT, digest, file_digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_activation_entry import bind, prefix
from src.research.tokenizer_activation_session import Session, reference, require
from src.research.tokenizer_activation_retention import verify_boundaries
from src.research.tokenizer_activation_rules import FAMILIES, FRESH_SEEDS, confirmation_specs
from src.research.tokenizer_activation_timing import CONTRACT as TIMING
from src.research.tokenizer_activation_final import CONTRACT as FINAL
from src.research.tokenizer_wide_hybrid_cleanup import _locks
from src.research.tokenizer_wide_hybrid_retention import RESERVE
from src.research.tokenizer_serial_process import _group_exists
from research.activation_inference_recovery_v1.census import synchronized_snapshot

ORIGINAL_ROOT = ROOT/'research_runs/tokenizer-activation-001'
PLAN_SHA256 = '508b922253f7eef8ae6fafe05cb393aefe4bb0f58225e8573857e920933016c4'
SOURCE_SHA256 = '7a4c02946a31820a7b1ff36b8cc4be42659a4fc4a62424cbc4ea5f32d6cbc487'
SELECTION_SHA256 = '9f3289f9d5d9806bb0fed52140020527ef4ef27b714b2720a9c18bbe8704a863'
FAILURE_REVIEW = ROOT/'research/tokenizer-evidence/activation-preparation-001/campaign001-inference-failure-review.json'
FAILURE_REVIEW_SHA256 = 'b83c77f3366619f49dae96edb2af76d95ac764b1523783dd3bfd449076502166'
BIND_SECONDS = 180.0


def _complete_state(state):
    require(state['next'] == {'kind': 'review_final', 'adopted': False}
            and state['pending'] is None and state['failed'] is False,
            'Complete nonpending training ledger required')
    require(len(state['pilots']) == 2 and len(state['records']) == 12
            and len(state['reviews']) == 14 and len(state['pairs']) == 14,
            'Exactly two pilots and twelve quality acceptances required')
    require(state['pilot_boundary'] is True and state['screen_boundary'] is True,
            'Both completed retention boundaries required')
    require(state['selection'] == dict.fromkeys(FAMILIES, .00125)
            and state['selection_ref']['path'] == 'evidence/meta/activation-selection.json'
            and state['selection_ref']['sha256'] == SELECTION_SHA256,
            'Exact independently saved original LR selection required')


def _failed_inference(ledger, seen):
    anchor = reference(ROOT, str(FAILURE_REVIEW.relative_to(ROOT)))
    require(anchor['sha256'] == FAILURE_REVIEW_SHA256, 'Preserved failure audit changed')
    audit = json.loads(FAILURE_REVIEW.read_text())
    require(audit['status'] == 'training_complete_inference_failed_no_retry'
            and audit['plan'] == ledger.plan and audit['source_sha256'] == SOURCE_SHA256
            and audit['holdout_scored'] is False and audit['retry_allowed'] is False
            and audit['formal_final_completed'] is False, 'Original failure audit contract differs')
    def failure_path(name):
        return name.startswith('evidence/inference/') or name.startswith('control/') and 'inference' in name
    require({name for name in seen['files'] if failure_path(name)} == set(audit['copied']),
            'Original inference attempt inventory changed; no retry or later slot allowed')
    for ref in audit['copied'].values():
        ledger.verify(ref)
    process = ledger.document(audit['copied']['evidence/inference/3511/process/process.json'])
    failure = ledger.document(audit['copied']['evidence/inference/3511/dispatch-failure.json'])
    require(process == audit['process'] and process['status'] == 'failed'
            and process['returncode'] == 1 and failure['retry_allowed'] is False,
            'Exact failed original process required')
    require(not _group_exists(process['pid']), 'Original inference process group remains live')
    return anchor, audit


def _pairs(ledger, state, plan, dispatches):
    bindings = {str(seed): {} for seed in FRESH_SEEDS}
    for index, spec in enumerate(confirmation_specs(state['selection']), 6):
        review_ref = state['reviews'][index+2]
        review = ledger.document(review_ref)
        request = ledger.document(review['request'])
        require(request['spec'] == spec and review['request']['path'] == f'evidence/quality/{index:02d}/request.json',
                'Exact ordered fresh selected request required')
        require(set(review['audits']) == {'training','evaluation','tensors','resources'},
                'All four independent audit roles required')
        for ref in review['audits'].values():
            require(ledger.document(ref).get('reviewer') == 'activation_independent_v1',
                    'Independent quality reviewer missing')
        require(set(review['weights']) == {'raw','h200'}, 'Raw and H200 pair required')
        for ref in review['weights'].values():
            ledger.verify(ref)
        seed = str(spec['seed'])
        require(spec['family'] not in bindings[seed], 'Duplicate fresh family')
        bindings[seed][spec['family']] = dict(request=review['request'], acceptance=review_ref, weights=review['weights'])
    require(all(set(pair) == set(FAMILIES) for pair in bindings.values()), 'Three complete fresh pairs required')
    # Same loader-facing shape as the original quality_binding result, but no
    # failed-root budget or original inference entry is invoked.
    return {seed:dict(plan=ledger.plan,seed=int(seed),selection=state['selection'],quality_reviews=state['reviews'],
                     dispatches=dispatches,inputs=plan['inputs'],bindings=pair,contract=TIMING)
            for seed,pair in bindings.items()}


def collect_source_import(ledger):
    """Revalidate original provenance and return deterministic exact import pins.

    Must finish inside 180s and preserve an 8GiB reserve. This does not run the
    old remaining-training storage gate: this extension allocates no weights.
    The future new-root controller must separately enforce its complete finite
    inference allocation and all unchanged inference resource limits.
    """
    started = time.monotonic()
    require(isinstance(ledger, Session) and ledger.root == ORIGINAL_ROOT,
            'Exact original activation001 session required')
    _locks(ledger.root/'evidence', ledger.writer.lock_fds)
    require(ledger.plan['path'] == 'evidence/meta/activation-plan.json'
            and ledger.plan['sha256'] == PLAN_SHA256, 'Exact original frozen plan required')
    def check():
        require(time.monotonic()-started < BIND_SECONDS, 'Source import time limit')
        require(shutil.disk_usage(ledger.root).free >= RESERVE, 'Source import disk reserve')
        require(not (ledger.root/'STOP').exists() and not (ledger.root/'STOP').is_symlink(),
                'Original STOP blocks recovery')
    check()
    before = synchronized_snapshot(ledger.writer)
    require(not any(name.endswith('.tmp') for name in before['files']), 'Partial original output blocks import')
    plan = bind(ledger, capacity=False)
    require(digest(plan['source']) == SOURCE_SHA256, 'Exact original source required')
    state = ledger.inspect()
    _complete_state(state)
    dispatches = prefix(ledger, state)
    require(len(dispatches) == 14, 'All fourteen completed original dispatches required')
    verify_boundaries(ledger, state)
    check()
    anchor, failure_audit = _failed_inference(ledger, before)
    protected = {ref['path']:ref for pair in state['pairs'] for ref in pair if ref['path'] not in state['deleted']}
    weight_names = {name for name,entry in before['files'].items() if entry['group'].endswith('/weights')}
    require(len(protected) == 16 and set(protected) == weight_names
            and protected == failure_audit['protected_checkpoints'],
            'Exactly sixteen original selected/fresh checkpoint files must survive')
    for ref in protected.values():
        check()
        ledger.verify(ref)
    pair_bindings = _pairs(ledger, state, plan, dispatches)
    manifest = dict(protected)
    for name in sorted(set(before['files'])-weight_names):
        check()
        manifest[name] = reference(ledger.root, name)
    require(synchronized_snapshot(ledger.writer) == before, 'Original inventory changed during import')
    # Rehash metadata after semantic review. Existing Session verification above
    # also rehashes retained training pins; no missing weight is inferred from
    # existence alone. Deleted files need the validated original boundaries.
    for name in sorted(set(manifest)-weight_names):
        check()
        ledger.verify(manifest[name])
    require(digest(source_fingerprint()) == SOURCE_SHA256, 'Original source changed during import')
    require(reference(ROOT,anchor['path']) == anchor, 'Failure audit changed during import')
    require(synchronized_snapshot(ledger.writer) == before, 'Original inventory changed at import completion')
    check()
    return dict(kind='activation001_readonly_import_v1', original_root=str(ledger.root), plan=ledger.plan,
        source_sha256=SOURCE_SHA256,runtime=plan['runtime'],inputs=plan['inputs'],
        selection=state['selection'],selection_ref=state['selection_ref'],quality_reviews=state['reviews'],
        dispatches=dispatches,retention={name:reference(ledger.root,'evidence/.retention/'+folder+'/activation-complete.json')
                                      for name,folder in [('pilot','pilot'),('screen','lr')]},
        protected_checkpoints=protected,deleted_checkpoints=state['deleted'],files=manifest,
        pair_bindings=pair_bindings,failure_audit=anchor,failed_inference=failure_audit['copied'],
        timing_contract=TIMING,final_contract=FINAL,training_complete=True,
        original_inference_failed=True,original_final_completed=False,dispatch_authorized=False)


def verify_source_import(ledger, expected):
    """A frozen import is accepted only after complete fresh independent replay."""
    actual = collect_source_import(ledger)
    require(type(expected) is dict and digest(actual) == digest(expected), 'Frozen original import differs')
    return actual
