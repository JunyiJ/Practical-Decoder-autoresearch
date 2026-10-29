"""Read-only complete prospective campaign identity gate; no freeze/dispatch CLI.

This binds numerical prerequisites and every conditional request before pilots.
It does not replace measured pilot review, complete current/future disk gates,
worker locks, producer caps or later quality/inference acceptance gates.
"""
import os
from pathlib import Path

from .protocol import digest, file_digest
from .runner import source_fingerprint
from .tokenizer_study import runtime_identity
from .tokenizer_wide_config import RECIPE
from .tokenizer_wide_hybrid_driver import CORPUS_SHA256, PROTOCOL, PROJECTION
from .tokenizer_wide_hybrid_config import rank_config
from .tokenizer_wide_hybrid_pilot_config import config, specs, CONTRACT as PILOT_V1
from .tokenizer_wide_hybrid_pilot_gate import panel_design as design_v1
from .tokenizer_wide_hybrid_pilot_cache import CACHE_CONTRACT
from .tokenizer_wide_hybrid_inference_gate import CONTRACT as SERVING_V1
from .tokenizer_wide_hybrid_retention import VERSION, storage_envelope
from .tokenizer_wide_hybrid_retention_audit import numerical_review, RESOURCE_LIMITS
from .tokenizer_wide_hybrid_retention_receipts import _document, _path, _sha
from .tokenizer_wide_hybrid_metadata import budgets, snapshot, MiB
from .tokenizer_wide_hybrid_scores import _json, _require
from .tokenizer_wide_hybrid_timing import token_digest
from . import tokenizer_wide_hybrid_rules as rules

KIND='wide_hybrid_campaign_v2'


def retention_contract():
    return dict(version=VERSION,boundaries=['pilot','lr','rank'],
        complete_stage_audits_before_cleanup=True,persist_selection_before_cleanup=True,
        require_completed_execution_chain=True,retain_selected_screens=True,
        retain_all_fresh_pairs=True,retain_failures_and_unreviewed=True,
        retain_all_non_checkpoint_evidence=True,no_retry=True)


def design():
    result=design_v1()
    result.update(retain_all_raw_h200=False,retention=retention_contract())
    return result


def layout_contract():
    from . import tokenizer_wide_hybrid_inventory_delta as inventory
    return dict(version='wide_hybrid_metadata_v2',immutable_root='evidence',
        plan='evidence/meta/plan.json',mutable_root='control',emergency_root='emergency',
        worker_lock='.lock',stop='STOP',metadata_groups=budgets(),
        inventory_wire=dict(version=inventory.VERSION,marker=inventory.MARKER,field='inventory',
            full_documents_compatible=True,immutable_metadata_bases=True,
            max_rows=inventory.MAX_ROWS,max_expanded_bytes=inventory.MAX_INVENTORY_BYTES,
            max_reference_depth=inventory.MAX_DEPTH,max_reference_bytes=inventory.MAX_CHAIN_BYTES))


def request_options(root):
    """Exactly 3 pilots and 108 conditional quality requests, not 111 trials.

    For each slot, only its selected family's LR/rank matters. The enumerated
    options cover all choices without enumerating irrelevant cross-family tuples.
    Scientific execution stays 3+9+6+9 trajectories, with original order/seeds.
    """
    root=Path(root);pilots=[];quality={str(i):[] for i in range(24)}
    for i,spec in enumerate(specs()):
        trial=root/'evidence/pilots'/f'{i:02d}'
        pilots.append(dict(spec=spec,config=config(trial,spec['family']),result_path=str(trial/'result.json')))
    def add(index,spec):
        trial=root/'evidence/quality'/f'{index:02d}'
        cfg=rank_config(trial,spec['family'],512,spec['seed'],spec['lr'],1200,'mps',
                        bottleneck_divisor=spec['divisor'])
        quality[str(index)].append(dict(spec=spec,config=cfg,result_path=str(trial/'result.json')))
    for i,spec in enumerate(rules.lr_specs()):add(i,spec)
    for rate in rules.RATES:
        for i,spec in enumerate(rules.rank_specs(dict.fromkeys(rules.FAMILIES,rate))):add(9+i,spec)
        for divisor in (8,4,2):
            selection={f:dict(lr=rate,divisor=divisor) for f in rules.FAMILIES}
            for i,spec in enumerate(rules.confirmation_specs(selection)):add(15+i,spec)
    return dict(pilots=pilots,quality=quality)


def request_identity(options):
    """Compact exact identity; reconstruct every request under frozen source.

    This only compresses the immutable plan representation. It does not change
    the request grid or remove any independently checked config from binding.
    """
    return dict(representation='wide_hybrid_request_identity_v1', sha256=digest(options),
        pilot_count=len(options['pilots']),
        quality_slot_counts=[len(options['quality'][str(i)]) for i in range(24)])


def bind_campaign(plan_path,plan_sha256):
    """Bind frozen identity and numerical evidence, never authorize a worker.

    Numerical review/trace paths are relative to the immutable evidence root,
    shared with v2 stage audit binding. Source/data/recipe files are absolute.
    Current storage fit and measured pilots remain separate mandatory gates.
    """
    path=Path(plan_path).absolute()
    _require(path==path.resolve() and path.is_file() and path.stat().st_size<=2*MiB,
             'Canonical bounded campaign plan required')
    plan=_json(path,_sha(plan_sha256))
    root=Path(plan['output'])
    _require(root.is_absolute() and root==root.resolve()
             and path==_path(root,'evidence/meta/plan.json'),'Campaign plan location differs')
    def stopped():return _path(root,'STOP').exists()
    _require(not stopped(),'Campaign STOP prevents binding')
    _require(plan.get('kind')==KIND and plan.get('status')=='frozen','Complete frozen v2 campaign required')
    _require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='1','Strict MPS required')
    source,runtime=source_fingerprint(),runtime_identity()
    _require(digest(plan.get('source'))==digest(source) and digest(plan.get('runtime'))==digest(runtime),
             'Frozen source/runtime differs')
    options=request_options(root)
    projection=_json(PROJECTION,file_digest(PROJECTION))
    expected=dict(protocol_sha256=file_digest(PROTOCOL),projection_sha256=file_digest(PROJECTION),
        panel_design=design(),layout=layout_contract(),storage_contract=storage_envelope(projection),
        pilot_contract=dict(PILOT_V1,version=2),quality_contract=RESOURCE_LIMITS,
        cache_contract=CACHE_CONTRACT,requests=(request_identity(options)
            if isinstance(plan.get('requests'),dict) and 'representation' in plan['requests'] else options))
    for key,value in expected.items():
        _require(digest(plan.get(key))==digest(value),'Frozen campaign contract differs: '+key)
    tokenizer=options['pilots'][0]['config']['model']['output_bias_prior']['tokenizer']
    _require(digest(plan.get('tokenizer'))==digest(tokenizer),'Frozen tokenizer differs from recipe')
    token_sha=_sha(plan.get('training_token_sha256'))
    serving=dict(SERVING_V1,version=2,tokenizer_sha256=digest(tokenizer),token_sha256=token_sha)
    _require(digest(plan.get('inference_contract'))==digest(serving),'Frozen serving workload differs')
    # Serving input provenance needs a separate numerical/preparation trace,
    # not a syntactically valid caller-chosen digest alone.
    context=dict(plan_sha256=plan_sha256,source_sha256=digest(source),
                 runtime_sha256=digest(runtime),corpus_sha256=CORPUS_SHA256)
    numeric=numerical_review(root/'evidence',plan['numerical_ref'],context)
    input_review=_document(root/'evidence',plan['input_ref'])
    _require(input_review.get('kind')=='wide_hybrid_training_input_v2'
             and input_review.get('status')=='passed'
             and input_review.get('source_sha256')==context['source_sha256']
             and input_review.get('runtime_sha256')==context['runtime_sha256']
             and input_review.get('corpus_sha256')==CORPUS_SHA256
             and input_review.get('tokenizer_sha256')==digest(tokenizer)
             and input_review.get('training_token_sha256')==token_sha
             and input_review.get('split')=='train' and input_review.get('token_count')==96
             and type(input_review.get('token_count')) is int
             and input_review.get('holdout_consumed') is False,'Training input provenance incomplete')
    input_traces=input_review.get('artifacts')
    _require(isinstance(input_traces,dict) and bool(input_traces),'Input review requires trace evidence')
    ids_ref=input_review.get('input_ids_ref')
    _require(isinstance(ids_ref,dict) and input_traces.get(ids_ref.get('path'))==ids_ref.get('sha256'),
             'Input IDs must be bound by the input review')
    ids=_document(root/'evidence',ids_ref)
    _require(ids.get('split')=='train' and ids.get('corpus_sha256')==CORPUS_SHA256
             and ids.get('tokenizer_sha256')==digest(tokenizer)
             and token_digest(ids.get('ids',[]))==token_sha,'Exact 96-token input trace differs')
    internal=dict(numeric)
    for name,sha in {plan['input_ref']['path']:plan['input_ref']['sha256'],**input_traces}.items():
        _require(name not in internal or internal[name]==sha,'Conflicting internal evidence binding')
        internal[name]=sha
    forbidden={'meta/plan.json',plan['input_ref']['path'],plan['numerical_ref']['path']}
    _require(not forbidden.intersection(input_traces),'Self-referential input trace refused')
    bound={str(path):plan_sha256}
    for item in plan['files']:
        _require(set(item)=={'path','sha256'},'Exact frozen file reference required')
        file=Path(item['path'])
        _require(file.is_absolute() and file==file.resolve() and str(file) not in bound,
                 'Duplicate or aliased frozen file')
        bound[str(file)]=_sha(item['sha256'])
    corpus=Path(options['pilots'][0]['config']['data']['path']).resolve()
    required={str(corpus):CORPUS_SHA256,str(PROTOCOL.resolve()):file_digest(PROTOCOL),
              str(PROJECTION.resolve()):file_digest(PROJECTION),str(RECIPE.resolve()):file_digest(RECIPE)}
    for name,sha in internal.items():required[str(_path(root/'evidence',name))]=sha
    _require(all(bound.get(name)==sha for name,sha in required.items()),'Missing required frozen evidence')
    for name,sha in bound.items():_require(file_digest(Path(name))==sha,'Frozen evidence changed')
    census=snapshot(root)
    _require(digest(source_fingerprint())==digest(source) and digest(runtime_identity())==digest(runtime)
             and not stopped(),'Source/runtime/STOP changed during binding')
    for name,sha in bound.items():_require(file_digest(Path(name))==sha,'Frozen evidence changed during binding')
    return dict(kind=KIND,context=context,files=bound,requests=options,
        storage_contract=expected['storage_contract'],metadata_bytes=census['metadata_bytes'],
        numerical_review_bound=True,training_input_bound=True,
        current_storage_fit_verified=False,measured_pilots_verified=False,
        producer_enforcement_verified=False,dispatch_authorized=False)
