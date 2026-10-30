"""Review bounded startup regression evidence and preserve closed004 identities."""
import base64,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import digest,file_digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_study import runtime_identity
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
E=Path(__file__).resolve().parent
RUN=ROOT/'research_runs/tokenizer-wide-hybrid-004'
with session_lock(ROOT/'research_state/tokenizer-worker-lock'),session_lock(RUN),CpuPrerequisiteGuard(E/'confirmation-census-v2-review-guard.json',ROOT,timeout_seconds=600):
    before=json.loads((E/'confirmation-census-v2-before.json').read_text())
    plan=json.loads((E/'campaign004-frozen/meta/plan.json').read_text())
    current=source_fingerprint();runtime=runtime_identity()
    changes={n:dict(before=plan['source'].get(n),after=current.get(n)) for n in set(plan['source'])|set(current) if plan['source'].get(n)!=current.get(n)}
    expected={'src/research/tokenizer_wide_hybrid_'+n+'_v2.py' for n in ('confirmation_startup','later_confirmation_startup','remaining_confirmation_startup')}
    assert set(changes)==expected and runtime==plan['runtime']
    assert all(changes[n]['before']==before['files'][n] for n in expected)
    assert file_digest(E/'confirmation-census-v2-red-test-source.py')==before['files']['tests/test_tokenizer_wide_hybrid_confirmation_census_v2.py']
    tests={}
    for phase,summary,exitcode in [('red','3 failed',1),('green','25 passed',0)]:
        stem='confirmation-census-v2-'+phase
        log=(E/(stem+'-tests.log')).read_text();guard=json.loads((E/(stem+'-guard.json')).read_text())
        assert summary in log and guard['status']=='finished' and guard['reason'] is None
        if phase=='red':
            for msg in ('Frozen slot16 request changed before receipt','Frozen slot17 request changed before receipt','Frozen startup request changed before receipt'):assert msg in log
        else:assert ' failed' not in log and ' error' not in log.lower()
        tests[phase]=dict(summary=summary,exitcode=exitcode,log_sha256=file_digest(E/(stem+'-tests.log')),guard=guard)
    closed=json.loads((E/'campaign004-fresh16-failure-review-v2.json').read_text())
    observed={}
    for p in RUN.rglob('*'):
        assert not p.is_symlink()
        if p.is_file():observed[str(p.relative_to(RUN))]=dict(size=p.stat().st_size,sha256=file_digest(p))
    assert observed==closed['files']
    for n,b in closed['noncheckpoint_bytes_base64'].items():assert (RUN/n).read_bytes()==base64.b64decode(b)
    names=sorted(expected|{'src/research/tokenizer_wide_hybrid_quality_readiness_v2.py','src/research/tokenizer_wide_hybrid_plan_v2.py','tests/test_tokenizer_wide_hybrid_confirmation_census_v2.py',*('tests/test_tokenizer_wide_hybrid_'+n+'_v2.py' for n in ('confirmation_startup','later_confirmation_startup','remaining_confirmation_startup'))})
    assert source_fingerprint()==current and runtime_identity()==runtime
    report=dict(status='startup_census_fix_regression_verified',before_source_sha256=digest(plan['source']),after_source_sha256=digest(current),runtime_sha256=digest(runtime),changed_files=changes,source_and_test_hashes={n:file_digest(ROOT/n) for n in names},tests=tests,
        preserved_closed_campaign=dict(files=len(observed),noncheckpoint_files=len(closed['noncheckpoint_bytes_base64']),checkpoint_files=sum(n.endswith('.pt') for n in observed),inventory_sha256=digest(observed)),
        claims=['All three affected startup paths fail with actual live census growth before the fix and pass after it.',
                'Only the metadata_bytes field is allowed to vary, with integer/range/presence validation; every other binding field stays exact.',
                'Live snapshot group/file/total caps, request checks, append audits, control checks, resource guards and no-retry checks remain unchanged.',
                'Late unexpected control appends still reject receipt publication for all three startup paths.'],
        limits=['Disposable CPU histories and mock frozen scientific inputs; no real model, training child, numerical rerun or end-to-end MPS reliability claim.',
                'Shared remaining-slot implementation exercised at slot18; slots19–23 use the same changed comparison, not new full-chain live-census runs.',
                'A future source-applicability/metadata/capacity review and finite plan are still required; campaigns001–004 remain closed.'],
        training=False,cleanup=False,closed_campaigns_reopened=False,holdout_consumed=False)
    with (E/'confirmation-census-v2-review.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ('status','after_source_sha256','preserved_closed_campaign')}),flush=True)
