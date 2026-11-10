"""Audit obsolete screens, then retire them once at a safe research-lock gap.

Never stops a process. A bounded waiting process exits after one pass or 12h.
Fresh confirmations, selected models, failed diagnostics and all metadata stay.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import OUT, ROOT, capacity, opens, protected_current, remove_entries, row, save, sha

STUDIES = {
    'tokenizer-moe-quality-v1-001': ('post-prefix-boundary-001/quality001-closure001.json', 9, 18),
    'tokenizer-mla-gate-quality-v1-001': ('post-diagnostic-boundary-001/gate-quality001-closure001/completed.json', 9, 18),
    'tokenizer-readout-mixture-quality-v1-001': ('readout-quality-preparation-001/readout-quality001-closure001/completed.json', 9, 18),
    'tokenizer-gpt2-prefix-quality-v1-001': ('tokenizer-capacity-boundary-001/quality001-closure002.json', 6, 12),
}


def active_dependency_check():
    state = json.loads((ROOT/'research_state/general-rsi-state.json').read_text())
    texts = [(ROOT/state['panel']).read_text(), (ROOT/state['plan']).read_text()]
    next_plan = ROOT/'research/GENERAL_RSI_NEXT_CAMPAIGNS.md'
    if next_plan.exists(): texts.append(next_plan.read_text())
    for study in STUDIES:
        assert not any(study in text for text in texts), ('Active plan references candidate study', study)
    return dict(panel=state['panel'], panel_sha256=sha(ROOT/state['panel']),
                next_plan_sha256=sha(next_plan) if next_plan.exists() else None,
                candidate_study_references_in_active_plans=0)


def audit():
    protected = protected_current()
    entries, evidence = [], []
    dependencies = active_dependency_check()
    for study, (closure_name, screen_count, total_count) in STUDIES.items():
        root = ROOT/'research_runs'/study
        closure = ROOT/'research/tokenizer-evidence'/closure_name
        closed = json.loads(closure.read_text())
        assert 'closed' in closed['status'].lower() and 'not_adopted' in closed['status'].lower()
        assert (root/'STOP').is_file()
        protected[str(closure)] = sha(closure)
        protected[str(root/'STOP')] = sha(root/'STOP')
        for slot in range(total_count):
            base = root/f'evidence/quality/{slot:02d}'
            accepted = json.loads((base/'accepted.json').read_text())
            request = json.loads((base/'request.json').read_text())
            assert accepted['slot'] == slot and request['slot'] == slot
            binding = accepted['acceptance']['binding']
            if slot == 0:
                plan = Path(request['plan']['path'])
                assert sha(plan) == request['plan']['sha256']
                original = json.loads(plan.read_text())
                assert original['design']['screen_trajectories'] == screen_count
                assert len(original['specs']) == screen_count
                protected[str(plan)] = sha(plan)
            for kind in ('parent','h200'):
                p = Path(binding[kind]['path'])
                assert p.parent == base/'checkpoints'
                actual = row(p)
                assert actual['sha256'] == binding[kind]['sha256'], p
                if slot < screen_count:
                    entries.append(actual)
                else:
                    protected[str(p)] = actual['sha256']
            # Preserve every original log/config/request/acceptance/guard record.
            for p in base.rglob('*'):
                if p.is_file() and not p.is_symlink() and 'checkpoints' not in p.relative_to(base).parts:
                    protected[str(p)] = sha(p)
        # Closure references anchor final statistics, selection, confirmation and
        # downstream analyses. Record all hash-addressed non-weight references.
        def visit(value):
            if isinstance(value, dict):
                if isinstance(value.get('path'),str) and 'sha256' in value:
                    p=Path(value['path'])
                    if p.is_file() and p.suffix not in ('.pt','.tmp'):
                        assert sha(p)==value['sha256'],p
                        protected[str(p)]=value['sha256']
                for child in value.values():visit(child)
            elif isinstance(value,list):
                for child in value:visit(child)
        visit(closed)
        evidence.append(dict(study=study,closure=str(closure),status=closed['status'],
                             obsolete_screen_slots=list(range(screen_count)),
                             preserved_fresh_slots=list(range(screen_count,total_count))))
        print(json.dumps({'audited':study,'candidate_bytes_so_far':sum(x['allocated'] for x in entries)}),flush=True)
    assert not any(e['path'] in protected for e in entries)
    manifest=dict(authorization='Direct human request on 2026-11-09 to safely free unused caches and obsolete checkpoints without interrupting research or deleting important models/results.',
                  policy_exception='Latest human authorization and AGENTS.md supersede old keep-all-screen wording only for these exact obsolete screening payloads. All four studies concluded without adoption; fresh independent confirmations supersede the LR screens. Historical source, STOP, logs, configs, scores, final decisions and original retention records remain unchanged.',
                  entries=entries,protected_sha256=protected,studies=evidence,dependencies=dependencies,
                  allocated_bytes=sum(x['allocated'] for x in entries),before=capacity(),
                  source_sha256={str(p):sha(p) for p in (Path(__file__).resolve(),OUT/'common.py')})
    save(OUT/'checkpoint-manifest.json',manifest)
    print(json.dumps({'status':'audited_waiting_for_lock','files':len(entries),'allocated_bytes':manifest['allocated_bytes']}),flush=True)


def wait_and_apply():
    manifest=json.loads((OUT/'checkpoint-manifest.json').read_text())
    for p,h in manifest['source_sha256'].items():assert sha(p)==h,p
    assert not (OUT/'checkpoint-completed.json').exists()
    with (OUT/'pending-cleanup.lock').open('a+') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(OUT/'checkpoint-worker.json',dict(pid=os.getpid(),status='waiting_for_safe_gap',
             started_unix=time.time(),deadline_unix=time.time()+12*3600,
             manifest_sha256=sha(OUT/'checkpoint-manifest.json')))
        deadline=time.monotonic()+12*3600
        while time.monotonic()<deadline:
            with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    state=json.loads((ROOT/'research_state/general-rsi-state.json').read_text())
                    pid=state.get('active_worker_pid')
                    ps=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
                    running={int(line.split(None,1)[0]):line for line in ps.splitlines() if line.strip()}
                    # Treat a live recorded worker as active even if it briefly
                    # releases its lock. Do not kill it or change its state.
                    if pid not in running:
                        assert not any(any(study in line for study in STUDIES) for p,line in running.items() if p!=os.getpid()), 'Historical study consumer exists'
                        dependencies=active_dependency_check()
                        protected=dict(manifest['protected_sha256'])
                        protected.update(protected_current())
                        for p,h in protected.items():assert sha(p)==h,p
                        opened=opens()
                        for e in manifest['entries']:
                            assert e['path'] not in opened and row(e['path'])==e,e['path']
                        before=capacity()
                        removed,skipped=remove_entries(manifest['entries'],OUT/'checkpoint-deletion-journal.jsonl')
                        for p,h in protected.items():assert sha(p)==h,p
                        result=dict(status='completed' if not skipped else 'partial',files_removed=len(removed),
                                    allocated_bytes_removed=sum(e['allocated'] for e in removed),
                                    before=before,after=capacity(),skipped=skipped,
                                    protected_hashes_verified=len(protected),dependencies=dependencies,
                                    no_workers_stopped=True,completed_unix=time.time())
                        save(OUT/'checkpoint-completed.json',result)
                        print(json.dumps(result),flush=True)
                        return
            time.sleep(20)
        save(OUT/'checkpoint-timeout.json',dict(status='not_removed_no_safe_gap',time_unix=time.time()))


if __name__=='__main__':
    if sys.argv[1:] == ['audit']:audit()
    elif sys.argv[1:] == ['wait']:wait_and_apply()
    else:raise SystemExit('Use audit or wait')
