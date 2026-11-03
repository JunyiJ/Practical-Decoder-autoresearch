"""Exact retirement of one dependency-closed failed generated payload."""
import importlib.util,json,shutil,sys
from pathlib import Path
R=Path(__file__).resolve().parents[4];E=Path(__file__).resolve().parent;sys.path.insert(0,str(R));B=E.parent
s=importlib.util.spec_from_file_location('primitives',R/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a);a.E=E
from research.adamw_epsilon_quality_v1.support import idle
TARGET=R/'research_runs/tokenizer-final-ln-scale-resume-001/learned/final.pt'
def paths():
 out=[]
 for root in ('research','research_runs','research_archives','research_state','src','data'):
  for p in (R/root).rglob('*'):
   if E in p.parents or '__pycache__' in p.parts or any(x.startswith('maintenance-cache-cleanup-') for x in p.parts) or p.name.endswith('.lock') or 'tokenizer-worker-lock' in p.parts or p.name=='tokenizer-quality-progress.json' or not p.is_file():continue
   out.append(p)
 return sorted(out)
def audit():
 f=json.loads((B/'model-resume001-failure.json').read_text());assert f['status']=='failed_closed' and f['reason']=='disk_reserve' and not f['accepted'];assert not (B/'model-resume001.json').exists()
 assert TARGET.stat().st_size==445084647;record=a.record(TARGET);protected=[a.record(p) for p in paths() if p!=TARGET];shards=[]
 for i in range(0,len(protected),500):
  p=E/f'survivors-{i//500:03d}.json';a.publish(p,protected[i:i+500]);shards.append(a.ref(p))
 a.publish(E/'plan.json',dict(source=a.ref(Path(__file__)),failure=a.ref(B/'model-resume001-failure.json'),delete=record,protected_shards=shards,authorization='Human permits obsolete failed numerical payload retirement after dependencies close; only this failed generated fixture payload,never selected/fresh scientific weights.',dependency_closure='No accepted resume result or downstream payload dependencies; original failure source/config/log/guard/trace/receipts retained. Closed fixture never restored or rerun.'))
def apply():
 p=json.loads((E/'plan.json').read_text());a.verify(p['source']);a.verify(p['failure']);assert p['delete']['path']==str(TARGET)
 protected=[r for s in p['protected_shards'] for r in a.bound(s)];assert set(map(str,paths()))=={r['path'] for r in protected}|{str(TARGET)}
 for r in protected:a.check(Path(r['path']),r)
 a.check(TARGET,p['delete']);a.journal(dict(event='intent_permanent_retirement',file=p['delete']));TARGET.unlink();a.syncdir(TARGET.parent);a.journal(dict(event='removed',path=str(TARGET)))
 assert set(map(str,paths()))=={r['path'] for r in protected}
 for r in protected:a.check(Path(r['path']),r)
 a.verify(p['source']);a.verify(p['failure']);a.publish(E/'completed.json',dict(status='completed',deleted_bytes=p['delete']['size'],deleted_path=str(TARGET),manifest=a.ref(E/'plan.json'),retained_files_verified=len(protected),free_after=shutil.disk_usage(R).free,selected_fresh_unchanged=True,original_failure_evidence_unchanged=True))
if __name__=='__main__':
 mode=sys.argv[1];assert mode in ('audit','apply')
 with a.session_lock(R/'research_state/tokenizer-worker-lock'),a.CpuPrerequisiteGuard(E/(mode+'-guard.json'),R,timeout_seconds=900):idle();audit() if mode=='audit' else apply()
