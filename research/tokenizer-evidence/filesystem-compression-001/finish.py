"""Finish the same explicit storage manifest after a harmless no-savings copy.

The original failure/guard/journal remain immutable. Verified no-savings staged
copies may be discarded while their original paths remain untouched. No research
campaign, test fixture or numerical execution is retried.
"""
import importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
E=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('compression',E/'compress.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=m.ROOT

def finish():
 p=json.loads((E/'plan.json').read_text());assert m.file_digest(E/'compress.py')==p['script_sha256']
 g=json.loads((E/'apply-guard.json').read_text());assert g['status']=='fixture_exception' and g['reason']=='AssertionError()'
 events=[json.loads(s) for s in (E/'journal.jsonl').read_text().splitlines()]
 assert len(events)==244 and all(events[i]['event']=='verified_staged_copy' and events[i+1]['event']=='replacement_verified' and events[i]['path']==events[i+1]['path'] for i in range(0,len(events),2))
 done={r['path']:r for r in events if r['event']=='replacement_verified'};assert len(done)==122
 for n,v in p['files'].items():
  w=m.check(ROOT/n,v,physical=n not in done)
  if n in done:assert w['flags']&32 and w['allocated_bytes']==done[n]['allocated_after']
 for n,v in p['protected'].items():m.check(ROOT/n,v,True)
 failed='research_state/position-quality-preparation001/pilot-repair-knowledge-snapshot001.md'
 temp=ROOT/(failed+'.apfs-compression-001.tmp');v=p['files'][failed];w=m.check(temp,v)
 assert w['size']==4011 and w['flags']==0 and w['allocated_bytes']==v['allocated_bytes']==4096
 before=shutil.disk_usage(ROOT).free;skipped=[]
 with (E/'continuation-journal.jsonl').open('x') as log:
  def record(v):log.write(json.dumps(v)+'\n');log.flush();os.fsync(log.fileno())
  temp.unlink();m.syncdir(temp.parent);record(dict(event='verified_no_savings_copy_discarded',path=str(temp.relative_to(ROOT)),original_unchanged=True,sha256=v['sha256']))
  for n,v in p['files'].items():
   if n in done:continue
   src=ROOT/n;tmp=src.with_name(src.name+'.apfs-compression-001.tmp');assert not tmp.exists();m.check(src,v,True)
   if n==failed:
    skipped.append(n);record(dict(event='original_retained_no_savings',path=n));continue
   assert shutil.disk_usage(ROOT).free>8*1024**3+v['size']
   subprocess.run(['/usr/bin/ditto','--hfsCompression','--noclone',str(src),str(tmp)],check=True,timeout=30)
   w=m.check(tmp,v);m.check(src,v,True)
   if w['allocated_bytes']>=v['allocated_bytes']:
    tmp.unlink();m.syncdir(src.parent);skipped.append(n);record(dict(event='original_retained_no_savings',path=n));continue
   assert w['flags']&32
   with tmp.open('rb') as f:os.fsync(f.fileno())
   record(dict(event='verified_staged_copy',path=n,sha256=v['sha256'],allocated_before=v['allocated_bytes'],allocated_after=w['allocated_bytes']))
   os.replace(tmp,src);m.syncdir(src.parent);m.check(src,v)
   r=dict(path=n,sha256=v['sha256'],allocated_before=v['allocated_bytes'],allocated_after=w['allocated_bytes']);done[n]=r
   record(dict(event='replacement_verified',**r))
 assert len(done)+len(skipped)==len(p['files'])
 for n,v in p['files'].items():m.check(ROOT/n,v,physical=n in skipped)
 for n,v in p['protected'].items():m.check(ROOT/n,v,True)
 m.publish(E/'completed.json',dict(status='completed',compressed_files=len(done),unchanged_no_savings_files=skipped,logical_bytes_unchanged=sum(v['size'] for v in p['files'].values()),allocated_bytes_reclaimed=sum(r['allocated_before']-r['allocated_after'] for r in done.values()),free_before=p['free_before'],continuation_free_before=before,free_after=shutil.disk_usage(ROOT).free,required_free_bytes=m.REQUIRED,protected_files_verified=len(p['protected']),protected_tokenizer_weights_verified=159,all_original_paths_and_readable_bytes_preserved=True,training=False,holdout_consumed=False,manifest_sha256=m.file_digest(E/'plan.json'),journal_sha256=m.file_digest(E/'journal.jsonl'),continuation_journal_sha256=m.file_digest(E/'continuation-journal.jsonl'),continuation_script_sha256=m.file_digest(Path(__file__)),original_partial_guard_sha256=m.file_digest(E/'apply-guard.json')))
 print((E/'completed.json').read_text(),flush=True)
if __name__=='__main__':
 with m.session_lock(ROOT/'research_state/tokenizer-worker-lock'),m.CpuPrerequisiteGuard(E/'continuation-guard.json',ROOT,timeout_seconds=1800):finish()
