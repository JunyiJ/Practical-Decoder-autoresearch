"""Exercise exact restoration to an owned temporary copy; no model evaluation."""
import importlib.util,json
from pathlib import Path
E=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('materializer',E/'materialize.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
with m.a.session_lock(m.R/'research_state/tokenizer-worker-lock'):
 m.idle()
 with m.a.CpuPrerequisiteGuard(E/'restoration-check-guard.json',m.R,timeout_seconds=120):
  d=json.loads((E/'raw-completed.json').read_text());row=d['files'][0];target=E/'verified-restoration-copy.pt'
  m.materialize(row['member'],target)
  assert not Path(row['original']['path']).exists()
  assert m.a.file_digest(target)==row['original']['sha256'] and target.stat().st_size==row['original']['size']
  m.a.verify(row['archive']);m.a.verify(row['h200'])
  result=dict(status='exact_restoration_verified',source=m.a.ref(Path(__file__)),materializer=m.a.ref(E/'materialize.py'),manifest=m.a.ref(E/'raw-completed.json'),archive=row['archive'],original=row['original'],verified_copy=m.a.record(target),archive_preserved=True,h200_unchanged=True,original_raw_remains_losslessly_archived=True,scientific_training=False)
  target.unlink();m.a.syncdir(E);result['owned_verified_temporary_copy_removed']=not target.exists()
  m.a.publish(E/'restoration-check.json',result)
 print('Exact raw bytes restored, hashed and verified; archive/H200 preserved; owned temporary copy removed.',flush=True)
