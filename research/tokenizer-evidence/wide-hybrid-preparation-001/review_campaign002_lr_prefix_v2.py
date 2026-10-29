"""Read-only LR prefix review and lossless non-weight archive; no dispatch or cleanup."""
import argparse,base64,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_quality_dispatch_v2 import _prior
from src.research.tokenizer_wide_hybrid_quality_lr_startup_v2 import _frozen
from src.research.tokenizer_wide_hybrid_quality_startup_v2 import _history
from src.research.tokenizer_wide_hybrid_quality_audit_v2 import bind_lr_prefix
from src.research.tokenizer_wide_hybrid_pilot_startup_v2 import _inventory,_check_inventory
from src.research.tokenizer_wide_hybrid_storage_v2 import bind_storage
HERE=Path(__file__).resolve().parent
RUN=ROOT/'research_runs/tokenizer-wide-hybrid-002'
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--completed',type=int,required=True);a=ap.parse_args()
    assert 1<=a.completed<=8
    stem=f'campaign002-lr-prefix-{a.completed:02d}-v2'
    freeze=json.loads((HERE/'campaign002-freeze-v2-review.json').read_text());sha=freeze['plan_sha256']
    boundary=json.loads((HERE/'campaign002-pilot-boundary-v2.json').read_text())['reference']
    projection=dict(path='meta/projection.json',sha256=file_digest(RUN/'evidence/meta/projection.json'))
    with session_lock(ROOT/'research_state/tokenizer-worker-lock') as g,session_lock(RUN) as c, CpuPrerequisiteGuard(HERE/(stem+'-guard.json'),ROOT,timeout_seconds=1800):
        writer=MetadataWriter(RUN,(g,c));root=RUN/'evidence'
        refs=_prior(writer,a.completed,sha,projection,boundary)
        bound,requests=_frozen(writer,a.completed,sha,projection)
        inventory=_inventory(root);plan=json.loads((root/'meta/plan.json').read_text())
        history=_history(writer,sha,boundary,inventory)
        prefix=bind_lr_prefix(root,context=bound['context'],requests=requests[:-1],
            evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')},
            numerical_ref=plan['numerical_ref'],projection_ref=projection,training_token_sha256=plan['training_token_sha256'])
        storage=bind_storage(RUN,json.loads((root/projection['path']).read_text()),pilots=history['records'],records=prefix['records'],rates=None,selection=None,stages=[history['chain_stage']],current_inventory=inventory,checkpoint_pairs={**history['checkpoint_pairs'],**prefix['checkpoint_pairs']},lock_fds=(g,c))
        files={str(p.relative_to(RUN)):dict(size=p.stat().st_size,sha256=file_digest(p)) for p in RUN.rglob('*') if p.is_file()}
        metadata={n:base64.b64encode((RUN/n).read_bytes()).decode() for n in files if not n.endswith('.pt')}
        _check_inventory(root,inventory)
        assert all(file_digest(RUN/n)==v['sha256'] for n,v in files.items())
        results=[json.loads((root/f'quality/{i:02d}/result.json').read_text()) for i in range(a.completed)]
        report=dict(status='independently_reviewed_lr_prefix',completed=a.completed,plan_sha256=sha,
            reviewer_sha256=file_digest(Path(__file__)),dispatch_references=refs,records=prefix['records'],checkpoint_pairs=prefix['checkpoint_pairs'],storage=storage,results=results,
            selection=None,adopted=False,holdout_consumed=False,files=files,noncheckpoint_bytes_base64=metadata)
        save(HERE/(stem+'-review.json'),report)
        print(json.dumps(dict(status=report['status'],records=prefix['records'],storage=storage)),flush=True)
if __name__=='__main__':main()
