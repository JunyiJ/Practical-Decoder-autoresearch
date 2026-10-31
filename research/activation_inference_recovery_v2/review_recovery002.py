"""One read-only portable prefix/closure review; no inference entry or retry."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from src.research.protocol import ROOT,digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock,MetadataWriter,reference
from src.research.tokenizer_activation_timing import FRESH_SEEDS,slot
from research.activation_inference_recovery_v2 import context as c,controller as e,final as f

parser=argparse.ArgumentParser();parser.add_argument('--count',type=int,choices=(2,3));parser.add_argument('--closed',action='store_true')
args=parser.parse_args();assert (args.count is None)==args.closed
root=c.RECOVERY_ROOT;base=ROOT/'research/tokenizer-evidence/activation-preparation-001'
stem='inference-recovery002-'+('closed' if args.closed else f'prefix-{args.count:02d}')
guard=base/(stem+'-guard.json');report=base/(stem+'-review.json');dest=base/(stem+'-portable')
if guard.exists() or report.exists() or dest.exists():raise FileExistsError('Existing review evidence; no overwrite')
with session_lock(GLOBAL_LOCK.parent) as g,session_lock(root) as r:
    idle()
    with CpuPrerequisiteGuard(guard,ROOT,timeout_seconds=180):
        ctx=c.Recovery(MetadataWriter(root,(g,r)),reference(root,c.PLAN_NAME))
        imported=c.bind(ctx);before=c.budget(ctx)
        with c.original_session(ctx) as source:
            c.verify_original(ctx,source,imported);prior=e.accepted_prefix(ctx,source,imported)
        assert len(prior)==(3 if args.closed else args.count)
        decision=None;receipt=None
        if args.closed:
            receipt=ctx.document(reference(root,f.BASE+'-complete.json'))
            assert receipt['kind']==c.KIND and receipt['status']=='closed' and receipt['plan']==ctx.plan
            saved=ctx.document(receipt['decision']);decision=f.build_report(ctx);assert saved==decision
            assert receipt['adopted']==decision['adopted']
            gr=ctx.document(receipt['guard']);c.verify_guard(ctx,receipt['guard'],gr['pid'])
        copied={}
        physical=slot(FRESH_SEEDS[args.count-1]) if not args.closed else None
        for name in before['files']:
            chosen=(name.startswith(f'evidence/inference/{physical}/') or name.startswith('control/') and name.endswith(f'inference-{physical}.json')) if not args.closed else (
                name.startswith(f.BASE) or name==e.SUMMARY or name.startswith('control/recovery-final-'))
            if chosen:
                ref=reference(root,name);target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream:stream.write((root/name).read_bytes())
                assert file_digest(target)==ref['sha256'];copied[name]=ref
        assert c.budget(ctx)==before
        idle()
value=dict(status='recovery_closed_independently_reviewed' if args.closed else 'recovery_prefix_independently_reviewed',
    observed_at=datetime.now(timezone.utc).isoformat(),root=str(root),plan=ctx.plan,
    accepted=prior,accepted_seeds=list(FRESH_SEEDS[:len(prior)]),original_import_sha256=digest(imported),
    original_source_sha256=imported['source_sha256'],protected_original_checkpoints=imported['protected_checkpoints'],
    copied=copied,decision=decision,terminal_receipt=receipt,metadata_bytes=before['metadata_bytes'],
    guard=reference(ROOT,str(guard.relative_to(ROOT))),procedure_sha256=file_digest(Path(__file__)),
    original_campaign_retried=False,closed_recovery001_retried=False,carried_first_repeated=False,
    carried_import_sha256=digest(c.carried_import(ctx)),new_accepted_seeds=list(c.REMAINING_SEEDS[:len(prior)-1]),holdout_scored=False)
with report.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps({k:value[k] for k in ('status','accepted_seeds','metadata_bytes')}))
