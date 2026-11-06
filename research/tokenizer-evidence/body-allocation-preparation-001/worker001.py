"""Three four-update real-data acceptance proofs, outside the scientific panel."""
import argparse,json,shutil
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,config,plain,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_worker_v1.execution import sources,checked
from research.body_allocation_worker_v1.dispatch import launch
from research.final_ln_scale_quality_v3.support import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import snapshot

B=Path(__file__).resolve().parent;RUN=ROOT/'research_runs/tokenizer-body-allocation-worker-v1-001';PLAN=B/'worker001-plan.json'
CAPS=[560*1024**2,432*1024**2,432*1024**2];FLOOR=8*1024**3+sum(CAPS)+64*1024**2
torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--slot',type=int,choices=range(3));a=p.parse_args()
def exclusive(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
with session_lock(GLOBAL_LOCK.parent) as g:
    idle();environment()
    if a.mode=='prepare':
        require(not RUN.exists() and not PLAN.exists(),'New attempt required')
        for arm in ('incumbent','shallow','deeper'):
            r=checked(reference(B/f'resume001-{arm}.json'));require(r['status']=='passed','All model resume proofs required')
        free=shutil.disk_usage(ROOT).free;require(free>=FLOOR,'Complete generated-worker capacity required')
        specs=[]
        for slot,arm in enumerate(('incumbent','shallow','deeper')):
            cfg=config(arm,4,91261+slot);t=cfg.training;base=RUN/f'evidence/pilots/{slot:02d}'
            t.checkpoint_dir=str(base/'checkpoints');t.metrics_file=str(base/'metrics.jsonl');t.summary_file=str(base/'worker.json')
            specs.append(dict(slot=slot,stage='pilots',arm=arm,config=plain(cfg),weight_cap=CAPS[slot]))
        plan=dict(kind='body_allocation_real_data_worker_fixture_v1',root=str(RUN),runner=reference(__file__),sources=sources(),environment=environment(),
            specs=specs,seconds=600,compare_cpu=True,required_free=FLOOR,cleanup_target=FLOOR+2*1024**3,free_bytes=free,metadata_cap=64*1024**2,
            scope='Four-update real-data worker proof only; fixture seeds are not scientific seeds; no screen/quality/adoption.',
            prerequisites={arm:reference(B/f'resume001-{arm}.json') for arm in ('incumbent','shallow','deeper')})
        RUN.mkdir();exclusive(PLAN,plan);print(json.dumps(dict(status='prepared',required=FLOOR,free=free,cleanup_target=plan['cleanup_target'])))
    else:
        plan=checked(reference(PLAN));require(plan['runner']==reference(__file__) and plan['sources']==sources(),'Frozen worker source differs')
        require(a.slot is not None and not (RUN/'STOP').exists(),'Exact unclosed fixture slot required')
        for i in range(a.slot):require((B/f'worker001-accepted-{i:02d}.json').exists(),'Serial prior acceptance required')
        with session_lock(RUN) as r:
            w=MetadataWriter(RUN,(g,r));used=snapshot(RUN)['checkpoint_logical_bytes'];required=max(8*1024**3,FLOOR-used)
            try:
                row=launch(w,reference(PLAN),plan['specs'][a.slot],required)
                exclusive(B/f'worker001-accepted-{a.slot:02d}.json',dict(status='accepted',accepted=reference(RUN/f'evidence/pilots/{a.slot:02d}/accepted.json'),scientific_quality=False,holdout_scored=False))
            except Exception as exc:
                if not (RUN/'STOP').exists():w.publish_json('STOP',dict(error=repr(exc),slot=a.slot,no_retry=True))
                raise
