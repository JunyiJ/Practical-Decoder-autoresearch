"""Freeze one real dual-EMA resource/acceptance pilot; quality remains blocked."""
import hashlib,json,shutil
from pathlib import Path
from types import SimpleNamespace
import torch
from research.ema_horizon_v1 import contract as c,execution as e
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter

B=e.EVIDENCE
torch.set_num_threads(4)
def reject(fn):
    try:fn()
    except (ValueError,TypeError):return
    raise AssertionError('Boundary accepted invalid input')
with session_lock(GLOBAL_LOCK.parent):
    e.idle()
    assert not e.ROOT.exists() and not e.freeze_path('pilots').exists()
    with CpuPrerequisiteGuard(B/'pilot001-prepare-guard.json',c.ROOT,timeout_seconds=120,poll_seconds=.25):
        census=read(reference(B/'scientific-seed-census-002/result.json'))
        assert census['status']=='fresh_seeds_reserved' and not census['matches']
        inventory=B/'scientific-seed-census-002/inventory.json'
        assert hashlib.sha256(inventory.read_bytes()).hexdigest()==census['inventory_sha256']
        seeds=census['seeds'];sources=c.source_inventory()
        prereqs={}
        for name in ['shadow-prerequisite-001-result.json','artifact-contract-001-result.json']:
            value=read(reference(B/name));assert value['status']=='passed'
            assert all(sources.get(p)==h for p,h in value['sources'].items())
            prereqs[name]=reference(B/name)
        for name in ['shadow-prerequisite-001-cpu-guard.json','shadow-prerequisite-001-mps-guard.json','artifact-contract-001-guard.json']:
            value=read(reference(B/name));assert value['status']=='finished' and value['reason'] is None
            prereqs[name]=reference(B/name)
        prereqs['seed_census']=reference(B/'scientific-seed-census-002/result.json')
        # Boundary tests do not create trajectories or launch subprocesses.
        reject(lambda:e.prefix('pilots',1));reject(lambda:e.prefix('quality',3));reject(lambda:e.Guard(B/'never-created.json',0))
        fake=SimpleNamespace(lock_fds=(7,8));args=e.argv(fake,'pilots',0,'test-digest')
        assert args[1:4]==['-m',e.MODULE,'worker'] and args[-4:]==['--global-fd','7','--campaign-fd','8']
        cfg=c.config(e.ROOT/'evidence/pilots/00','pilot',0,seeds)
        sink=e.Sink.__new__(e.Sink);sink.base='evidence/pilots/00';sink.raw=None;sink.cfg=cfg
        written=[];sink.weights=SimpleNamespace(save=lambda name,value:written.append((name,value)))
        path=e.ROOT/'evidence/pilots/00/checkpoints/unit.pt'
        reject(lambda:sink.checkpoint(path,dict(step=19)))
        reject(lambda:sink.checkpoint(e.ROOT/'wrong.pt',dict(step=20)))
        sink.checkpoint(path,dict(step=20));assert written==[('unit.pt',dict(step=20))]
        reject(lambda:sink.checkpoint(path,dict(step=20)))
        assert not path.exists()
        free=shutil.disk_usage(c.ROOT).free;assert free>=c.REQUIRED_FREE
        boundary=dict(status='passed',checks=8,scope='Finite slot, deadline, argv and final-only sink boundaries; positive sink uses in-memory capture, not a real checkpoint.',source_inventory=sources)
        e.exclusive(B/'pilot001-boundary-tests.json',boundary);prereqs['boundary_tests']=reference(B/'pilot001-boundary-tests.json')
        plan=dict(kind=c.KIND,stage='pilots',status='frozen_one_pilot_only',root=str(e.ROOT),
            sources=sources,environment=c.inherited.environment(),data=c.inherited.bind_data(),seeds=seeds,
            specs=[c.spec('pilot',0,seeds)],configs=[cfg],limits=dict(c.inherited.LIMITS),
            required_free_bytes=c.REQUIRED_FREE,trajectory_allowance_bytes=c.TRAJECTORY_ALLOWANCE,
            free_bytes=free,prerequisites=prereqs,quality_plan_frozen=False,quality_launch_permitted=False,
            real_artifact_gate='This one20step batch8/context128 pilot is the first real dual-EMA export/worker/independent-parent/full-validation/cache acceptance test. Failure stops this attempt and preserves all evidence; no quality launch or automatic retry.',
            retention='Keep this raw/H100/H200 triple plus all three future fresh triples; zero cleanup credit. No extra full-size generated artifact fixture.',
            holdout='No test scoring; unchanged training loader materializes test tokens, while scoring uses train/validation-only view.')
        e.exclusive(B/'pilot001-plan.json',plan)
        e.ROOT.mkdir();(e.ROOT/'evidence/meta').mkdir(parents=True);(e.ROOT/'control').mkdir()
        e.exclusive(e.freeze_path('pilots'),dict(plan=reference(B/'pilot001-plan.json'),quality_launch_permitted=False))
        with session_lock(e.ROOT) as lock:
            # Shared global lock is still held; full entry binding is checked by
            # the launcher before any request or child is created.
            assert e.prefix('pilots',0)==([],0)
        assert c.source_inventory()==sources
    e.exclusive(B/'pilot001-preflight.json',dict(status='prepared_one_pilot_only',freeze=reference(e.freeze_path('pilots')),
        guard=reference(B/'pilot001-prepare-guard.json'),quality_launch_permitted=False))
    print(json.dumps(reference(e.freeze_path('pilots'))))
