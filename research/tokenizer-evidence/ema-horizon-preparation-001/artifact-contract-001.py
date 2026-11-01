"""CPU artifact-contract tests: synthetic tensors, real bounded file reading."""
import copy,hashlib,json,os
from pathlib import Path
from types import SimpleNamespace
import torch
from research.ema_horizon_v1 import contract as c,artifacts as a
from research.block_norm_v1 import artifact_acceptance as old
from research.block_norm_v1.runtime.models.block_norm import CHECKPOINT_KEY
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock

B=c.ROOT/'research/tokenizer-evidence/ema-horizon-preparation-001'
STEM='artifact-contract-001'
seeds=dict(pilot=375921,screen=375922,fresh=[375931,375932,375933],evaluation=375923,bootstrap=375924)
torch.set_num_threads(4)
checks=[]
def reject(name,fn):
    try:fn()
    except (ValueError,TypeError,KeyError):checks.append(name);return
    raise AssertionError('Accepted mutation: '+name)
with session_lock(GLOBAL_LOCK.parent):
    idle()
    assert not (B/(STEM+'-result.json')).exists()
    with CpuPrerequisiteGuard(B/(STEM+'-guard.json'),c.ROOT,timeout_seconds=120,poll_seconds=.25):
        sources=c.source_inventory()
        output=c.ROOT/'research_runs/tokenizer-ema-horizon-v1-fixture/evidence/quality/00'
        cfg=c.config(output,'quality',0,seeds);spec=cfg['training']['weight_averaging']
        req=dict(scope=a.SCOPE,sources=sources,stage='quality',slot=0,seeds=seeds,config=cfg,
            files={k:dict(path=str(output/'checkpoints'/(k+'.pt')),sha256='synthetic-no-file') for k in ('raw','h100','h200')})
        req['files']['config']=dict(path=str(output/'config.json'),sha256='synthetic-no-file')
        raw=dict(model_state={'w':torch.tensor([1.,2.]),'buffer':torch.tensor([3.])},
            training_state=dict(weight_averaging_state=dict(spec=spec,parameter_names=['w'],
                averages={'100':{'w':torch.tensor([.5,1.5])},'200':{'w':torch.tensor([.25,1.25])}})),
            **{CHECKPOINT_KEY:dict(synthetic=True)})
        data=SimpleNamespace(tokenizer_contract=dict(synthetic=True))
        for half in (100,200):
            avg=dict(cfg=copy.deepcopy(cfg),parent_checkpoint=req['files']['raw'],step=1200,
                model_math_version=2,parameter_layout_version=2,tokenizer_contract=data.tokenizer_contract,
                model_state=dict(w=raw['training_state']['weight_averaging_state']['averages'][str(half)]['w'].clone(),buffer=raw['model_state']['buffer'].clone()),
                averaged_weights=dict(spec=spec,half_life=half,snapshots=41,completed_steps=1200,parameter_names=['w'],purpose='inference_only_not_optimizer_resume'),
                **{CHECKPOINT_KEY:raw[CHECKPOINT_KEY]})
            export=dict(checkpoint=req['files'][f'h{half}']['path'],checkpoint_sha256=req['files'][f'h{half}']['sha256'],parent_checkpoint=avg['parent_checkpoint'],averaged_weights=avg['averaged_weights'])
            a.audit_export(raw,avg,req,half,export,data);checks.append(f'H{half}_synthetic_export')
            bad=copy.deepcopy(avg);bad['averaged_weights']['half_life']=200 if half==100 else 100
            reject(f'H{half}_wrong_half',lambda:a.audit_export(raw,bad,req,half,export,data))
            bad=copy.deepcopy(avg);bad['model_state']['w'][0]+=1
            reject(f'H{half}_changed_weight',lambda:a.audit_export(raw,bad,req,half,export,data))
            bad=copy.deepcopy(avg);bad['model_state']['buffer'][0]+=1
            reject(f'H{half}_changed_buffer',lambda:a.audit_export(raw,bad,req,half,export,data))
            bad=copy.deepcopy(avg);bad['optimizer_state']={}
            reject(f'H{half}_optimizer_export',lambda:a.audit_export(raw,bad,req,half,export,data))
            bad=copy.deepcopy(avg);bad['cfg']['training']['learning_rate']=.0025
            reject(f'H{half}_config_change',lambda:a.audit_export(raw,bad,req,half,export,data))
            bad=copy.deepcopy(avg);bad['parent_checkpoint']['sha256']='wrong'
            reject(f'H{half}_wrong_parent',lambda:a.audit_export(raw,bad,req,half,export,data))
            bad=copy.deepcopy(avg);bad['averaged_weights']['snapshots']=40
            reject(f'H{half}_wrong_snapshots',lambda:a.audit_export(raw,bad,req,half,export,data))
            badraw=copy.deepcopy(raw);del badraw['training_state']['weight_averaging_state']['averages']['100']
            reject(f'H{half}_missing_average',lambda:a.audit_export(badraw,avg,req,half,export,data))
        reject('old_scope_rejects_new_request',lambda:old.verify_request(req))
        reject('new_scope_rejects_old_request',lambda:a.verify_request(dict(scope=old.TRIAL_SCOPE)))
        # A sparse all-zero size probe is not a serialized model/checkpoint.
        # Retain it on failure; remove only the exact verified successful probe.
        probe=c.ROOT/'research_state/ema-artifact-sizeprobe-001.bin'
        fd=os.open(probe,os.O_RDWR|os.O_CREAT|os.O_EXCL,0o600)
        try:os.ftruncate(fd,558903984);os.fsync(fd)
        finally:os.close(fd)
        stat=probe.stat();allocated=stat.st_blocks*512
        reject('legacy_512MiB_rejects_dual_raw_size',lambda:c.inherited.file_sha(probe,root=c.ROOT,max_bytes=512*1024**2))
        digest=c.inherited.file_sha(probe,root=c.ROOT,max_bytes=c.RAW_LIMIT)
        ref=dict(path=str(probe),sha256=digest);c.check_artifact('raw',ref);checks.append('new640MiB_reads_complete_probe')
        reject('export128MiB_rejects_raw_size',lambda:c.check_artifact('h100',ref))
        manifest=dict(path=str(probe),size=stat.st_size,sha256=digest,device=stat.st_dev,inode=stat.st_ino,
            allocated_bytes=allocated,kind='synthetic_sparse_zero_size_probe_not_checkpoint',reason='Successful bounded file-reader test; no model tensors or research evidence in probe')
        with (B/(STEM+'-probe-manifest.json')).open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
        actual=probe.stat();assert (actual.st_dev,actual.st_ino,actual.st_size)==(stat.st_dev,stat.st_ino,stat.st_size)
        assert c.inherited.file_sha(probe,root=c.ROOT,max_bytes=c.RAW_LIMIT)==digest
        probe.unlink()
        assert not probe.exists() and c.source_inventory()==sources
        result=dict(status='passed',checks=checks,check_count=len(checks),sources=sources,fixture_seeds=seeds,
            sparse_probe_manifest=manifest,sparse_probe_removed=True,real_checkpoints_tested=False,
            full_model_artifact_acceptance_pending=True,scientific_training_launched=False,holdout_read=False,
            research_checkpoint_deletion=False,scope='Synthetic in-memory tensor/metadata rejection tests plus real bounded file-size/hash reading; does not establish real model export/acceptance.')
    with (B/(STEM+'-result.json')).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='passed',checks=len(checks),probe_allocated_bytes=allocated)))
