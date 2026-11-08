"""Exactly twenty actual updates per arm through accepted training consumers."""
import copy
import gc
from pathlib import Path
import time
import torch
from research import byte_input_training_v2 as training
from research import byte_input_inference_v1 as inference
from research.byte_input_cache_v1 import Session
from research.byte_input_model_v3 import Spec,ByteInputModel,build_codebook,build_optimizer

def clean():gc.collect();torch.mps.synchronize();torch.mps.empty_cache()

def run(plan,archives,publish):
    torch.set_num_threads(4)
    cpu=torch.random.get_rng_state().clone();mps=torch.mps.get_rng_state().clone()
    assets=[Path(r['path']).read_bytes() for r in plan['codebook_assets']]
    results=[]
    try:
        for variant in ('native','raw_bytes','nonlexical_codes'):
            book=None if variant=='native' else build_codebook(*assets,variant,
                plan['roles']['nonlexical_key'] if variant=='nonlexical_codes' else None)
            old_mps=torch.mps.get_rng_state().clone()
            try:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(plan['roles']['native_initializer'])
                    spec=Spec(variant) if book is None else Spec(variant,plan['roles']['composer_initializer'],book.identity)
                    model=ByteInputModel(spec,book).to('mps')
            finally:torch.mps.set_rng_state(old_mps)
            view=training.load_view(plan['roles'],'mps')
            c=training.Context(model,build_optimizer(model,.00125),view,20,.00125,plan['roles'],plan['binding'])
            del model
            identity=copy.deepcopy(c._identity);rows=[]
            for step in range(1,21):
                torch.mps.synchronize();started=time.perf_counter()
                row=c.step();torch.mps.synchronize()
                row=dict(row,elapsed_seconds=time.perf_counter()-started,
                    mps_driver_bytes=torch.mps.driver_allocated_memory(),mps_current_bytes=torch.mps.current_allocated_memory())
                assert row['step']==step and row['gradient_tensors']==(88 if variant=='native' else 105)
                publish(variant+'-step'+str(step).zfill(2)+'.json',row);rows.append(row)
            assert c.optimizer._steps==c.averages.completed_steps==20 and c.averages.snapshots==8
            assert all(g['lr']==.00125*(.5 if i==0 else 1.)*.1 for i,child in enumerate(c.optimizer._optimizers) for g in child.param_groups)
            clean();payload=training.capture(c)
            parent=training.write_new(archives/(variant+'-raw.pt'),payload)
            assert parent['size']<=(512 if variant=='native' else 320)*1024**2
            del payload
            averaged=inference.export(c,parent)
            h200=inference.write_new(archives/(variant+'-h200.pt'),averaged)
            assert h200['size']<=(128 if variant=='native' else 80)*1024**2
            publish(variant+'-artifacts.json',dict(parent=parent,h200=h200,identity=identity,
                completed_steps=20,averaging_snapshots=8,parameters=len(c.model.ordered_parameters()),
                parameter_elements=sum(p.numel() for _,p in c.model.ordered_parameters())))
            del c,averaged;clean()
            model=inference.load(h200,identity,parent,book,'mps')
            started=time.perf_counter();score=inference.score(model,view,identity,h200);score_seconds=time.perf_counter()-started
            publish(variant+'-validation.json',score)
            cache_rows=[]
            for batch in (1,8):
                ids=view.train_data[:128].unsqueeze(0).repeat(batch,1).to('mps')
                with torch.no_grad():full=model.primary_logits(ids).values
                session=Session(model,batch);maximum=0.;gate=0.
                for lo,hi in [(0,32)]+[(i,i+1) for i in range(32,96)]+[(96,128)]:
                    actual=session.logits(ids[:,lo:hi],lo);target=full[:,lo:hi]
                    torch.testing.assert_close(actual,target,atol=3e-5,rtol=3e-4,equal_nan=False)
                    error=(actual-target).abs();maximum=max(maximum,float(error.max()))
                    gate=max(gate,float((error/(3e-5+3e-4*target.abs())).max()))
                    del actual,target,error
                cache_rows.append(dict(batch=batch,cache_bytes=session.storage_bytes,max_absolute_error=maximum,max_gate_ratio=gate))
                del full,session,ids;clean()
            row=dict(variant=variant,backend='mps',completed_steps=20,targets_seen=20480,rows=rows,
                parent=parent,h200=h200,validation_targets=score['report']['target']['targets'],
                validation_seconds=score_seconds,cached_correctness=cache_rows,
                total_update_seconds=sum(r['elapsed_seconds'] for r in rows),
                max_update_seconds=max(r['elapsed_seconds'] for r in rows),
                mean_update_seconds=sum(r['elapsed_seconds'] for r in rows)/20,
                max_mps_driver_bytes=max(r['mps_driver_bytes'] for r in rows),quality_selected=False)
            publish(variant+'-complete.json',row);results.append(row)
            del model,view,book,score;clean()
        publish('result.json',dict(accepted=True,arms=results,updates_executed=60,
            scope='Three fresh-role actual-corpus20-update resource pilots. No quality comparison/selection; no1200-step scientific launch.'))
    finally:torch.random.set_rng_state(cpu);torch.mps.set_rng_state(mps);clean()
