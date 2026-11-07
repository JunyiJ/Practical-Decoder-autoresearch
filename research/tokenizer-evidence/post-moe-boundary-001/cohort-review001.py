"""Independent scalar review of completed cohort metadata; no model/gradient replay."""
import hashlib,json,math,shutil
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from research.tokenizer_gradient_cohorts_v1 import contract as K
from research.tokenizer_moe_worker_v1.execution import idle
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
E=K.C.E;R=K.RUN

def ref(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(r):
    p=Path(r['path']);require(ref(p)=={k:r[k] for k in ('path','sha256')},'Immutable reference changed')
    return json.loads(p.read_text())
def publish(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

with session_lock(GLOBAL_LOCK.parent):
    idle();require(not(E/'cohort-review001.json').exists() and not(E/'cohort-review001-guard.json').exists(),'New terminal scalar review only;no retry')
    with CpuPrerequisiteGuard(E/'cohort-review001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
        terminal=ref(R/'STOP');stop=checked(terminal);result=checked(stop['result']);plan=checked(ref(R/'plan.json'));guard=checked(ref(R/'guard.json'))
        require(stop['status']=='complete_training_free_gradient_cohorts' and stop['never_rerun'],'Complete terminal cohort diagnostic required')
        require(result['status']=='complete_training_free_three_endpoint_four_cohort_diagnostic' and result['total_target_evaluations']==393216 and result['optimizer_steps']==result['weight_files_written']==0 and result['old_probe_excluded'] and result['global_cpu_mps_rng_unchanged'],'Complete training-free diagnostic required')
        require(not result['validation_encoded'] and not result['holdout_encoded'] and guard['status']=='finished' and guard['reason'] is None,'Original closed resource/data guards required')
        require(plan['sources']==result['sources']==K.sources() and len(result['sources'])==1047,'All1047 frozen bindings required')
        require(plan['cohort_seeds']==list(K.SEEDS) and plan['initializer']==K.INITIALIZER and plan['combined_draws_per_endpoint']==128 and plan['cohorts_per_endpoint']==4 and plan['draws_per_cohort']==32,'Original finite four-cohort role/count design required')
        first=checked(plan['prerequisites']['first_review']);checked(first['terminal']);first_rows=first['endpoints'] if 'endpoints' in first else first['rows']
        rows=[];streams=[];maxerror=0.
        for item in result['endpoints']:
            d=checked(item);draws=d['draws'];reports=d['cohorts'];combined=d['combined']
            require(len(draws)==combined['draws']==128 and len(reports)==4 and len(combined['parameter_statistics'])==88 and d['model_before']==d['model_after'] and d['model_unchanged'] and d['optimizer_steps']==0 and d['old_probe_excluded'],'Complete unchanged fixed endpoint required')
            require(d['training_only_identity']['validation_encoded'] is False and d['training_only_identity']['holdout_encoded'] is False,'Training-only data required')
            def review_report(r,ds):
                global maxerror
                n=len(ds);require(r['draws']==n and n in (32,128) and set(r['parameter_statistics'])==set(combined['parameter_statistics']),'Complete parameter inventory and count required')
                for name,s in r['parameter_statistics'].items():
                    require(all(set(x['gradient_squared_norms'])==set(r['parameter_statistics']) for x in ds),'Perdraw complete named norms required')
                    q=math.fsum(x['gradient_squared_norms'][name] for x in ds);cov=(q-n*s['mean_gradient_squared_norm'])/(n-1);signal=s['mean_gradient_squared_norm']-cov/n
                    error=max(abs(cov-s['trace_sample_covariance']),abs(signal-s['signal_squared_unbiased']));maxerror=max(maxerror,error);require(error<1e-10,'Independent norm/covariance/signal identity differs')
                norm=math.fsum(s['mean_gradient_squared_norm'] for s in r['parameter_statistics'].values());cov=math.fsum(s['trace_sample_covariance'] for s in r['parameter_statistics'].values());signal=norm-cov/n;ratio=cov/signal if signal>0 else None
                require(abs(norm-r['mean_gradient_squared_norm'])<1e-12 and abs(cov-r['trace_sample_covariance'])<1e-12 and abs(signal-r['signal_squared_unbiased'])<1e-12 and ratio==r['ratio_in_independent_B8_draw_units'],'Independent aggregate or unclamped ratio differs')
                return dict(draws=n,trace_sample_covariance=cov,signal_squared_unbiased=signal,ratio_in_independent_B8_draw_units=ratio,noise_correction_fraction=(cov/n/norm if norm>0 else None))
            cohort_rows=[]
            for i,c in enumerate(reports):
                ds=draws[i*32:(i+1)*32];require(c['cohort']==i and c['seed']==K.SEEDS[i] and [(x['cohort'],x['seed'],x['draw'],x['targets']) for x in ds]==[(i,K.SEEDS[i],j,1024) for j in range(1,33)],'Exact original complete cohort/order/seed/targets required')
                cohort_rows.append(dict(cohort=i,seed=c['seed'],**review_report(c['statistics'],ds)))
            combined_row=review_report(combined,draws)
            # Independent within+between-cohort total covariance decomposition.
            for name,s in combined['parameter_statistics'].items():
                parts=[c['statistics']['parameter_statistics'][name] for c in reports]
                m2=math.fsum(31*p['trace_sample_covariance'] for p in parts)+32*(math.fsum(p['mean_gradient_squared_norm'] for p in parts)-4*s['mean_gradient_squared_norm'])
                error=abs(m2/127-s['trace_sample_covariance']);maxerror=max(maxerror,error);require(error<1e-10,'Independent cohort/combined decomposition differs')
            groups={}
            for group in ('input','readout','body'):
                parts=[s for name,s in combined['parameter_statistics'].items() if ('input' if name.startswith('transformer.wte.') else 'readout' if name.startswith('lm_head.') else 'body')==group]
                norm=math.fsum(p['mean_gradient_squared_norm'] for p in parts);cov=math.fsum(p['trace_sample_covariance'] for p in parts);signal=norm-cov/128
                expected=dict(parameters=len(parts),mean_gradient_squared_norm=norm,trace_sample_covariance=cov,signal_squared_unbiased=signal,ratio_in_independent_B8_draw_units=cov/signal if signal>0 else None)
                require(expected==d['combined_groups'][group],'Independent combined group differs');groups[group]=expected
            stream=[(x['input_sha256'],x['target_sha256']) for x in draws];streams.append(stream)
            require(len(set(tuple(stream[i*32:(i+1)*32]) for i in range(4)))==4,'Four distinct private cohort streams required')
            K.C.verify_artifact(d['raw'])
            rows.append(dict(slot=d['slot'],raw=d['raw'],cohorts=cohort_rows,combined=combined_row,combined_groups=groups,first_probe_reported_separately=next(x for x in first_rows if x['slot']==d['slot'])))
        require([x['slot'] for x in rows]==[9,14,16] and streams[0]==streams[1]==streams[2],'All three complete models and controlled common cohort streams required')
        mirror=E/'cohort-metadata-mirror001';mirror.mkdir();files=[]
        for p in [R/'plan.json',R/'guard.json',R/'result.json',R/'STOP',*[R/f'endpoint-{s:02d}.json' for s in (9,14,16)]]:
            original=ref(p);destination=mirror/p.name;shutil.copyfile(p,destination)
            require(ref(p)==original and ref(destination)['sha256']==original['sha256'],'Closed metadata mirror differs');files.append(dict(source=original,destination=ref(destination),size=p.stat().st_size))
        require(ref(R/'STOP')==terminal and len(files)==7 and not any(p.suffix in ('.pt','.tmp') for p in R.rglob('*')),'No terminal mutation or tensor output allowed')
        publish(mirror/'completion.json',dict(files=files,total_bytes=sum(x['size'] for x in files),weight_copies=0))
    publish(E/'cohort-review001.json',dict(status='complete_training_free_gradient_cohorts_independently_reviewed',terminal=terminal,plan=ref(R/'plan.json'),result=stop['result'],guard=ref(R/'guard.json'),source=ref(Path(__file__)),closed_metadata_mirror=ref(mirror/'completion.json'),rows=rows,max_scalar_identity_error=maxerror,models_loaded=0,gradients_replayed=False,optimizer_steps=0,validation_encoded=False,holdout_encoded=False,quality_result=False,adopted=False,scope='All cohorts kept. Descriptive sampling variation only;biased endpoint raw-gradient ratios are not optimal batch,preconditioned update noise or quality evidence.'))
    print(json.dumps(dict(status='reviewed',maxerror=maxerror,rows=[dict(slot=x['slot'],cohort_ratios=[c['ratio_in_independent_B8_draw_units'] for c in x['cohorts']],combined=x['combined']) for x in rows])),flush=True)
