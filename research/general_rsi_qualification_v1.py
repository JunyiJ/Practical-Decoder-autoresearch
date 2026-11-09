"""Bounded exact restart and independent scoring qualification for new data."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/tokenizer-evidence/general-rsi-qualification-001'
WEIGHTS = ROOT / 'research_runs/general-rsi-qualification-001'


def put(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare():
    from research.general_rsi_training_v1 import sha
    prior = json.loads((ROOT / 'research/tokenizer-evidence/general-rsi-pilot-001/plan.json').read_text())
    sources = dict(prior['sources'])
    for name in ('research/general_rsi_training_v1.py','research/general_rsi_qualification_v1.py'):
        sources[name] = sha(ROOT / name)
    plan = dict(kind='general_rsi_complete_restart_qualification_v1', sources=sources,
        codebook_assets=prior['codebook_assets'], torch_version=prior['torch_version'],
        roles=dict(zip(('native_initializer','composer_initializer','train_sampler','validation_sampler'), range(62110200001,62110200005))),
        prior_role_audit='rg before assignment: exit1, no JSON/JSONL matches',
        arms=['native','raw_bytes'], learning_rate=.00125, horizon=40, restart_step=35,
        restart_requirement='Exact tensor and scalar equality of complete final state, including history/RNG/EMA/optimizer.',
        score_mean_atol=3e-5, timeout_seconds=600, rss_limit_bytes=4*1024**3,
        mps_limit_bytes=6*1024**3, reserve_bytes=10*1024**3,
        checkpoint_cap_bytes=1024**3, metadata_cap_bytes=16*1024**2,
        scientific_selection=False, holdout_opened=False,
        checkpoint_disposition='Numerical prerequisite artifacts only; eligible for exact audited retirement after successful restart/scoring receipt and dependency closure. Never scientific confirmation weights.')
    assert not OUT.exists() and not WEIGHTS.exists()
    OUT.mkdir()
    put(OUT / 'plan.json', plan)
    print(sha(OUT / 'plan.json'), flush=True)


def equal(actual, expected, path='root'):
    import torch
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype
        assert actual.shape == expected.shape and torch.equal(actual, expected), path
    elif isinstance(expected, dict):
        assert type(actual) is dict and list(actual) == list(expected), path
        for k, v in expected.items(): equal(actual[k], v, path+'.'+str(k))
    elif isinstance(expected, (list,tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for i, (a,b) in enumerate(zip(actual,expected)): equal(a,b,path+f'[{i}]')
    else:
        assert type(actual) is type(expected) and actual == expected, path


def oracle(context, primary, tolerance):
    import torch
    from torch.nn import functional as F
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    ids = [0, next(i for i,r in enumerate(context.data.indexes['val']) if r['source']=='wikipedia')]
    model = context.model
    model.to('cpu'); model.eval()
    errors = []
    for doc in ids:
        row = context.data.indexes['val'][doc]
        document = context.data.view.val_data[row['offset']:row['offset']+row['tokens']]
        target_ids = []
        losses = []
        # Independent document slicing and FP64 CE; no production score reducer.
        for start in range(0, len(document)-1, 128):
            piece = document[start:min(start+129,len(document))]
            with torch.no_grad(), FP32RMSNormInference():
                logits = model.primary_logits(piece[:-1][None,:]).values[0]
            loss = F.cross_entropy(logits.double(), piece[1:], reduction='none')
            target_ids.extend(piece[1:].tolist()); losses.extend(loss.tolist())
        assert target_ids == document[1:].tolist()
        expected = sum(losses)/len(losses)
        actual_row = primary['documents'][doc]
        actual = actual_row['loss_sum']/actual_row['count']
        error = abs(expected-actual)
        assert error <= tolerance
        errors.append(dict(document=doc, count=len(losses), absolute_mean_ce_error=error))
    return errors


def run(plan, binding):
    import torch
    from research.general_rsi_training_v1 import Context, write_checkpoint, read_checkpoint, score, sha
    from research.byte_input_model_v6 import build_codebook
    assert torch.__version__ == plan['torch_version'] and torch.backends.mps.is_available()
    torch.set_num_threads(4)
    raw=[]
    for asset in plan['codebook_assets']:
        assert sha(asset['path']) == asset['sha256']
        raw.append(Path(asset['path']).read_bytes())
    book=build_codebook(*raw,'raw_bytes')
    reports=[]
    for arm in plan['arms']:
        context=Context(arm,plan['learning_rate'],plan['horizon'],plan['roles'],book,binding)
        while context.step < plan['restart_step']: context.update()
        assert context.snapshots == 1 and context.ema
        payload=context.capture()
        started=time.perf_counter()
        receipt=write_checkpoint(WEIGHTS/f'{arm}-step35.pt',payload)
        save_seconds=time.perf_counter()-started
        del payload
        while context.step < plan['horizon']: context.update()
        expected=context.capture()
        assert expected['snapshots'] == 2 and expected['history'][-1]['lr_ratio']==.1
        del context
        gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()
        restored=Context(arm,plan['learning_rate'],plan['horizon'],plan['roles'],book,binding)
        started=time.perf_counter(); saved=read_checkpoint(receipt)
        restored.restore(saved); restore_seconds=time.perf_counter()-started
        equal(restored.capture(),saved)
        del saved
        while restored.step < plan['horizon']: restored.update()
        actual=restored.capture()
        equal(actual,expected)
        del actual, expected
        assert torch.mps.driver_allocated_memory() < plan['mps_limit_bytes']
        started=time.perf_counter()
        primary=score(restored.model,restored.data)
        score_seconds=time.perf_counter()-started
        assert primary['count']==100416 and len(primary['documents'])==108
        put(OUT/f'{arm}-full-validation-numerical.json',primary)
        errors=oracle(restored,primary,plan['score_mean_atol'])
        report=dict(arm=arm, checkpoint=receipt, save_seconds=save_seconds,
            restore_seconds=restore_seconds, exact_restore_at35=True, exact_complete_state_at40=True,
            ema_snapshots=2, full_validation_seconds=score_seconds, full_validation_targets=100416,
            fp64_cpu_oracle=errors, scientific_selection=False)
        put(OUT/f'{arm}-result.json',report); reports.append(report)
        print(json.dumps(report),flush=True)
        del restored
        gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()
    return dict(arms=reports, complete_restart_qualified=True, document_scoring_qualified=True,
        scientific_selection=False, test_opened=False,
        averaging_alignment='For H2048, first snapshot1370 and last2040; no forced endpoint snapshot. H10240 snapshots6830..10240. This implements the frozen interval10 plan.')


def main():
    from research.general_rsi_training_v1 import sha
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','run']);parser.add_argument('plan_sha',nargs='?')
    args=parser.parse_args()
    if args.action=='prepare': return prepare()
    assert sha(OUT/'plan.json')==args.plan_sha
    plan=json.loads((OUT/'plan.json').read_text())
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free>=plan['reserve_bytes']+plan['checkpoint_cap_bytes']+plan['metadata_cap_bytes']
        for path,digest in plan['sources'].items(): assert sha(ROOT/path)==digest,path
        WEIGHTS.mkdir()
        put(OUT/'attempt.json',dict(pid=os.getpid(),plan_sha256=args.plan_sha,started=time.time()))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=plan['timeout_seconds'],
                    rss_limit_bytes=plan['rss_limit_bytes'],reserve_bytes=plan['reserve_bytes']):
                result=run(plan,args.plan_sha)
            for path,digest in plan['sources'].items(): assert sha(ROOT/path)==digest,path
            assert sum(p.stat().st_size for p in WEIGHTS.iterdir())<=plan['checkpoint_cap_bytes']
            assert sum(p.stat().st_size for p in OUT.iterdir())<=plan['metadata_cap_bytes']
            put(OUT/'result.json',result);print('QUALIFIED',flush=True)
        except BaseException:
            put(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
