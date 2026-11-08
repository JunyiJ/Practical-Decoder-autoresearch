"""Bounded fresh experiment. Exclusive attempts; immutable plan and saved outcomes."""
import argparse
import fcntl
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/tokenizer-evidence/output-anchored-tying-001'
WEIGHTS = ROOT/'research_runs/output-anchored-tying-001'
CAP = int(2.28125*1024**3)
TORCH = None
ARMS = ('native', 'matched', 'tied')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(1024**2):
            h.update(data)
    return h.hexdigest()


def put(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def read(path):
    return json.loads(path.read_text())


def roles(bundle):
    return dict(zip(('native_initializer', 'composer_initializer', 'nonlexical_key',
                     'train_sampler', 'validation_sampler'), range(bundle+1, bundle+6)))


def runtime():
    import torch, tiktoken
    return dict(python=sys.version, torch=torch.__version__, torch_git=torch.version.git_version,
                tiktoken=tiktoken.__version__, macos=platform.mac_ver()[0], machine=platform.machine())


def pins(plan):
    for path, digest in plan['sources'].items():
        assert sha(ROOT/path) == digest, path
    assert runtime() == plan['runtime']


def prepare():
    assert not BASE.exists() and not WEIGHTS.exists()
    assert shutil.disk_usage(ROOT).free >= CAP+10*1024**3
    # Bind all project Python source, including inherited namespace overlays.
    paths = sorted(p for folder in ('research', 'src') for p in (ROOT/folder).rglob('*.py'))
    paths += [ROOT/'research/OUTPUT_ANCHORED_TYING_PROTOCOL.md',
              ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001/quality002-slot03-evidence001/config.json',
              ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001/quality002-plan.json']
    prior = subprocess.run(['rg', '-l', '6210170000[1-5]|6210180000[1-5]|6210190000[1-5]|6210200000[1-6]',
                            'research', 'research_state', '-g', '*.json', '-g', '*.jsonl'], cwd=ROOT,
                           text=True, capture_output=True)
    assert prior.returncode == 1 and not prior.stdout, ('Roles already used', prior.stdout, prior.stderr)
    specs = []
    for i, bundle in enumerate((62101800000, 62101900000, 62102000000)):
        for arm in ARMS[i:]+ARMS[:i]:
            specs.append(dict(slot=len(specs), arm=arm, roles=roles(bundle), steps=1200, learning_rate=.00125))
    plan = dict(kind='output_anchored_tying_v1', sources={str(p.relative_to(ROOT)):sha(p) for p in paths},
                runtime=runtime(), specs=specs, pilot_roles=roles(62101700000),
                bootstrap_seed=62102000006, output_cap=CAP, reserve=8*1024**3, extra_headroom=2*1024**3,
                initial_free=shutil.disk_usage(ROOT).free, holdout_scored=False,
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],text=True).strip(),
                role_census=dict(previous_json_matches=[], pattern=prior.args[2]),
                primary_min_gain=.02, bootstrap_quantile=1-.05/6)
    BASE.mkdir(); WEIGHTS.mkdir()
    put(BASE/'plan.json', plan)
    print(json.dumps(dict(plan_sha256=sha(BASE/'plan.json'), slots=len(specs), initial_free=plan['initial_free'])), flush=True)


def clean():
    gc.collect()
    if TORCH.backends.mps.is_available():
        TORCH.mps.synchronize(); TORCH.mps.empty_cache()


def sync(device):
    if device == 'mps': TORCH.mps.synchronize()


def training_objects(model, view, steps):
    from omegaconf import OmegaConf
    from research.output_anchored_tying_v1 import Optimizer
    from research.byte_input_training_v6 import recipe
    from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
    from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
    cfg = OmegaConf.create(recipe(steps, .00125))
    optimizer = Optimizer(model)
    return optimizer, LearningRateSchedule(optimizer, cfg), TailWeightAverages(model, cfg)


def update(model, view, optimizer, schedule, averages):
    from research.gradient_clip_model_v1.clipping import clip_gradients
    schedule.apply(optimizer.steps); model.train(); optimizer.zero_grad()
    x, y = view.get_batch('train')
    _, loss = model(x, y)
    value = float(loss.detach()); assert math.isfinite(value)
    loss.backward()
    observed = clip_gradients(model.parameters(), 1.)
    assert observed['gradient_tensors'] == len(tuple(model.parameters()))
    optimizer.step(); averages.update(optimizer.steps); optimizer.zero_grad()
    return dict(step=optimizer.steps, loss=value, **observed)


def pilot(plan, out):
    import torch
    from research.output_anchored_tying_v1 import construct, GPT, native_config
    from research.byte_input_training_v6 import load_view
    r = plan['pilot_roles']; reports = []
    for device in ('cpu', 'mps'):
        view = load_view(r, device); x, y = view.get_batch('train')
        # Full training shape; native reference and identical-function controls.
        gradients = {}; logits = {}
        for arm in ('reference', *ARMS):
            if arm == 'reference':
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(r['native_initializer']); model = GPT(native_config()).to(device)
            else: model = construct(arm, r['native_initializer'], device)
            output, loss = model(x, y); loss.backward()
            logits[arm] = output.detach().cpu()
            gradients[arm] = {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
            del output, loss, model; clean()
        assert torch.equal(logits['reference'], logits['native'])
        assert torch.equal(logits['matched'], logits['tied'])
        for name, value in gradients['native'].items():
            assert torch.equal(value, gradients['reference'][name]), name
        maximum = 0.
        for name, value in gradients['tied'].items():
            expected = gradients['matched'][name]
            if name == 'transformer.wte.weight':
                expected = expected + gradients['matched']['lm_head.weight']
            torch.testing.assert_close(value, expected, atol=2e-6, rtol=2e-4)
            maximum = max(maximum, float((value-expected).abs().max()))
        del gradients, logits, x, y, view; clean()
        for arm in ARMS:
            model = construct(arm, r['native_initializer'], device)
            view = load_view(r, device)
            optimizer, schedule, averages = training_objects(model, view, 4)
            start = time.perf_counter()
            rows = [update(model, view, optimizer, schedule, averages) for _ in range(4)]
            sync(device); seconds = time.perf_counter()-start
            model.validate(); assert averages.snapshots == 3
            state = {n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
            buffer = io.BytesIO(); torch.save(dict(arm=arm, weights=state), buffer); buffer.seek(0)
            saved = torch.load(buffer, map_location='cpu', weights_only=True)
            restored = construct(arm, r['native_initializer'], 'cpu')
            restored.load_state_dict(saved['weights'], strict=True); restored.validate()
            assert all(torch.equal(p, saved['weights'][n]) for n,p in restored.state_dict().items())
            assert len(optimizer.children[1].state) == (43 if arm == 'tied' else 44)
            assert len(averages.averages['200']) == (87 if arm == 'tied' else 88)
            reports.append(dict(device=device, arm=arm, updates=4, seconds=seconds,
                                parameters=sum(p.numel() for p in model.parameters()),
                                native_exact=True, matched_tied_initial_exact=True,
                                shared_gradient_max_error=maximum, roundtrip_exact=True, rows=rows))
            del model, restored, view, optimizer, schedule, averages, state, saved, buffer; clean()
    put(out/'result.json', dict(passed=True, quality_scored=False, updates=24, cases=reports))


def score(model, view):
    import torch
    from torch.nn import functional as F
    mode = model.training; model.eval(); losses = []
    generator = view.generators['train'].get_state().clone()
    with torch.no_grad():
        for start in range(0, 34278, 128):
            count = min(128, 34278-start)
            x = view.val_data[start:start+count].unsqueeze(0).to(view.device)
            y = view.val_data[start+1:start+count+1].to(view.device)
            logits, _ = model(x)
            values = F.cross_entropy(logits[0], y, reduction='none').cpu().to(torch.float64)
            assert bool(torch.isfinite(values).all()) and bool((values >= 0).all())
            losses.extend(values.tolist())
    model.train(mode)
    assert torch.equal(generator, view.generators['train'].get_state()) and len(losses) == 34278
    return dict(ce=math.fsum(losses)/34278, target_count=34278, per_target_losses=losses, holdout_scored=False)


def save_weights(path, model, spec, plan_digest, kind):
    import torch
    weights = {n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
    payload = dict(kind=kind, spec=spec, plan_sha256=plan_digest, weights=weights)
    with path.open('xb') as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['kind'] == kind and saved['spec'] == spec and saved['plan_sha256'] == plan_digest
    assert saved['weights'].keys() == weights.keys()
    assert all(torch.equal(saved['weights'][n], p) for n,p in weights.items())
    return dict(path=str(path), sha256=sha(path), size=path.stat().st_size, exact_readback=True)


def trial(plan, index, out):
    import torch
    from research.output_anchored_tying_v1 import construct
    from research.byte_input_training_v6 import load_view, check_view
    spec = plan['specs'][index]; r = spec['roles']; arm = spec['arm']
    model = construct(arm, r['native_initializer'], 'mps'); view = load_view(r, 'mps')
    optimizer, schedule, averages = training_objects(model, view, 1200)
    put(out/'config.json', dict(spec=spec, data=view.data_identity, parameters=sum(p.numel() for p in model.parameters())))
    curves = []; started = time.perf_counter()
    with (out/'training.jsonl').open('x') as stream:
        for step in range(1,1201):
            tick = time.perf_counter(); row = update(model, view, optimizer, schedule, averages)
            sync('mps'); row['seconds'] = time.perf_counter()-tick
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n'); stream.flush()
            if step in (300,600,900,1200):
                online = score(model, view)
                curves.append(dict(step=step, ce=online['ce']))
                put(out/f'curve-{step}.json', curves[-1]); del online
                print(f'{index:02d} {arm} step{step} online_validation={curves[-1]["ce"]:.6f}', flush=True)
        os.fsync(stream.fileno())
    elapsed = time.perf_counter()-started
    assert optimizer.steps == averages.completed_steps == 1200 and averages.snapshots == 41
    model.validate(); check_view(view, r)
    folder = WEIGHTS/f'{index:02d}'; folder.mkdir()
    raw = save_weights(folder/'raw.pt', model, spec, sha(BASE/'plan.json'), 'final_raw_weights_not_resume')
    with torch.no_grad():
        for name, p in model.named_parameters(): p.copy_(averages.averages['200'][name])
    model.validate()
    averaged = save_weights(folder/'h200.pt', model, spec, sha(BASE/'plan.json'), 'h200_inference_weights')
    # Score the reloaded saved artifact, rather than an unbound in-memory model.
    del optimizer, averages, schedule; clean()
    payload = torch.load(averaged['path'], map_location='cpu', weights_only=True)
    model.load_state_dict(payload['weights'], strict=True); del payload
    model.validate(); result = score(model, view)
    put(out/'validation.json', result)
    check_view(view, r)
    put(out/'result.json', dict(completed=True, spec=spec, steps=1200, targets_seen=1228800,
         averaging_snapshots=41, ce=result['ce'], raw=raw, h200=averaged,
         parameter_count=sum(p.numel() for p in model.parameters()), elapsed_seconds=elapsed,
         train_sampler_sha256=hashlib.sha256(view.generators['train'].get_state().numpy().tobytes()).hexdigest(),
         curves=curves, holdout_scored=False, adopted=False))


def worker(name):
    global TORCH
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    class Guard(CpuPrerequisiteGuard):
        def _sample(self):
            sample, reason = super()._sample()
            if TORCH is not None:
                sample['mps_driver_bytes'] = TORCH.mps.driver_allocated_memory()
                if sample['mps_driver_bytes'] >= 6*1024**3: reason = 'mps_limit'
            # No temporary renames/removals in this namespace; completed files persist.
            sample['retained_bytes'] = sum(p.stat().st_size for root in (BASE,WEIGHTS) for p in root.rglob('*') if p.is_file())
            if sample['retained_bytes'] > CAP: reason = 'output_limit'
            if (BASE/'STOP').exists(): reason = 'explicit_stop'
            return sample, reason
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = read(BASE/'plan.json'); pins(plan)
        used = sum(p.stat().st_size for root in (BASE,WEIGHTS) for p in root.rglob('*') if p.is_file())
        assert shutil.disk_usage(ROOT).free >= CAP-used+10*1024**3
        out = BASE/name; out.mkdir()
        put(out/'attempt.json', dict(pid=os.getpid(), plan_sha256=sha(BASE/'plan.json'), name=name))
        try:
            with Guard(out/'guard.json', ROOT, timeout_seconds=7200, poll_seconds=.5):
                import torch
                TORCH = torch; torch.set_num_threads(4)
                assert torch.backends.mps.is_available() and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
                assert sys.flags.optimize == 0
                if name == 'pilot': pilot(plan, out)
                else:
                    assert read(BASE/'pilot/result.json')['passed'] and read(BASE/'pilot/guard.json')['status'] == 'finished'
                    trial(plan, int(name), out)
                pins(plan)
            put(out/'complete.json', dict(accepted=True, result_sha256=sha(out/'result.json'), guard_sha256=sha(out/'guard.json')))
        except BaseException:
            put(out/'failure.json', dict(traceback=traceback.format_exc(), no_scientific_retry=True))
            raise


def finish(plan):
    import numpy as np
    rows = []; scores = []
    for index in range(9):
        out = BASE/f'{index:02d}'; acceptance = read(out/'complete.json')
        assert acceptance['accepted'] and acceptance['result_sha256'] == sha(out/'result.json')
        assert acceptance['guard_sha256'] == sha(out/'guard.json') and read(out/'guard.json')['status'] == 'finished'
        result = read(out/'result.json'); assert result['spec'] == plan['specs'][index]
        for key in ('raw','h200'):
            assert sha(result[key]['path']) == result[key]['sha256']
        score_row = read(out/'validation.json'); assert score_row['ce'] == math.fsum(score_row['per_target_losses'])/34278 == result['ce']
        rows.append(result); scores.append(score_row['per_target_losses'])
    pairs = []
    for start in (0,3,6):
        assert len({r['train_sampler_sha256'] for r in rows[start:start+3]}) == 1
        pairs.append({rows[i]['spec']['arm']:np.array(scores[i],dtype=np.float64) for i in range(start,start+3)})
    comparisons = {}
    for candidate,control in (('tied','native'),('tied','matched'),('matched','native')):
        diffs = np.stack([p[candidate]-p[control] for p in pairs]); deltas=diffs.mean(axis=1); bounds=[]
        for width in (2048,4096,8192):
            starts=range(0,34278,width); counts=np.array([min(width,34278-i) for i in starts])
            blocks=np.stack([diffs[:,i:i+width].sum(axis=1) for i in starts],axis=1)
            draws=np.random.Generator(np.random.PCG64(plan['bootstrap_seed'])).integers(0,len(counts),(2000,len(counts)))
            estimates=(blocks[:,draws].sum(axis=2)/counts[draws].sum(axis=1)).mean(axis=0)
            bounds.append(dict(block_tokens=width,upper=float(np.quantile(estimates,plan['bootstrap_quantile']))))
        gates=dict(mean_gain=float(deltas.mean())<-.02,every_seed=bool((deltas<0).all()),all_bounds=all(b['upper']<0 for b in bounds))
        comparisons[candidate+'_vs_'+control]=dict(deltas=deltas.tolist(),mean=float(deltas.mean()),bounds=bounds,gates=gates,passes=all(gates.values()))
    put(BASE/'final.json',dict(completed=True,trajectories=9,updates=10800,
        means={a:sum(r['ce'] for r in rows if r['spec']['arm']==a)/3 for a in ARMS},comparisons=comparisons,
        quality_success=all(comparisons['tied_vs_'+a]['passes'] for a in ('native','matched')),
        adopted=False,byte_incumbent_unchanged=True,holdout_scored=False,
        limitation='Fixed LR; conditional text intervals on three seeds and reused validation; no direct byte comparison.'))


def run():
    plan = read(BASE/'plan.json'); pins(plan)
    put(BASE/'controller-attempt.json',dict(pid=os.getpid(),plan_sha256=sha(BASE/'plan.json')))
    for name in ('pilot', *(f'{i:02d}' for i in range(9))):
        assert not (BASE/name).exists(), 'No automatic replay'
        result = subprocess.run([sys.executable,'-m','research.output_anchored_tying_run_v1','worker',name],cwd=ROOT)
        if result.returncode:
            put(BASE/'controller-failure.json',dict(name=name,returncode=result.returncode)); return result.returncode
    pins(plan); finish(plan)
    print(json.dumps(read(BASE/'final.json')),flush=True)
    return 0


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run','worker'));parser.add_argument('name',nargs='?')
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='worker': worker(args.name)
    else: sys.exit(run())
