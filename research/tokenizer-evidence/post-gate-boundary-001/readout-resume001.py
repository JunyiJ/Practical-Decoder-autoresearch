"""Distinct generated same-MPS full-state resume and own scorer proof."""
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import require
from research.readout_mixture_model_v1 import GPT, build_optimizer
from research.readout_mixture_resume_v1.data import config, model_spec, View
from research.readout_mixture_resume_v1 import checkpoint as C
from research.readout_mixture_resume_v1 import scoring as S
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.gradient_clip_model_v1.clipping import clip_gradients
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def publish(name, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    require(sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw) <= 8*1024**2, 'Metadata cap')
    with (R/name).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def tree_hash(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous(); h.update(str((t.dtype, tuple(t.shape))).encode()); h.update(t.numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b'dict')
            for k in sorted(v, key=lambda x: (type(x).__name__, str(x))):
                visit(k); visit(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            for x in v:
                visit(x)
        else:
            h.update(json.dumps([type(v).__name__, v], sort_keys=True, allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def clean():
    gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()


def view(cfg):
    tensors = {n: ((torch.arange(length, dtype=torch.int64)*factor+31) % 50257)
               for n, length, factor in [('train', 4096, 17), ('val', 2048, 29)]}
    return View(train_data=tensors['train'], val_data=tensors['val'], splits=tensors,
        generators={n: torch.Generator(device='cpu').manual_seed(s) for n, s in [('train', cfg.training.seed), ('val', cfg.training.eval_seed)]},
        batch_size=8, block_size=128, batch_layout='paired_blocks512', vocab_size=50257, device='mps',
        data_identity=dict(kind='generated_readout_resume_fixture_only', ids_sha256=tree_hash(tensors)), tokenizer_contract=None)


def setup(cfg, label):
    with CpuPrerequisiteGuard(R/('cpu-construction-'+label+'.json'), ROOT, timeout_seconds=120, poll_seconds=.25):
        m = GPT(model_spec(cfg), initializer_seed=cfg.initializer_seed)
        data = view(cfg)
    m = m.to('mps')
    opt = build_optimizer(m, cfg.training); ema = TailWeightAverages(m, cfg.training)
    return m, opt, ema, LearningRateSchedule(opt, cfg.training), data


def updates(m, opt, ema, schedule, data, start, end, history):
    trace = []
    for i in range(start, end):
        m.train(); x, y = data.get_batch('train'); opt.zero_grad()
        batch_sha = tree_hash((x, y)); loss = m.loss(x, y); loss.backward()
        active = [n for n, p in m.named_parameters() if p.grad is not None]
        require(active == [n for n, _ in m.named_parameters()], 'Complete gradient exposure required')
        history.append(dict(step=i+1, **clip_gradients(m.parameters(), 1.), active_parameter_names=active, ce_loss=float(loss.detach())))
        ratio = schedule.apply(i); opt.step(); ema.update(i+1)
        m.eval()
        with torch.no_grad():
            vx, vy = data.get_batch('val'); val = float(m.loss(vx, vy))
        trace.append(dict(step=i+1, train_batch_sha256=batch_sha, val_batch_sha256=tree_hash((vx, vy)), val_ce=val, lr_multiplier=ratio))
    opt.zero_grad()
    return trace


def alter(value, keys, replacement):
    if not keys:
        return replacement
    out = value.copy() if isinstance(value, dict) else list(value)
    out[keys[0]] = alter(value[keys[0]], keys[1:], replacement)
    return out


def live_hash(m, opt, ema, data):
    return tree_hash(dict(model=m.state_dict(), optimizer=opt.state_dict(), ema_metadata=[ema.completed_steps, ema.snapshots],
        averages=ema.averages, samplers={n: g.get_state() for n, g in data.generators.items()}, cpu=torch.get_rng_state(),
        mps=torch.mps.get_rng_state(), cache_epoch=m.weight_epoch))


def adversaries(payload):
    adam = payload['optimizer_state']['optimizers'][1]; aid = next(iter(adam['state']))
    muon = payload['optimizer_state']['optimizers'][0]; mid = next(iter(muon['state']))
    param = next(iter(payload['ema']['averages']['200']))
    cases = [('kind', ['kind'], C.H200), ('initializer', ['initializer', 'backbone'], 0),
        ('architecture', ['architecture', 'arm'], 'foreign'), ('cfg', ['cfg', 'training', 'learning_rate'], .0025),
        ('source_identity', ['training_identity', 'source_fingerprint'], '0'*64), ('step', ['step'], True),
        ('clipping_missing', ['clipping_history'], payload['clipping_history'][:-1]),
        ('exposure', ['optimizer_state', 'gradient_steps', param], 0), ('optimizer_kind', ['optimizer_state', 'readout_mixture_signature', 'arm'], 'foreign'),
        ('scheduled_lr', ['optimizer_state', 'optimizers', 1, 'param_groups', 0, 'lr'], .0025),
        ('adam_step', ['optimizer_state', 'optimizers', 1, 'state', aid, 'step'], torch.tensor(1.)),
        ('missing_moment', ['optimizer_state', 'optimizers', 1, 'state', aid], {'step': torch.tensor(2.), 'exp_avg': adam['state'][aid]['exp_avg']}),
        ('moment_shape', ['optimizer_state', 'optimizers', 0, 'state', mid, 'momentum_buffer'], torch.zeros(1)),
        ('moment_dtype', ['optimizer_state', 'optimizers', 0, 'state', mid, 'momentum_buffer'], torch.zeros_like(muon['state'][mid]['momentum_buffer'], dtype=torch.float64)),
        ('moment_nonfinite', ['optimizer_state', 'optimizers', 0, 'state', mid, 'momentum_buffer'], torch.full_like(muon['state'][mid]['momentum_buffer'], float('nan'))),
        ('second_moment_negative', ['optimizer_state', 'optimizers', 1, 'state', aid, 'exp_avg_sq'], -torch.ones_like(adam['state'][aid]['exp_avg_sq'])),
        ('ema_count', ['ema', 'snapshots'], 0), ('ema_shape', ['ema', 'averages', '200', param], torch.zeros(1)),
        ('model_shape', ['model_state', 'backbone.transformer.ln_f.bias'], torch.zeros(1)),
        ('rope_values', ['model_state', 'backbone.transformer.h.0.freq_cos'], torch.zeros(128, 16)),
        ('cpu_rng', ['cpu_rng'], torch.zeros(1, dtype=torch.uint8)), ('mps_rng', ['mps_rng'], torch.zeros(1, dtype=torch.uint8)),
        ('sampler_inventory', ['sampler_state'], {'train': payload['sampler_state']['train']}),
        ('sampler_encoding', ['sampler_state', 'train'], torch.zeros_like(payload['sampler_state']['train']))]
    return [(name, alter(payload, path, value)) for name, path, value in cases]


def scorer_math():
    # Literal binary-fraction losses: arithmetic/coverage fixture, never real validation results.
    n = 34278; values = (torch.arange(n, dtype=torch.float64) % 17+1)/32
    target = dict(policy=S.POLICY, targets=n, target_bytes=111536, window_counts=[c for _, c in S.windows(n)],
        block_counts=[min(2048, n-i) for i in range(0, n, 2048)], context=128, first_target_offset=1, tail_retained=True, holdout_scored=False)
    report = S.aggregate([values[s:s+c] for s, c in S.windows(n)], target); S.validate_report(report, target)
    require(sum(c for _, c in S.windows(n)) == n and S.windows(n)[-1] == (34176, 102), 'All target/tail indices required')
    cases = [(['ce'], report['ce']+.01), (['bits_per_byte'], report['bits_per_byte']+.01), (['total_loss'], float('nan')),
        (['window_loss_sums'], report['window_loss_sums'][:-1]), (['block_loss_sums'], report['block_loss_sums'][:-1]),
        (['loss_dtype'], 'old_generic_logits_CE'), (['holdout_scored'], True), (['aux_loss_included'], True),
        (['target', 'first_target_offset'], 0), (['target', 'target_bytes'], 111475)]
    for path, value in cases:
        try:
            S.validate_report(alter(report, path, value), target)
        except ValueError:
            continue
        raise AssertionError('Scorer forgery accepted')
    return dict(generated_loss_math_only=True, targets=n, windows=len(S.windows(n)), tail=102, forged_reports_rejected=len(cases),
                literal_loss_sha256=tree_hash(values), report_sha256=tree_hash(report))


def run():
    cpu_initial, mps_initial = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
    torch.random.default_generator.manual_seed(600563); torch.mps.manual_seed(600564)
    branch_cpu, branch_mps = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
    rows = []
    try:
        for arm in ('native', 'context_mix2', 'probability_mix2'):
            cfg = config(arm, 4, 600560, 600561, 600562)
            torch.set_rng_state(branch_cpu); torch.mps.set_rng_state(branch_mps)
            m, opt, ema, schedule, data = setup(cfg, arm+'-full'); history = []
            trace = updates(m, opt, ema, schedule, data, 0, 4, history)
            full = C.raw_payload(m, opt, ema, cfg, 4, history, data); full_sha = tree_hash(full)
            exported = C.h200_payload(m, ema, cfg, dict(kind=C.RAW, fixture_kind='in_memory_no_weight_file', raw_state_sha256=full_sha))
            export_sha = tree_hash(exported)
            del full, exported, m, opt, ema, schedule, data; clean()
            torch.set_rng_state(branch_cpu); torch.mps.set_rng_state(branch_mps)
            m, opt, ema, schedule, data = setup(cfg, arm+'-prefix'); history = []
            partial_trace = updates(m, opt, ema, schedule, data, 0, 2, history)
            prefix = C.raw_payload(m, opt, ema, cfg, 2, history, data)
            del m, opt, ema, schedule, data; clean()
            m, opt, ema, schedule, data = setup(cfg, arm+'-destination')
            # Deliberately advance the destination caller/private streams before restore.
            torch.rand(3); torch.rand(3, device='mps'); data.get_batch('train'); data.get_batch('val')
            before = live_hash(m, opt, ema, data); rejected = []
            for name, bad in adversaries(prefix):
                try:
                    C.resume(m, opt, ema, bad, cfg, data)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Full-state adversary accepted: '+name)
                require(live_hash(m, opt, ema, data) == before, 'Rejected resume mutated destination: '+name)
                rejected.append(name)
            del bad
            step, history = C.resume(m, opt, ema, prefix, cfg, data)
            require(step == 2 and tree_hash(C.raw_payload(m, opt, ema, cfg, step, history, data)) == tree_hash(prefix), 'Complete restored prefix differs')
            del prefix
            partial_trace += updates(m, opt, ema, schedule, data, 2, 4, history)
            resumed = C.raw_payload(m, opt, ema, cfg, 4, history, data)
            require(tree_hash(resumed) == full_sha and partial_trace == trace, 'Uninterrupted/split-resumed full state or batch trace differs')
            require(opt.completed_steps == 4 and all(v == 4 for v in opt.gradient_steps.values()) and ema.snapshots == 3, 'Full horizon/exposure/EMA differs')
            exported = C.h200_payload(m, ema, cfg, dict(kind=C.RAW, fixture_kind='in_memory_no_weight_file', raw_state_sha256=full_sha))
            require(tree_hash(exported) == export_sha, 'Uninterrupted/resumed H200 differs')
            before = live_hash(m, opt, ema, data)
            for forged in [dict(exported, parent=dict(kind='foreign')), dict(exported, kind=C.RAW)]:
                try:
                    C.load_export(m, forged, cfg)
                except ValueError:
                    pass
                else:
                    raise AssertionError('H200 ancestry/kind forgery accepted')
                require(live_hash(m, opt, ema, data) == before, 'Rejected H200 changed destination')
            try:
                C.resume(m, opt, ema, exported, cfg, data)
            except ValueError:
                pass
            else:
                raise AssertionError('H200 resumed optimizer')
            C.load_export(m, exported, cfg)
            require(all(torch.equal(p.cpu(), exported['model_state'][n]) for n, p in m.named_parameters()), 'H200 consumer values differ')
            del opt, ema, resumed, exported, forged; clean()
            m.eval(); ids = (torch.arange(260, dtype=torch.int64)*43+11) % 50257
            values = S.window_losses(m, ids); actual = torch.cat(values)
            gather_error = 0.
            with torch.no_grad():
                for (start, count), loss in zip(S.windows(259), values):
                    x = ids[start:start+count].unsqueeze(0).to('mps'); y = ids[start+1:start+count+1].to('mps')
                    lp = m(x).log_probs()[0]
                    require(float(torch.logsumexp(lp, -1).abs().max()) <= 3e-5, 'Own scorer distribution not normalized')
                    oracle = -lp.gather(-1, y[:, None]).reshape(-1).cpu().to(torch.float64)
                    gather_error = max(gather_error, float((oracle-loss).abs().max()))
            require(gather_error <= 3e-5 and [v.numel() for v in values] == [128, 128, 3], 'Own generated scorer target/gather/tail differs')
            m.cpu(); clean()
            with CpuPrerequisiteGuard(R/('cpu-scorer-'+arm+'.json'), ROOT, timeout_seconds=120, poll_seconds=.25):
                cpu_losses = torch.cat(S.window_losses(m, ids))
                ce_error = float((cpu_losses-actual).abs().max()); require(ce_error <= 3e-5, 'Own scorer CPU/MPS per-target loss differs')
            row = dict(arm=arm, full_raw_sha256=full_sha, exact_resume=True, exact_h200_sha256=export_sha,
                optimizer_steps=8, targets_per_branch=4096, ema_snapshots=3, per_parameter_exposure=4,
                resume_forgeries=rejected, h200_forgeries=3, full_private_batch_trace_exact=True,
                generated_scored_targets=259, target_only_full_gather_max_error=gather_error, scorer_cpu_mps_max_error=ce_error,
                real_corpus_scoring=False, tensor_files=0)
            rows.append(row); publish('arm-'+arm+'.json', row)
            print(json.dumps(dict(completed_arm=arm)), flush=True)
            del m, actual, values, cpu_losses, lp, oracle; clean()
        math_result = scorer_math()
    finally:
        torch.set_rng_state(cpu_initial); torch.mps.set_rng_state(mps_initial)
    require(torch.equal(torch.get_rng_state(), cpu_initial) and torch.equal(torch.mps.get_rng_state(), mps_initial), 'Caller RNG not restored')
    return rows, math_result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan-sha', required=True); arg = ap.parse_args()
    require(sha(R/'plan.json') == arg.plan_sha and set(p.name for p in R.iterdir()) == {'plan.json'}, 'One pristine pinned attempt')
    plan = json.loads((R/'plan.json').read_text()); require(len(plan['sources']) == 1101, 'Exact 1095+five sources+harness')
    for p, h in plan['sources'].items():
        require(sha(ROOT/p) == h, 'Source differs: '+p)
    for ref in plan['census_evidence']:
        require(sha(ref['path']) == ref['sha256'], 'Original role evidence differs')
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); torch.set_num_threads(4)
        require(all(os.environ.get(k) == '4' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')), 'Four thread flags')
        publish('attempt.json', dict(pid=os.getpid(), plan_sha256=arg.plan_sha, roles=plan['roles']))
        try:
            with StrictMpsGuard(R/'mps-guard.json', ROOT):
                rows, math_result = run()
                for p, h in plan['sources'].items():
                    require(sha(ROOT/p) == h, 'Source changed: '+p)
                publish('result.json', dict(status='complete_generated_readout_resume_scorer_proof', rows=rows, scorer_math=math_result,
                    source_global_cpu_mps_rng_preserved=True, actual_generated_optimizer_steps=24,
                    corpus_checkpoint_holdout_quality_seed_calls=0, tensor_files=0, optimizer_cpu_mps_transport_claim=False))
            publish('STOP', dict(status='complete_generated_readout_resume_scorer_proof', no_replay=True))
        except BaseException as exc:
            if not (R/'STOP').exists():
                publish('STOP', dict(status='closed_failed', no_retry=True, exception=repr(exc)))
            raise


if __name__ == '__main__':
    main()
