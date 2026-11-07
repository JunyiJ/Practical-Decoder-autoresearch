"""NEW serialized same-MPS complete state and own native scorer proof."""
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import math
import shutil
import os
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import require
from research.relu_squared_model_v1.model import GPT
from research.relu_squared_model_v1.optimizer import build_optimizer
from research.relu_squared_resume_v1.data import config, model_spec, View
from research.relu_squared_resume_v1 import checkpoint as C
from research.relu_squared_resume_v1 import scoring as S
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.gradient_clip_model_v1.clipping import clip_gradients
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')
A = ROOT/'research_runs/tokenizer-relu-squared-resume-v1-001'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def publish(name, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    io_bytes = sum(p.lstat().st_size for p in (A/'io').iterdir()) if (A/'io').exists() else 0
    root_stop = (A/'STOP').stat().st_size if (A/'STOP').exists() else 0
    require(sum(p.stat().st_size for p in R.rglob('*') if p.is_file())+io_bytes+root_stop+len(raw) <= 8*1024**2, 'Complete proof metadata and non-tensor fixture cap')
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
    tensors = {n: ((torch.arange(length, dtype=torch.int64)*factor+83) % 50257)
               for n, length, factor in [('train', 6144, 59), ('val', 3072, 73)]}
    return View(train_data=tensors['train'], val_data=tensors['val'], splits=tensors,
        generators={n: torch.Generator(device='cpu').manual_seed(s) for n, s in [('train', cfg.training.seed), ('val', cfg.training.eval_seed)]},
        batch_size=8, block_size=128, batch_layout='paired_blocks512', vocab_size=50257, device='mps',
        data_identity=dict(kind='generated_ffn_serialized_resume_fixture_only', ids_sha256=tree_hash(tensors)), tokenizer_contract=dict(kind='literal_native_ID_fixture_not_physical_codec'))


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
        # NEW explicit audited global-stream draws exercise counter restoration.
        global_cpu_draw = tree_hash(torch.rand(3))
        global_mps_draw = tree_hash(torch.rand(7, device='mps'))
        m.train(); x, y = data.get_batch('train'); opt.zero_grad()
        batch_sha = tree_hash((x, y)); loss = m.loss(x, y); loss.backward()
        active = [n for n, p in m.named_parameters() if p.grad is not None]
        require(active == [n for n, _ in m.named_parameters()], 'Complete gradient exposure required')
        history.append(dict(step=i+1, **clip_gradients(m.parameters(), 1.), active_parameter_names=active, ce_loss=float(loss.detach())))
        ratio = schedule.apply(i); opt.step(); ema.update(i+1)
        m.eval()
        with torch.no_grad():
            vx, vy = data.get_batch('val'); val = float(m.loss(vx, vy))
        trace.append(dict(step=i+1, train_batch_sha256=batch_sha, val_batch_sha256=tree_hash((vx, vy)),
                          global_cpu_draw_sha256=global_cpu_draw, global_mps_draw_sha256=global_mps_draw,
                          val_ce=val, lr_multiplier=ratio))
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
    cases = [('kind', ['kind'], C.H200), ('initializer', ['initializer', 'native_initializer'], 0),
        ('architecture', ['architecture', 'arm'], 'foreign'), ('cfg', ['cfg', 'training', 'learning_rate'], .0025),
        ('source_identity', ['training_identity', 'source_fingerprint'], '0'*64), ('step', ['step'], True),
        ('clipping_missing', ['clipping_history'], payload['clipping_history'][:-1]),
        ('exposure', ['optimizer_state', 'gradient_steps', param], 0), ('optimizer_kind', ['optimizer_state', 'relu_squared_signature', 'arm'], 'foreign'),
        ('scheduled_lr', ['optimizer_state', 'optimizers', 1, 'param_groups', 0, 'lr'], .0025),
        ('adam_step', ['optimizer_state', 'optimizers', 1, 'state', aid, 'step'], torch.tensor(1.)),
        ('missing_moment', ['optimizer_state', 'optimizers', 1, 'state', aid], {'step': torch.tensor(2.), 'exp_avg': adam['state'][aid]['exp_avg']}),
        ('moment_shape', ['optimizer_state', 'optimizers', 0, 'state', mid, 'momentum_buffer'], torch.zeros(1)),
        ('moment_dtype', ['optimizer_state', 'optimizers', 0, 'state', mid, 'momentum_buffer'], torch.zeros_like(muon['state'][mid]['momentum_buffer'], dtype=torch.float64)),
        ('moment_nonfinite', ['optimizer_state', 'optimizers', 0, 'state', mid, 'momentum_buffer'], torch.full_like(muon['state'][mid]['momentum_buffer'], float('nan'))),
        ('second_moment_negative', ['optimizer_state', 'optimizers', 1, 'state', aid, 'exp_avg_sq'], -torch.ones_like(adam['state'][aid]['exp_avg_sq'])),
        ('ema_count', ['ema', 'snapshots'], 0), ('ema_shape', ['ema', 'averages', '200', param], torch.zeros(1)),
        ('model_shape', ['model_state', 'transformer.ln_f.bias'], torch.zeros(1)),
        ('rope_values', ['model_state', 'transformer.h.0.freq_cos'], torch.zeros(128, 16)),
        ('cpu_rng', ['cpu_rng'], torch.zeros(1, dtype=torch.uint8)), ('mps_rng', ['mps_rng'], torch.zeros(1, dtype=torch.uint8)),
        ('sampler_inventory', ['sampler_state'], {'train': payload['sampler_state']['train']}),
        ('sampler_encoding', ['sampler_state', 'train'], torch.zeros_like(payload['sampler_state']['train']))]
    cases += [('private_initializer', ['initializer', 'private_ungated'], 0),
              ('clipping_coefficient', ['clipping_history', 0, 'coefficient'], 1.25),
              ('ema_names', ['ema', 'parameter_names'], payload['ema']['parameter_names'][:-1]),
              ('ordered_model_inventory', ['model_state'], dict(reversed(list(payload['model_state'].items())))),
              ('adam_epsilon', ['optimizer_state', 'optimizers', 1, 'param_groups', 0, 'eps'], 1e-8),
              ('sampler_seed', ['sampler_state', 'train'], torch.Generator(device='cpu').manual_seed(payload['cfg']['training']['eval_seed']).get_state())]
    live_mps = torch.mps.get_rng_state().clone()
    try:
        # Already audited role deliberately used as a WRONG seed in an invalid state;
        # no draw/model/trajectory is launched from this counterexample.
        torch.mps.manual_seed(payload['cfg']['rng_identity']['global_cpu'])
        wrong_mps_role = torch.mps.get_rng_state().clone()
    finally:
        torch.mps.set_rng_state(live_mps)
    cases += [('global_cpu_role', ['cpu_rng'], payload['sampler_state']['train']),
              ('global_mps_role', ['mps_rng'], wrong_mps_role),
              ('cfg_global_role_bool', ['cfg', 'rng_identity', 'global_cpu'], True),
              ('clipping_count_bool', ['clipping_history', 0, 'gradient_tensors'], True),
              ('ema_half_life', ['ema', 'spec', 'half_lives'], [100]),
              ('moment_device', ['optimizer_state', 'optimizers', 0, 'state', mid, 'momentum_buffer'], muon['state'][mid]['momentum_buffer'].to('mps'))]
    result = [(name, alter(payload, path, value)) for name, path, value in cases]+[('incomplete_raw', dict(kind=C.RAW, literal='new-ffn-nontensor-state-fixture'))]
    require(len(result) == 37, 'Complete NEW state-rejection inventory')
    return result


def file_manifest(p, *, non_tensor=False):
    st = p.lstat()
    return dict(path=str(p), bytes=st.st_size, inode=st.st_ino, device=st.st_dev, links=st.st_nlink,
                symlink=p.is_symlink(), symlink_target=os.readlink(p) if p.is_symlink() else None,
                sha256=None if p.is_symlink() else sha(p), non_tensor_fixture=non_tensor)


def write_archive(label, path, payload):
    with CpuPrerequisiteGuard(R/('cpu-write-'+label+'.json'), ROOT, timeout_seconds=120, poll_seconds=.25):
        class Counter:
            def __init__(self):
                self.count = 0
            def write(self, data):
                self.count += len(data)
                require(self.count <= C.FILE_CAPS[payload['kind']], 'Actual serialization counter cap exceeded')
                return len(data)
            def flush(self):
                pass
            def tell(self):
                return self.count
        counter = Counter(); torch.save(payload, counter)
        reference = C.publish_payload(path, payload)
        require(reference['bytes'] == counter.count, 'Counted versus published serialization bytes differ')
        return reference


def read_archive(label, reference):
    with CpuPrerequisiteGuard(R/('cpu-read-'+label+'.json'), ROOT, timeout_seconds=120, poll_seconds=.25):
        return C.read_payload(reference)


def io_fixtures():
    folder = A/'io'; folder.mkdir(); rejected = []
    payload = dict(kind=C.RAW, literal='ffn-serialization-nontensor-io-001')
    good = C.publish_payload(folder/'literal.pt', payload, cap_bytes=4096)
    require(C.read_payload(good) == payload, 'Literal own archive roundtrip failed')
    def reject(name, call):
        try:
            call()
        except (ValueError, RuntimeError):
            rejected.append(name)
        else:
            raise AssertionError('Archive adversary accepted: '+name)
    reject('overwrite', lambda: C.publish_payload(folder/'literal.pt', payload, cap_bytes=4096))
    require(sha(folder/'literal.pt') == good['sha256'], 'Overwrite changed retained fixture')
    reject('hash', lambda: C.read_payload(dict(good, sha256='0'*64)))
    reject('size', lambda: C.read_payload(dict(good, bytes=good['bytes']+1)))
    reject('kind', lambda: C.read_payload(dict(good, kind=C.H200)))
    reject('cap_increase', lambda: C.publish_payload(folder/'invalid-cap.pt', payload, cap_bytes=C.FILE_CAPS[C.RAW]+1))
    reject('cap_bool', lambda: C.publish_payload(folder/'invalid-bool.pt', payload, cap_bytes=True))
    reject('cap_overflow', lambda: C.publish_payload(folder/'partial.pt', dict(payload, literal='v'*6144), cap_bytes=128))
    require(not (folder/'partial.pt').exists() and (folder/'partial.pt.tmp').is_file()
            and (folder/'partial.pt.tmp').stat().st_size <= 128, 'Oversized publication did not retain bounded partial evidence')
    (folder/'reserved.pt.tmp').write_bytes(b'ffn-new-reserved-io-only-v1')
    reject('preexisting_temporary', lambda: C.publish_payload(folder/'reserved.pt', payload, cap_bytes=4096))
    require((folder/'reserved.pt.tmp').read_bytes() == b'ffn-new-reserved-io-only-v1', 'Rejected publication altered existing temporary')
    outside = ROOT/'research/unlaunched-ffn-io.pt'; require(not outside.exists(), 'New outside-path fixture only')
    reject('outside_run', lambda: C.publish_payload(outside, payload, cap_bytes=4096))
    require(not outside.exists(), 'Rejected outside path was written')
    reject('wrong_suffix', lambda: C.publish_payload(folder/'suffix.bin', payload, cap_bytes=4096))
    reject('cap_float', lambda: C.publish_payload(folder/'float-cap.pt', payload, cap_bytes=4096.0))
    (folder/'literal-link.pt').symlink_to(folder/'literal.pt')
    reject('symlink', lambda: C.read_payload(dict(good, path=str(folder/'literal-link.pt'))))
    os.link(folder/'literal.pt', folder/'literal-alias.pt')
    reject('hardlink_read', lambda: C.read_payload(good))
    reject('noncanonical', lambda: C.publish_payload(folder/'..'/'io'/'noncanonical.pt', payload, cap_bytes=4096))
    require(sha(folder/'literal.pt') == good['sha256'] and not any((folder/n).exists() for n in ['invalid-cap.pt', 'invalid-bool.pt', 'partial.pt', 'reserved.pt']), 'Rejected I/O changed protected fixture or published invalid file')
    return dict(non_tensor_fixtures_only=True, rejected_cases=rejected,
                preserved_outputs=[file_manifest(p, non_tensor=True) for p in sorted(folder.iterdir())])


def scorer_math():
    # NEW exact binary-fraction arithmetic, not corpus losses or an old fixture.
    n = 34278; literal = [(((i*7) % 37)+5)/128 for i in range(n)]
    values = torch.tensor(literal, dtype=torch.float64)
    target = dict(policy=S.POLICY, targets=n, target_bytes=111536, window_counts=[c for _, c in S.windows(n)],
        block_counts=[min(2048, n-i) for i in range(0, n, 2048)], context=128, first_target_offset=1, tail_retained=True, holdout_scored=False)
    report = S.aggregate([values[s:s+c] for s, c in S.windows(n)], target); S.validate_report(report, target)
    total = math.fsum(literal)
    require(report['total_loss'] == total and report['ce'] == total/n
            and report['bits_per_byte'] == total/math.log(2)/111536, 'Independent scalar score denominators differ')
    require(report['window_loss_sums'] == [math.fsum(literal[s:s+c]) for s, c in S.windows(n)]
            and report['block_loss_sums'] == [math.fsum(literal[i:i+2048]) for i in range(0, n, 2048)]
            and S.windows(n)[-1] == (34176, 102), 'Exact window/block/tail arithmetic differs')
    cases = [(['ce'], report['ce']+.03125), (['bits_per_byte'], report['bits_per_byte']+.03125), (['total_loss'], float('nan')),
        (['window_loss_sums'], report['window_loss_sums'][:-1]), (['block_loss_sums'], report['block_loss_sums'][:-1]),
        (['loss_dtype'], 'generic_logits_ce'), (['holdout_scored'], True), (['aux_loss_included'], True),
        (['target', 'first_target_offset'], 0), (['target', 'target_bytes'], 111475),
        (['target', 'policy'], 'readout_projection_native_distribution_validation_v1'), (['target', 'targets'], 34257)]
    for path, value in cases:
        try:
            S.validate_report(alter(report, path, value), target)
        except ValueError:
            continue
        raise AssertionError('Score report forgery accepted')
    return dict(literal_loss_math_only=True, targets=n, windows=268, tail=102,
        independent_math_fsum_exact=True, rejected_reports=len(cases), literal_sha256=tree_hash(values), report_sha256=tree_hash(report))


def pair_fixtures():
    folder = R/'pair-fixtures'; folder.mkdir()
    target = dict(policy=S.POLICY, targets=34278, target_bytes=111536, context=128,
                  first_target_offset=1, tail_retained=True, holdout_scored=False,
                  literal_fixture='new_ffn_pair_metadata_not_physical_codec')
    provenance = dict(scope='new_ffn_state_pair_consumer_fixture', no_model_or_corpus=True)
    def report(ce=.25):
        return dict(ce=ce, target=copy.deepcopy(target), literal_values=[.0625, .1875, .4375])
    rows = []
    def case(name, cpu, mps, accepted, saved, *, prov=None, suffix='.json'):
        path = folder/(name+suffix)
        p = provenance if prov is None else prov
        before = json.dumps([cpu, mps, p], sort_keys=True)
        try:
            value = S.publish_pair_then_require(path, cpu, mps, p)
        except (ValueError, FileExistsError):
            require(not accepted, 'Valid own pair rejected')
        else:
            require(accepted and value['accepted'], 'Invalid own pair accepted')
        require(path.exists() is saved, 'Own pair durability predicate differs')
        if saved:
            value = json.loads(path.read_text())
            require(value['cpu'] == cpu and value['mps'] == mps and value['accepted'] is accepted
                    and value['absolute_ce_tolerance'] == 3e-5 and value['physical_scorer_or_checkpoint_acceptance'] is False,
                    'Both own pair reports/decision must be durable')
        require(json.dumps([cpu, mps, p], sort_keys=True) == before, 'Pair caller inputs changed')
        rows.append(dict(case=name, accepted=accepted, saved=saved, all_predicates_passed=True))
    case('identical', report(), report(), True, True)
    case('inside', report(), report(.25002), True, True)
    case('exact_boundary', report(0.), report(3e-5), True, True)
    case('outside', report(0.), report(.000030002), False, True)
    changed = report(); changed['target']['literal_fixture'] = 'distinct_new_ffn_literal_target'
    case('target_mismatch', report(), changed, False, True)
    changed = report(); changed['target']['policy'] = 'readout_projection_native_distribution_validation_v1'
    case('foreign_projection_policy', changed, copy.deepcopy(changed), False, True)
    case('nonfinite_ce', report(float('inf')), report(), False, False)
    case('integer_ce', report(1), report(), False, False)
    changed = report(); changed['extra_literal'] = 's'*(512*1024)
    case('overflow_metadata', changed, report(), False, False)
    case('suffix', report(), report(), False, False, suffix='.txt')
    case('provenance', report(), report(), False, False, prov={})
    original = (folder/'identical.json').read_bytes()
    try:
        S.publish_pair_then_require(folder/'identical.json', report(), report(.25002), provenance)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Own paired reports overwritten')
    require((folder/'identical.json').read_bytes() == original, 'Overwrite altered original pair')
    rows.append(dict(case='overwrite', existing_bytes_unchanged=True, all_predicates_passed=True))
    require(len(rows) == 12, 'Complete own pair metadata fixture count')
    return dict(new_own_policy_metadata_only=True, cases=rows, extra_models_corpus_or_rng_draws=0)


def run(roles):
    cpu_initial, mps_initial = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
    torch.random.default_generator.manual_seed(roles['global_cpu']); torch.mps.manual_seed(roles['global_mps'])
    branch_cpu, branch_mps = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
    rows = []; archives = []
    try:
        with CpuPrerequisiteGuard(R/'cpu-io-fixtures.json', ROOT, timeout_seconds=120, poll_seconds=.25):
            io_result = io_fixtures()
        with CpuPrerequisiteGuard(R/'cpu-pair-fixtures.json', ROOT, timeout_seconds=120, poll_seconds=.25):
            pair_result = pair_fixtures()
        for arm in ('native_swiglu', 'ungated_silu', 'ungated_relu_squared'):
            cfg = config(arm, 4, roles['initializer'], roles['train'], roles['eval'], global_cpu_seed=roles['global_cpu'], global_mps_seed=roles['global_mps'])
            torch.set_rng_state(branch_cpu); torch.mps.set_rng_state(branch_mps)
            m, opt, ema, schedule, data = setup(cfg, arm+'-full'); history = []
            trace = updates(m, opt, ema, schedule, data, 0, 4, history)
            payload = C.raw_payload(m, opt, ema, cfg, 4, history, data); full_sha = tree_hash(payload)
            raw_ref = write_archive(arm+'-final-raw', A/(arm+'-final-raw.pt'), payload); del payload; clean()
            loaded = read_archive(arm+'-control-raw', raw_ref); require(tree_hash(loaded) == full_sha, 'Serialized uninterrupted RAW roundtrip differs')
            del loaded; clean()
            payload = C.h200_payload(m, ema, cfg, raw_ref); export_sha = tree_hash(payload)
            h_ref = write_archive(arm+'-h200', A/(arm+'-h200.pt'), payload); del payload; clean()
            loaded = read_archive(arm+'-control-h200', h_ref); require(tree_hash(loaded) == export_sha, 'Serialized H200 roundtrip differs')
            del loaded, m, opt, ema, schedule, data; clean()
            torch.set_rng_state(branch_cpu); torch.mps.set_rng_state(branch_mps)
            m, opt, ema, schedule, data = setup(cfg, arm+'-prefix'); history = []
            partial_trace = updates(m, opt, ema, schedule, data, 0, 2, history)
            payload = C.raw_payload(m, opt, ema, cfg, 2, history, data); prefix_sha = tree_hash(payload)
            prefix_ref = write_archive(arm+'-prefix-raw', A/(arm+'-prefix-raw.pt'), payload)
            del payload, m, opt, ema, schedule, data; clean()
            m, opt, ema, schedule, data = setup(cfg, arm+'-destination')
            prefix = read_archive(arm+'-resume-prefix', prefix_ref); require(tree_hash(prefix) == prefix_sha, 'Serialized prefix RAW differs')
            torch.rand(11); torch.rand(13, device='mps'); data.get_batch('train'); data.get_batch('val')
            before = live_hash(m, opt, ema, data); rejected = []
            for name, bad in adversaries(prefix):
                try:
                    C.resume(m, opt, ema, bad, cfg, data)
                except ValueError:
                    pass
                else:
                    raise AssertionError('State forgery accepted: '+name)
                require(live_hash(m, opt, ema, data) == before, 'Rejected state mutated destination: '+name)
                rejected.append(name)
            del bad
            step, history = C.resume(m, opt, ema, prefix, cfg, data); del prefix; clean()
            restored = C.raw_payload(m, opt, ema, cfg, step, history, data)
            require(step == 2 and tree_hash(restored) == prefix_sha, 'Complete serialized prefix restore differs')
            del restored; clean()
            partial_trace += updates(m, opt, ema, schedule, data, 2, 4, history)
            resumed = C.raw_payload(m, opt, ema, cfg, 4, history, data)
            require(tree_hash(resumed) == full_sha and partial_trace == trace, 'Serialized4vs2+2 complete RAW/trace differs')
            require(opt.completed_steps == 4 and all(v == 4 for v in opt.gradient_steps.values()) and ema.snapshots == 3, 'Complete exposure/EMA differs')
            del resumed; clean()
            exported = C.h200_payload(m, ema, cfg, raw_ref)
            require(tree_hash(exported) == export_sha, 'Resumed H200 differs from serialized control')
            before = live_hash(m, opt, ema, data); h_rejected = []
            h_cases = [('kind', dict(exported, kind=C.RAW)), ('parent_kind', dict(exported, parent=dict(kind='foreign'))),
                ('parent_hash', dict(exported, parent=dict(raw_ref, sha256='0'*64))),
                ('parent_size', dict(exported, parent=dict(raw_ref, bytes=raw_ref['bytes']+1))),
                ('snapshots', dict(exported, averaged_weights=dict(exported['averaged_weights'], snapshots=0))),
                ('parent_prefix_horizon', dict(exported, parent=prefix_ref)),
                ('parent_actual_archive_kind', dict(exported, parent=dict(h_ref, kind=C.RAW))),
                ('export_tensor_parent_mismatch', alter(exported, ['model_state', 'transformer.ln_f.bias'],
                    exported['model_state']['transformer.ln_f.bias']+.125))]
            for name, forged in h_cases:
                try:
                    C.load_export(m, forged, cfg)
                except ValueError:
                    pass
                else:
                    raise AssertionError('H200 forgery accepted: '+name)
                require(live_hash(m, opt, ema, data) == before, 'Rejected H200 changed destination')
                h_rejected.append(name)
            try:
                C.resume(m, opt, ema, exported, cfg, data)
            except ValueError:
                pass
            else:
                raise AssertionError('H200 resumed optimizer')
            require(live_hash(m, opt, ema, data) == before, 'Rejected H200 resume mutated destination')
            h_rejected.append('optimizer_resume')
            key = 'transformer.ln_f.bias'; original_average = ema.averages['200'][key].clone()
            with torch.no_grad():
                ema.averages['200'][key].add_(.125)
            bad_ema_before = live_hash(m, opt, ema, data)
            try:
                C.h200_payload(m, ema, cfg, raw_ref)
            except ValueError:
                pass
            else:
                raise AssertionError('Live EMA inconsistent with parent accepted')
            require(live_hash(m, opt, ema, data) == bad_ema_before, 'Rejected live EMA producer mutated destination')
            with torch.no_grad():
                ema.averages['200'][key].copy_(original_average)
            require(live_hash(m, opt, ema, data) == before, 'Intentional EMA adversary restoration differs')
            h_rejected.append('producer_live_ema_parent_mismatch')
            del original_average
            require(len(h_rejected) == 10, 'Complete retained-parent H200 rejection count')
            del exported, h_cases, forged; clean()
            loaded = read_archive(arm+'-inference-h200', h_ref); C.load_export(m, loaded, cfg)
            require(all(torch.equal(p.cpu(), loaded['model_state'][n]) for n, p in m.named_parameters()), 'Serialized H200 consumer tensor values differ')
            del loaded, opt, ema, schedule; clean()
            m.eval(); ids = (torch.arange(276, dtype=torch.int64)*89+31) % 50257
            values = S.window_losses(m, ids); actual = torch.cat(values); gather_error = 0.
            with torch.no_grad():
                for (start, count), losses in zip(S.windows(275), values):
                    x = ids[start:start+count].unsqueeze(0).to('mps'); y = ids[start+1:start+count+1].to('mps')
                    lp = m(x).log_probs()[0]
                    require(float(torch.logsumexp(lp, -1).abs().max()) <= 3e-5, 'Scorer distribution not normalized')
                    oracle = -lp.gather(-1, y[:, None]).reshape(-1).cpu().to(torch.float64)
                    error = float((oracle-losses).abs().max()); require(torch.allclose(oracle, losses, atol=3e-5, rtol=0), 'Per-target gather predicate failed')
                    gather_error = max(gather_error, error)
            require([v.numel() for v in values] == [128, 128, 19], 'Generated scorer window coverage differs')
            del x, y, lp, oracle
            m.cpu(); clean()
            with CpuPrerequisiteGuard(R/('cpu-scorer-'+arm+'.json'), ROOT, timeout_seconds=120, poll_seconds=.25):
                cpu_losses = torch.cat(S.window_losses(m, ids)); ce_error = float((cpu_losses-actual).abs().max())
                require(torch.allclose(cpu_losses, actual, atol=3e-5, rtol=0), 'CPU/MPS per-target scorer predicate failed')
            arm_archives = [prefix_ref, raw_ref, h_ref]; archives.extend(arm_archives)
            require(sum(v['bytes'] for v in archives) <= 3456*1024**2, 'Complete retained archive cap')
            row = dict(arm=arm, full_raw_sha256=full_sha, prefix_raw_sha256=prefix_sha, exact_h200_sha256=export_sha,
                exact_serialized_resume=True, full_private_batch_trace_exact=True, optimizer_steps=8, targets_per_branch=4096,
                ema_snapshots=3, per_parameter_exposure=4, parameter_tensors=88 if arm=='native_swiglu' else 84,
                state_forgeries=rejected, h200_forgeries=h_rejected, archive_references=arm_archives,
                generated_scored_targets=275, target_only_full_gather_max_error=gather_error, scorer_cpu_mps_max_error=ce_error,
                all_per_target_predicates_passed=True, real_corpus_scoring=False)
            rows.append(row); publish('arm-'+arm+'.json', row); print(json.dumps(dict(completed_arm=arm, archives=len(archives))), flush=True)
            del m, data, values, actual, cpu_losses, losses; clean()
        with CpuPrerequisiteGuard(R/'cpu-scorer-math.json', ROOT, timeout_seconds=120, poll_seconds=.25):
            math_result = scorer_math()
        require(len(archives) == 9, 'All9 retained checkpoint archives required')
        manifests = []
        for ref in archives:
            p = Path(ref['path']); require(p.stat().st_nlink == 1 and p.stat().st_size == ref['bytes'] and sha(p) == ref['sha256'], 'Original retained archive changed')
            manifests.append(file_manifest(p))
    finally:
        torch.set_rng_state(cpu_initial); torch.mps.set_rng_state(mps_initial)
    require(torch.equal(cpu_initial, torch.get_rng_state()) and torch.equal(mps_initial, torch.mps.get_rng_state()), 'Caller global RNG not restored')
    return rows, math_result, io_result, pair_result, manifests


def close_archive_root(value):
    raw = (json.dumps(value, sort_keys=True)+'\n').encode()
    metadata = sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
    metadata += sum(p.lstat().st_size for p in (A/'io').iterdir()) if (A/'io').exists() else 0
    require(metadata+len(raw) <= 8*1024**2, 'Archive-root STOP metadata cap')
    with (A/'STOP').open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan-sha', required=True); arg = parser.parse_args()
    require(sha(R/'plan.json') == arg.plan_sha and set(p.name for p in R.iterdir()) == {'plan.json', 'source-review.json'}, 'One pristine exact pinned attempt')
    plan = json.loads((R/'plan.json').read_text()); require(len(plan['sources']) == 1191, 'All1189 predecessors plus2new harnesses required')
    for p, h in plan['sources'].items():
        require(sha(ROOT/p) == h, 'Source differs: '+p)
    for ref in plan['census_evidence']:
        require(sha(ref['path']) == ref['sha256'], 'Original census evidence differs')
    census = json.loads(Path(plan['census_result']).read_text())
    require(census['status'] == 'new_ffn_resume_roles_unused_before_construction' and census['prior_matches'] == [], 'New unused six-role census required')
    require(not A.exists() and shutil.disk_usage(ROOT).free >= plan['complete_next_chain_target_bytes'], 'Complete new retained outputs/reserve/extra2GiB required')
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); torch.set_num_threads(4)
        env = dict(torch_version=str(torch.__version__), torch_git=torch.version.git_version, threads=torch.get_num_threads(),
                   flags={k:os.environ.get(k) for k in plan['environment']['flags']})
        require(env == plan['environment'], 'Exact admitted runtime and strict four-thread flags required')
        A.mkdir(); publish('attempt.json', dict(pid=os.getpid(), plan_sha256=arg.plan_sha, roles=census['roles'], environment=env, archive_root=str(A)))
        try:
            with StrictMpsGuard(R/'mps-guard.json', ROOT):
                rows, math_result, io_result, pair_result, manifests = run(census['roles'])
                for p, h in plan['sources'].items():
                    require(sha(ROOT/p) == h, 'Source changed: '+p)
                publish('result.json', dict(status='complete_serialized_ffn_resume_scorer_proof', rows=rows, scorer_math=math_result,
                    io_fixtures=io_result, own_pair_metadata=pair_result, archive_manifests=manifests, retained_archive_bytes=sum(m['bytes'] for m in manifests),
                    source_global_cpu_mps_rng_preserved=True, actual_generated_optimizer_steps=24, model_archives=9,
                    real_corpus_holdout_quality_seed_calls=0, optimizer_cpu_mps_transport_claim=False, quality_gain_claim=False))
            publish('STOP', dict(status='complete_serialized_ffn_resume_scorer_proof', no_replay=True, no_retry=True))
            close_archive_root(dict(status='complete_accepted_generated_serialized_ffn_resume_fixture', no_replay=True, no_retry=True))
        except BaseException as exc:
            if not (R/'STOP').exists():
                publish('STOP', dict(status='closed_failed', error=repr(exc), no_retry=True))
            if not (A/'STOP').exists():
                close_archive_root(dict(status='closed_failed', error=repr(exc), no_retry=True))
            raise


if __name__ == '__main__':
    main()
