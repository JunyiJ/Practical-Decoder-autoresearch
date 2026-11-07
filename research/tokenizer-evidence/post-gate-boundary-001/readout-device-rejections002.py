"""Distinct uncovered device/autocast adversaries; no positive-case replay."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import torch
from research.readout_mixture_v1.component import nll
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent
R = E/'readout-device-rejections002'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def publish(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()
    require(sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw) <= 2*1024**2, 'Metadata cap')
    with (R/name).open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def fixture(device):
    # New geometry N2,D2,V3; no original fixture/oracle/gradient execution.
    return [torch.tensor(x, device=device, dtype=torch.float32) for x in
            ([[.125, -.5], [.75, .25]], [[.3, -.2], [.4, .1], [-.5, .6]],
             [[[.2, .4], [-.3, .5]], [[.6, -.1], [.3, -.4]]], [[.7, -.2], [-.3, .1]])]


def reject(kind, args, target):
    snapshot = [x.detach().cpu().clone() for x in args if x.device.type != 'meta']
    try:
        nll(kind, args[0], args[1], target, **({} if kind == 'native' else dict(a=args[2], r=args[3])))
    except ValueError:
        pass
    else:
        raise AssertionError('Device/autocast adversary accepted')
    actual = [x.detach().cpu() for x in args if x.device.type != 'meta']
    require(all(torch.equal(a, b) for a, b in zip(actual, snapshot)), 'Rejected input mutated')


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--plan-sha', required=True)
    arg = cli.parse_args()
    require(digest(R/'plan.json') == arg.plan_sha, 'Plan digest differs')
    require(set(p.name for p in R.iterdir()) == {'plan.json'}, 'One pristine attempt only')
    plan = json.loads((R/'plan.json').read_text())
    for p, h in plan['sources'].items():
        require(digest(ROOT/p) == h, 'Source differs')
    require(len(plan['sources']) == 1088, '1087 predecessors plus distinct harness')
    for ref in plan['closed_positive_proof']:
        require(digest(ref['path']) == ref['sha256'], 'Closed evidence differs')
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        require(all(os.environ.get(k) == '4' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')), 'Thread flags')
        publish('attempt.json', dict(pid=os.getpid(), plan_sha256=arg.plan_sha))
        try:
            with StrictMpsGuard(R/'mps-guard.json', ROOT), CpuPrerequisiteGuard(
                    R/'cpu-guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
                mps_rng = torch.mps.get_rng_state().clone()
                cpu_rng = torch.get_rng_state().clone()
                rows = []
                for kind in ('context_mix2', 'probability_mix2'):
                    for index in range(5):
                        args = fixture('mps')
                        target = torch.tensor([0, 2], dtype=torch.int64, device='mps')
                        if index == 4:
                            target = target.cpu()
                        else:
                            args[index] = args[index].cpu()
                        reject(kind, args, target)
                        rows.append(dict(kind=kind, mismatch_index=index, rejected=True))
                for target_mismatch in (False, True):
                    args = fixture('mps')
                    target = torch.tensor([0, 2], dtype=torch.int64, device='cpu' if target_mismatch else 'mps')
                    if not target_mismatch:
                        args[1] = args[1].cpu()
                    reject('native', args, target)
                    rows.append(dict(kind='native', target_mismatch=target_mismatch, rejected=True))
                args = fixture('cpu')
                args[0] = torch.empty(2, 2, device='meta')
                reject('native', args, torch.tensor([0, 2], dtype=torch.int64))
                rows.append(dict(kind='native', unsupported_meta_device=True, rejected=True))
                for kind in ('native', 'context_mix2', 'probability_mix2'):
                    with torch.autocast('mps', dtype=torch.float16):
                        reject(kind, fixture('mps'), torch.tensor([0, 2], dtype=torch.int64, device='mps'))
                    rows.append(dict(kind=kind, autocast=True, rejected=True))
                torch.mps.synchronize()
                require(torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(mps_rng, torch.mps.get_rng_state()), 'RNG changed')
                for p, h in plan['sources'].items():
                    require(digest(ROOT/p) == h, 'Source changed')
                publish('result.json', dict(status='complete_distinct_device_autocast_rejections', rows=rows,
                    rejected=len(rows), positive_case_replay=False, model_optimizer_corpus_checkpoint_calls=0,
                    sources_inputs_global_cpu_mps_rng_unchanged=True, tensor_files=0))
            publish('STOP', dict(status='complete_distinct_device_autocast_rejections', no_replay=True))
            print(json.dumps(dict(status='complete', rejected=len(rows))), flush=True)
        except BaseException as exc:
            if not (R/'STOP').exists():
                publish('STOP', dict(status='closed_failed', no_retry=True, exception=repr(exc)))
            raise


if __name__ == '__main__':
    main()
