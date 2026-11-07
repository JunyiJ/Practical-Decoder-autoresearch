"""NEW literal V50257 CE/analytic-gradient oracle; no models/data/seeds/replay."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import torch
from research.native_vocabulary_ce_v1 import losses, VOCAB
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')
FP64 = dict(forward_atol=1e-11, gradient_atol=1e-12, gradient_rtol=1e-10)
FP32 = dict(forward_atol=3e-5, gradient_atol=4e-6, gradient_rtol=4e-5)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw) <= 2*1024**2
    with (R/name).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def literals():
    periodic = [(((j*37+11) % 257)-128)/32 for j in range(VOCAB)]
    correct = [(((j*13+19) % 31)-15)/64 for j in range(VOCAB)]
    correct[32003] = 14.
    wrong = [(((j*23+7) % 43)-21)/32 for j in range(VOCAB)]
    wrong[27011] = 20.; wrong[40009] = -12.
    rows = [[0.]*VOCAB, periodic, correct, wrong,
            [v+768 for v in periodic], [v-768 for v in periodic]]
    names = ['flat50257', 'dyadic_period257', 'correct_peak14', 'wrong_peak20_target_minus12',
             'period257_plus768', 'period257_minus768']
    targets = [25128, 40103, 32003, 40009, 40103, 40103]
    delta = [1/2048 if j % 3 == 0 else -1/4096 for j in range(VOCAB)]
    perturbed = [v+d for v, d in zip(periodic, delta)]
    return rows, targets, names, perturbed, delta


def oracle(row, target):
    # Literal scalar implementation, independent of torch CE/logsumexp/softmax/autograd.
    peak = max(row); exponentials = [math.exp(v-peak) for v in row]
    total = math.fsum(exponentials)
    value = (peak-row[target])+math.log(total)
    gradient = [v/total for v in exponentials]; gradient[target] -= 1.
    assert math.isfinite(value) and all(math.isfinite(v) for v in gradient)
    return value, gradient


def evaluate(rows, targets, expected, device, dtype, label):
    x = torch.tensor(rows, dtype=dtype, device=device, requires_grad=True)
    y = torch.tensor(targets, dtype=torch.int64, device=device)
    x.grad = torch.full_like(x, 1/64)
    saved_x = x.detach().clone(); saved_y = y.clone(); saved_grad = x.grad.clone()
    values = losses(x, y)
    gradient, = torch.autograd.grad(values.sum(), x)
    assert torch.equal(x, saved_x) and torch.equal(y, saved_y) and torch.equal(x.grad, saved_grad)
    values_cpu = values.detach().cpu().to(torch.float64)
    gradient_cpu = gradient.detach().cpu().to(torch.float64)
    reference_values = torch.tensor([v[0] for v in expected], dtype=torch.float64)
    reference_gradient = torch.tensor([v[1] for v in expected], dtype=torch.float64)
    tolerance = FP64 if dtype == torch.float64 else FP32
    rows_report = []
    for i in range(len(rows)):
        forward_error = float(abs(values_cpu[i]-reference_values[i]))
        error = (gradient_cpu[i]-reference_gradient[i]).abs()
        grad_ok = bool((error <= tolerance['gradient_atol']+tolerance['gradient_rtol']*reference_gradient[i].abs()).all())
        rows_report.append(dict(row=i, target=targets[i], ce=float(values_cpu[i]), oracle_ce=expected[i][0],
            forward_abs_error=forward_error, forward_pass=forward_error <= tolerance['forward_atol'],
            gradient_abs_max=float(error.max()), gradient_elementwise_pass=grad_ok))
    # Save each complete backend measurement BEFORE numerical predicate rejection.
    publish(label+'.json', dict(device=str(device), dtype=str(dtype), tolerances=tolerance,
        rows=rows_report, caller_input_targets_populated_gradslots_unchanged=True))
    return values_cpu, gradient_cpu, rows_report


def invalids(device, dtype, label):
    x = torch.full((1, VOCAB), 1/8, device=device, dtype=dtype, requires_grad=True)
    y = torch.tensor([17011], device=device, dtype=torch.int64)
    x.grad = torch.full_like(x, 1/32)
    saved = (x.detach().clone(), y.clone(), x.grad.clone()); rejected = []
    def probe(a, b):
        snapshots = []
        for value in (a, b):
            if isinstance(value, torch.Tensor):
                if value.device.type == 'meta':
                    snapshots.append((value, None, None, tuple(value.shape)))
                elif value.layout == torch.strided:
                    grad = value.grad.detach().clone() if value.is_leaf and value.grad is not None else None
                    snapshots.append((value, value.detach().clone(), grad, tuple(value.stride())))
                else:
                    snapshots.append((value, value.detach().clone(), None, None))
        try:
            return losses(a, b)
        finally:
            for value, content, grad, layout in snapshots:
                if content is None:
                    assert tuple(value.shape) == layout
                elif value.layout == torch.strided:
                    assert tuple(value.stride()) == layout
                    torch.testing.assert_close(value, content, atol=0., rtol=0., equal_nan=True)
                    if value.is_leaf:
                        assert (value.grad is None) == (grad is None)
                        if grad is not None:
                            torch.testing.assert_close(value.grad, grad, atol=0., rtol=0., equal_nan=True)
                else:
                    torch.testing.assert_close(value.to_dense(), content.to_dense(), atol=0., rtol=0., equal_nan=True)
    def reject(name, call):
        try:
            call()
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Invalid input accepted: '+name)
        assert torch.equal(x, saved[0]) and torch.equal(y, saved[1]) and torch.equal(x.grad, saved[2])
    reject('logits_list', lambda: probe([[0.]], y))
    reject('targets_list', lambda: probe(x, [0]))
    reject('parameter_subclass', lambda: probe(torch.nn.Parameter(x.detach()), y))
    reject('rank1', lambda: probe(x[0], y))
    reject('rank3', lambda: probe(x.unsqueeze(0), y))
    reject('zero_rows', lambda: probe(x[:0], y[:0]))
    reject('seven_rows', lambda: probe(x.detach().repeat(7, 1), y.repeat(7)))
    reject('wrong_vocab', lambda: probe(x[:, :-1], y))
    reject('target_rank', lambda: probe(x, y.unsqueeze(0)))
    reject('target_length', lambda: probe(x, y.repeat(2)))
    reject('logits_int', lambda: probe(x.detach().to(torch.int64), y))
    reject('logits_half', lambda: probe(x.detach().to(torch.float16), y))
    reject('targets_float', lambda: probe(x, y.to(torch.float32)))
    reject('targets_bool', lambda: probe(x, y.to(torch.bool)))
    reject('target_negative', lambda: probe(x, torch.tensor([-1], device=device)))
    reject('target_vocab', lambda: probe(x, torch.tensor([VOCAB], device=device)))
    reject('noncontiguous_logits', lambda: probe(torch.zeros((VOCAB, 2), dtype=dtype, device=device).T, y.repeat(2)))
    reject('noncontiguous_targets', lambda: probe(x.detach().repeat(2, 1), y.repeat(4)[::2]))
    for name, value in [('nan', float('nan')), ('positive_inf', float('inf')), ('negative_inf', -float('inf'))]:
        bad = x.detach().clone(); bad[0, 37] = value
        reject(name, lambda bad=bad: probe(bad, y))
    # Finite FP32 inputs can overflow CE; no mutation or nonfinite result allowed.
    if dtype == torch.float32:
        overflow = torch.full_like(x, 3e38); overflow[0, 17011] = -3e38
        reject('finite_CE_overflow', lambda: probe(overflow, y))
    if str(device) == 'mps':
        reject('foreign_CPU_targets', lambda: probe(x, y.cpu()))
    else:
        reject('meta_device', lambda: probe(torch.empty((1, VOCAB), device='meta'), torch.empty(1, dtype=torch.int64, device='meta')))
        reject('sparse_layout', lambda: probe(x.detach().to_sparse(), y))
    with torch.autocast(device_type=str(device)):
        reject('autocast', lambda: probe(x, y))
    publish(label+'-invalids.json', dict(rejected=rejected, caller_input_targets_populated_gradslots_unchanged=True))
    return rejected


def close_rows(rows):
    return all(r['forward_pass'] and r['gradient_elementwise_pass'] for r in rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan-sha', required=True); args = parser.parse_args()
    assert sha(R/'plan.json') == args.plan_sha and set(p.name for p in R.iterdir()) == {'plan.json', 'source-review.json'}
    plan = json.loads((R/'plan.json').read_text())
    for p, h in plan['sources'].items():
        assert sha(ROOT/p) == h, p
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == os.environ.get('PYTORCH_MPS_FAST_MATH') == '0'
    assert all(os.environ.get(k) == '4' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'])
    assert torch.__version__ == plan['runtime']['torch'] and torch.version.git_version == plan['runtime']['torch_git']
    torch.set_num_threads(4)
    with session_lock(GLOBAL_LOCK.parent):
        publish('attempt.json', dict(pid=os.getpid(), plan_sha256=args.plan_sha))
        try:
            with StrictMpsGuard(R/'mps-guard.json', ROOT):
                cpu_rng = torch.get_rng_state().clone(); mps_rng = torch.mps.get_rng_state().clone()
                with CpuPrerequisiteGuard(R/'cpu-oracle-guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
                    rows, targets, names, perturbed, delta = literals()
                    expected = [oracle(z, t) for z, t in zip(rows, targets)]
                    perturb_expected = [oracle(perturbed, targets[1])]
                    epsilon = max(abs(v) for v in delta)
                    assert expected[1][0] == expected[4][0] == expected[5][0]
                    assert abs(perturb_expected[0][0]-expected[1][0]) <= 2*epsilon+1e-12
                    publish('oracle.json', dict(names=names, targets=targets, values=[v[0] for v in expected],
                        perturbed_value=perturb_expected[0][0], perturbation_epsilon=epsilon,
                        ideal_CE_two_epsilon_bound=2*epsilon, common_shift_reference_exact=True,
                        no_tensor_model_optimizer_data_rng_oracle_calls=True))
                backend={}; negative={}; predicates={}
                for label, device, dtype in [('cpu64','cpu',torch.float64),('cpu32','cpu',torch.float32),('mps32','mps',torch.float32)]:
                    def phase():
                        a,g,report=evaluate(rows,targets,expected,device,dtype,label+'-six')
                        p,pg,pr=evaluate([perturbed],[targets[1]],perturb_expected,device,dtype,label+'-perturb')
                        single,sg,sr=evaluate([rows[1]],[targets[1]],[expected[1]],device,dtype,label+'-slice')
                        negative[label]=invalids(device,dtype,label)
                        return a,g,p,report,pr,single,sr
                    if device=='cpu':
                        with CpuPrerequisiteGuard(R/(label+'-guard.json'), ROOT, timeout_seconds=120, poll_seconds=.25):
                            a,g,p,report,pr,single,sr=phase()
                    else:
                        a,g,p,report,pr,single,sr=phase();torch.mps.synchronize()
                    tolerance=FP64 if dtype==torch.float64 else FP32
                    predicates[label+'_oracle']=close_rows(report+pr+sr)
                    predicates[label+'_common_shift']=bool((a[[4,5]]-a[1]).abs().max()<=tolerance['forward_atol'])
                    predicates[label+'_slice']=abs(float(single[0]-a[1]))<=tolerance['forward_atol']
                    predicates[label+'_perturbation']=abs(float(p[0]-a[1]))<=2*epsilon+2*tolerance['forward_atol']
                    backend[label]=(a,g,p)
                a,g,p=backend['cpu32'];b,h,q=backend['mps32']
                ce_errors=(a-b).abs();gradient_errors=(g-h).abs()
                predicates['CPU_MPS_same_input_CE']=bool(ce_errors.max()<=3e-5) and abs(float(p[0]-q[0]))<=3e-5
                predicates['CPU_MPS_gradient']=bool((gradient_errors<=4e-6+4e-5*g.abs()).all())
                pairs=dict(names=names,cpu32_CE=a.tolist(),mps32_CE=b.tolist(),per_row_abs_ce_error=ce_errors.tolist(),
                    fixed_CE_atol=3e-5,fixed_CE_rtol=0.,gradient_max=float(gradient_errors.max()),
                    perturb_cpu32_CE=float(p[0]),perturb_mps32_CE=float(q[0]))
                publish('paired-comparison.json',pairs)
                assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(mps_rng,torch.mps.get_rng_state())
                for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,path
                publish('result.json',dict(status='accepted_new_literal_full_vocabulary_CE_probe' if all(predicates.values()) else 'fixed_numerical_predicate_failed',
                    predicates=predicates,negative_counts={k:len(v) for k,v in negative.items()},pairs=pairs,
                    forward_oracle_max={k:max(r['forward_abs_error'] for part in ['six','perturb','slice'] for r in json.loads((R/(k+'-'+part+'.json')).read_text())['rows']) for k in backend},
                    caller_CPU_MPS_RNG_unchanged=True,source_pins_unchanged=True,no_models_optimizer_corpus_checkpoints_seeds_rng_draws=True,
                    no_closed_fixture_replay_or_failed_model_cause_or_quality_adoption_claim=True))
                assert all(predicates.values()), 'Fixed prespecified numerical predicate failed;see durable result/pairs before rejection'
            publish('STOP',dict(status='complete_accepted_new_literal_full_vocabulary_CE_probe',no_replay=True,no_retry=True))
        except BaseException as exc:
            if not (R/'STOP').exists():publish('STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
            raise

if __name__=='__main__':main()
