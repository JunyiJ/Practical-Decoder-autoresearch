"""Original finite actual-shape component proof. UNADMITTED until runtime receipt.

Stdlib whole-process guard starts before Torch, oracle and candidate imports.
"""
from array import array
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zlib
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
OUT = BASE/'component-math002'
CAP = 268435456
ATOL, RTOL = 4e-6, 4e-5
T = None
NEGATIVES = (
 'codes_list','lengths_list','codes_float','codes_bool','codes_int32','lengths_float',
 'codes_rank1','codes_rank3','empty_batch','batch_1025','columns_2','columns_131',
 'lengths_rank2','lengths_count','code_negative','code_260','length_2','length_past_end',
 'begin_wrong','end_wrong','bad_padding','interior_pad','interior_begin','interior_end',
 'mixed_special','special_length','codes_foreign_device','lengths_foreign_device',
 'autocast','parameter_nan','parameter_inf','parameter_shape','parameter_replaced',
 'parameter_frozen','parameter_alias','extra_parameter','extra_buffer','identity_changed',
 'parameter_float16','parameter_noncontiguous','parameter_missing','parameter_device',
 'hidden_module','default_mps','lengths_bool','lengths_int32',
 'state_export','state_load','seed_bool','seed_float','seed_negative','seed_upper',
)
CPU_ONLY = ('codes_sparse','codes_meta','parameter_float64','default_float64',
            'codes_float64','lengths_sparse','parameter_sparse','parameter_meta','lengths_meta')


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def publish(name, value):
    payload = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    need(sum(p.lstat().st_size for p in OUT.iterdir())+len(payload) <= CAP, 'Whole 256MiB report cap')
    with (OUT/name).open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


class WholeGuard(CpuPrerequisiteGuard):
    def _sample(self):
        sample, reason = super()._sample()
        entries = list(OUT.iterdir())
        if any(p.is_symlink() or not p.is_file() for p in entries):
            reason = 'unsafe_output'
        sample['output_bytes'] = sum(p.lstat().st_size for p in entries)
        if sample['output_bytes'] > CAP:
            reason = 'output_limit'
        if T is not None:
            sample['MPS_driver_bytes'] = T.mps.driver_allocated_memory()
            if sample['MPS_driver_bytes'] >= 6442450944:
                reason = 'MPS_limit'
        return sample, reason


def packed(values, code):
    data = array(code, values)
    if sys.byteorder != 'little':
        data.byteswap()
    raw = data.tobytes()
    return {'format': 'little_'+code, 'count': len(data), 'sha256': sha(raw),
            'zlib_base64': base64.b64encode(zlib.compress(raw)).decode()}


def comparison(actual, expected, shape):
    a = array('d', actual); e = array('d', expected)
    need(len(a) == len(e) == math.prod(shape), 'All coordinate shapes before comparison')
    mask = [math.isfinite(x) and math.isfinite(y) and abs(x-y) <= ATOL+RTOL*abs(y)
            for x, y in zip(a, e)]
    return {'shape': list(shape), 'atol': ATOL, 'rtol': RTOL,
            'actual': packed(a, 'd'), 'expected': packed(e, 'd'), 'mask': packed(mask, 'B'),
            'all_pass': all(mask), 'max_abs': max((abs(x-y) for x, y in zip(a, e)), default=0.)}


def cpu_values(tensor):
    # CPU transfer precedes every FP64 conversion; no MPS FP64 construction.
    return tensor.detach().to(device='cpu').to(dtype=T.float64).reshape(-1).tolist()


def tensor_digest(tensor):
    record = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
              'device': str(tensor.device), 'layout': str(tensor.layout)}
    if tensor.device.type == 'meta':
        record['payload'] = 'no_meta_storage'
    elif tensor.layout == T.sparse_coo:
        x = tensor.coalesce()
        record['indices'] = tensor_digest(x.indices()); record['values'] = tensor_digest(x.values())
    else:
        x = tensor.detach().to(device='cpu').contiguous()
        record['sha256'] = sha(bytes(x.view(T.uint8).reshape(-1).tolist()))
        record['storage_ptr'] = tensor.untyped_storage().data_ptr()
    return record


def rng_snapshot():
    # These APIs return state copies: preserve bytes, shape and dtype, not copy allocation addresses.
    result = {'cpu': tensor_digest(T.random.get_rng_state()), 'mps': tensor_digest(T.mps.get_rng_state())}
    for state in result.values():
        state.pop('storage_ptr', None)
    return result


def snapshot(module, codes, lengths):
    def value(x):
        return tensor_digest(x) if isinstance(x, T.Tensor) else repr(x)
    return {'inputs': [value(codes), value(lengths)], 'identity': repr(module.identity),
            'training': module.training, 'default_dtype': str(T.get_default_dtype()),
            'bindings': [(n,id(p)) for n,p in module._bindings],
            'default_device': str(T.get_default_device()),
            'modules': [(n,type(m).__module__+'.'+type(m).__qualname__) for n,m in module.named_modules()],
            'parameters': [(name, id(p), id(p.grad), tensor_digest(p), p.requires_grad,
                            tensor_digest(p.grad) if p.grad is not None else None)
                           for name, p in module.named_parameters()],
            'unregistered_bindings': [(n,tensor_digest(p),tensor_digest(p.grad) if p.grad is not None else None) for n,p in module._bindings if id(p) not in {id(q) for _,q in module.named_parameters()}],
            'buffers': [(name, tensor_digest(p)) for name, p in module.named_buffers()],
            'rng': rng_snapshot()}


def fill_literals(module, parameters, shapes):
    with T.no_grad():
        for name, p in module.named_parameters():
            p.copy_(T.tensor(parameters[name], dtype=T.float32).reshape(shapes[name]))
            p.grad = T.full_like(p, .125)


def invalid_transaction(label, backend, module, Composer, seed):
    codes = T.tensor([[257, 0, 255, 258, 256, 256]], dtype=T.int64, device=backend)
    lengths = T.tensor([4], dtype=T.int64, device=backend)
    undo = lambda: None
    operation = lambda: module(codes, lengths)
    autocast = False
    if label == 'codes_list': codes = [[257, 0, 258]]
    elif label == 'lengths_list': lengths = [4]
    elif label == 'codes_float': codes = codes.float()
    elif label == 'codes_bool': codes = codes.bool()
    elif label == 'codes_int32': codes = codes.int()
    elif label == 'lengths_float': lengths = lengths.float()
    elif label == 'lengths_bool': lengths = lengths.bool()
    elif label == 'lengths_int32': lengths = lengths.int()
    elif label == 'codes_float64': codes = codes.double()
    elif label == 'codes_rank1': codes = codes[0]
    elif label == 'codes_rank3': codes = codes[None]
    elif label == 'empty_batch': codes = codes[:0]; lengths = lengths[:0]
    elif label == 'batch_1025': codes = codes.repeat(1025, 1); lengths = lengths.repeat(1025)
    elif label == 'columns_2': codes = codes[:, :2]
    elif label == 'columns_131': codes = T.full((1,131), 256, dtype=T.int64, device=backend)
    elif label == 'lengths_rank2': lengths = lengths[:,None]
    elif label == 'lengths_count': lengths = lengths.repeat(2)
    elif label == 'code_negative': codes[0,1] = -1
    elif label == 'code_260': codes[0,1] = 260
    elif label == 'length_2': lengths[0] = 2
    elif label == 'length_past_end': lengths[0] = 7
    elif label == 'begin_wrong': codes[0,0] = 0
    elif label == 'end_wrong': codes[0,3] = 0
    elif label == 'bad_padding': codes[0,5] = 0
    elif label == 'interior_pad': codes[0,1] = 256
    elif label == 'interior_begin': codes[0,1] = 257
    elif label == 'interior_end': codes[0,1] = 258
    elif label == 'mixed_special': codes[0,1] = 259
    elif label == 'special_length': codes = T.tensor([[257,259,259,258]],device=backend); lengths[0] = 4
    elif label in ('codes_foreign_device','lengths_foreign_device'):
        other = 'mps' if backend == 'cpu' else 'cpu'
        if label == 'codes_foreign_device': codes = codes.to(other)
        else: lengths = lengths.to(other)
    elif label == 'codes_sparse': codes = codes.to_sparse()
    elif label == 'codes_meta': codes = T.empty((1,6),dtype=T.int64,device='meta')
    elif label == 'lengths_meta': lengths = T.empty((1,),dtype=T.int64,device='meta')
    elif label == 'lengths_sparse': lengths = lengths.to_sparse()
    elif label == 'autocast': autocast = True
    elif label in ('parameter_nan','parameter_inf'):
        p = module.highway_bias; original = p.detach().clone()
        with T.no_grad(): p[0] = float('nan') if label == 'parameter_nan' else float('inf')
        def undo():
            with T.no_grad(): p.copy_(original)
    elif label in ('parameter_shape','parameter_replaced','parameter_alias','parameter_float64','parameter_float16','parameter_noncontiguous','parameter_device','parameter_sparse','parameter_meta'):
        original = module.highway_weight
        if label in ('parameter_replaced','parameter_device','parameter_sparse','parameter_meta'):
            value = original.detach().clone()
            if label == 'parameter_device': value = value.to('mps' if backend=='cpu' else 'cpu')
            elif label == 'parameter_sparse': value = value.to_sparse()
            elif label == 'parameter_meta': value = T.empty_like(value,device='meta')
            module.highway_weight = T.nn.Parameter(value)
            def undo(): module.highway_weight = original
        else:
            data = original.detach().clone()
            if label == 'parameter_shape': original.data = original.detach()[:-1]
            elif label == 'parameter_alias': original.data = module.transform_weight.detach()
            elif label == 'parameter_float16': original.data = original.detach().half()
            elif label == 'parameter_noncontiguous': original.data = original.detach().T
            else: original.data = original.detach().double()
            def undo(): original.data = data
    elif label == 'parameter_missing':
        original_registry = module._parameters.copy(); del module.highway_bias
        def undo(): module._parameters = original_registry
    elif label == 'hidden_module':
        module.hidden = T.nn.Identity()
        def undo(): del module.hidden
    elif label == 'default_mps':
        T.set_default_device('mps')
        operation = lambda: Composer(seed)
        def undo(): T.set_default_device('cpu')
    elif label == 'parameter_frozen':
        module.highway_weight.requires_grad_(False)
        def undo(): module.highway_weight.requires_grad_(True)
    elif label == 'extra_parameter':
        module.extra = T.nn.Parameter(T.zeros(1,device=backend))
        def undo(): del module.extra
    elif label == 'extra_buffer':
        module.register_buffer('extra_buffer',T.zeros(1,device=backend))
        def undo(): del module.extra_buffer
    elif label == 'identity_changed':
        old = module._signature; module._signature = ('foreign',)
        def undo(): module._signature = old
    elif label == 'state_export': operation = lambda: module.state_dict()
    elif label == 'state_load': operation = lambda: module.load_state_dict({})
    elif label.startswith('seed_'):
        bad = {'seed_bool':True,'seed_float':1.,'seed_negative':-1,'seed_upper':2**63}[label]
        operation = lambda: Composer(bad)
    elif label == 'default_float64':
        T.set_default_dtype(T.float64)
        operation = lambda: Composer(seed)
        def undo(): T.set_default_dtype(T.float32)
    else: raise ValueError('Undeclared invalid family')
    before = snapshot(module,codes,lengths)
    caught = None
    try:
        try:
            if autocast:
                with T.autocast(device_type=backend,dtype=T.float16): operation()
            else: operation()
        except ValueError as exc:
            caught = repr(exc)
        after = snapshot(module,codes,lengths)
        report = {'id':backend+'-invalid-'+label,'family':label,'ValueError':caught,
                  'preserved':before == after,'before':before,'after':after}
        publish(report['id']+'.json', report)
        need(caught is not None and before == after, 'Invalid transactional preservation: '+label)
    finally:
        undo()


def main(expected_sha):
    global T
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need({p.name for p in OUT.iterdir()} == {'plan.json'}, 'Original proof attempt only')
        publish('attempt.json',dict(pid=os.getpid(),plan_sha256=expected_sha,no_retry=True,no_replay=True))
        try:
            with WholeGuard(OUT/'whole-guard.json',ROOT,timeout_seconds=600,
                            rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
                raw = (OUT/'plan.json').read_bytes(); need(sha(raw)==expected_sha,'Original numerical plan')
                plan = json.loads(raw)
                receipt = json.loads((BASE/'component-runtime002/review.json').read_text())
                need(receipt['status']=='complete_NEW_byte_component_runtime_admission' and
                     receipt['accepted'] is True and receipt['numerical_plan_sha256']==expected_sha,
                     'Independent complete runtime admission first')
                need(json.loads((BASE/'component-runtime002/guard.json').read_text())['status']=='finished' and
                     json.loads((BASE/'component-runtime002/STOP').read_text())['status']=='complete_original_admission',
                     'All original admission guards complete')
                need(os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize >= 11005853696,
                     '256MiB plus reserve and 2GiB extra entry capacity')
                for pin in plan['execution_sources']:
                    payload = Path(pin['path']).read_bytes()
                    need(len(payload)==pin['size'] and sha(payload)==pin['sha256'],'All execution source/runtime pins')
                for key,val in plan['environment'].items():
                    need(os.environ.get(key)==val,'Pinned execution environment: '+key)
                sys.path.insert(0,str(BASE))
                import component_oracle001 as oracle
                # No oracle function is executed before the whole guard/admission.
                import torch
                T = torch
                from research.byte_composer_v1 import ByteComposer
                need(T.__version__=='2.14.1' and T.version.git_version=='5c4886908584029761b579af026dcfb627c84070',
                     'Exact installed Torch runtime')
                need(T.backends.mps.is_available(),'MPS required, no fallback')
                T.set_num_threads(4)
                initial_cpu = T.random.get_rng_state().clone(); initial_mps = T.mps.get_rng_state().clone()
                saved_cpu_outputs = {}
                oracle_results = {}
                try:
                    for case in oracle.CASES:
                        with CpuPrerequisiteGuard(OUT/(case[0]+'-oracle-guard.json'),ROOT,timeout_seconds=120):
                            parameters = oracle.literal_parameters(case[3])
                            oracle_results[case[0]] = oracle.reference(case,parameters)
                    for backend in ('cpu','mps'):
                        for case in oracle.CASES:
                            name,sequences,padded,_ = case
                            with CpuPrerequisiteGuard(OUT/(backend+'-'+name+'-guard.json'),ROOT,timeout_seconds=120):
                                rng_before = rng_snapshot()
                                module = ByteComposer(plan['initializer']).to(backend)
                                init_rng_preserved = rng_before == rng_snapshot()
                                parameters = oracle.literal_parameters(case[3])
                                fill_literals(module,parameters,oracle.SHAPES)
                                rows = [list(s)+[256]*(padded-len(s)) for s in sequences]
                                codes = T.tensor(rows,dtype=T.int64,device=backend)
                                lengths = T.tensor([len(s) for s in sequences],dtype=T.int64,device=backend)
                                before = snapshot(module,codes,lengths)
                                output = module(codes,lengths)
                                ref = oracle_results[name]
                                coefficients = T.tensor(ref['coefficients'],dtype=T.float32,device=backend)
                                loss = (output*coefficients).sum()
                                items = tuple(module.named_parameters())
                                gradients = T.autograd.grad(loss,[p for _,p in items])
                                comparisons = {'output':comparison(cpu_values(output),ref['output'],(len(rows),256))}
                                comparisons.update({n:comparison(cpu_values(g),ref['gradients'][n],oracle.SHAPES[n])
                                                    for (n,_),g in zip(items,gradients)})
                                after = snapshot(module,codes,lengths)
                                values = cpu_values(output)
                                pair = None
                                if backend=='cpu': saved_cpu_outputs[name] = values
                                else:
                                    pair = {'atol':3e-5,'rtol':0.,'all_coordinates':
                                            all(abs(a-b)<=3e-5 for a,b in zip(values,saved_cpu_outputs[name]))}
                                report = dict(id=backend+'-'+name,backend=backend,case=name,
                                    comparisons=comparisons,parameter_tensor_count=len(items),
                                    parameter_elements=sum(p.numel() for _,p in items),preserved=before==after,
                                    before=before,after=after,constructor_global_RNG_preserved=init_rng_preserved,
                                    oracle_valid_max_tie_channels=ref['tie_channels'],CPU_MPS_pair=pair)
                                publish(backend+'-'+name+'.json',report)
                                need(init_rng_preserved and before==after and len(items)==18 and
                                     all(x['all_pass'] for x in comparisons.values()) and
                                     (pair is None or pair['all_coordinates']), 'Fixed all-coordinate component predicates: '+backend+'/'+name)
                                del module, gradients, output, loss, comparisons, report, parameters
                        with CpuPrerequisiteGuard(OUT/(backend+'-invalid-guard.json'),ROOT,timeout_seconds=120):
                            module = ByteComposer(plan['initializer']).to(backend)
                            parameters = oracle.literal_parameters(False)
                            fill_literals(module,parameters,oracle.SHAPES)
                            for label in NEGATIVES+(CPU_ONLY if backend=='cpu' else ()):
                                invalid_transaction(label,backend,module,ByteComposer,plan['initializer'])
                            del module,parameters
                finally:
                    T.random.set_rng_state(initial_cpu); T.mps.set_rng_state(initial_mps)
                need(T.equal(T.random.get_rng_state(),initial_cpu) and T.equal(T.mps.get_rng_state(),initial_mps),
                     'Final caller CPU/MPS RNG restored')
                for pin in plan['execution_sources']:
                    need(sha(Path(pin['path']).read_bytes())==pin['sha256'],'Final source preservation')
                publish('result.json',dict(status='complete_original_finite_byte_component_math',
                    positives=12,invalids=2*len(NEGATIVES)+len(CPU_ONLY),parameter_tensors_each=18,
                    parameter_gradient_coordinates=12*725235,caller_CPU_MPS_RNG_restored=True,
                    no_model_optimizer_corpus_tokenizer_scoring_or_tensor_files=True,
                    initializer=plan['initializer'],initializer_status='USED_EXCLUSIVE_ORIGINAL_COMPONENT_MATH002_ONLY',
                    no_quality_or_model_adoption=True))
            publish('STOP',dict(status='complete_original_finite_component_math',no_retry=True,no_replay=True))
        except BaseException as exc:
            if not (OUT/'STOP').exists():
                publish('STOP',dict(status='closed_failed_original_component_math',error=repr(exc),no_retry=True,no_replay=True))
            raise
if __name__=='__main__': main(sys.argv[1])
