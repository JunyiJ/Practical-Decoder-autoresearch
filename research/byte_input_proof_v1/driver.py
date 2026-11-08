"""Full actual-model numerical proof. Imported only by guarded admitted entry."""
import gc
import hashlib
import json
import math
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.byte_input_model_v2 import Spec, ByteInputModel, build_codebook, build_optimizer
from research.byte_input_model_v2 import codebook as implementation
from research.final_ln_scale_model_v1.runtime.models.gpt import GPT as Native
from . import cases, codebook_reference as code_ref, model_reference as reference
from .coordinates import compare
from .transactions import run as invalid, tensor_record


def private_parameters(seed):
    """Independent declared initializer draws in construction order."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    values = {}
    def draw(name, shape, lo=-.05, hi=.05):
        values[name] = torch.nn.Parameter(torch.empty(shape).uniform_(lo, hi, generator=generator))
    draw('byte_weight', (259, 15))
    for i, channels in enumerate((25, 50, 75, 100, 125, 150)):
        draw('filters.'+str(i), (channels, 15*(i+1)))
    for i, channels in enumerate((25, 50, 75, 100, 125, 150)):
        draw('filter_biases.'+str(i), (channels,))
    draw('highway_weight', (525, 525)); draw('highway_bias', (525,))
    draw('transform_weight', (525, 525)); draw('transform_bias', (525,), -2.05, -1.95)
    draw('projection_weight', (256, 525), -1/math.sqrt(525), 1/math.sqrt(525))
    # Module parameter registration traverses direct parameters before children.
    order = ('byte_weight', 'highway_weight', 'highway_bias', 'transform_weight',
             'transform_bias', 'projection_weight')
    return dict([(n, values[n]) for n in order] +
                [(n, values[n]) for n in values if n.startswith(('filters.', 'filter_biases.'))])


def cleanup():
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()


def native_construct(seed, config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        native = Native(OmegaConf.create(config))
        state = torch.random.get_rng_state().clone()
    return native, state


def candidate_construct(seed, spec, codebook):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        candidate = ByteInputModel(spec, codebook)
        state = torch.random.get_rng_state().clone()
    return candidate, state


def optimizer_inventory(children, parameters):
    by_id = {id(p):n for n,p in parameters.items()}
    return [dict(type=type(child).__name__, groups=[
        dict(names=[by_id[id(p)] for p in group['params']],
             options={k:v for k,v in group.items() if k != 'params'})
        for group in child.param_groups]) for child in children]


def inspect_codebooks(plan, out, publish):
    payloads = [Path(pin['path']).read_bytes() for pin in plan['assets']]
    rows = code_ref.raw_rows(*payloads)
    books, expected = {}, {}
    for variant in ('raw_bytes', 'nonlexical_codes'):
        key = None if variant == 'raw_bytes' else plan['roles']['nonlexical_key']
        codes, lengths = code_ref.expected_codes(rows, variant, key)
        book = build_codebook(*payloads, variant, key)
        compare(out, 'codebook-'+variant+'-rows',
                torch.frombuffer(bytearray(book.codes), dtype=torch.uint16).reshape(50257,130),
                torch.tensor(codes, dtype=torch.int64).reshape(50257,130), 0., 0.)
        compare(out, 'codebook-'+variant+'-lengths',
                torch.frombuffer(bytearray(book.lengths), dtype=torch.uint16),
                torch.tensor(lengths, dtype=torch.int64), 0., 0.)
        assert book.codes_i64_sha256 == code_ref.staged_digest(codes)
        assert book.lengths_i64_sha256 == code_ref.staged_digest(lengths)
        books[variant], expected[variant] = book, (codes, lengths)
        publish('codebook-'+variant+'.json', dict(identity=book.identity, exact_all_rows=True,
                length_rows=len(lengths), code_coordinates=len(codes), staged_hashes_verified=True))
    key = plan['roles']['nonlexical_key']
    checked = 0
    for length in (1, 2):
        actual, expect, recovered, independent_inverse = [], [], [], []
        for number in range(1 << (8*length)):
            value = implementation.permute(number, length, key)
            expected_value = code_ref.feistel(number, length, key)
            inverse = implementation.inverse(value, length, key)
            actual.append(value); expect.append(expected_value); recovered.append(inverse)
            independent_inverse.append(code_ref.feistel(value, length, key, reverse=True))
            checked += 1
        compare(out, 'feistel'+str(length)+'-forward', torch.tensor(actual), torch.tensor(expect), 0., 0.)
        compare(out, 'feistel'+str(length)+'-inverse', torch.tensor(recovered), torch.arange(len(actual)), 0., 0.)
        compare(out, 'feistel'+str(length)+'-reference-inverse', torch.tensor(independent_inverse), torch.arange(len(actual)), 0., 0.)
        assert len(set(actual)) == 1 << (8*length)
    # Every vocabulary ordinal was checked by both builders, including long
    # domains. Exhaustive small domains are additional, not universal coverage.
    publish('bijection.json', dict(exhaustive_inputs=checked, lengths=[1,2],
             full_vocab_inverse_rows=50256, symbolic_argument=code_ref.bijection_argument(),
             nonlexical_byte_marginals_matched=False))
    return rows, books, expected


def one_model(plan, out, publish, rows, books, expected, backend, variant, common_initializers):
    prefix = backend+'-'+variant
    book = books.get(variant)
    spec = Spec(variant) if book is None else Spec(variant, plan['roles']['composer_initializer'], book.identity)
    native, native_rng = native_construct(plan['roles']['native_initializer'], plan['native_config'])
    model, candidate_rng = candidate_construct(plan['roles']['native_initializer'], spec, book)
    assert torch.equal(native_rng, candidate_rng), 'Native-first global RNG draw parity'
    composer = None if book is None else private_parameters(plan['roles']['composer_initializer'])
    ref_parameters = reference.parameter_inventory(native, composer)
    actual_parameters = dict(model.ordered_parameters())
    inventory = [(n, list(p.shape)) for n,p in actual_parameters.items()]
    expected_count = 88 if book is None else 105
    expected_elements = 28492416 if book is None else 16351859
    assert list(actual_parameters) == list(ref_parameters)
    assert len(actual_parameters) == expected_count
    assert sum(p.numel() for p in actual_parameters.values()) == expected_elements
    initial = {n:tensor_record(p, False) for n,p in actual_parameters.items()}
    exact_init = {n:torch.equal(p, ref_parameters[n]) for n,p in actual_parameters.items()}
    publish(prefix+'-initialization.json', dict(inventory=inventory, exact_values=exact_init,
            initializer_digests=initial, native_first_rng_equal=True, config=model.config,
            root_modules=list(model._modules), registered_parameters=len(list(model.parameters()))))
    assert all(exact_init.values())
    for n, record in initial.items():
        if n == 'transformer.wte.weight': continue
        old = common_initializers.setdefault(n, record)
        assert old == record, 'Paired native body/readout/private composer initialization'
    if book is not None:
        # Reference body no longer owns a displaced input table. Its canonical
        # inventory was obtained first and excludes that parameter.
        del native.transformer.wte
    model = model.to(backend)
    native = native.to(backend)
    if composer is not None:
        composer = {n:torch.nn.Parameter(p.detach().to(backend)) for n,p in composer.items()}
        ref_parameters = {n:p for n,p in native.named_parameters()}
        ref_parameters.update(('composer.'+n,p) for n,p in composer.items())
    else:
        ref_parameters = dict(native.named_parameters())
    actual_parameters = dict(model.ordered_parameters())
    codes = lengths = None
    if book is not None:
        raw_codes, raw_lengths = expected[variant]
        codes = torch.tensor(raw_codes, dtype=torch.int64, device=backend).reshape(50257,130)
        lengths = torch.tensor(raw_lengths, dtype=torch.int64, device=backend)
        compare(out, prefix+'-code-ids', model.code_ids, codes, 0., 0.)
        compare(out, prefix+'-code-lengths', model.code_lengths, lengths, 0., 0.)
    optimizer = build_optimizer(model, 0.00125)
    ref_optimizers = reference.optimizer(ref_parameters, 0.00125)
    groups = optimizer_inventory(optimizer._optimizers, actual_parameters)
    ref_groups = optimizer_inventory(ref_optimizers, ref_parameters)
    publish(prefix+'-optimizer-groups.json', dict(actual=groups, reference=ref_groups))
    assert groups == ref_groups
    assert [len(c.param_groups) for c in optimizer._optimizers] == ([44,44] if book is None else [53,52])
    inputs, labels = cases.inputs(torch, rows, backend)
    assert tuple(inputs.shape) == (8,128)
    # Sequential graphs: save CPU outputs/gradients from the actual path, release
    # its graph, and only then construct the independent reference graph.
    result = model.forward_train(inputs, labels)
    assert tuple(result.primary.values.shape) == cases.SHAPE
    actual_logits = result.primary.values.detach().cpu().clone()
    actual_loss = result.loss.detach().cpu().clone()
    result.loss.backward()
    del result
    cleanup()
    ref_logits, ref_loss = reference.primary(native, composer, codes, lengths, inputs, labels)
    compare(out, prefix+'-logits', actual_logits, ref_logits, cases.ATOL, cases.RTOL)
    compare(out, prefix+'-loss', actual_loss, ref_loss, cases.ATOL, cases.RTOL)
    ref_loss.backward()
    del actual_logits, actual_loss, ref_logits, ref_loss
    cleanup()
    for n,p in actual_parameters.items():
        assert p.grad is not None and ref_parameters[n].grad is not None
        compare(out, prefix+'-gradient-'+n, p.grad, ref_parameters[n].grad, cases.ATOL, cases.RTOL)
    for step in (1,2):
        # The second update deliberately reuses the fixed first-step gradients
        # to isolate optimizer state evolution; it is not a second training step.
        optimizer.step()
        reference.update(ref_parameters, ref_optimizers)
        optimizer.validate(require_gradients=True)
        for n,p in actual_parameters.items():
            q = ref_parameters[n]
            compare(out, prefix+'-update'+str(step)+'-'+n, p, q, cases.ATOL, cases.RTOL)
            ai = 0 if any(p is g['params'][0] for g in optimizer._optimizers[0].param_groups) else 1
            state = optimizer._optimizers[ai].state[p]
            other = ref_optimizers[ai].state[q]
            assert set(state) == set(other)
            for name,value in state.items():
                compare(out, prefix+'-state'+str(step)+'-'+n+'-'+name, value, other[name], cases.ATOL, cases.RTOL)
    # Short synthetic causal prefixes are a separate forward invariant; all
    # actual-shape forward/backward/update evidence above remains mandatory.
    with torch.no_grad():
        short = inputs[:1,:8].clone()
        changed = short.clone(); changed[:,4:] = (changed[:,4:]+1) % 50257
        original = model.primary_logits(short).values
        altered = model.primary_logits(changed).values
        compare(out, prefix+'-causality', original[:,:4], altered[:,:4], 0., 0.)
    del original, altered, native, ref_optimizers, ref_parameters, composer, codes, lengths
    cleanup()
    names = cases.INPUT_REJECTIONS + cases.OPTIMIZER_REJECTIONS
    if backend == 'cpu': names += cases.CPU_ONLY_REJECTIONS
    for name in names:
        invalid(name, model, optimizer, inputs, labels, publish, prefix)
    optimizer.validate(require_gradients=True)
    publish(prefix+'-complete.json', dict(backend=backend, variant=variant,
            forward_shape=list(cases.SHAPE), parameter_elements=expected_elements,
            parameter_tensors=expected_count, updates=2, rejected=len(names),
            tolerances=[cases.ATOL,cases.RTOL], scope='implementation_only_no_quality'))
    del model, optimizer, actual_parameters, inputs, labels
    cleanup()


def run(plan, out, publish):
    assert torch.__version__ == plan['runtime']['torch_version']
    assert torch.version.git_version == plan['runtime']['torch_git']
    assert torch.backends.mps.is_available(), 'Required MPS; no CPU fallback'
    torch.set_num_threads(4)
    initial_cpu, initial_mps = torch.random.get_rng_state().clone(), torch.mps.get_rng_state().clone()
    common = {}
    try:
        rows, books, expected = inspect_codebooks(plan, out, publish)
        for backend in cases.BACKENDS:
            for variant in cases.VARIANTS:
                one_model(plan, out, publish, rows, books, expected, backend, variant, common)
    finally:
        torch.random.set_rng_state(initial_cpu)
        torch.mps.set_rng_state(initial_mps)
    assert torch.equal(initial_cpu, torch.random.get_rng_state())
    assert torch.equal(initial_mps, torch.mps.get_rng_state())
    publish('result.json', dict(status='complete_original_actual_byte_input_model_proof',
            model_cases=6, full_size=[8,128,50257], roles=plan['roles'],
            caller_rng_restored=True, no_corpus_holdout_or_quality=True,
            numerical_adoption=False))
