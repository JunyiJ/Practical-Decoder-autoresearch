"""Independent NEW TOP saved-report reconciliation SOURCE; no numerical replay.

Decode saved coordinate arrays/masks only. Parameter vector digests do not permit
recomputation of unretained parameter values; that limit is explicit in output.
"""
import array
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import zlib
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')
MAX_FILE = 67108864
MAX_COORDINATES = 12865792


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''): digest.update(chunk)
    return digest.hexdigest()


def pairs_unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result,'Duplicate JSON key'); result[key] = value
    return result


def read(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_FILE,'Bounded plain JSON file required')
    return json.loads(path.read_text(),object_pairs_hook=pairs_unique,
                      parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON literal')))


def write(name, value):
    raw = (json.dumps(value,sort_keys=True,allow_nan=False) + '\n').encode()
    require(sum(p.stat().st_size for p in R.iterdir() if p.is_file()) + len(raw) + 65536 < 2097152,'Review2MiB cap')
    with (R / name).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def same(a,b):
    if type(a) is not type(b): return False
    if type(a) is dict: return a.keys() == b.keys() and all(same(a[k],b[k]) for k in a)
    if type(a) is list: return len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
    return a == b


def hex_digest(value): return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def expand(encoded, expected):
    require(type(encoded) is str and 0 <= expected <= MAX_COORDINATES * 8,'Bounded encoded coordinates')
    compressed = base64.b64decode(encoded,validate=True)
    decoder = zlib.decompressobj(); raw = decoder.decompress(compressed,expected + 1)
    require(len(raw) == expected and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,'Exact bounded compressed payload')
    return raw


def dense(value, shape, digest):
    require(same(value['shape'],shape) and type(value['elements']) is int and value['elements'] == math.prod(shape),'Dense shape/count')
    require(value['encoding'] == 'zlib-base64-little-endian' and sys.byteorder == 'little','Exact coordinate encoding')
    require(value['dtype'] in ('torch.float32','torch.float64'),'Only declared CPU float coordinates')
    width, code = (4,'f') if value['dtype'] == 'torch.float32' else (8,'d')
    raw = expand(value['coordinates'],math.prod(shape) * width)
    require(sha_bytes(raw) == value['sha256'] == digest,'Dense coordinate digest')
    result = array.array(code); result.frombytes(raw)
    require(all(math.isfinite(x) for x in result),'Finite dense coordinates')
    return result


def sha_bytes(raw): return hashlib.sha256(raw).hexdigest()


def compare(value, shape, tolerance, counters, *, full=False):
    count = math.prod(shape)
    require(0 < count <= MAX_COORDINATES and same(value['shape'],shape) and type(value['coordinates']) is int and value['coordinates'] == count,'Complete comparison shape/count')
    require(type(value['atol']) is float and type(value['rtol']) is float and (value['atol'],value['rtol']) == tuple(tolerance),'Fixed predicates')
    require(value['mask_encoding'] == 'zlib-base64-bool8','Boolean mask encoding')
    mask = expand(value['all_coordinate_mask'],count)
    require(all(byte in (0,1) for byte in set(mask)),'Every saved predicate is bool8')
    failures = mask.count(0)
    require(type(value['passed']) is bool and value['passed'] == (failures == 0),'Full-mask flag')
    require(type(value['failed_coordinates']) is int and value['failed_coordinates'] == failures and failures == 0 and value['first_64_failures'] == [],'All original elementwise predicates must pass')
    require(hex_digest(value['actual_sha256']) and hex_digest(value['expected_sha256']),'Both vector identities')
    require(type(value['max_abs']) is float and math.isfinite(value['max_abs']) and value['max_abs'] >= 0.,'Finite diagnostic maximum')
    if full:
        a = dense(value['actual'],shape,value['actual_sha256']); e = dense(value['expected'],shape,value['expected_sha256'])
        maximum = 0.; atol,rtol = tolerance
        for index,(actual,expected) in enumerate(zip(a,e)):
            error = abs(actual-expected); maximum = max(maximum,error)
            predicate = math.isfinite(actual) and math.isfinite(expected) and error <= atol + rtol * abs(expected)
            require(predicate == bool(mask[index]),'Independent saved-coordinate predicate differs')
        require(maximum == value['max_abs'],'Saved dense diagnostic maximum differs')
        counters['independently_recomputed_dense_coordinates'] += count
    else:
        require('actual' not in value and 'expected' not in value,'Parameter/digest-only evidence must not be relabeled dense')
        counters['mask_and_digest_only_coordinates'] += count
    counters['comparison_masks'] += 1


def shapes():
    result = {'transformer.wte.weight':[50257,256]}
    attention = {'dkv':[64,256],'uk':[256,64],'uv':[256,64],'dq':[64,256],
                 'uq':[256,64],'qr':[256,64],'kr':[32,256],'out':[256,256]}
    for layer in range(4):
        prefix = 'transformer.h.' + str(layer) + '.'
        result[prefix+'ln1.weight'] = [256]; result[prefix+'ln2.weight'] = [256]
        for name,shape in attention.items():
            result[prefix+'attn.'+name+'.weight'] = shape
            result[prefix+'attn.'+name+'.bias'] = [shape[0]]
        for name,shape in {'linear_gate':[672,256],'linear_up':[672,256],'linear_down':[256,672]}.items():
            result[prefix+'mlp.'+name+'.weight'] = shape
    result.update({'transformer.ln_f.weight':[256],'transformer.ln_f.bias':[256],'lm_head.weight':[50257,256]})
    require(len(result) == 88 and sum(math.prod(s) for s in result.values()) == 28492416,'Independent native shape literal inventory')
    return result


def descriptor(value, shape=None, dtype=None):
    require(type(value) is list and len(value) == 3 and hex_digest(value[2]),'Tensor descriptor')
    require(type(value[0]) is str and type(value[1]) is list and all(type(n) is int and n >= 0 for n in value[1]),'Descriptor shape/dtype')
    if shape is not None: require(value[1] == shape,'Descriptor expected shape')
    if dtype is not None: require(value[0] == dtype,'Descriptor expected dtype')


def snapshot(value, inventory, raw_shape, cfg, spec):
    require(same(value['cfg'],cfg) and value['spec'] == spec and value['modes'] == [True,True],'Genuine config/spec/modes')
    require(value['optimizer_state_sizes'] == [0,0] and value['foreign_parameters'] == [],'Pre-update state/ownership')
    parameters = value['parameters_and_populated_gradslots']; own = value['own_inventory']
    names = [item[0] for item in parameters]
    require(len(names) == len(set(names)) == len(inventory) and set(names) == set(inventory),'Complete canonical parameter names')
    require([item[0] for item in own] == names,'Own canonical object order')
    ids = []
    for (name,p,g), (own_name,ident,pointer,shape,dtype,trainable) in zip(parameters,own):
        descriptor(p,inventory[name],'torch.float32'); descriptor(g,inventory[name],'torch.float32')
        require(own_name == name and shape == inventory[name] and dtype == 'torch.float32' and trainable is True,'Own parameter shape/type')
        require(type(ident) is int and type(pointer) is int and ident > 0 and pointer > 0,'Parameter object/storage identity'); ids.append(ident)
    require(len(set(ids)) == len(ids) and len({row[2] for row in own}) == len(own),'Separate parameter objects/storage')
    groups = value['groups']; require(len(groups) == len(inventory),'One complete group per tensor')
    group_ids = []
    for pairs in groups:
        group = dict(pairs); require(len(group['params']) == 1,'One parameter/group'); group_ids.extend(group['params'])
    require(len(set(group_ids)) == len(ids) and set(group_ids) == set(ids),'Optimizer parameter identity coverage')
    descriptor(value['raw'],raw_shape,'torch.int64'); descriptor(value['CPU_RNG']); descriptor(value['MPS_RNG'])
    require(value['extra_input_payload'] is None,'Valid math input payload')
    for name,item in value['buffers']: require(type(name) is str,'Buffer name'); descriptor(item)
    return names


def guard(value, seconds, *, mps=False, outer=False):
    require(value['status'] == 'finished' and value['reason'] is None,'Original guard must finish')
    require(value['elapsed_seconds'] < seconds and value['limits']['timeout_seconds'] == seconds,'Original time gate')
    require(value['limits']['rss_limit_bytes'] == 4294967296 and value['limits']['reserve_bytes'] == 8589934592,'Original RSS/reserve gate')
    observation = value['observation']; require(observation['peak_rss_bytes'] < 4294967296 and observation['free_bytes'] >= 8589934592,'Actual guard observations')
    if mps:
        require(value['limits']['mps_driver_limit_bytes'] == 6442450944 and observation['peak_observed_mps_driver_bytes'] < 6442450944,'Actual MPS guard')
    if outer:
        require(observation['elapsed_since_entry_seconds'] < 600 and observation['output_bytes'] < 536870912 and observation['unsafe_output_entry'] is False,'Whole entry/output guard')


def review(plan):
    numerical_path = ROOT / plan['numerical_plan']; require(sha(numerical_path) == plan['numerical_plan_sha256'],'Original numerical plan SHA')
    numerical = read(numerical_path); run = numerical_path.parent
    contract = numerical['report_contract']; tolerance = contract['tolerances']; expected_counts = numerical['expected_reports']
    require(expected_counts == {'arms':10,'transactions':89,'tail_pairs':4,'updates':4},'Exact finite counts')
    expected_files = set(contract['output_files'])
    require(len(expected_files) == 126 and {p.name for p in run.iterdir()} == expected_files,'Complete exact original flat output set')
    require(all(p.is_file() and not p.is_symlink() for p in run.iterdir()),'Plain owned original reports')
    require(sum(p.stat().st_size for p in run.iterdir()) < 536870912,'Whole output cap')
    require(read(run/'STOP')['status'] == 'complete_NEW_TOP_model_math_finite_scope_only','Inner completion')
    require(read(run/'outer-STOP')['status'] == 'complete_original_NEW_TOP_outer_guarded_math_entry','Outer completion')
    for name in ('attempt.json','outer-attempt.json'): require(read(run/name)['plan_sha256'] == plan['numerical_plan_sha256'],'Same original command identity')
    guard(read(run/'mps-guard.json'),600,mps=True); guard(read(run/'outer-guard.json'),600,outer=True)
    result = read(run/'result.json'); preservation = read(run/'preservation-and-counts.json')
    require(result['counts'] == preservation['counts'] == expected_counts and result['no_state_cache_scoring_quality_qualification'] is True,'Final counts/scope')
    require(preservation['CPU_RNG'] is True and preservation['MPS_RNG'] is True and preservation['passed'] is True,'Caller global RNG preservation')
    admission_path = ROOT/numerical['runtime_admission_review']; admission = read(admission_path)
    require(admission['accepted'] is True and admission['status'] == 'complete_NEW_TOP_model_math_runtime_admission' and admission['numerical_plan_sha256'] == plan['numerical_plan_sha256'],'Original full admission identity')
    guard(read(admission_path.parent/'guard.json'),120)
    require(read(admission_path.parent/'STOP')['status'] == 'complete_original_NEW_TOP_runtime_admission','Completed original admission gate')
    for name,digest in numerical['metadata_pins'].items(): require(sha(ROOT/name) == digest,'Original source/census acceptance metadata pin')
    census_path = ROOT/numerical['initializer_census']; census = read(census_path)
    require(sha(census_path) == numerical['initializer_census_sha256'] and census['roles'] == {'initializer':numerical['initializer']} and census['prior_matches'] == 0 and census['accepted'] is True,'Exclusive original initializer identity')
    require(read(census_path.parent/'STOP')['status'] == 'complete_original_NEW_TOP_initializer_census','Original unused census completed')
    guard(read(census_path.parent/'guard.json'),120)
    native = shapes(); counters = {'comparison_masks':0,'independently_recomputed_dense_coordinates':0,'mask_and_digest_only_coordinates':0}
    math_ids,updates,transactions,tail_pairs = [],[],[],[]
    for item in contract['arms']:
        ident,variant,case,batch,context,window = item
        require(ident not in math_ids,'Unique arm ID'); math_ids.append(ident)
        value = read(run/(ident+'-math.json')); guard(read(run/(ident+'-phase-guard.json')),120)
        inventory = dict(native)
        if variant != 'native': inventory['aux_weight'] = [50257,256]
        spec = "Spec(variant=" + repr(variant) + ", window=" + str(window) + ")"
        caller = value['caller_preservation']; require(same(caller['before'],caller['after']),'ALL caller model/grad/input/group/config/RNG descriptors preserved')
        names = snapshot(caller['before'],inventory,[batch,context+window],numerical['native_config'],spec)
        for key in ('raw_preserved','identity','primary_identity','training_kind','primary_kind','passed'): require(value[key] is True,'Required original valid-entry boolean')
        init = value['initialization']; require(init['RNG_draws_exact'] is True and init['native_count_exact'] is True,'Native CPU initialization draws/count')
        require(set(init['parameters']) == set(native),'ALL88 initialization tensors')
        for name,shape in native.items(): compare(init['parameters'][name],shape,tolerance['exact'],counters)
        full_shape = [batch,context,50257]
        compare(value['genuine_primary'],full_shape,tolerance['exact'],counters,full=True)
        compare(value['uncached_primary'],full_shape,tolerance['exact'],counters)
        for name in ('primary_loss','total_loss'): compare(value[name],[],tolerance['loss'],counters,full=True)
        require(value['optimizer_groups'] == {'passed':True,'Muon':44,'AdamW':44 if variant == 'native' else 45},'Own grouping')
        if variant == 'native': require(value['no_aux'] is True,'Native no auxiliary')
        else:
            require(init['separate_storage'] is True and value['gradient_coverage'] is True and value['aux_nonzero_gradient'] is True,'Separate trainable readout/all gradients')
            compare(init['aux'],[50257,256],tolerance['exact'],counters)
            compare(value['auxiliary_loss'],[],tolerance['loss'],counters,full=True)
            compare(value['auxiliary_logits'],full_shape,tolerance['logits'],counters,full=True)
            for name in ('primary_logit_gradient','auxiliary_logit_gradient'): compare(value[name],full_shape,tolerance['logit_gradient'],counters,full=True)
            compare(value['summed_hidden_gradient'],[batch,context,256],tolerance['hidden_gradient'],counters,full=True)
            graph = value['one_shared_final_norm']; require(graph == {'passed':True,'primary_finalnorm_nodes':1,'aux_finalnorm_nodes':1,'shared_finalnorm_nodes':1,'hooks_or_candidate_hidden_recompute':False},'One shared final-normalization graph')
            require(set(value['parameter_gradients']) == set(inventory),'ALL89 independent parameter-gradient predicates')
            for name,shape in inventory.items(): compare(value['parameter_gradients'][name],shape,tolerance['parameter_gradient'],counters)
            rows = value['represented_targets']; require(len(rows) == batch*context,'All positions represented')
            for row in rows:
                ids = [pair[0] for pair in row]; require(1 <= len(row) <= window and len(ids) == len(set(ids)),'Distinct nearest token support')
                require(all(type(token) is int and 0 <= token < 50257 and type(mass) is float and math.isfinite(mass) and mass >= 0 for token,mass in row),'Finite represented target descriptors')
            if case == 'repeated_tail_001':
                updates.append(ident); update = read(run/(ident+'-update.json'))
                compare(update['head_update'],[50257,256],tolerance['head_update'],counters)
                require(update['aux_changed'] is True and update['gradient_exposure'] == names and update['Muon_groups'] == 44 and update['AdamW_groups'] == 45,'Full finite update exposure/change')
                require(type(update['grad_norm_before_clip']) is float and math.isfinite(update['grad_norm_before_clip']) and update['grad_norm_before_clip'] > 0,'Finite full clip norm')
    for ident in contract['negative_ids']:
        require(ident not in transactions,'Unique negative ID'); transactions.append(ident); value = read(run/(ident+'.json'))
        require(value['id'] == ident and value['rejected_ValueError'] is True and value['passed'] is True and type(value['error']) is str and value['error'],'Concrete original ValueError rejection')
        require(same(value['before'],value['after']),'ALL invalid caller-owned state descriptors preserved')
        require(value['before']['optimizer_state_sizes'] == [0,0],'No preflight optimizer mutation')
    for ident in contract['tail_ids']:
        require(ident not in tail_pairs,'Unique tail pair'); tail_pairs.append(ident)
        compare(read(run/(ident+'.json')),[2,3,50257],tolerance['exact'],counters,full=True)
    require([len(math_ids),len(transactions),len(updates),len(tail_pairs)] == [10,89,4,4],'ALL original report coverage')
    for path,digest in numerical['sources'].items(): require(sha(ROOT/path) == digest,'Original execution source changed: '+path)
    return {'accepted':True,'status':'complete_NEW_TOP_original_saved_report_reconciliation_finite_scope_only',
            'counts':expected_counts,'coordinate_evidence':counters,'no_Torch_model_oracle_target_objective_scoring_or_RNG_replay':True,
            'parameter_comparison_limit':'Complete masks/digests/coverage reconciled; unretained parameter vectors cannot be independently recomputed',
            'numerical_plan_sha256':plan['numerical_plan_sha256'],'no_full_batch_state_cache_quality_qualification':True}


def main():
    require(len(sys.argv) == 2 and sha(R/'plan.json') == sys.argv[1],'Exact original reviewer plan SHA')
    require({p.name for p in R.iterdir()} == {'plan.json'},'Original saved-report review is once-only')
    plan = read(R/'plan.json'); require(shutil.disk_usage(ROOT).free >= 10739515392,'Review entry capacity')
    require(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules),'No Torch import')
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write('attempt.json',{'pid':os.getpid(),'plan_sha256':sys.argv[1],'no_retry':True,'no_replay':True})
        try:
            with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
                result = review(plan)
                write('review.json',result)
        except BaseException as error:
            if not (R/'review.json').exists(): write('review.json',{'accepted':False,'status':'original_NEW_TOP_metadata_incomplete_or_rejected','error':repr(error),'no_numerical_replay':True})
            write('STOP',{'status':'closed_unqualified_original_NEW_TOP_saved_report_review','no_retry':True,'no_replay':True})
            raise
        else:
            write('STOP',{'status':'complete_original_NEW_TOP_saved_report_review','no_retry':True,'no_replay':True})


if __name__ == '__main__': main()
