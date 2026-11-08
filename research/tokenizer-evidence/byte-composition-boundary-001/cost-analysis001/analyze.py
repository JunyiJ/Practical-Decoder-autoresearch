"""Read completed resource pilots and count static costs; no model execution."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'research/tokenizer-evidence/byte-composition-boundary-001'

def receipt(path):
    return dict(path=str(path.relative_to(ROOT)), size=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def function(path, name):
    tree = ast.parse(path.read_text())
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    return matches[0]

def calls(node, expression):
    return sum(isinstance(child, ast.Call) and ast.unparse(child.func) == expression
               for child in ast.walk(node))

training = ROOT / 'research/byte_input_training_v2.py'
state = ROOT / 'research/byte_input_state_v2.py'
optimizer = ROOT / 'research/byte_input_model_v3/optimizer.py'
model = ROOT / 'research/byte_input_model_v3/model.py'
step = function(training, 'step')
assert calls(step, 'self.check') == 2
assert calls(function(training, 'check'), 'numerical.live') == 1
assert calls(function(state, 'live'), 'optimizer.validate') == 1
assert calls(function(optimizer, 'validate'), 'self.model.validate') == 1
assert calls(step, 'self.model.forward_train') == 1
assert calls(function(model, 'forward_train'), 'self.validate') == 1
assert calls(step, 'self.optimizer.validate') == 1
for operation in ('step', 'zero_grad'):
    assert calls(step, 'self.optimizer.' + operation) == 1
    assert calls(function(optimizer, operation), 'self.validate') == 1

rows = []
inputs = [training, state, optimizer, model,
          ROOT / 'research/byte_input_model_v3/contract.py',
          ROOT / 'research/byte_composer_v2.py', ROOT / 'research/byte_embedding_v1.py']
for arm in ('native', 'raw_bytes', 'nonlexical_codes'):
    path = BASE / 'pilot-math002' / (arm + '-complete.json')
    inputs.append(path)
    pilot = json.loads(path.read_text())
    assert pilot['completed_steps'] == 20 and not pilot['quality_selected']
    row = {key: pilot[key] for key in ('variant', 'mean_update_seconds',
           'max_update_seconds', 'max_mps_driver_bytes', 'validation_seconds')}
    row['parameters'] = 28492416 if arm == 'native' else 16351859
    row['targets_per_second'] = 1024 / row['mean_update_seconds']
    row['raw_archive_bytes'] = pilot['parent']['size']
    row['h200_archive_bytes'] = pilot['h200']['size']
    rows.append(row)
for row in rows:
    row['update_ratio_to_native'] = row['mean_update_seconds'] / rows[0]['mean_update_seconds']
    row['observed_driver_ratio_to_native'] = row['max_mps_driver_bytes'] / rows[0]['max_mps_driver_bytes']
codebook_bytes = 50257 * (130 + 1) * 8
parameter_saved = (28492416 - 16351859) * 4
result = dict(
    kind='saved_pilot_and_static_source_cost_analysis', no_new_training=True,
    no_quality_scores_used=True, inputs=[receipt(path) for path in inputs], pilot_rows=rows,
    static=dict(composer_parameters=725235, displaced_embedding_parameters=50257*256,
        parameter_reduction_fraction=1-16351859/28492416,
        fp32_parameter_bytes_saved=parameter_saved,
        dense_i64_codebook_bytes=codebook_bytes,
        parameters_plus_extra_codebook_byte_delta=codebook_bytes-parameter_saved,
        full_model_validation_calls_per_update_lower_bound=6,
        codebook_bytes_hashed_per_update_lower_bound=6*codebook_bytes,
        composed_updates_in_frozen_panel=12*1200,
        codebook_bytes_hashed_across_composed_updates_lower_bound=6*codebook_bytes*12*1200),
    limits=['Static byte counts are logical reads/conversions, not measured hardware traffic or elapsed cost.',
            'Pilot timings include validation and implementation overhead; no kernel-level attribution.',
            'Driver observations are step-end samples, not allocator peaks or total machine memory.',
            'Parameter-plus-codebook accounting excludes optimizer, activation and other buffers.',
            'No equal-wall-time quality comparison, long-run throughput estimate or adoption result.'])
print(json.dumps(result, indent=2, sort_keys=True))
