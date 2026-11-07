"""Prospective finite integration cases; no initialization or corpus access.

Case construction and predicates must be frozen with the eventual driver before
any numerical entry. No role values are assigned by these source declarations.
"""
VARIANTS = ('native', 'raw_bytes', 'nonlexical_codes')
BACKENDS = ('cpu', 'mps')
ATOL, RTOL = 4e-6, 4e-5
SHAPE = (8, 128, 50257)
LIMITS = dict(rss_bytes=4294967296, mps_driver_bytes=6442450944,
              reserve_bytes=8589934592, extra_headroom_bytes=2147483648,
              output_bytes=17179869184, timeout_seconds=3600, cpu_threads=4)
ROLES = ('native_initializer', 'composer_initializer', 'nonlexical_key')
# Every name denotes a distinct prospective rejection transaction. Snapshot
# values/identity before the rejected call, after injecting the invalidity.
# Restoration is fixture cleanup, never a pass predicate or an update retry.
INPUT_REJECTIONS = (
    'ids_float', 'ids_bool', 'ids_int32', 'ids_list', 'ids_rank1', 'ids_rank3',
    'ids_empty_batch', 'ids_batch9', 'ids_time129', 'ids_empty_time',
    'ids_negative', 'ids_50257', 'labels_wrong_shape', 'labels_float',
    'labels_negative', 'labels_50257', 'foreign_device_ids', 'foreign_device_labels',
    'autocast', 'changed_config', 'changed_spec', 'replaced_parameter',
    'nonfinite_parameter', 'extra_parameter', 'replaced_buffer', 'mutated_buffer',
    'model_forward', 'model_export', 'model_load', 'optimizer_export', 'optimizer_load',
)
OPTIMIZER_REJECTIONS = (
    'missing_gradient', 'nan_gradient', 'inf_gradient', 'foreign_state_key',
    'missing_state_parameter', 'missing_state_field', 'extra_state_field',
    'nan_momentum', 'inf_adam_average', 'negative_second_moment',
    'wrong_state_shape', 'wrong_state_dtype', 'wrong_state_device',
    'state_alias_parameter', 'state_alias_gradient', 'state_alias_other_state',
    'adam_step_nan', 'adam_step_wrong', 'adam_step_rank1', 'state_before_first_step',
    'replaced_child', 'child_hook', 'changed_group_option', 'inconsistent_base_lr',
    'duplicate_parameter_group', 'foreign_parameter_group',
)
CPU_ONLY_REJECTIONS = ('ids_sparse', 'ids_meta', 'state_sparse', 'state_meta', 'state_float64')


def inputs(torch, rows, device):
    # All synthetic IDs; explicitly include special, longest token, duplicate,
    # single-byte extremes and NUL. No corpus or holdout is read.
    maximum = max(range(50256), key=lambda i: len(rows[i]))
    nul = rows.index(b'\0')
    last_byte = rows.index(b'\xff')
    ids = (torch.arange(1024, dtype=torch.int64) * 49) % 50256
    ids[:8] = torch.tensor([50256, maximum, nul, last_byte, 0, 0, 1, 2])
    labels = (ids * 31 + 17) % 50257
    return ids.reshape(8, 128).to(device), labels.reshape(8, 128).to(device)
