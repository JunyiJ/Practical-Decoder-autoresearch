"""Prospective checkpoint cases, no numerical imports or assigned roles."""
BACKENDS = ('cpu', 'mps')
VARIANTS = ('native', 'raw_bytes', 'nonlexical_codes')
SHAPE = (8, 128, 50257)
REJECTIONS = (
    'extra_field', 'missing_field', 'kind', 'identity', 'binding', 'backend',
    'training_type', 'negative_step', 'bool_step', 'missing_parameter',
    'reordered_parameters', 'nonfinite_parameter', 'parameter_shape',
    'parameter_dtype', 'parameter_requires_grad', 'parameter_alias',
    'buffer_identity', 'child_kind', 'missing_child', 'group_name',
    'group_option', 'lr_mismatch', 'missing_state', 'extra_state_field',
    'state_shape', 'state_dtype', 'state_nonfinite', 'negative_variance',
    'adam_step', 'state_alias', 'cpu_rng_shape', 'cpu_rng_encoding', 'mps_rng_dtype',
)
IO_REJECTIONS = ('receipt_hash', 'receipt_size', 'receipt_kind', 'receipt_path', 'overwrite')
