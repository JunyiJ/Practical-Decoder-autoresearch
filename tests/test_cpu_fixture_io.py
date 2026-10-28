"""Small synthetic serialization checks, not model continuation retries."""
import io

import pytest
import torch

from src.research.cpu_fixture_io import consume_checkpoint_buffer


def test_loaded_tensors_optimizer_and_rng_survive_buffer_close():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(3487)
    weights = torch.randn(16, 8, generator=generator)
    payload = dict(model_math_version=2, parameter_layout_version=2,
        model_state={'weight': weights, 'alias': weights.view(8, 16)},
        optimizer_state={'exp_avg': weights.clone(), 'exp_avg_sq': weights.square()},
        training_state={'rng': generator.get_state(), 'ema': weights.clone()}, step=1)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    assert buffer.getbuffer().nbytes < 64*1024
    loaded = consume_checkpoint_buffer(buffer)
    assert buffer.closed
    for section in ('model_state', 'optimizer_state', 'training_state'):
        for name, original in payload[section].items():
            torch.testing.assert_close(loaded[section][name], original, rtol=0, atol=0)
    assert loaded['model_state']['weight'].untyped_storage().data_ptr() == loaded['model_state']['alias'].untyped_storage().data_ptr()
    loaded['model_state']['weight'][0, 0] += 1
    assert loaded['model_state']['alias'][0, 0] == weights[0, 0]+1
    assert loaded['training_state']['ema'][0, 0] == weights[0, 0]
    with pytest.raises(ValueError, match='already consumed'):
        consume_checkpoint_buffer(buffer)


@pytest.mark.parametrize('kind', ['incompatible', 'truncated'])
def test_invalid_synthetic_buffer_is_closed_and_error_propagates(kind):
    buffer = io.BytesIO()
    if kind == 'incompatible':
        torch.save({'model_math_version': 1}, buffer)
    else:
        buffer.write(b'not a serialized checkpoint')
    with pytest.raises(Exception):
        consume_checkpoint_buffer(buffer)
    assert buffer.closed


def test_retained_file_paths_are_not_accepted(tmp_path):
    path = tmp_path/'untouched.txt'
    path.write_text('retained evidence')
    with pytest.raises(TypeError, match='owned BytesIO'):
        consume_checkpoint_buffer(path)
    assert path.read_text() == 'retained evidence'
