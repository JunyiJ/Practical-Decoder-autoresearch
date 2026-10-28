"""Consume scratch CPU fixture bytes; never modifies a retained checkpoint file."""
import io

from ..utils.checkpoint import read_checkpoint


def consume_checkpoint_buffer(buffer):
    """Fully deserialize on CPU, then close the owned BytesIO before returning.

    Only for scratch test serialization: no mmap, files, production resume hook,
    or numerical validation dispatcher. The caller relinquishes this buffer.
    Closing releases its serialized storage; it does not guarantee lower RSS.
    """
    if type(buffer) is not io.BytesIO:
        raise TypeError('CPU fixture requires an owned BytesIO buffer')
    if buffer.closed:
        raise ValueError('CPU fixture buffer was already consumed')
    try:
        buffer.seek(0)
        return read_checkpoint(buffer, map_location='cpu')
    finally:
        buffer.close()
