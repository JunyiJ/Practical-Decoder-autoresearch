"""Independent public-asset and Feistel equations. No candidate imports or I/O.

These functions are proof sources, not yet numerically admitted. The decoder
walks Unicode code points, and the permutation uses an interleaved list of
halves rather than calling or extracting the implementation under test.
"""
from array import array
import hashlib
import json
import sys


def raw_rows(encoder_bytes, merges_bytes):
    encoder = json.loads(encoder_bytes)
    assert len(encoder) == 50257
    direct = {x for x in range(256) if 33 <= x <= 126 or 161 <= x <= 172 or 174 <= x}
    missing = tuple(x for x in range(256) if x not in direct)

    def decode(spelling):
        result = bytearray()
        for character in spelling:
            point = ord(character)
            if point in direct:
                result.append(point)
            else:
                assert 256 <= point < 256 + len(missing)
                result.append(missing[point - 256])
        return bytes(result)

    rows = [None] * 50257
    for spelling, identifier in encoder.items():
        assert type(identifier) is int and 0 <= identifier < 50257
        assert rows[identifier] is None
        rows[identifier] = None if identifier == 50256 else decode(spelling)
    assert encoder['<|endoftext|>'] == 50256
    assert all(type(row) is bytes and 1 <= len(row) <= 128 for row in rows[:-1])
    assert len(set(rows[:-1])) == 50256
    # Independently recover token order from the merge program, then compare
    # the complete vector, rather than trusting the encoder's numeric values.
    merged = [bytes([x]) for x in sorted(direct)] + [bytes([x]) for x in missing]
    known = set(merged)
    lines = merges_bytes.decode('utf8').splitlines()
    assert len(lines) == 50001 and lines.pop(0) == '#version: 0.2'
    for line in lines:
        a, b = map(decode, line.split(' '))
        assert a in known and b in known and a + b not in known
        merged.append(a + b)
        known.add(a + b)
    assert merged == rows[:-1]
    return rows


def feistel(value, length, private_key, reverse=False):
    """Balanced eight-round permutation, independently expressed as a history."""
    bits = length * 4
    modulus = 2 ** bits
    size = (bits + 7) // 8
    halves = [value // modulus, value % modulus]
    if reverse:
        halves.reverse()
    rounds = reversed(range(8)) if reverse else range(8)
    for round_number in rounds:
        message = b''.join((b'rsi_byte_codebook_feistel_v1\0',
                            str(private_key).encode('ascii'), b'\0',
                            str(length).encode('ascii'), b'\0',
                            bytes((round_number,)), halves[-1].to_bytes(size, 'big')))
        digest = hashlib.shake_256(message).digest(size)
        halves.append(halves[-2] ^ (int.from_bytes(digest, 'big') % modulus))
    result = halves[-2:]
    if reverse:
        result.reverse()
    return result[0] * modulus + result[1]


def expected_codes(rows, variant, private_key):
    assert variant in ('raw_bytes', 'nonlexical_codes')
    assert (variant == 'raw_bytes' and private_key is None) or (
        variant == 'nonlexical_codes' and type(private_key) is int and 0 <= private_key < 2**63)
    codes = array('H')
    lengths = array('H')
    ordinals = {}
    distinct = set()
    for token, raw in enumerate(rows):
        if token == 50256:
            content = [259]
        else:
            if variant == 'nonlexical_codes':
                length = len(raw)
                ordinal = ordinals.get(length, 0)
                ordinals[length] = ordinal + 1
                encoded = feistel(ordinal, length, private_key)
                assert feistel(encoded, length, private_key, reverse=True) == ordinal
                raw = encoded.to_bytes(length, 'big')
            assert raw not in distinct
            distinct.add(raw)
            content = list(raw)
        length = len(content) + 2
        codes.extend([257] + content + [258] + [256] * (130 - length))
        lengths.append(length)
    assert len(codes) == 50257 * 130 and len(lengths) == 50257
    return codes, lengths


def little_bytes(values, typecode):
    data = array(typecode, values)
    if sys.byteorder != 'little':
        data.byteswap()
    return data.tobytes()


def staged_digest(values):
    digest = hashlib.sha256()
    for start in range(0, len(values), 65536):
        digest.update(little_bytes(values[start:start + 65536], 'q'))
    return digest.hexdigest()


def bijection_argument():
    """Symbolic scope: full domains are too large to enumerate for length128.

    Each round maps (L,R) to (R,L xor F(R)); the inverse maps (U,V) to
    (V xor F(U),U). XOR cancellation establishes a bijection for any F.
    Composition of eight such rounds is a bijection on every 8*length-bit
    domain. Finite tests still check every vocabulary ordinal, plus exhaustive
    length1/2 domains. The symbolic claim is distinct from finite coverage.
    """
    return 'Eight bijective XOR rounds compose; fixed-length serialization is injective.'
