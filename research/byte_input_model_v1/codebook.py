"""Builder-owned fixed-vocabulary input codes; source only, no I/O or RNG."""
from array import array
from dataclasses import dataclass, field
import hashlib
import json
import sys

NATIVE_HASH = 'bdc599f0a294cff99d60cf83aed3870857b30bc0d1d06164cae137831b55257c'
ENCODER_HASH = '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783'
MERGES_HASH = '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5'
_FACTORY_TOKEN = object()
DOMAIN = b'rsi_byte_codebook_feistel_v1\0'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def _key(value, length, private_key):
    require(type(value) is int and type(length) is int and 1 <= length <= 128
            and 0 <= value < 1 << (8*length), 'Bounded integer permutation input')
    require(type(private_key) is int and 0 <= private_key < 2**63, 'Explicit bounded private code key')


def _round(right, length, private_key, index):
    bits = 4*length
    size = (bits+7)//8
    message = (DOMAIN + str(private_key).encode() + b'\0' + str(length).encode()
               + b'\0' + bytes([index]) + right.to_bytes(size, 'big'))
    return int.from_bytes(hashlib.shake_256(message).digest(size), 'big') & ((1 << bits)-1)


def permute(value, length, private_key):
    _key(value, length, private_key)
    bits = 4*length; mask = (1 << bits)-1
    left, right = value >> bits, value & mask
    for index in range(8):
        left, right = right, left ^ _round(right, length, private_key, index)
    return (left << bits) | right


def inverse(value, length, private_key):
    _key(value, length, private_key)
    bits = 4*length; mask = (1 << bits)-1
    left, right = value >> bits, value & mask
    for index in range(7, -1, -1):
        left, right = right ^ _round(left, length, private_key, index), left
    return (left << bits) | right


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _little(values):
    copied = array(values.typecode, values)
    if sys.byteorder != 'little':
        copied.byteswap()
    return copied.tobytes()


def _i64_sha(values):
    result = hashlib.sha256()
    for start in range(0, len(values), 131072):
        result.update(_little(array('q', values[start:start+131072])))
    return result.hexdigest()


@dataclass(frozen=True)
class Codebook:
    variant: str
    private_key: object
    codes: bytes
    lengths: bytes
    codes_i64_sha256: str
    lengths_i64_sha256: str
    _factory_token: object = field(repr=False, compare=False)
    _identity: tuple = field(init=False, repr=False)

    def __post_init__(self):
        require(self._factory_token is _FACTORY_TOKEN, 'Builder-owned codebook required')
        self._validate_fields()
        object.__setattr__(self, '_identity', self._signature())

    def _validate_fields(self):
        require(type(self.variant) is str and self.variant in ('raw_bytes', 'nonlexical_codes'), 'Own code variant')
        require((self.variant == 'raw_bytes' and self.private_key is None) or
                (self.variant == 'nonlexical_codes' and type(self.private_key) is int
                 and 0 <= self.private_key < 2**63), 'Variant-specific private key')
        require(type(self.codes) is bytes and len(self.codes) == 13066820 and
                type(self.lengths) is bytes and len(self.lengths) == 100514, 'Exact packed codebook shape')
        for value in (self.codes_i64_sha256, self.lengths_i64_sha256):
            require(type(value) is str and len(value) == 64 and
                    all(c in '0123456789abcdef' for c in value), 'Exact staged-buffer digest')

    def _signature(self):
        return ('rsi_byte_codebook_v1', self.variant, self.private_key, NATIVE_HASH,
                50257, 130, _sha(self.codes), _sha(self.lengths),
                self.codes_i64_sha256, self.lengths_i64_sha256)

    @property
    def identity(self):
        return self._identity

    def validate(self):
        require(self._factory_token is _FACTORY_TOKEN, 'Foreign codebook factory identity')
        self._validate_fields()
        require(self._signature() == self._identity, 'Mutated codebook identity or payload')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate encoder key')
        result[key] = value
    return result


def build_codebook(encoder_bytes, merges_bytes, variant, private_key=None):
    require(type(encoder_bytes) is bytes and type(merges_bytes) is bytes and
            len(encoder_bytes) == 1042301 and len(merges_bytes) == 456318,
            'Exact supplied public asset bytes')
    require(_sha(encoder_bytes) == ENCODER_HASH and _sha(merges_bytes) == MERGES_HASH,
            'Pinned public assets before parsing')
    require(type(variant) is str and variant in ('raw_bytes', 'nonlexical_codes'), 'Exact input code variant')
    require((variant == 'raw_bytes' and private_key is None) or
            (variant == 'nonlexical_codes' and type(private_key) is int and 0 <= private_key < 2**63),
            'Explicit variant-specific private code key')
    visible = list(range(33, 127))+list(range(161, 173))+list(range(174, 256))
    omitted = [b for b in range(256) if b not in visible]
    alphabet = {chr(b): b for b in visible}
    alphabet.update({chr(256+i): b for i, b in enumerate(omitted)})
    encoder = json.loads(encoder_bytes.decode('utf8', errors='strict'), object_pairs_hook=_unique)
    require(type(encoder) is dict and len(encoder) == 50257 and
            all(type(k) is str and type(v) is int for k, v in encoder.items()) and
            set(encoder.values()) == set(range(50257)) and encoder.get('<|endoftext|>') == 50256,
            'Full dense public IDs and explicit special')
    regular = {}
    for spelling, token in encoder.items():
        if token == 50256:
            continue
        require(spelling and all(c in alphabet for c in spelling), 'Canonical public byte alphabet')
        raw = bytes(alphabet[c] for c in spelling)
        require(raw not in regular and 1 <= len(raw) <= 128, 'Unique bounded regular bytes')
        regular[raw] = token
    lines = merges_bytes.decode('utf8', errors='strict').splitlines()
    require(len(lines) == 50001 and lines[0] == '#version: 0.2', 'Exact public merge stream')
    ranks = {bytes([b]): i for i, b in enumerate(visible+omitted)}
    for rank, line in enumerate(lines[1:], 256):
        pair = line.split(' ')
        require(len(pair) == 2 and all(pair) and all(all(c in alphabet for c in x) for x in pair), 'Exact merge components')
        left, right = (bytes(alphabet[c] for c in x) for x in pair)
        require(left in ranks and right in ranks and left+right not in ranks, 'Ordered unique merge operands')
        ranks[left+right] = rank
    require(ranks == regular and len(regular) == 50256, 'Full public merge agreement')
    native_hash = _sha(json.dumps(sorted((raw.hex(), idx) for raw, idx in regular.items())).encode())
    require(native_hash == NATIVE_HASH, 'Exact native vocabulary identity')
    by_id = [None]*50256
    for raw, token in regular.items():
        by_id[token] = raw
    codes = array('H', [256])*(50257*130)
    lengths = array('H', [0])*50257
    require(codes.itemsize == 2 and array('q').itemsize == 8, 'Exact packed/staged integer widths')
    ordinals = {}; seen = set()
    for token, raw in enumerate(by_id):
        length = len(raw)
        if variant == 'nonlexical_codes':
            ordinal = ordinals.get(length, 0); ordinals[length] = ordinal+1
            value = permute(ordinal, length, private_key)
            require(inverse(value, length, private_key) == ordinal, 'Exact full inverse control mapping')
            raw = value.to_bytes(length, 'big')
        require(raw not in seen, 'Distinct whole token codes')
        seen.add(raw)
        row = array('H', [257, *raw, 258])
        codes[token*130:token*130+len(row)] = row
        lengths[token] = len(row)
    codes[50256*130:50256*130+3] = array('H', [257, 259, 258]); lengths[50256] = 3
    require(len(codes) == 50257*130 and min(lengths) == 3 and max(lengths) == 130,
            'Canonical fixed byte codes')
    return Codebook(variant, private_key, _little(codes), _little(lengths),
                    _i64_sha(codes), _i64_sha(lengths), _FACTORY_TOKEN)
