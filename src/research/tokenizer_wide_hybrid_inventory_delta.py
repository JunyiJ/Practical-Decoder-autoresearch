"""Lossless inventory wire format; never an artifact audit or cleanup permission.

Only new immutable metadata documents may use this encoding. Readers must bind
both physical document hashes and the reconstructed inventory digest. Existing
bytes are never rewritten, and ordinary full inventory documents remain valid.
"""
from .protocol import digest
from .tokenizer_wide_hybrid_retention_receipts import _relative, _sha

VERSION='wide_hybrid_inventory_delta_v1'
MARKER='_inventory_encoding'
MAX_ROWS=2048
MAX_INVENTORY_BYTES=2*1024**2
MAX_DEPTH=16
MAX_CHAIN_BYTES=8*1024**2


def _inventory(value):
    if type(value) is not dict or len(value)>MAX_ROWS:raise ValueError('Bounded inventory map required')
    for name,binding in value.items():
        _relative(name)
        if len(name)>1024 or type(binding) is not dict or set(binding)!={'size','sha256'}:
            raise ValueError('Exact bounded inventory binding required')
        if type(binding['size']) is not int or binding['size']<0:raise ValueError('Exact inventory byte size required')
        _sha(binding['sha256'])
    # Bound the expanded representation independently of compressed input size.
    import json
    if len(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True))>MAX_INVENTORY_BYTES:
        raise ValueError('Expanded inventory byte cap exceeded')
    return value


def _identity(value,base):
    if not all(key in value and key in base and digest(value[key])==digest(base[key])
               for key in ('plan_sha256','context')):
        raise ValueError('Inventory base campaign identity differs')
    _sha(value['plan_sha256'])


def encode_inventory(root,value,base_ref):
    """Return a new wire document, leaving its source and base bytes unchanged."""
    from .tokenizer_wide_hybrid_retention_receipts import _document
    if type(value) is not dict or MARKER in value:raise ValueError('Unencoded report required')
    if not base_ref['path'].startswith('meta/'):raise ValueError('Inventory base must be immutable metadata')
    base=_document(root,base_ref);_identity(value,base)
    current=_inventory(value['inventory']);previous=_inventory(base['inventory'])
    wire=dict(value)
    wire[MARKER]=VERSION
    wire['inventory']=dict(base=dict(base_ref),removed=sorted(set(previous)-set(current)),
        added={n:v for n,v in current.items() if previous.get(n)!=v},
        full_count=len(current),full_sha256=digest(current))
    return wire


def decode_inventory(value,load_base):
    """Reconstruct all rows; downstream audits still verify actual file bytes."""
    if value.get(MARKER)!=VERSION:raise ValueError('Unknown inventory wire version')
    delta=value.get('inventory')
    if type(delta) is not dict or set(delta)!={'base','removed','added','full_count','full_sha256'}:
        raise ValueError('Exact inventory delta fields required')
    ref=delta['base']
    if type(ref) is not dict or set(ref)!={'path','sha256'} or not _relative(ref['path']).startswith('meta/'):
        raise ValueError('Exact immutable inventory base reference required')
    _sha(ref['sha256']);_sha(delta['full_sha256'])
    base=load_base(ref);_identity(value,base)
    previous=_inventory(base['inventory']);added=_inventory(delta['added']);removed=delta['removed']
    if (type(removed) is not list or any(type(n) is not str for n in removed)
            or removed!=sorted(set(removed)) or not set(removed)<=set(previous)
            or set(removed)&set(added) or any(previous.get(n)==v for n,v in added.items())):
        raise ValueError('Noncanonical inventory delta')
    current={n:v for n,v in previous.items() if n not in set(removed)};current.update(added)
    _inventory(current)
    if type(delta['full_count']) is not int or delta['full_count']!=len(current) or digest(current)!=delta['full_sha256']:
        raise ValueError('Reconstructed inventory count/hash differs')
    result=dict(value);result.pop(MARKER);result['inventory']=current
    return result
