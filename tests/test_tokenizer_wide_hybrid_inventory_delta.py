"""Bounded immutable inventory references; no training, weights or deletion."""
import copy
import json
from pathlib import Path

import pytest

from src.research import tokenizer_wide_hybrid_inventory_delta as codec
from src.research import tokenizer_wide_hybrid_retention_receipts as receipts
from src.research.protocol import digest,file_digest


def write(root,name,value):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n')
    return dict(path=name,sha256=file_digest(p))


def row(size=1):return dict(size=size,sha256='a'*64)


def setup(tmp_path):
    root=tmp_path.resolve();base=dict(plan_sha256='b'*64,context={'identity':'fixture'},
        inventory={'quality/00/a.json':row(),'quality/00/b.json':row(2)})
    ref=write(root,'meta/base.json',base)
    value=dict(base,inventory={'quality/00/a.json':row(3),'quality/01/new.json':row(4)},dispatch_authorized=False)
    return root,base,ref,value


def test_exact_roundtrip_preserves_original_physical_bytes_and_claims(tmp_path):
    root,base,ref,value=setup(tmp_path);old=(root/ref['path']).read_bytes();before=copy.deepcopy(value)
    wire=codec.encode_inventory(root,value,ref)
    assert wire[codec.MARKER]==codec.VERSION and wire['inventory']['removed']==['quality/00/b.json']
    assert wire['inventory']['full_count']==2 and wire['inventory']['full_sha256']==digest(value['inventory'])
    out=write(root,'meta/new.json',wire)
    assert receipts._document(root,out)==before and value==before
    assert (root/ref['path']).read_bytes()==old and receipts._document(root,ref)==base
    assert not receipts._document(root,out)['dispatch_authorized']


@pytest.mark.parametrize('bad',['version','extra','base_hash','base_path','count_bool','count','digest','duplicate_remove',
    'missing_remove','remove_and_add','redundant_add','negative_size','bool_size','bad_sha','path_escape','context','plan'])
def test_rehashed_bad_wire_documents_rejected(tmp_path,bad):
    root,base,ref,value=setup(tmp_path);wire=codec.encode_inventory(root,value,ref);delta=wire['inventory']
    if bad=='version':wire[codec.MARKER]='unknown'
    elif bad=='extra':delta['unreviewed']=True
    elif bad=='base_hash':delta['base']['sha256']='0'*64
    elif bad=='base_path':delta['base']['path']='../base.json'
    elif bad=='count_bool':delta['full_count']=True
    elif bad=='count':delta['full_count']=3
    elif bad=='digest':delta['full_sha256']='0'*64
    elif bad=='duplicate_remove':delta['removed']*=2
    elif bad=='missing_remove':delta['removed']=['missing.json']
    elif bad=='remove_and_add':delta['removed'].append('quality/00/a.json');delta['removed'].sort()
    elif bad=='redundant_add':delta['added']['quality/00/a.json']=row()
    elif bad=='negative_size':delta['added']['quality/00/a.json']['size']=-1
    elif bad=='bool_size':delta['added']['quality/00/a.json']['size']=True
    elif bad=='bad_sha':delta['added']['quality/00/a.json']['sha256']='bad'
    elif bad=='path_escape':delta['added']['../outside']=row()
    elif bad=='context':wire['context']={'identity':'other'}
    else:wire['plan_sha256']='c'*64
    with pytest.raises(ValueError):receipts._document(root,write(root,'meta/bad.json',wire))


def test_changed_missing_and_symlink_base_rejected(tmp_path):
    root,base,ref,value=setup(tmp_path);out=write(root,'meta/new.json',codec.encode_inventory(root,value,ref))
    p=root/ref['path'];old=p.read_bytes();p.write_bytes(b'{}')
    with pytest.raises(ValueError):receipts._document(root,out)
    p.unlink()
    with pytest.raises(ValueError):receipts._document(root,out)
    other=root/'other.json';other.write_bytes(old);p.symlink_to(other)
    with pytest.raises(ValueError,match='Symlink'):receipts._document(root,out)


def test_no_encoding_outside_metadata_and_no_reencoding(tmp_path):
    root,base,ref,value=setup(tmp_path);wire=codec.encode_inventory(root,value,ref)
    with pytest.raises(ValueError):codec.encode_inventory(root,wire,ref)
    with pytest.raises(ValueError):receipts._document(root,write(root,'quality/00/bad.json',wire))


def test_chain_depth_is_bounded_and_plain_documents_stay_compatible(tmp_path):
    root,base,ref,value=setup(tmp_path)
    for i in range(codec.MAX_DEPTH-1):
        ref=write(root,f'meta/chain-{i}.json',codec.encode_inventory(root,base,ref))
    assert receipts._document(root,ref)==base
    wire=codec.encode_inventory(root,base,ref)
    with pytest.raises(ValueError,match='depth'):receipts._document(root,write(root,'meta/too-deep.json',wire))
    assert receipts._document(root,write(root,'meta/plain.json',[1,True,None]))==[1,True,None]


def test_expanded_rows_and_bytes_bounded_without_large_allocation(tmp_path,monkeypatch):
    root,base,ref,value=setup(tmp_path)
    monkeypatch.setattr(codec,'MAX_ROWS',1)
    with pytest.raises(ValueError,match='Bounded inventory'):codec.encode_inventory(root,value,ref)
    monkeypatch.setattr(codec,'MAX_ROWS',2048);monkeypatch.setattr(codec,'MAX_INVENTORY_BYTES',10)
    with pytest.raises(ValueError,match='byte cap'):codec.encode_inventory(root,value,ref)


def test_chain_bytes_bounded_and_parent_rechecked_after_expansion(tmp_path,monkeypatch):
    root,base,ref,value=setup(tmp_path);out=write(root,'meta/new.json',codec.encode_inventory(root,value,ref))
    limit=(root/ref['path']).stat().st_size+(root/out['path']).stat().st_size
    monkeypatch.setattr(codec,'MAX_CHAIN_BYTES',limit-1)
    with pytest.raises(ValueError,match='chain byte'):receipts._document(root,out)
    monkeypatch.setattr(codec,'MAX_CHAIN_BYTES',limit)
    original=codec.decode_inventory
    def mutate(*a,**kw):
        result=original(*a,**kw);(root/out['path']).write_bytes(b'{}');return result
    monkeypatch.setattr(codec,'decode_inventory',mutate)
    with pytest.raises(ValueError,match='hash changed'):receipts._document(root,out)


def test_prospective_plan_declares_wire_format_and_all_reader_limits():
    from src.research.tokenizer_wide_hybrid_plan_v2 import layout_contract
    assert layout_contract()['inventory_wire']==dict(version=codec.VERSION,marker=codec.MARKER,
        field='inventory',full_documents_compatible=True,immutable_metadata_bases=True,
        max_rows=codec.MAX_ROWS,max_expanded_bytes=codec.MAX_INVENTORY_BYTES,
        max_reference_depth=codec.MAX_DEPTH,max_reference_bytes=codec.MAX_CHAIN_BYTES)
