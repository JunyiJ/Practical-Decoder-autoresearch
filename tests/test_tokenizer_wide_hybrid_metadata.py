"""Small and sparse disposable files only; no training/model/corpus access."""
import os

import pytest

from src.research import tokenizer_wide_hybrid_metadata as metadata
from src.research.tokenizer_wide_hybrid_retention import METADATA


def write(root,name,size):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as stream:stream.truncate(size)
    return path


def test_all_caps_reconcile_including_serving_and_atomic_control():
    caps=metadata.budgets()
    assert sum(caps.values())==METADATA==256*1024**2
    assert sum(v for k,v in caps.items() if '/inference/' in k)==12.75*1024**2
    assert sum(v for k,v in caps.items() if '/.retention/' in k)==15*1024**2
    assert sum(v for k,v in caps.items() if '/quality/' in k or '/pilots/' in k)==222.75*1024**2


@pytest.mark.parametrize('name,group',[
    ('.lock','control'),('STOP','control'),('control/state.json.tmp','control'),
    ('emergency/failure.json','emergency'),('evidence/meta/plan.json','evidence/meta'),
    ('evidence/pilots/02/request.json','evidence/pilots/02/other'),
    ('evidence/quality/23/result.json.tmp','evidence/quality/23/other'),
    ('evidence/quality/00/process/worker.log','evidence/quality/00/log'),
    ('evidence/quality/00/resource-events.jsonl','evidence/quality/00/events'),
    ('evidence/quality/00/resource-audit.json.tmp','evidence/quality/00/resources'),
    ('evidence/inference/3513/process/worker.log','evidence/inference/3513/log'),
    ('evidence/inference/3511/result.json','evidence/inference/3511/other'),
    ('evidence/.retention/rank/receipt.json','evidence/.retention/rank'),
    ('evidence/quality/00/checkpoints/final_001200_123.pt','evidence/quality/00/weights'),
    ('evidence/quality/00/checkpoints/final_001200_123.pt.tmp','evidence/quality/00/weights'),
])
def test_layout_classification(name,group):
    assert metadata.classify(name)==group


@pytest.mark.parametrize('name',[
    '../outside','/absolute','control//state.json','undeclared.json',
    'evidence/quality/24/result.json','evidence/pilots/03/result.json',
    'evidence/quality/1/result.json','evidence/inference/3514/result.json',
    'evidence/.retention/confirm/receipt.json','evidence/quality/00/checkpoints/extra.txt',
    'evidence/quality/00/raw.pt','evidence/inference/3511/checkpoints/raw.pt',
])
def test_unknown_or_misplaced_files_fail_closed(name):
    with pytest.raises(ValueError):metadata.classify(name)


def test_census_counts_all_metadata_and_reports_weights_separately(tmp_path):
    write(tmp_path,'evidence/quality/00/request.json',100)
    write(tmp_path,'evidence/quality/00/training-summary.json',200)
    write(tmp_path,'evidence/quality/00/checkpoints/raw.pt',300)
    write(tmp_path,'evidence/quality/00/checkpoints/h200.pt',400)
    write(tmp_path,'evidence/inference/3511/process/worker.log',500)
    write(tmp_path,'control/state.json',600)
    observed=metadata.snapshot(tmp_path)
    assert observed['metadata_bytes']==1400 and observed['checkpoint_logical_bytes']==700
    assert observed['remaining_metadata_bytes']==METADATA-1400
    assert observed['groups']['evidence/quality/00/other']==300
    assert observed['dispatch_authorized'] is False


@pytest.mark.parametrize('name,cap',[
    ('evidence/quality/00/result.json',256*1024),
    ('evidence/quality/00/process/worker.log',4*1024**2),
    ('evidence/quality/00/resource-events.jsonl',2*1024**2),
    ('evidence/quality/00/resource-audit.json',2*1024**2),
    ('evidence/inference/3511/process/worker.log',4*1024**2),
    ('evidence/inference/3511/report.json',256*1024),
    ('evidence/.retention/pilot/deletions.jsonl',5*1024**2),
    ('evidence/meta/plan.json',2*1024**2),
    ('control/state.json',3*1024**2),
    ('emergency/failure.json',512*1024),
])
def test_every_group_cap_rejects_one_byte_over_without_borrowing(tmp_path,name,cap):
    path=write(tmp_path,name,cap)
    assert metadata.snapshot(tmp_path)['metadata_bytes']==cap
    with path.open('ab') as stream:stream.write(b'x')
    with pytest.raises(ValueError,match='group allowance exceeded'):metadata.snapshot(tmp_path)


def test_atomic_replacement_counts_both_distinct_payloads(tmp_path):
    write(tmp_path,'control/state.json',2*1024**2)
    with pytest.raises(ValueError,match='overlap'):
        metadata.require_write_capacity(tmp_path,'control/state.json',2*1024**2)
    assert metadata.require_write_capacity(tmp_path,'control/state.json',1024**2)['additional_bytes']==1024**2
    write(tmp_path,'control/state.json.tmp',2*1024**2)
    with pytest.raises(ValueError,match='group allowance'):metadata.snapshot(tmp_path)


def test_exclusive_publisher_temporary_hardlink_counts_one_inode(tmp_path):
    original=write(tmp_path,'evidence/quality/00/result.json.tmp',200*1024)
    os.link(original,original.with_suffix(''))
    observed=metadata.snapshot(tmp_path)
    assert observed['metadata_bytes']==200*1024 and len(observed['files'])==2


@pytest.mark.parametrize('alias',['same_group','different_group','checkpoint','external'])
def test_other_hardlinks_are_rejected(tmp_path,alias):
    root=tmp_path/'campaign';root.mkdir()
    original=write(root,'control/state.json',10)
    if alias=='same_group':target=root/'control/duplicate.json'
    elif alias=='different_group':target=root/'emergency/failure.json';target.parent.mkdir()
    elif alias=='external':target=tmp_path/'external'
    else:
        original=write(root,'evidence/quality/00/checkpoints/raw.pt',10)
        target=original.parent/'h200.pt'
    os.link(original,target)
    with pytest.raises(ValueError,match='hard link|inode alias'):metadata.snapshot(root)


@pytest.mark.parametrize('kind',['symlink','directory_symlink','fifo','missing_root','root_symlink'])
def test_unscannable_outputs_rejected(tmp_path,kind):
    root=tmp_path/'campaign';root.mkdir()
    if kind=='symlink':(root/'STOP').symlink_to(tmp_path/'missing')
    elif kind=='directory_symlink':(root/'control').symlink_to(tmp_path,target_is_directory=True)
    elif kind=='fifo':os.mkfifo(root/'STOP')
    elif kind=='missing_root':root=tmp_path/'missing'
    else:
        alias=tmp_path/'alias';alias.symlink_to(root,target_is_directory=True);root=alias
    with pytest.raises(ValueError):metadata.snapshot(root)


def test_group_cap_covers_all_small_outputs_together(tmp_path):
    write(tmp_path,'evidence/pilots/00/request.json',200*1024)
    write(tmp_path,'evidence/pilots/00/metrics.jsonl',60*1024)
    with pytest.raises(ValueError,match='group allowance'):metadata.snapshot(tmp_path)


@pytest.mark.parametrize('size',[-1,True,1.5,None])
def test_write_requires_integer_byte_count(tmp_path,size):
    with pytest.raises(ValueError):metadata.require_write_capacity(tmp_path,'control/state.json',size)


def test_weight_write_cannot_use_metadata_capacity(tmp_path):
    with pytest.raises(ValueError,match='full trajectory gate'):
        metadata.require_write_capacity(tmp_path,'evidence/pilots/00/checkpoints/raw.pt',1)


def test_file_count_cap_includes_checkpoint_and_metadata(tmp_path,monkeypatch):
    monkeypatch.setattr(metadata,'MAX_FILES',1)
    write(tmp_path,'control/state.json',1)
    write(tmp_path,'evidence/pilots/00/checkpoints/raw.pt',1)
    with pytest.raises(ValueError,match='file-count cap'):metadata.snapshot(tmp_path)


def test_write_capacity_rejects_symlink_campaign_root(tmp_path):
    root=tmp_path/'campaign';root.mkdir()
    alias=tmp_path/'alias';alias.symlink_to(root,target_is_directory=True)
    with pytest.raises(ValueError,match='nonsymlink'):
        metadata.require_write_capacity(alias,'control/state.json',10)


def test_checkpoint_atomic_overlap_is_reported_without_cleanup_credit(tmp_path):
    write(tmp_path,'evidence/quality/00/checkpoints/raw.pt',1000)
    write(tmp_path,'evidence/quality/00/checkpoints/raw.pt.tmp',1200)
    report=metadata.snapshot(tmp_path)
    assert report['checkpoint_logical_bytes']==2200
    assert report['metadata_bytes']==0 and report['remaining_metadata_bytes']==METADATA
    assert report['dispatch_authorized'] is False
