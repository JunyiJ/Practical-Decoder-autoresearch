"""Pinned transfer acquisition, fixed excerpt selection and exact-span exclusion."""
import argparse
import base64
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
from urllib.parse import quote
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/tokenizer-evidence/general-rsi-transfer-001'
D=ROOT/'data/raw/rsi-transfer-001'
BINDING='cf798226383b22b0bf115d5cea851438b19cddb01eda330f99f88dbc3bb61b89'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1024**2):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def put(path,obj):
    with Path(path).open('x') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def append(path,obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def contract():
    assert sha(P/'protocol.json')==BINDING
    plan=read(P/'protocol.json')
    for f,h in plan['source_metadata_sha256'].items():assert sha(ROOT/f)==h
    return plan

def key(domain,name,suffix=''):
    s=f'62111000001\n{domain}\n{name}'+suffix
    return hashlib.sha256(s.encode()).hexdigest()

def window(ids,domain,name):
    assert len(ids)>=3201
    start=int(key(domain,name,'\nwindow'),16)%(len(ids)-3201+1)
    return start,np.asarray(ids[start:start+3201],dtype='<u2')

def digests(ids):
    ids=np.asarray(ids,dtype='<u2')
    return np.array([hashlib.blake2b(ids[i:i+32].tobytes(),digest_size=16).digest() for i in range(max(0,len(ids)-31))],dtype='V16')

def intersects(a,index):
    if not len(a) or not len(index):return False
    at=np.searchsorted(index,a);valid=at<len(index)
    return bool(np.any(index[at[valid]]==a[valid]))

class TransferData:
    def __init__(self,values,rows,check=None):
        import torch
        self.indexes={'val':rows}
        self.view=SimpleNamespace(val_data=torch.tensor(values,dtype=torch.long),generators={})
        self.check=check
        offset=0
        for row in rows:
            assert row['offset']==offset and row['tokens']>1
            offset+=row['tokens']
        assert offset==len(values) and int(min(values))>=0 and int(max(values))<50257
    def validation_chunks(self):
        from research.general_rsi_data_v1 import GeneralData
        for row in GeneralData.validation_chunks(self):
            if self.check:self.check()
            yield row


def qualify():
    # Independent-parent boundaries must not enter the exact-span index.
    a=np.arange(100,140,dtype='<u2');b=np.arange(1000,1040,dtype='<u2')
    index=np.unique(np.concatenate([digests(a),digests(b)]))
    assert intersects(digests(a[4:36]),index)
    assert not intersects(digests(a[:31]),index)
    assert not intersects(digests(np.concatenate([a[-16:],b[:16]])),index)
    changed=a[4:36].copy();changed[15]=5000
    assert not intersects(digests(changed),index)
    assert intersects(digests(changed),np.unique(digests(changed)))
    values=np.arange(6402,dtype='<u2');rows=[{'offset':0,'tokens':3201,'source':'a'},{'offset':3201,'tokens':3201,'source':'b'}]
    data=TransferData(values,rows);observed={0:[],1:[]};chunks=0
    for i,src,x,y in data.validation_chunks():
        assert len(x[0])<=128 and np.array_equal(x.numpy()[0]+1,y.numpy()[0])
        observed[i]+=y[0].tolist();chunks+=1
    assert observed[0]==list(range(1,3201)) and observed[1]==list(range(3202,6402)) and chunks==50
    start,chosen=window(np.arange(5000),'fixture','a');s2,c2=window(np.arange(5000),'fixture','a')
    assert start==s2 and np.array_equal(chosen,c2) and len(chosen)==3201
    put(P/'data-qualification.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=BINDING,source_sha256=sha(__file__),passed=True,checks=['exact32-token hit;31-token non-hit; changed-token non-hit','no cross-parent32-token span; selected-excerpt duplicate hit','deterministic window3201 tokens','new adapter covers6400 targets exactly once in50 chunks without cross-parent targets'],new_model_scores=0,test_opened=False))
    print('New data adapter and exclusion boundaries qualified.',flush=True)


def fetch(url,path,size=None,git_sha=None,md5=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        receipt=next(r for r in [json.loads(l) for l in (P/'acquisition.jsonl').read_text().splitlines()] if r.get('path')==str(path.relative_to(ROOT)) and r['status']=='verified')
        assert sha(path)==receipt['sha256'];return receipt
    attempt=1
    while path.with_name(path.name+f'.attempt-{attempt:02d}.partial').exists():attempt+=1
    assert attempt<=2
    temp=path.with_name(path.name+f'.attempt-{attempt:02d}.partial')
    t=time.monotonic()
    r=subprocess.run(['curl','--fail','--silent','--show-error','--location','--max-time','30','--max-filesize',str(size if size is not None else 1024**2),url,'-o',str(temp)],capture_output=True,text=True)
    if r.returncode:
        append(P/'acquisition.jsonl',dict(status='failed',url=url,attempt=attempt,returncode=r.returncode,stderr=r.stderr,partial_path=str(temp.relative_to(ROOT)),seconds=time.monotonic()-t))
        raise RuntimeError(r.stderr)
    payload=temp.read_bytes()
    assert size is None or len(payload)==size
    assert git_sha is None or hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()==git_sha
    assert md5 is None or base64.b64encode(hashlib.md5(payload).digest()).decode()==md5
    os.link(temp,path);temp.unlink()
    out=dict(status='verified',url=url,path=str(path.relative_to(ROOT)),size=len(payload),sha256=sha(path),git_blob_sha=git_sha,md5=md5,seconds=time.monotonic()-t)
    append(P/'acquisition.jsonl',out);return out


def build():
    import tiktoken
    plan=contract();D.mkdir(parents=True,exist_ok=True)
    assert not (P/'dataset.json').exists()
    enc=tiktoken.get_encoding('gpt2')
    for a in plan['inference']['codebook_assets']:assert sha(a['path'])==a['sha256']
    receipts=[];candidates={}
    licenses=[('pg19','https://raw.githubusercontent.com/google-deepmind/pg19/b0e1f2925359688af19f8784c23c3deeda52d3ea/LICENSE'),('python','https://raw.githubusercontent.com/python/cpython/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/LICENSE')]
    for name,url in licenses:receipts.append(fetch(url,D/'licenses'/name))
    # Complete source pools are retained. Selection cannot depend on model loss.
    for domain in plan['domains']:
        meta=read(ROOT/plan['source_pools'][domain]['metadata'])
        rows=meta['items'] if domain=='pg19_books' else meta['files']
        items=[]
        for x in rows:
            name=x['name'] if domain=='pg19_books' else x['path']
            path=D/'sources'/domain/Path(name).name
            if domain=='pg19_books':
                url=f"https://storage.googleapis.com/download/storage/v1/b/deepmind-gutenberg/o/{quote(name,safe='')}?generation={x['generation']}&alt=media"
                receipt=fetch(url,path,int(x['size']),md5=x['md5Hash'])
            else:
                url=f"https://raw.githubusercontent.com/python/cpython/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/{name}"
                receipt=fetch(url,path,x['size'],git_sha=x['sha'])
            receipts.append(receipt);items.append((name,path,receipt))
            assert sum(r['size'] for r in receipts)<=plan['resource']['source_and_license_cap_bytes']
        candidates[domain]=sorted(items,key=lambda x:(key(domain,x[0]),x[0]))
        print('Acquired',domain,len(items),flush=True)
    # Build within-document digest index from only original train and validation.
    old=ROOT/'data/raw/rsi-general-10m-v1';manifest=read(old/'manifest.json')
    assert sha(old/'manifest.json')=='b1849a5fed1fb648f507e407e5794946bba64328f519e6b0d6c360fa4e2e4fa8'
    refs=[];index_file=D/'train-val-span32.npy'
    if index_file.exists():
        prior=read(P/'overlap-index.json');assert sha(index_file)==prior['sha256'];index=np.load(index_file,allow_pickle=False)
    else:
        total=sum(manifest['tokens'][s] for s in ['train','val']);index=np.empty(total,dtype='V16');used=0
        for split in ['train','val']:
            for suffix in ['bin','index.jsonl']:
                f=old/f'{split}.{suffix}';assert sha(f)==manifest['files'][f.name]['sha256'];refs.append(dict(path=str(f.relative_to(ROOT)),sha256=sha(f)))
            ids=np.fromfile(old/f'{split}.bin',dtype='<u2');rows=[json.loads(l) for l in (old/f'{split}.index.jsonl').read_text().splitlines()]
            for row in rows:
                ds=digests(ids[row['offset']:row['offset']+row['tokens']]);index[used:used+len(ds)]=ds;used+=len(ds)
            print('Indexed',split,used,'spans',flush=True)
        index=np.unique(index[:used]);temp=D/'train-val-span32.partial.npy';assert not temp.exists()
        np.save(temp,index,allow_pickle=False);assert temp.stat().st_size<=plan['resource']['overlap_index_retained_cap_bytes']
        os.link(temp,index_file);temp.unlink()
        put(P/'overlap-index.json',dict(path=str(index_file.relative_to(ROOT)),sha256=sha(index_file),size=index_file.stat().st_size,unique_spans=len(index),input_spans=used,references=refs,original_test_opened=False))
    selected=[];parts=[];offset=0;previous=np.array([],dtype='V16');counts={}
    for domain,items in candidates.items():
        accepted=0
        for name,path,receipt in items:
            ids=enc.encode_ordinary(path.read_text(encoding='utf-8'))
            row=dict(domain=domain,name=name,source_sha256=receipt['sha256'],full_tokens=len(ids))
            if len(ids)<3201:row['decision']='exclude_short'
            else:
                start,excerpt=window(ids,domain,name);ds=digests(excerpt);row.update(excerpt_start=start,excerpt_tokens=3201)
                if intersects(ds,index):row['decision']='exclude_original_train_or_val_overlap'
                elif intersects(ds,previous):row['decision']='exclude_selected_excerpt_overlap'
                else:
                    row['decision']='selected';row['excerpt_sha256']=hashlib.sha256(excerpt.tobytes()).hexdigest()
                    selected.append(dict(document=len(selected),source=domain,name=name,offset=offset,tokens=3201,excerpt_start=start,source_sha256=receipt['sha256'],excerpt_sha256=row['excerpt_sha256']))
                    parts.append(excerpt);offset+=3201;accepted+=1;previous=np.unique(np.concatenate([previous,ds]))
            append(P/'selection.jsonl',row)
            if accepted==32:break
        counts[domain]=accepted
        print('Selected',domain,accepted,flush=True)
    assert counts=={domain:32 for domain in plan['domains']},f'UNDERFILLED: {counts}; panel not admitted'
    values=np.concatenate(parts);values.tofile(D/'excerpts.bin');put(D/'index.json',selected)
    # New adapter validation is structural, not a model score.
    data=TransferData(values,selected);coverage=[0]*64
    for i,source,x,y in data.validation_chunks():coverage[i]+=y.numel()
    assert coverage==[3200]*64 and sum(coverage)==204800
    put(P/'dataset.json',dict(protocol_sha256=BINDING,source_sha256=sha(__file__),tokenizer_version=tiktoken.__version__,sources=receipts,
        payloads=[dict(path=str(f.relative_to(ROOT)),size=f.stat().st_size,sha256=sha(f)) for f in [D/'excerpts.bin',D/'index.json',index_file]],documents=selected,counts=counts,targets=204800,coverage=coverage,selection_sha256=sha(P/'selection.jsonl'),overlap_index_receipt_sha256=sha(P/'overlap-index.json'),test_opened=False))
    print('Dataset built and bound; scientific scoring still requires admission.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['qualify','build']);args=parser.parse_args()
    contract()
    if args.action=='qualify':qualify();return
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=sum(read(f)['elapsed_seconds'] for f in P.glob('*-guard-*.json'));remaining=7200-prior;assert remaining>0
        attempt=1
        while (P/f'build-guard-{attempt:02d}.json').exists():attempt+=1
        assert attempt<=2
        with CpuPrerequisiteGuard(P/f'build-guard-{attempt:02d}.json',ROOT,timeout_seconds=remaining,rss_limit_bytes=4*1024**3,reserve_bytes=10*1024**3):build()
if __name__=='__main__':main()
