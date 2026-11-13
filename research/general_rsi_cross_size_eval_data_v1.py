"""Fresh-parent broader evaluation from pinned local pools; no model scores."""
import argparse,datetime,fcntl,hashlib,json,re
from pathlib import Path
import numpy as np
import tiktoken
from research.general_rsi_transfer_data_v1 import digests,intersects,TransferData
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-eval-001'
DATA=ROOT/'data/raw/rsi-cross-size-eval-001'
OLD=ROOT/'research/tokenizer-evidence/general-rsi-transfer-001'
OLD_DATA=ROOT/'data/raw/rsi-transfer-001'
SEED=62111312020201

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def key(domain,name,suffix=''):return hashlib.sha256(f'{SEED}\n{domain}\n{name}{suffix}'.encode()).hexdigest()
def prose(text):
    paragraphs=[]
    for block in re.split(r'\n\s*\n',text):
        lines=block.splitlines()
        if not lines or any(line[:1].isspace() for line in lines):continue
        value=' '.join(line.strip() for line in lines)
        if len(value)<200 or any(x in value for x in ['`','::','.. ','*','|','<','>']):continue
        if sum(c.isalpha() or c.isspace() for c in value)/len(value)<.85:continue
        paragraphs.append(value)
    return '\n\n'.join(paragraphs)
def prepare():
    refs=[Path(__file__),ROOT/'research/general_rsi_transfer_data_v1.py',OLD/'protocol.json',OLD/'acquisition.jsonl',OLD_DATA/'index.json',OLD/'overlap-index.json',ROOT/'research/tokenizer-evidence/general-rsi-transfer-preparation-001/pg19_validation_objects.json',ROOT/'research/tokenizer-evidence/general-rsi-transfer-preparation-001/cpython-library-tree.json']
    p=dict(kind='cross_size_fresh_parent_eval_v1',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources={str(f.relative_to(ROOT)):sha(f) for f in refs},selection_seed=SEED,roles_audit='rg research/research_state JSON/JSONL returned no previous matches',domains=['pg19_books','python_library_prose'],documents_per_domain=8,tokens_per_document=2049,targets_per_domain=16384,tokenizer='tiktoken gpt2 encode_ordinary',tokenizer_version=tiktoken.__version__,parent_exclusion='Exclude all64 parents from old transfer data/index.json, regardless of old scores; no model scores read.',python_preprocessing='Split blank-line paragraphs. Reject any leading-indented line, paragraph under200 characters, backtick/::/.. /asterisk/pipe/angle bracket, or alphabetic+whitespace ratio below.85. Join retained paragraphs with blank lines; one filtered parent remains one document.',selection='Domains in listed order. Sort remaining parents by SHA256(seed newline domain newline original source name), name tie-break. Deterministic2049-token window at hash(... newline window) modulo eligible offsets. No EOT added.',dedup='Reject any32-token exact BLAKE2b128 span matching original train/val within-document index or any previously selected excerpt in either domain. No alternative window after rejection. Original test never consulted.',underfill='Reject whole dataset if either domain lacks8 accepted fresh parents. No model scoring before complete fixed manifest.',limits='Exact32-token overlap only; paraphrases/shorter overlaps remain possible. PG19 historical books and mechanically filtered Python prose are narrow domain probes. Filtered paragraph joins are artificial. Existing source pools were selected historically, but these parents have not entered prior model evaluation.',model_scores_used=0,test_opened=False)
    OUT.mkdir();put(OUT/'protocol.json',p)
def build():
    p=read(OUT/'protocol.json')
    for f,h in p['sources'].items():assert sha(ROOT/f)==h,f
    assert tiktoken.__version__==p['tokenizer_version'];enc=tiktoken.get_encoding('gpt2')
    old=read(OLD_DATA/'index.json');excluded={(r['source'],r['name']) for r in old};assert len(excluded)==64
    receipt=read(OLD/'overlap-index.json');index_path=ROOT/receipt['path'];assert sha(index_path)==receipt['sha256'];index=np.load(index_path,allow_pickle=False)
    acquisitions=[json.loads(line) for line in (OLD/'acquisition.jsonl').read_text().splitlines()];acquisitions={r['path']:r for r in acquisitions if r['status']=='verified'}
    selected=[];decisions=[];parts=[];previous=np.array([],dtype='V16');offset=0
    for domain in p['domains']:
        orig='python_library_rst' if domain=='python_library_prose' else domain
        meta=read(ROOT/('research/tokenizer-evidence/general-rsi-transfer-preparation-001/'+('pg19_validation_objects.json' if domain=='pg19_books' else 'cpython-library-tree.json')))
        names=[r['name'] for r in meta['items']] if domain=='pg19_books' else [r['path'] for r in meta['files']]
        names=sorted(names,key=lambda n:(key(domain,n),n));accepted=0
        for name in names:
            entry=dict(domain=domain,name=name)
            if (orig,name) in excluded:entry['decision']='exclude_previously_evaluated_parent';decisions.append(entry);continue
            path=OLD_DATA/'sources'/orig/Path(name).name;receipt=acquisitions[str(path.relative_to(ROOT))];assert sha(path)==receipt['sha256']
            text=path.read_text(encoding='utf-8');filtered=prose(text) if domain=='python_library_prose' else text
            ids=enc.encode_ordinary(filtered);entry.update(source_sha256=receipt['sha256'],filtered_text_sha256=hashlib.sha256(filtered.encode()).hexdigest(),eligible_tokens=len(ids))
            if len(ids)<p['tokens_per_document']:entry['decision']='exclude_short'
            else:
                start=int(key(domain,name,'\nwindow'),16)%(len(ids)-p['tokens_per_document']+1)
                excerpt=np.array(ids[start:start+p['tokens_per_document']],dtype='<u2');ds=digests(excerpt);entry['excerpt_start']=start
                if intersects(ds,index):entry['decision']='exclude_train_val_overlap'
                elif intersects(ds,previous):entry['decision']='exclude_selected_overlap'
                else:
                    entry['decision']='selected';entry['excerpt_sha256']=hashlib.sha256(excerpt.tobytes()).hexdigest()
                    selected.append(dict(document=len(selected),source=domain,name=name,offset=offset,tokens=len(excerpt),excerpt_start=start,source_sha256=receipt['sha256'],filtered_text_sha256=entry['filtered_text_sha256'],excerpt_sha256=entry['excerpt_sha256']))
                    parts.append(excerpt);offset+=len(excerpt);accepted+=1;previous=np.unique(np.concatenate([previous,ds]))
            decisions.append(entry)
            if accepted==p['documents_per_domain']:break
        put(OUT/(domain+'-selection.json'),[r for r in decisions if r['domain']==domain]);assert accepted==p['documents_per_domain'],f'underfilled {domain}: {accepted}'
    DATA.mkdir();values=np.concatenate(parts);values.tofile(DATA/'excerpts.bin');put(DATA/'index.json',selected)
    adapter=TransferData(values,selected);coverage={i:[] for i in range(len(selected))}
    for i,_,x,y in adapter.validation_chunks():coverage[i]+=y[0].tolist()
    for i,r in enumerate(selected):assert coverage[i]==values[r['offset']+1:r['offset']+r['tokens']].tolist()
    put(OUT/'dataset.json',dict(protocol_sha256=sha(OUT/'protocol.json'),payload_sha256={str(f.relative_to(ROOT)):sha(f) for f in [DATA/'excerpts.bin',DATA/'index.json']},documents=selected,targets=32768,coverage_exact=True,prior_evaluated_parents_excluded=64,within_train_val_and_selected_exact32_exclusion=True,overlap_index_sha256=receipt['sha256'],model_scores_used=0,test_opened=False))
    print('Frozen16 fresh-parent excerpts;32768targets; no model scores.')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','build']);a=ap.parse_args()
    if a.action=='prepare':prepare();return
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);build()
if __name__=='__main__':main()
