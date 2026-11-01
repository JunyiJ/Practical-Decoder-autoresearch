"""Six-screen adapter over the tested score retention executor."""
from pathlib import Path
from . import campaign_contract as c,campaign_rules as rules
from research.local_mix_campaign_v1 import retention as tested
from research.block_norm_v1.fixture_dispatch import read,reference
from src.research.protocol import digest
require=c.require

def inventory(root):
    result={}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(),'Unaliased evidence inventory required')
        if p.is_file():result[str(p)]=dict(size=p.stat().st_size,sha256=c.file_sha(p,max_bytes=512*1024**2))
    require(0<len(result)<2048,'Bounded complete inventory required');return result

def check_entry(name,row):
    p=Path(name);require(p.stat().st_size==row['size'] and c.file_sha(p,max_bytes=512*1024**2)==row['sha256'],'Retained file changed')

def inputs(root,seeds,records,selection,requests):
    rules.screening_retention(seeds,records,selection);require(len(requests)==6,'Complete six-screen requests required')
    e=root/'evidence';pairs={}
    for slot,req in enumerate(requests):
        expected=root/f'evidence/quality/{slot:02d}';c.validate_trial_config(req['config'],expected,rules.screen_specs(seeds)[slot],seeds);pairs[slot]={}
        for role in ('parent','h200'):
            ref=req['files'][role];p=Path(ref['path']);require(p.parent==expected/'checkpoints' and c.file_sha(p,max_bytes=512*1024**2)==ref['sha256'],'Exact audited screen weight required')
            pairs[slot]['raw' if role=='parent' else role]=dict(path=str(p.relative_to(e)),size=p.stat().st_size,sha256=ref['sha256'])
    nonweights={str(p.relative_to(e)):c.file_sha(p,max_bytes=2*1024**2) for p in e.rglob('*') if p.is_file() and p.suffix!='.pt'}
    simple=[dict(slot=i,arm=r['arm'],lr=r['lr'],status=r['status'],h200_ce=r['h200_ce']) for i,r in enumerate(records)]
    return dict(boundary='lr',records=simple,selection=selection,pairs=pairs,evidence=nonweights)

def execute(writer,destination,seeds,records,selection,requests,context,revalidate):
    require(destination==writer.root/'evidence/.retention/lr','Declared in-budget retention path required')
    args=inputs(writer.root,seeds,records,selection,requests);e=writer.root/'evidence'
    before={str(Path(n).relative_to(e)):v for n,v in inventory(e).items()}
    tested.execute_boundary(e,'lr',writer.lock_fds,lambda:dict(inputs=args,inventory=before),revalidate)
    return reference(destination/'execution-review.json')

def verify(ref,context):
    e=Path(ref['path']).parents[2];base=e/'.retention/lr';manifest=read(reference(base/'manifest.json'))
    result=tested.recheck_execution(e,manifest,dict(path='.retention/lr/execution-review.json',sha256=ref['sha256']),digest(context))
    return {str(e/x['path']):dict(size=x['size'],sha256=x['sha256']) for x in manifest['delete']}
