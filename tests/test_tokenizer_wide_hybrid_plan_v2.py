"""Synthetic prerequisite documents; no real numerical pass or MPS launch."""
import copy
import json
import itertools
from pathlib import Path

import pytest

from src.research import tokenizer_wide_hybrid_plan_v2 as gate
from src.research.protocol import digest,file_digest
from src.research.tokenizer_wide_hybrid_pilot_gate import numerical_cases,CHECKS


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    return dict(path=str(path),sha256=file_digest(path))


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    root=tmp_path/'campaign';meta=root/'evidence/meta';meta.mkdir(parents=True)
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK',raising=False)
    source={'synthetic':'source'};runtime={'synthetic':'runtime'}
    monkeypatch.setattr(gate,'source_fingerprint',lambda:source)
    monkeypatch.setattr(gate,'runtime_identity',lambda:runtime)
    for key in ('PROTOCOL','PROJECTION','RECIPE'):
        original=getattr(gate,key);path=tmp_path/key;path.write_bytes(original.read_bytes())
        monkeypatch.setattr(gate,key,path)
    corpus=tmp_path/'corpus';corpus.write_text('Synthetic corpus identity, not research data')
    monkeypatch.setattr(gate,'CORPUS_SHA256',file_digest(corpus))
    pilot_config=gate.config;quality_config=gate.rank_config
    def pilot(*a,**kw):
        cfg=pilot_config(*a,**kw);cfg['data']['path']=str(corpus);return cfg
    def quality(*a,**kw):
        cfg=quality_config(*a,**kw);cfg['data']['path']=str(corpus);return cfg
    monkeypatch.setattr(gate,'config',pilot);monkeypatch.setattr(gate,'rank_config',quality)
    options=gate.request_options(root)
    tokenizer=options['pilots'][0]['config']['model']['output_bias_prior']['tokenizer']
    context=dict(source_sha256=digest(source),runtime_sha256=digest(runtime),corpus_sha256=file_digest(corpus))
    log=meta/'numerical.log';log.write_text('Synthetic prerequisite trace only')
    numeric=dict(kind='numerical',status='passed',**context,cases=numerical_cases(),
        checks=dict.fromkeys(('all_declared_ranks',)+CHECKS,True),artifacts={'meta/numerical.log':file_digest(log)})
    write(meta/'numerical.json',numeric)
    ids=dict(ids=list(range(96)),split='train',corpus_sha256=file_digest(corpus),tokenizer_sha256=digest(tokenizer))
    write(meta/'ids.json',ids);token_sha=gate.token_digest(ids['ids'])
    input_review=dict(kind='wide_hybrid_training_input_v2',status='passed',**context,
        tokenizer_sha256=digest(tokenizer),training_token_sha256=token_sha,split='train',token_count=96,
        holdout_consumed=False,input_ids_ref=dict(path='meta/ids.json',sha256=file_digest(meta/'ids.json')),
        artifacts={'meta/ids.json':file_digest(meta/'ids.json')})
    write(meta/'input.json',input_review)
    files=[gate.PROTOCOL,gate.PROJECTION,gate.RECIPE,corpus,log,meta/'numerical.json',meta/'input.json',meta/'ids.json']
    plan=dict(kind=gate.KIND,status='frozen',output=str(root),source=source,runtime=runtime,
        protocol_sha256=file_digest(gate.PROTOCOL),projection_sha256=file_digest(gate.PROJECTION),
        panel_design=gate.design(),layout=gate.layout_contract(),
        storage_contract=gate.storage_envelope(json.loads(gate.PROJECTION.read_text())),
        pilot_contract=dict(gate.PILOT_V1,version=2),quality_contract=copy.deepcopy(gate.RESOURCE_LIMITS),
        cache_contract=copy.deepcopy(gate.CACHE_CONTRACT),requests=options,tokenizer=tokenizer,
        training_token_sha256=token_sha,
        inference_contract=dict(gate.SERVING_V1,version=2,tokenizer_sha256=digest(tokenizer),token_sha256=token_sha),
        numerical_ref=dict(path='meta/numerical.json',sha256=file_digest(meta/'numerical.json')),
        input_ref=dict(path='meta/input.json',sha256=file_digest(meta/'input.json')),
        files=[dict(path=str(p),sha256=file_digest(p)) for p in files])
    return root,plan,numeric,input_review


def bind(fixture):
    root,plan,_,_=fixture;path=root/'evidence/meta/plan.json';write(path,plan)
    return gate.bind_campaign(path,file_digest(path))


def rebind(fixture,name,value,ref_key):
    root,plan,_,_=fixture;path=root/'evidence/meta'/name;write(path,value)
    plan[ref_key]['sha256']=file_digest(path)
    for ref in plan['files']:
        if ref['path']==str(path):ref['sha256']=file_digest(path)


def test_complete_identity_binds_but_does_not_launch(fixture):
    report=bind(fixture);root,plan,_,_=fixture
    assert len(report['requests']['pilots'])==3
    assert [len(v) for v in report['requests']['quality'].values()]==[1]*9+[3]*6+[9]*9
    assert sum(map(len,report['requests']['quality'].values()))==108
    assert report['storage_contract']['required_free_bytes']==30_631_281_412
    assert report['numerical_review_bound'] and report['training_input_bound']
    assert not report['dispatch_authorized'] and not report['current_storage_fit_verified']
    assert not report['measured_pilots_verified'] and not report['producer_enforcement_verified']
    assert not (root/'control').exists() and not (root/'evidence/pilots').exists()
    assert (root/'evidence/meta/plan.json').stat().st_size<2*1024**2


def test_conditional_requests_cover_all_729_family_selection_combinations(fixture):
    report=bind(fixture);options=report['requests']['quality']
    for rates_tuple in itertools.product(gate.rules.RATES,repeat=3):
        rates=dict(zip(gate.rules.FAMILIES,rates_tuple))
        for divisors in itertools.product((8,4,2),repeat=3):
            selection={f:dict(lr=rates[f],divisor=d) for f,d in zip(gate.rules.FAMILIES,divisors)}
            queue=gate.rules.lr_specs()+gate.rules.rank_specs(rates)+gate.rules.confirmation_specs(selection)
            assert len(queue)==24
            for index,spec in enumerate(queue):
                matches=[r for r in options[str(index)] if r['spec']==spec]
                assert len(matches)==1
                cfg=matches[0]['config']
                assert cfg['training']['learning_rate']==spec['lr']
                assert cfg['training']['seed']==spec['seed']
                assert cfg['model']['attention_bottleneck_divisor']==spec['divisor']


def test_wrong_plan_hash_is_rejected(fixture):
    root,plan,_,_=fixture;path=root/'evidence/meta/plan.json';write(path,plan)
    with pytest.raises(ValueError,match='hash mismatch'):gate.bind_campaign(path,'0'*64)


def test_stop_created_during_review_rejected(fixture,monkeypatch):
    root,_,_,_=fixture;calls=[]
    def runtime():
        calls.append(1)
        if len(calls)>1:(root/'STOP').touch()
        return {'synthetic':'runtime'}
    monkeypatch.setattr(gate,'runtime_identity',runtime)
    with pytest.raises(ValueError,match='STOP changed during'):bind(fixture)


@pytest.mark.parametrize('bad',[
    'draft','v1_kind','source','runtime','protocol','projection','storage','layout',
    'cleanup','retain_fresh','neutrality','seeds','quality_order','quality_option',
    'quality_config','pilot_config','pilot_timeout','quality_timeout','memory','reserve',
    'serving','holdout','tokenizer','token_sha','numerical_binding','trace_binding','duplicate',
])
def test_changed_frozen_contract_rejected(fixture,bad):
    _,plan,_,_=fixture
    if bad=='draft':plan['status']='draft'
    elif bad=='v1_kind':plan['kind']='wide_hybrid_pilot_v1'
    elif bad in ('source','runtime'):plan[bad]={}
    elif bad in ('protocol','projection'):plan[bad+'_sha256']='0'*64
    elif bad=='storage':plan['storage_contract']['required_free_bytes']-=1
    elif bad=='layout':plan['layout']['metadata_groups']['control']+=1
    elif bad=='cleanup':plan['panel_design']['retention']['boundaries'].reverse()
    elif bad=='retain_fresh':plan['panel_design']['retention']['retain_all_fresh_pairs']=False
    elif bad=='neutrality':plan['panel_design']['neutral_delta']=.02
    elif bad=='seeds':plan['requests']['quality']['23'][0]['spec']['seed']=3501
    elif bad=='quality_order':plan['requests']['quality']['9'].reverse()
    elif bad=='quality_option':plan['requests']['quality']['15'].pop()
    elif bad=='quality_config':plan['requests']['quality']['0'][0]['config']['model']['rope']=1
    elif bad=='pilot_config':plan['requests']['pilots'][0]['config']['training']['max_iters']=21
    elif bad=='pilot_timeout':plan['pilot_contract']['job_seconds']=601
    elif bad=='quality_timeout':plan['quality_contract']['timeout_seconds']=7201
    elif bad=='memory':plan['quality_contract']['rss_limit_bytes']+=1
    elif bad=='reserve':plan['storage_contract']['reserve_bytes']-=1
    elif bad=='serving':plan['inference_contract']['decode_tokens']=32
    elif bad=='holdout':plan['panel_design']['test_holdout']='open'
    elif bad=='tokenizer':plan['tokenizer']={'name':'character'}
    elif bad=='token_sha':plan['training_token_sha256']='0'*64
    elif bad in ('numerical_binding','trace_binding'):
        suffix='numerical.json' if bad=='numerical_binding' else 'numerical.log'
        plan['files']=[r for r in plan['files'] if not r['path'].endswith(suffix)]
    elif bad=='duplicate':plan['files'].append(copy.deepcopy(plan['files'][0]))
    with pytest.raises(ValueError):bind(fixture)


@pytest.mark.parametrize('bad',['failed','missing_case','geometry_only','resume','source','empty_traces'])
def test_numerical_rehash_cannot_replace_nine_passed_cases(fixture,bad):
    _,_,numeric,_=fixture
    if bad=='failed':numeric['status']='failed'
    elif bad=='missing_case':numeric['cases'].pop()
    elif bad=='geometry_only':numeric['checks']={'geometry':True}
    elif bad=='resume':numeric['cases'][0]['checks']['resume']=False
    elif bad=='source':numeric['source_sha256']='0'*64
    elif bad=='empty_traces':numeric['artifacts']={}
    rebind(fixture,'numerical.json',numeric,'numerical_ref')
    with pytest.raises(ValueError):bind(fixture)


@pytest.mark.parametrize('bad',['test_split','holdout','count','float_count','source','trace','ids_ref'])
def test_input_provenance_rehash_does_not_bypass_contract(fixture,bad):
    _,_,_,review=fixture
    if bad=='test_split':review['split']='test'
    elif bad=='holdout':review['holdout_consumed']=True
    elif bad=='count':review['token_count']=95
    elif bad=='float_count':review['token_count']=96.0
    elif bad=='source':review['source_sha256']='0'*64
    elif bad=='trace':review['artifacts']={}
    elif bad=='ids_ref':review['input_ids_ref']['sha256']='0'*64
    rebind(fixture,'input.json',review,'input_ref')
    with pytest.raises(ValueError):bind(fixture)


def test_altered_input_ids_fail_even_with_updated_trace_hashes(fixture):
    root,plan,_,review=fixture;p=root/'evidence/meta/ids.json'
    ids=json.loads(p.read_text());ids['ids'][0]=400;write(p,ids)
    review['input_ids_ref']['sha256']=file_digest(p);review['artifacts']['meta/ids.json']=file_digest(p)
    for ref in plan['files']:
        if ref['path']==str(p):ref['sha256']=file_digest(p)
    rebind(fixture,'input.json',review,'input_ref')
    with pytest.raises(ValueError,match='96-token'):bind(fixture)


@pytest.mark.parametrize('bad',['stop','broken_stop_link','fallback','source_changes','trace_changes','unknown_output'])
def test_live_identity_and_output_guards(fixture,monkeypatch,bad):
    root,_,_,_=fixture
    if bad=='stop':(root/'STOP').touch()
    elif bad=='broken_stop_link':(root/'STOP').symlink_to(root/'missing')
    elif bad=='fallback':monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','1')
    elif bad=='source_changes':
        calls=[]
        def source():calls.append(1);return {'synthetic':'source' if len(calls)==1 else 'changed'}
        monkeypatch.setattr(gate,'source_fingerprint',source)
    elif bad=='trace_changes':(root/'evidence/meta/numerical.log').write_text('changed')
    elif bad=='unknown_output':(root/'unexpected').touch()
    with pytest.raises(ValueError):bind(fixture)



def test_compact_request_identity_preserves_every_reconstructed_option(fixture):
    root,plan,_,_=fixture
    expected=copy.deepcopy(plan['requests'])
    before=len(json.dumps(plan,sort_keys=True,separators=(',',':')).encode())
    plan['requests']=gate.request_identity(expected)
    after=len(json.dumps(plan,sort_keys=True,separators=(',',':')).encode())
    report=bind(fixture)
    assert report['requests']==expected
    assert before-after>250000
    assert len(json.dumps(plan['requests']).encode())<300
    assert not report['dispatch_authorized']


@pytest.mark.parametrize('bad',['hash','version','count','order','missing_slot','extra','reconstructed_config'])
def test_compact_identity_cannot_hide_request_changes(fixture,monkeypatch,bad):
    _,plan,_,_=fixture
    plan['requests']=gate.request_identity(plan['requests'])
    identity=plan['requests']
    if bad=='hash':identity['sha256']='0'*64
    elif bad=='version':identity['representation']='unreviewed'
    elif bad=='count':identity['pilot_count']=True
    elif bad=='order':identity['quality_slot_counts'].reverse()
    elif bad=='missing_slot':identity['quality_slot_counts'].pop()
    elif bad=='extra':identity['ignored_requests']=[]
    else:
        original=gate.rank_config
        def changed(*a,**kw):
            cfg=original(*a,**kw);cfg['model']['rope']=1;return cfg
        monkeypatch.setattr(gate,'rank_config',changed)
    with pytest.raises(ValueError,match='requests'):bind(fixture)


def test_live_census_locks_only_snapshot_not_frozen_hashing(fixture,monkeypatch):
    import fcntl
    from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
    from src.research import tokenizer_wide_hybrid_cleanup as cleanup
    root,plan,_,_=fixture;path=root/'evidence/meta/plan.json';write(path,plan)
    sha=file_digest(path);global_lock=root.parent/'global.lock';global_lock.touch();(root/'.lock').touch()
    monkeypatch.setattr(cleanup,'GLOBAL_LOCK',global_lock)
    original_snapshot=gate.snapshot;original_hash=gate.file_digest;hashed=[];censuses=[]
    with global_lock.open('rb') as g,(root/'.lock').open('rb') as c:
        fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(c,fcntl.LOCK_EX|fcntl.LOCK_NB)
        writer=MetadataWriter(root,(g.fileno(),c.fileno()))
        def snapshot(campaign):
            with (root/'control/.metadata.lock').open('rb') as lock:
                with pytest.raises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            censuses.append(True);return original_snapshot(campaign)
        def hashing(target):
            with (root/'control/.metadata.lock').open('rb') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            hashed.append(str(target));return original_hash(target)
        monkeypatch.setattr(gate,'snapshot',snapshot);monkeypatch.setattr(gate,'file_digest',hashing)
        report=gate.bind_campaign(path,sha,live_writer=writer)
        assert report['numerical_review_bound'] and censuses==[True] and hashed
        with pytest.raises(ValueError,match='exact locked campaign'):gate.bind_campaign(path,sha,live_writer=object())
        with monkeypatch.context() as patch:
            patch.setattr(writer,'root',root.parent/'wrong-campaign')
            with pytest.raises(ValueError,match='exact locked campaign'):gate.bind_campaign(path,sha,live_writer=writer)
