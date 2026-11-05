"""Closed, one-shot exact Docker image removal under explicit human authorization."""
import fcntl, hashlib, http.client, json, os, socket, subprocess, time
from pathlib import Path

ROOT = Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI')
HERE = Path(__file__).resolve().parent
DOCKER = '/usr/local/bin/docker'
PRIOR = ROOT / 'research/maintenance-cache-cleanup-007/image-candidates.json'
BACKING = Path('/Users/junyi/Library/Group Containers/HUAQ24HBR6.dev.orbstack/data/data.img.raw')

def run(*args):
    return subprocess.run([DOCKER, *args], capture_output=True, text=True, check=True, timeout=60).stdout

def save(name, obj):
    with (HERE / name).open('x') as f:
        json.dump(obj, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())

def image_row(image_id):
    x = json.loads(run('image', 'inspect', image_id))[0]
    return dict(id=x['Id'], tags=x.get('RepoTags') or [], size=x['Size'], layers=x['RootFS']['Layers'])

def containers():
    ids = run('ps', '-aq', '--no-trunc').split()
    if not ids: return []
    rows = json.loads(run('container', 'inspect', *ids))
    return sorted([dict(id=x['Id'], image=x['Image'], name=x['Name'], running=x['State']['Running'],
                        started_at=x['State']['StartedAt'], mounts=x['Mounts']) for x in rows], key=lambda x:x['id'])

def free():
    s = os.statvfs(ROOT); return s.f_bavail*s.f_frsize

def backing():
    assert not any(x.is_symlink() for x in (BACKING, *BACKING.parents))
    s=BACKING.lstat()
    return dict(path=str(BACKING), device=s.st_dev, inode=s.st_ino, uid=s.st_uid,
                gid=s.st_gid, mode=s.st_mode, links=s.st_nlink, logical_bytes=s.st_size,
                allocated_bytes=s.st_blocks*512, disposition='Preserved; measured only, never directly modified')

def usage():
    host=json.loads(run('context', 'inspect', '--format', '{{json .Endpoints.docker.Host}}'))
    assert host=='unix:///Users/junyi/.orbstack/run/docker.sock'
    c=http.client.HTTPConnection('localhost', timeout=30)
    c.sock=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM);c.sock.settimeout(30);c.sock.connect(host[7:])
    c.request('GET','/system/df');r=c.getresponse();assert r.status==200
    d=json.loads(r.read());c.close()
    return dict(layers_size=d['LayersSize'], images=[{k:x.get(k) for k in ['Id','RepoTags','Size','SharedSize','Containers']} for x in d['Images']],
                build_cache=[{k:x.get(k) for k in ['ID','InUse','Shared','Size']} for x in d.get('BuildCache',[])])

assert not (HERE/'manifest.json').exists(), 'One-shot operation already planned; inspect records instead of rerunning'
lock=ROOT/'research_state/external-cache-maintenance.lock'
assert not lock.is_symlink()
with lock.open('r') as held:
    fcntl.flock(held, fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=json.loads(PRIOR.read_text()); ids=old['candidate_ids']; assert len(ids)==4
    expected={x['id']:x for x in old['images'] if x['id'] in ids}
    rows=[image_row(i) for i in ids];assert all(x==expected[x['id']] for x in rows)
    before_containers=containers();assert all(x['image'] not in ids for x in before_containers)
    before_images=set(run('image','ls','-aq','--no-trunc').split())
    before_volumes=sorted(run('volume','ls','-q').split())
    before_usage=usage();assert not any(x['InUse'] for x in before_usage['build_cache'])
    before_backing=backing();before_free=free()
    manifest=dict(authorization='Human explicitly selected Remove those unused images for browserless/chrome and old openclaw-docker plus the two inventoried untagged descendants.',
                  source_manifest_sha256=hashlib.sha256(PRIOR.read_bytes()).hexdigest(),
                  candidates=rows, containers=before_containers, volumes=before_volumes,
                  before_usage=before_usage, before_backing=before_backing, before_free=before_free,
                  method='Docker image rm --no-prune exact SHA256 IDs, without force; daemon enforces reference safety. API objects are content-addressed; VM backing inode is measurement only.',
                  observed_unix=time.time())
    save('manifest.json',manifest)
    removed=[];errors=[]
    with (HERE/'deletion-journal.jsonl').open('x') as j:
        def event(obj):
            j.write(json.dumps(dict(time=time.time(),**obj))+'\n');j.flush();os.fsync(j.fileno())
        for i in ids:
            assert image_row(i)==expected[i]
            current=containers();assert all(x['image']!=i for x in current)
            assert current==before_containers, 'Container state/dependencies changed'
            event(dict(event='intent',image=expected[i],reason='Explicitly identified unused image; no container consumer'))
            r=subprocess.run([DOCKER,'image','rm','--no-prune',i],capture_output=True,text=True,timeout=60)
            event(dict(event='result',id=i,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
            if r.returncode: errors.append(dict(id=i,error=r.stderr));continue
            assert i not in set(run('image','ls','-aq','--no-trunc').split());removed.append(i)
    after_containers=containers();assert after_containers==before_containers
    after_images=set(run('image','ls','-aq','--no-trunc').split())
    assert before_images-set(removed)<=after_images, 'Unexpected image removed'
    assert sorted(run('volume','ls','-q').split())==before_volumes
    after_usage=usage();after_backing=backing();after_free=free()
    receipt=dict(status='completed' if not errors else 'partial',removed=removed,errors=errors,
                 after_usage=after_usage,after_backing=after_backing,after_free=after_free,
                 docker_unique_layers_bytes_removed=before_usage['layers_size']-after_usage['layers_size'],
                 backing_allocation_reduction=before_backing['allocated_bytes']-after_backing['allocated_bytes'],
                 observed_host_free_increase=after_free-before_free,containers_unchanged=True,volumes_unchanged=True,
                 non_allowlisted_images_preserved=True,limitation='Docker layers, VM allocated blocks, and host free space have different accounting. Shared layers counted once by daemon. Asynchronous trim, APFS and concurrent writes can affect measurements.',completed_unix=time.time())
    save('completed.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['after_usage']},indent=2))
