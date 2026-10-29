import argparse,contextlib,json,resource,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from src.research import cpu_prerequisite_guard as guard
# Capture system readers before importing pytest or test fixtures. Some legacy
# fixtures replace the global shutil module's disk_usage attribute.
disk_usage=shutil.disk_usage
getrusage=resource.getrusage
platform=sys.platform

def actual_observation(root):
    rss=getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(peak_rss_bytes=rss if platform=='darwin' else rss*1024,
                free_bytes=disk_usage(root).free)

guard.observation=actual_observation
p=argparse.ArgumentParser()
p.add_argument('--name',required=True)
p.add_argument('--timeout',required=True,type=int)
p.add_argument('tests',nargs='+')
a=p.parse_args()
evidence=Path('research/tokenizer-evidence/wide-hybrid-preparation-001')
with (evidence/(a.name+'-tests.log')).open('x') as out:
    with contextlib.redirect_stdout(out),contextlib.redirect_stderr(out):
        with guard.CpuPrerequisiteGuard(evidence/(a.name+'-guard.json'),Path.cwd(),timeout_seconds=a.timeout):
            import pytest
            result=pytest.main(['-q',*a.tests,'--basetemp=/tmp/'+a.name+'-fixtures'])
raise SystemExit(result)
