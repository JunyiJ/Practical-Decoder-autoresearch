"""Finite inference follow-up for reviewed context-interaction GQA finalists."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from . import inference_bench as bench
from .protocol import digest, file_digest
from .runner import atomic_json
from .team.suite import source_digest

WORKER_MODULE = 'src.research.context_inference'
STUDY = 'context-interaction-001'
PANEL = [(2, 128), (2, 512), (4, 512)]
SEEDS = [2031, 2032, 2033]
POLICY = dict(decode_ratio_max=.90, decode_iqr_upper_max=1., cache_ratio_max=.75,
              latency_ratio_max=1.05, latency_iqr_upper_max=1.10,
              scope='Every seed and workload must pass; engineering thresholds, not statistical confidence. No automatic adoption or historical quality reselection.')


def panel_jobs(root):
    directory = Path(root)/'research_runs'/STUDY
    report = json.loads((directory/'report.json').read_text())
    original = json.loads((directory/'plan.json').read_text())
    jobs = []
    for layers, capacity in PANEL:
        candidate = f'd{layers}_gqa_c{capacity}'
        reference = f'd{layers}_mha_c{capacity}'
        comparison, = [r for r in report['comparisons'] if r['contrast']=='gqa_minus_mha'
                       and r['candidate']==candidate and r['width']==256]
        if not comparison['quality_eligible'] or comparison['reference'] != reference:
            raise ValueError('Reviewed quality prerequisite failed')
        for seed in SEEDS:
            models = {}
            for role, family, attention in [('reference',reference,'mha'),('candidate',candidate,'gqa')]:
                row, = [r for r in report['records'] if r['family']==family and r['width']==256
                        and r['seed']==seed and r['stage']=='confirmation'
                        and r['lr']==report['selection']['choices'][family]['lr']]
                if row['status'] != 'completed':
                    raise ValueError('Incomplete historical checkpoint')
                models[role] = {k:row[k] for k in ['family','checkpoint','checkpoint_sha256','trial_id','lr']}
                # Request config is small and immutable; verify actual saved config in worker.
                request = json.loads((directory/'trials'/row['trial_id']/'request.json').read_text())
                request['config']['model']['vocab_size'] = 65
                models[role]['attention'] = attention
                models[role]['request_sha256'] = file_digest(directory/'trials'/row['trial_id']/'request.json')
                if models[role]['request_sha256'] != row['request_sha256']:
                    raise ValueError('Historical request changed')
                models[role]['cfg_sha256'] = digest(request['config'])
            for batch in [1,8]:
                for length in ([32,64] if capacity==128 else [128,256]):
                    jobs.append(dict(id=f'{candidate}_s{seed}_b{batch}_p{length}', study=STUDY,
                        width=256,seed=seed,layers=layers,capacity=capacity,models=models,
                        batch=batch,prompt=length,decode=length,warmups=3,repeats=9,
                        first_role_offset=len(jobs)%2,atol=1e-4,rtol=2e-5,data_sha256=original['data_sha256']))
    return jobs


def evaluate_job(job):
    return bench.evaluate_job(job)


def validate_result(job, result):
    bench.validate_result(job, result)


def summarize_cells(records):
    cells = []
    for layers,capacity in PANEL:
        name=f'd{layers}_gqa_c{capacity}'
        rows=[r for r in records if r['models']['candidate']['family']==name]
        if len(rows)!=12 or len({r['id'] for r in rows})!=12:
            raise ValueError('Incomplete cell')
        decode_pass=all(r['summary']['decode']['paired_ratio_median']<=POLICY['decode_ratio_max']
                        and r['summary']['decode']['paired_ratio_iqr'][1]<=POLICY['decode_iqr_upper_max'] for r in rows)
        cache_pass=all(r['correctness']['candidate']['cache_allocated_bytes']/r['correctness']['reference']['cache_allocated_bytes']<=POLICY['cache_ratio_max'] for r in rows)
        latency_guard=all(r['summary'][p]['paired_ratio_median']<=POLICY['latency_ratio_max']
                          and r['summary'][p]['paired_ratio_iqr'][1]<=POLICY['latency_iqr_upper_max']
                          for r in rows for p in ['prefill','decode'])
        cells.append(dict(candidate=name,reference=f'd{layers}_mha_c{capacity}',width=256,
                          quality_eligible_on_common_context128=True,
                          decode_benefit_all_workloads=decode_pass,
                          cache_benefit_all_workloads=cache_pass,
                          latency_guard_all_workloads=latency_guard,
                          operational_efficiency_eligible=latency_guard and (decode_pass or cache_pass),
                          automatic_adoption=False))
    return cells


def verify_plan(plan, check_checkpoints=True):
    source=Path(plan['source_root']);root=Path(plan['research_root'])
    if (plan['study'],plan['job_seconds'],plan['session_seconds'],plan['max_disk_mb'],plan['max_jobs']) != ('context-inference-001',120,3600,512,36):
        raise ValueError('Frozen limits changed')
    if plan['benefit_policy']!=POLICY or plan['holdout_consumed']:
        raise ValueError('Selection policy changed')
    if plan['runtime']!=dict(torch=str(torch.__version__),device='mps',cpu_threads=4) or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':
        raise ValueError('Runtime changed')
    if (source_digest(source)!=plan['source_sha256']
        or subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=plan['git_commit']
        or subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()):
        raise ValueError('Source changed or dirty')
    for item in [plan['knowledge_snapshot'],*plan['artifacts']]:
        if file_digest(Path(item['path']))!=item['sha256']:
            raise ValueError('Frozen artifact changed')
    directory=root/'research_runs'/STUDY
    required={str(directory/f) for f in ['plan.json','selection.json','report.json','review.json']}
    required.add(str(source/'research/CONTEXT_INFERENCE_PROTOCOL.md'))
    required.add(str(root/'research_state/context-inference-validation.json'))
    if not required.issubset({x['path'] for x in plan['artifacts']}):
        raise ValueError('Prerequisite artifact missing')
    if json.loads((directory/'review.json').read_text())['status']!='verified':
        raise ValueError('Historical review missing')
    if plan['jobs']!=panel_jobs(root):
        raise ValueError('Frozen panel changed')
    if check_checkpoints:
        checked=set()
        for job in plan['jobs']:
            for spec in job['models'].values():
                if spec['checkpoint'] in checked:continue
                checked.add(spec['checkpoint'])
                if file_digest(Path(spec['checkpoint']))!=spec['checkpoint_sha256']:
                    raise ValueError('Checkpoint changed')


def build_report(plan, records):
    if len(records)!=36 or [r['id'] for r in records]!=[j['id'] for j in plan['jobs']] or any(r['status']!='completed' for r in records):
        raise ValueError('Incomplete benchmark')
    for job,row in zip(plan['jobs'],records):
        if any(row[k]!=v for k,v in job.items()):raise ValueError('Job changed')
        validate_result(job,row)
    return dict(plan_sha256=digest(plan),records=records,cells=summarize_cells(records),
                benefit_policy=POLICY,holdout_consumed=False,historical_incumbent_unchanged=True,
                scope='Adaptive saved-weight FP32 MPS teacher-forced inference. Quality eligibility is common-context128 only. Timing IQR is descriptive, not population uncertainty. Both models resident; peak device memory unavailable. No automatic adoption.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['worker','run'])
    parser.add_argument('--request',type=Path);parser.add_argument('--plan',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='worker':atomic_json(args.request.parent/'result.json',evaluate_job(json.loads(args.request.read_text())))
    else:bench.run(json.loads(args.plan.read_text()),args.output,backend=sys.modules[__name__])

if __name__=='__main__':main()
