"""Serial, restartable finite GPT-2 tokenizer transfer experiment.

The controller never accesses test losses, silently retries failures, or modifies
its frozen plan. A new research round is a human/agent decision after review.
"""
import argparse
import copy
import gc
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from omegaconf import OmegaConf

from .protocol import file_digest, digest
from .runner import atomic_json, session_lock, source_fingerprint

ROOT = Path(__file__).resolve().parents[2]


def runtime_identity():
    return dict(python=sys.version, executable=sys.executable,
                packages={p: importlib.metadata.version(p) for p in ('torch','numpy','tiktoken','omegaconf')})


def trial_config(base, output, width, seed, lr, steps):
    cfg = copy.deepcopy(base)
    cfg['model'].update(dim=width, n_heads=width//32)
    t = cfg['training']
    t.update(seed=seed, learning_rate=lr, max_iters=steps,
             eval_every=300 if steps==3000 else steps,
             checkpoint_dir=str(output/'checkpoints'), metrics_file=str(output/'metrics.jsonl'),
             summary_file=str(output/'training-summary.json'))
    t['lr_schedule'].update(total_steps=steps,start_step=2000 if steps==3000 else steps//2)
    t['weight_averaging'].update(total_steps=steps,start_step=t['lr_schedule']['start_step'],
                                 interval=10 if steps==3000 else 1)
    return cfg


def make_config(output, width=64, seed=2601, lr=.01, steps=3000, device='mps'):
    from ..train.train import _make_loader
    from ..models.output_bias import tokenizer_no_bias_prior, TOKENIZER_POLICY
    template = ROOT/'research/snapshots/2026-10-27/incumbent-request.original.json'
    cfg = OmegaConf.create(json.loads(template.read_text())['config'])
    cfg.data.path = str(ROOT/'data/raw/shakespeare.txt')
    cfg.data.tokenizer = 'gpt2'
    cfg.model.dim, cfg.model.n_heads = width, width//32
    cfg.training.seed = seed
    cfg.training.learning_rate = lr
    cfg.training.device = device
    cfg.training.max_iters = steps
    cfg.training.eval_every = 300 if steps == 3000 else steps
    cfg.training.eval_iters = 10  # Diagnostic only; full validation is primary.
    cfg.training.checkpoint_dir = str(output/'checkpoints')
    cfg.training.metrics_file = str(output/'metrics.jsonl')
    cfg.training.summary_file = str(output/'training-summary.json')
    cfg.training.lr_schedule.total_steps = steps
    cfg.training.lr_schedule.start_step = 2000 if steps == 3000 else steps//2
    cfg.training.weight_averaging.total_steps = steps
    cfg.training.weight_averaging.start_step = cfg.training.lr_schedule.start_step
    cfg.training.weight_averaging.interval = 10 if steps == 3000 else 1
    loader = _make_loader(cfg, 'cpu')
    cfg.model.vocab_size = loader.vocab_size
    cfg.model.output_bias_width_policy = TOKENIZER_POLICY
    cfg.model.output_bias_prior = tokenizer_no_bias_prior(loader, cfg.data.path)
    return OmegaConf.to_container(cfg, resolve=True)


def trial_id(stage, width, seed, lr):
    return f'{stage}-w{width}-s{seed}-lr{lr:g}'


def choose_screen(records, plan):
    """Frozen choice uses only paired small-width screen seeds."""
    rows = {r['id']: r for r in records}
    baseline = plan['baseline_lr']
    means, eligible = {}, []
    for lr in plan['learning_rates']:
        losses, deltas = [], []
        for seed in plan['screen_seeds']:
            a = rows[trial_id('screen', 64, seed, lr)]['h200']['mean_ce']
            b = rows[trial_id('screen', 64, seed, baseline)]['h200']['mean_ce']
            losses.append(a); deltas.append(a-b)
        means[str(lr)] = sum(losses)/len(losses)
        if sum(deltas)/len(deltas) < -plan['min_gain'] and all(d < 0 for d in deltas):
            eligible.append(lr)
    selected = min(eligible, key=lambda lr: (means[str(lr)], lr)) if eligible else baseline
    return dict(selected_lr=selected, screen_means=means,
                rule='Mean CE gain > 0.02 and every paired screen seed improves; otherwise retain 0.01.')


def specs(plan, selection=None):
    rows = [('screen', 64, s, lr) for i, s in enumerate(plan['screen_seeds'])
            for lr in (plan['learning_rates'] if i % 2 == 0 else list(reversed(plan['learning_rates'])))]
    if selection is not None:
        rates = sorted({plan['baseline_lr'], selection['selected_lr']})
        rows += [('confirm', w, s, lr) for w in plan['widths']
                 for i, s in enumerate(plan['confirm_seeds'])
                 for lr in (rates if i % 2 == 0 else list(reversed(rates)))]
    return [dict(id=trial_id(stage,w,s,lr), stage=stage, width=w, seed=s, lr=lr)
            for stage,w,s,lr in rows]


def prepare(output, pilot=False):
    output.mkdir(parents=True, exist_ok=False)
    cfg = make_config(output/'template')
    from ..train.train import _make_loader
    loader = _make_loader(OmegaConf.create(cfg), 'cpu')
    plan = dict(version=1, root=str(ROOT), output=str(output), pilot=pilot,
        source=source_fingerprint(), runtime=runtime_identity(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config=cfg, baseline_lr=.01, learning_rates=[.01,.005,.0025],
        screen_seeds=[2601,2602], confirm_seeds=[2611,2612,2613], widths=[64,256],
        steps=40 if pilot else 3000, min_gain=.02, job_seconds=600 if pilot else 7200,
        min_free_disk_mb=8192, max_output_mb=12288, cpu_threads=4,
        corpus_sha256=file_digest(Path(cfg['data']['path'])),
        split_tokens={k:len(v) for k,v in loader.splits.items()},
        split_bytes={k:len(v.encode()) for k,v in loader.split_texts.items()},
        tokenizer=loader.tokenizer_contract,
        prior_evidence={str(p.relative_to(ROOT)):file_digest(p) for p in [
            ROOT/'research/RESEARCH_HANDOFF.md',ROOT/'research/LEARNING_GUIDE.md',
            ROOT/'research/snapshots/2026-10-27/research-metadata.tar.gz']},
        scope='New GPT-2 subword experiment. Character CE is not comparable. Test losses remain closed.')
    atomic_json(output/'plan.json', plan)
    atomic_json(output/'state.json',dict(status='prepared',records=[],selection=None,plan_sha256=digest(plan)))
    return plan


def worker(request_path):
    import torch
    from .memory_guard import start_memory_guard
    from .coverage_eval import score_validation
    from ..train.train import run_training, _make_loader
    from ..train.weight_averaging import export_averaged_checkpoint
    from ..utils.checkpoint import read_checkpoint, load_model_weights
    from ..models.gpt import GPT
    request=json.loads(request_path.read_text())
    torch.set_num_threads(4)
    start_memory_guard()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    cfg=OmegaConf.create(request['config'])
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1': raise ValueError('MPS fallback forbidden')
    started=time.monotonic()
    result=run_training(cfg)
    if result['completed_steps'] != cfg.training.max_iters:
        raise ValueError('Incomplete training budget')
    raw=Path(result['final_checkpoint'])
    gc.collect(); torch.mps.empty_cache()
    averaged=export_averaged_checkpoint(raw,raw.parent/'h200.pt',200)
    loader=_make_loader(cfg,'cpu')
    evaluations={}
    for name,path in [('raw',raw),('h200',Path(averaged['checkpoint']))]:
        checkpoint=read_checkpoint(path)
        model=GPT(OmegaConf.create(checkpoint['cfg']['model'])).to(cfg.training.device)
        load_model_weights(model,checkpoint); del checkpoint
        score=score_validation(model,loader.val_data,128,8,2048)
        target_bytes=sum(len(loader.enc.decode_single_token_bytes(int(i))) for i in loader.val_data[1:])
        score['target_bytes']=target_bytes
        score['bits_per_byte']=sum(score['block_loss_sums'])/target_bytes/math.log(2)
        score['perplexity']=math.exp(score['mean_ce'])
        score['checkpoint']=str(path); score['checkpoint_sha256']=file_digest(path)
        score['parameters']=sum(p.numel() for p in model.parameters())
        if name == 'h200':
            from ..utils.generate import generate
            torch.manual_seed(12345)
            prompt='ROMEO:\n'
            ids=torch.tensor([loader.encode(prompt)],device=cfg.training.device)
            score['sample']=loader.decode(generate(model,ids,96,temperature=.8,top_k=40)[0])
            score['sample_settings']=dict(prompt=prompt,seed=12345,temperature=.8,top_k=40,new_tokens=96)
        evaluations[name]=score
        del model
        gc.collect();torch.mps.empty_cache()
    atomic_json(Path(request['result_path']),dict(**request['spec'],status='completed',
        **evaluations,training=result,wall_seconds=time.monotonic()-started,
        holdout_consumed=False,request_sha256=file_digest(request_path)))


def conclusion(records,selection,plan):
    from .coverage_eval import paired_blocks
    if selection['selected_lr']==plan['baseline_lr']:
        return dict(adopted=False,reason='No screen candidate qualified; inherited LR retained.', comparisons=[])
    comparisons=[]
    for w in plan['widths']:
        arms={lr:{r['seed']:r['h200'] for r in records if r['stage']=='confirm' and r['width']==w and r['lr']==lr}
              for lr in [selection['selected_lr'],plan['baseline_lr']]}
        a,b=arms[selection['selected_lr']],arms[plan['baseline_lr']]
        intervals=[paired_blocks(a,b,n) for n in [2048,4096,8192]]
        good=(intervals[0]['mean_delta'] < -plan['min_gain'] and
              all(d<0 for d in intervals[0]['paired_seed_deltas'].values()) and
              all(p['conditional_text_bootstrap_95pct'][1]<0 for p in intervals))
        comparisons.append(dict(width=w,passes=good,intervals=intervals))
    return dict(adopted=all(c['passes'] for c in comparisons),comparisons=comparisons,
                reason='Frozen paired-seed and conditional-text rule; adaptive validation, not independent test evidence.')


def run(output):
    plan=json.loads((output/'plan.json').read_text())
    state=json.loads((output/'state.json').read_text())
    if state['status'] in {'failed','completed'}: return state
    if state['plan_sha256']!=digest(plan) or source_fingerprint()!=plan['source']:
        raise ValueError('Frozen plan/source changed; use a new reviewed campaign')
    if runtime_identity()!=plan['runtime']:
        raise ValueError('Runtime changed')
    if file_digest(Path(plan['config']['data']['path']))!=plan['corpus_sha256']:
        raise ValueError('Corpus changed')
    def stop(signum, frame):
        (output/'STOP').write_text(f'Human/process stop signal {signum}\n')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    global_lock = ROOT/'research_state/tokenizer-worker-lock'
    global_lock.mkdir(parents=True, exist_ok=True)
    with session_lock(output) as study_fd, session_lock(global_lock) as worker_fd:
        while True:
            if source_fingerprint()!=plan['source'] or file_digest(Path(plan['config']['data']['path']))!=plan['corpus_sha256']:
                raise ValueError('Frozen source or corpus changed between trials')
            if (output/'STOP').exists():
                state['status']='paused';atomic_json(output/'state.json',state);return state
            queue=specs(plan,state['selection']) if not plan['pilot'] else [
                dict(id=f'pilot-w{w}',stage='pilot',width=w,seed=2591,lr=.01) for w in plan['widths']]
            done={r['id'] for r in state['records']}
            pending=[s for s in queue if s['id'] not in done]
            if not pending:
                if not plan['pilot'] and state['selection'] is None:
                    state['selection']=choose_screen(state['records'],plan)
                    atomic_json(output/'selection.json',state['selection'])
                    atomic_json(output/'state.json',state);continue
                state['status']='completed'
                state['conclusion']=None if plan['pilot'] else conclusion(state['records'],state['selection'],plan)
                atomic_json(output/'state.json',state);return state
            spec=pending[0]
            if shutil.disk_usage(output).free/1024**2 < plan['min_free_disk_mb']:
                raise RuntimeError('Free disk below frozen reserve')
            size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())/1024**2
            if size>plan['max_output_mb']:raise RuntimeError('Output disk budget exhausted')
            trial=output/spec['id']
            # An interrupted/incomplete worker requires diagnosis, never overwrite it.
            if trial.exists():raise RuntimeError(f'Existing incomplete trial needs review: {trial}')
            trial.mkdir()
            config=trial_config(plan['config'],trial,spec['width'],spec['seed'],spec['lr'],plan['steps'])
            request=dict(spec=spec,config=config,result_path=str(trial/'result.json'))
            atomic_json(trial/'request.json',request)
            state.update(status='running',active=spec,controller_pid=os.getpid())
            atomic_json(output/'state.json',state)
            print(f'START {spec["id"]}',flush=True)
            started=time.monotonic()
            with (trial/'worker.log').open('x') as log:
                process=subprocess.Popen([sys.executable,'-m',__name__ if __name__!='__main__' else 'src.research.tokenizer_study',
                    'worker','--output',str(trial/'request.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,pass_fds=(study_fd,worker_fd))
                state['worker_pid']=process.pid;atomic_json(output/'state.json',state)
                try: code=process.wait(timeout=plan['job_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    code=124
            if code or not (trial/'result.json').exists():
                record=dict(**spec,status='failed',exit_code=code,wall_seconds=time.monotonic()-started,log=str(trial/'worker.log'))
                state['records'].append(record);state['status']='failed'
                atomic_json(output/'state.json',state);print(json.dumps(record),flush=True);return state
            record=json.loads((trial/'result.json').read_text())
            state['records'].append(record);state['active']=None
            atomic_json(output/'state.json',state)
            print(f'DONE {spec["id"]}: H200 CE={record["h200"]["mean_ce"]:.6f}, BPB={record["h200"]["bits_per_byte"]:.6f}',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','worker','status'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',action='store_true')
    args=p.parse_args();output=args.output.resolve()
    if args.command=='prepare':prepare(output,args.pilot)
    elif args.command=='worker':worker(output)
    elif args.command=='run':run(output)
    else:
        state=json.loads((output/'state.json').read_text())
        print(json.dumps({k:v for k,v in state.items() if k!='records'},indent=2))
        for r in state['records']:print(r['id'],r['status'],r.get('h200',{}).get('mean_ce'))


if __name__=='__main__':main()
