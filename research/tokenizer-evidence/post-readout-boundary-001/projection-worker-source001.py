"""New finite config/serializer gate; no model, corpus or checkpoint construction."""
import argparse
import copy
import io
import json
import os
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT, require, fingerprint
from research.body_allocation_model_v1.evidence import reference
from research.readout_projection_model_v1.contract import ARMS
from research.readout_projection_resume_v1.data import config
from research.readout_projection_resume_v1.checkpoint import FILE_CAPS, RAW, H200
from research.readout_projection_resume_v1.contract import runtime
from research.readout_projection_worker_v1.training import validate_config, Counter
from research.readout_projection_worker_v1.execution import sources, checked, idle
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock

E = Path(__file__).resolve().parent
R = E/'projection-worker-source001'
PREREQUISITES = ['projection-resume001/result.json', 'projection-resume001/mps-guard.json',
    'projection-resume001/STOP', 'projection-resume001-review.json', 'projection-resume001-archive-closure.json']


def publish(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    require(sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw) <= 2*1024**2, 'Metadata cap')
    with (R/name).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan-sha', required=True); arg = ap.parse_args()
    require(reference(R/'plan.json')['sha256'] == arg.plan_sha and set(p.name for p in R.iterdir()) == {'plan.json'},
            'One pristine exact-plan source gate required')
    with session_lock(GLOBAL_LOCK.parent):
        idle(); torch.set_num_threads(4); runtime()
        plan = json.loads((R/'plan.json').read_text()); require(sources() == plan['sources'], 'Pinned source inventory differs')
        census = E/'projection-worker-seed-census001'
        receipt = checked(plan['census'])
        require(plan['census'] == reference(census/'result.json') and not receipt['prior_matches']
            and receipt['status'] == 'new_projection_worker_chain_roles_unused_before_construction'
            and checked(reference(census/'STOP'))['status'] == 'complete_new_projection_worker_chain_role_census',
            'Original successful eight-role census required')
        publish('attempt.json', dict(pid=os.getpid(), plan_sha256=arg.plan_sha))
        try:
            with CpuPrerequisiteGuard(R/'guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
                initial = torch.get_rng_state().clone(); roles = receipt['roles']; rows = []; rejections = 0
                for arm in ARMS:
                    for steps in [4, 20]:
                        prefix = 'four' if steps == 4 else 'twenty'
                        cfg = config(arm, steps, roles[prefix+'_initializer'], roles[prefix+'_train_global_rng'], roles[prefix+'_eval'])
                        cfg.training.eval_every = 1 if steps == 4 else 5; cfg.training.eval_iters = 1
                        cfg.readout_projection_execution = dict(kind='projection_prerequisite_execution_v1',
                            sources_sha256=fingerprint(plan['sources']), runner_sha256=reference(E/'projection-worker001.py')['sha256'])
                        require(validate_config(cfg) == steps, 'Exact finite config rejected')
                        for kind in ['horizon', 'legacy', 'binding', 'clipping', 'cadence']:
                            bad = copy.deepcopy(cfg)
                            if kind == 'horizon': bad.training.max_iters = 1200
                            elif kind == 'legacy': bad.model = {}
                            elif kind == 'binding': bad.readout_projection_execution.kind = 'legacy'
                            elif kind == 'clipping': bad.training.grad_clip = .5
                            else: bad.training.eval_every = 2
                            try: validate_config(bad)
                            except ValueError: rejections += 1
                            else: raise AssertionError('Forbidden config accepted: '+kind)
                        # Four RAW copies: model + conservative two moments + one EMA.
                        p = ARMS[arm][0]; overhead = 8*1024**2; margin = 8*1024**2
                        raw_bound = 4*p*4+overhead+margin; h200_bound = p*4+overhead+margin
                        require(raw_bound <= FILE_CAPS[RAW] and h200_bound <= FILE_CAPS[H200]
                            and raw_bound+h200_bound <= 640*1024**2, 'Each archive and complete pair must fit strict caps')
                        rows.append(dict(arm=arm, steps=steps, parameters=p, raw_upper_bytes=raw_bound, h200_upper_bytes=h200_bound,
                            raw_cap=FILE_CAPS[RAW], h200_cap=FILE_CAPS[H200], pair_cap=640*1024**2,
                            overhead_per_archive=overhead, additional_margin_per_archive=margin))
                payload = dict(kind='generated_projection_counter_only', x=torch.tensor([.375, -.625, 1.75], dtype=torch.float32))
                counter = Counter(2*1024**2); torch.save(payload, counter); buffer = io.BytesIO(); torch.save(payload, buffer)
                require(counter.size == len(buffer.getvalue()), 'Serializer counter differs')
                small = Counter(5); require(small.write(b'pqrsu') == 5, 'Counter boundary differs')
                try: small.write(b'!')
                except ValueError: pass
                else: raise AssertionError('Counter exceeded cap')
                require(torch.equal(initial, torch.get_rng_state()) and sources() == plan['sources'], 'Source/global CPU RNG changed')
                refs = {n: reference(E/n) for n in PREREQUISITES}
                publish('result.json', dict(status='passed_projection_worker_source_scope_and_output_bounds', sources=plan['sources'],
                    prepared_builder=reference(E/'projection-worker001.py'), prerequisites=refs, census=plan['census'], rows=rows,
                    forbidden_configs_rejected=rejections, counter_bytes=counter.size, counter_overflow_rejected=True,
                    model_optimizer_corpus_checkpoint_weightfile_quality_seed_calls=0, private_generator_seed_calls=0,
                    source_global_cpu_rng_unchanged=True,
                    limit='Analytical four/twenty payload bounds and tiny in-memory serializer only. Actual archives and public native scorer remain unqualified until independent real-worker acceptance.'))
            publish('STOP', dict(status='complete_projection_worker_source_scope_output_gate', no_replay=True))
        except BaseException as exc:
            if not (R/'STOP').exists(): publish('STOP', dict(status='closed_failed', no_retry=True, error=repr(exc)))
            raise


if __name__ == '__main__':
    main()
