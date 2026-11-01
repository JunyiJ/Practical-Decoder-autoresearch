"""Freeze exactly six FFN screens and six paired fresh confirmations after real pilots."""
import json,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone
import torch
from research.ffn_quality_v1 import quality_execution as q,campaign_contract as c,campaign_rules as rules
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
B=q.EVIDENCE;torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g:
    q.idle()
    with CpuPrerequisiteGuard(B/'quality001-freeze-guard.json',c.ROOT,timeout_seconds=120,poll_seconds=.25):
        assert not q.ROOT.exists() and not q.FREEZE.exists()
        integration=read(reference(B/'quality-integration002-review.json'))
        assert integration['status']=='passed' and len(integration['tests'])==76 and integration['protected_pilot_roots_unchanged']
        checked=read(integration['plan']);sources=c.source_inventory();assert sources==checked['sources']
        assert read(reference(B/'pilot001-inference-result.json'))['status']=='resource_pilot_pair_inference_passed'
        prior=read(reference(B/'pilot001-plan.json'));seeds=prior['seeds'];c.frozen_predecessors(sources)
        protected=[]
        for slot,root in enumerate(sorted(checked['protected_pilot_roots'])):
            inventory=checked['protected_pilot_roots'][root]
            assert q.retention.inventory(Path(root))==inventory
            path=B/f'quality001-protected-pilot-{slot:02d}.json'
            q.exclusive(path,dict(root=root,inventory={str(Path(n).relative_to(root)):row for n,row in inventory.items()}));protected.append(reference(path))
        capacity=dict(required_free_bytes=q.FULL_FREE,trajectory_allowances=q.ALLOWANCES,
            aggregate_checkpoint_cap_per_trajectory_bytes=728*1024**2,per_trajectory_process_output_cap_bytes=788953140,
            round_metadata_allowance_bytes=256*1024**2,reserve_bytes=8*1024**3,
            retained_quality_peak_pairs=8,existing_pilot_pairs=2,retained_pilot_weight_bytes=1118322228,
            free_bytes=shutil.disk_usage(c.ROOT).free,
            serialization='Streaming temporary checkpoint exclusively published using a hard link then unlink under census lock. Two names share one inode, not a second physical payload. Aggregate raw+H200 cap728MiB plus bounded per-slot metadata fits788953140B output allowance;256MiB round metadata includes transient control-document overlap.',
            requirement='Carry the conservative remaining pilot projection15629411630B; do not discount unused pilot headroom. Subtract only verified retained quality weight bytes at entry. Losing-screen deletion requires completed exact audit before any fresh run.',
            metadata='Bounded MetadataWriter and process capture reused; scientific outputs capped at788953140B per slot including log and terminal report. No source or corpus copy in trial root.',
            cleanup_credit='No pilot/fixture/failed cleanup credited. At boundary exactly four losing screen pairs eligible after selection; no early screen deletion.')
        assert capacity['free_bytes']>=q.FULL_FREE
        assert 728*1024**2+16*1024**2<min(q.ALLOWANCES.values())
        q.exclusive(B/'quality001-capacity.json',capacity)
        plan=dict(kind=q.KIND,status='frozen_quality_only',root=str(q.ROOT),observed_at=datetime.now(timezone.utc).isoformat(),
            source_commit_before_freeze=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=sources,added_sources=checked['added_sources'],
            environment=c.environment(),data=c.bind_data(),seeds=seeds,screens=rules.screen_specs(seeds),limits=dict(c.LIMITS),allowances=q.ALLOWANCES,
            required_free_bytes=q.FULL_FREE,observed_free_bytes=capacity['free_bytes'],pilot_plan=reference(B/'pilot001-plan.json'),protected_pilot_inventories=protected,
            prerequisites={n:reference(B/path) for n,path in dict(integration='quality-integration002-review.json',reporting_failure='quality-integration001-failure.json',
                inference='pilot001-inference-result.json',pilots='pilot001-review.json',capacity='quality001-capacity.json',seeds='scientific-seed-census-001/result.json').items()},
            finite_contract=dict(screens=6,confirmations=6,steps_per_trial=1200,targets_per_trial=1228800,learning_rates=list(rules.RATES),
                selection='Within each arm minimum full-validation H200 CE; lower LR tie-break. Persist only after all six accepted screens.',
                confirmation='Three reserved fresh pairs with alternating arm order, saved per-arm LR. No grid extension, duplicate control, retries, replacement seeds or interim stopping.',
                statistics=dict(blocks=list(rules.BLOCKS),draws=rules.DRAWS,quantile=rules.GATE_QUANTILE,
                    gate='UngatedSiLU minus SwiGLU: mean<-0.02, every fresh delta<0, each simultaneous conditional-text upper<0. No cost-only adoption. Separate final review; population-seed/holdout inference unsupported.',adoption_automatic=False)),
            confirmation_sequences=[dict(selection=dict(zip(rules.ARMS,(a,b))),specs=rules.confirmation_specs(seeds,dict(zip(rules.ARMS,(a,b))))) for a in rules.RATES for b in rules.RATES],
            capacity_accounting=capacity,
            retention='After complete independently re-audited screens and saved LR selection, delete only exact four losing raw/H200 pairs via size/SHA/path allowlist, fsynced intent/deletion journal and protected-evidence verification under shared/campaign locks. All selected/fresh pairs and both pilots retained. Failed/partial cleanup blocks all fresh entry; no retry.',
            provenance='Actual child argv/request/previous acceptance bound; clean process reaping, guarded worker output and independent parent raw/optimizer/EMA/export/full-validation/cache recomputation precede every anchor. Existing unaccepted attempts cannot retry. No historical source mutation or campaign reopening.',
            holdout='No test scoring, training test batch, or test-based decision; unchanged trainer materializes test tokens. Evaluator and inference use train/validation-only view.',
            future_inference='Measure all three final confirmation pairs with frozen FFN inference contract, descriptive only; no cost-only adoption.',
            quality_adopted=False,builder=reference(Path(__file__).resolve()))
        q.ROOT.mkdir()
        with session_lock(q.ROOT) as p:
            writer=MetadataWriter(q.ROOT,(g,p));path=writer.publish_json('evidence/meta/quality-plan.json',plan)
            with (B/'quality001-plan.json').open('xb') as f:f.write(path.read_bytes())
            q.exclusive(q.FREEZE,dict(kind=q.KIND,plan=reference(path),portable=reference(B/'quality001-plan.json'),builder=reference(Path(__file__).resolve())))
            digest=reference(q.FREEZE)['sha256'];assert q.bind(writer,digest)==plan
            current=q.state(plan,digest);assert not current['records'] and q.next_spec(plan,current)==rules.screen_specs(seeds)[0]
            q.exclusive(B/'quality001-preflight.json',dict(status='frozen_quality_entry_bound',freeze=reference(q.FREEZE),accepted_prefix=0,next_spec=q.next_spec(plan,current),
                sources=len(sources),free_bytes=capacity['free_bytes'],training_launched=False))
    print(json.dumps(dict(status='finite_quality_frozen',freeze=reference(q.FREEZE),free_bytes=capacity['free_bytes'])),flush=True)
