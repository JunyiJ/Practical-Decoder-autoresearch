"""Preserve and verify the completed frozen six-screen boundary; no deletion/training."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import torch
from research.ffn_norm_quality_v1 import quality_execution as q, campaign_contract as c
from research.block_norm_v1.fixture_dispatch import reference, read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

SHA = "4a10df86ada4f27e89911f6ee231946713e277f9c3eb5f33b683f09e8fbdad26"
B = q.EVIDENCE
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g, session_lock(q.ROOT) as lock:
    q.idle()
    guardpath = B / "quality001-boundary-record-guard.json"
    resultpath = B / "quality001-boundary-result.json"
    c.require(not guardpath.exists() and not resultpath.exists(), "Existing recording cannot retry")
    with CpuPrerequisiteGuard(guardpath, c.ROOT, timeout_seconds=120, poll_seconds=.25):
        writer = MetadataWriter(q.ROOT, (g, lock))
        plan = q.bind(writer, SHA)
        current = q.state(plan, SHA)
        c.require(len(current["records"]) == 6 and current["cleanup"] is not None, "Completed six-screen boundary required")
        execution = read(reference(q.cleanup_path()))
        guard = read(reference(q.ROOT / "control/screen-boundary-guard.json"))
        c.require(guard["status"] == "finished" and guard["reason"] is None, "Boundary guard failed")
        c.require(execution["status"] == "completed" and execution["deleted_count"] == 8, "Exactly four losing pairs required")
        selection = read(reference(q.selection_path()))
        c.require(selection["value"] == q.rules.choose_rates(plan["seeds"], current["records"]), "Selection differs")
        dest = B / "quality001-screen-retention"
        dest.mkdir()
        copies = []
        sources = list(sorted(q.cleanup_path().parent.rglob("*"))) + [q.selection_path(), q.ROOT / "control/screen-boundary-guard.json"]
        for source in sources:
            if not source.is_file():
                continue
            c.require(source.suffix != ".pt" and source.stat().st_size <= 4 * 1024**2 and not source.is_symlink(), "Bounded nonweight evidence only")
            used = sum(p.stat().st_size for p in B.rglob("*") if p.is_file() and p.relative_to(B).parts[0].startswith("quality001"))
            c.require(used + source.stat().st_size <= 32 * 1024**2, "External evidence cap")
            target = dest / source.name
            with target.open("xb") as f:
                f.write(source.read_bytes())
            c.require(reference(target)["sha256"] == reference(source)["sha256"], "Copied evidence changed")
            copies.append(dict(original=reference(source), preserved=reference(target)))
        required = max(8 * 1024**3, q.FULL_FREE - current["used_weights"])
        free = shutil.disk_usage(c.ROOT).free
        value = dict(status="completed", at=datetime.now(timezone.utc).isoformat(), runner=reference(Path(__file__).resolve()),
            freeze=dict(path=str(q.FREEZE), sha256=SHA), accepted_prefix=6, records=current["records"],
            selection=selection, execution=execution, copies=copies, retained_quality_weight_bytes=current["used_weights"],
            remaining_required_free_bytes=required, observed_free_bytes=free, capacity_fits=free >= required,
            next_spec=q.next_spec(plan, current), quality_adopted=False, holdout_scored=False,
            training_launched=False, deletion_performed_by_this_recording=False)
        q.exclusive(resultpath, value)
    print(json.dumps(dict(status=value["status"], selection=selection["value"], reclaimed_bytes=execution["deleted_logical_bytes"], capacity_fits=value["capacity_fits"], next_spec=value["next_spec"])), flush=True)
