"""Disposable metadata subprocess regression; not an inference worker/entry."""
import json
from pathlib import Path
import sys
import time

from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_activation_timing import slot
from research.activation_inference_recovery_v2.inventory import global_observation,slot_inventory,require_slot_unchanged

root=Path(sys.argv[1]);cleanup.GLOBAL_LOCK=Path(sys.argv[2]);fds=tuple(map(int,sys.argv[3:5]))
prior_seed,current_seed=map(int,sys.argv[5:7]);case=sys.argv[7]
writer=MetadataWriter(root,fds)
current=slot(current_seed)
# Parent has a real process group, bounded pipe, periodic control replacement and
# output accounting. No model/corpus/weights are involved in this probe.
before=global_observation(writer);proof=slot_inventory(writer,prior_seed)
control=f'control/process-inference-{current}.json'
with writer._serialized():control_before=(root/control).read_bytes()
print('x'*8192,flush=True)
started=time.monotonic()
while True:
    with writer._serialized():
        ack=json.loads((root/f'control/recovery-entry-inference-{current}.json').read_text())
        monitor=json.loads((root/control).read_text())
    if ack['ack'] and monitor['observation']['log_bytes']>=8193:break
    if time.monotonic()-started>=5:raise TimeoutError('Disposable monitor handshake timeout')
    time.sleep(.01)
with writer._serialized():control_after=(root/control).read_bytes()
assert control_before!=control_after,'Real parent monitor must change during prior review'
if case=='allowed':
    after=global_observation(writer)
    assert before!=after,'The old whole-root equality would reject this valid overlap'
    assert require_slot_unchanged(writer,prior_seed,proof)==proof
    result=dict(case=case,status='passed',old_whole_root_equality=False,immutable_slot_equal=True,
                real_parent_monitor_changed=True,prior_seed=prior_seed,current_seed=current_seed)
else:
    try:require_slot_unchanged(writer,prior_seed,proof)
    except ValueError as exc:
        result=dict(case=case,status='rejected_expected',error=str(exc),real_parent_monitor_changed=True)
    else:raise AssertionError('Injected immutable/unknown output change was accepted')
# Fault cases include forbidden outputs, so emit a small report through the
# supervisor's bounded pipe rather than invoking an acceptance-capable writer.
print('PROBE_RESULT '+json.dumps(result,sort_keys=True),flush=True)
