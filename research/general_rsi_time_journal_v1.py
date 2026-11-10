"""Durable interval evidence for equal-time workers, outside frozen namespaces.

A completed interval has an exact measured duration. An unfinished interval
requires a same-boot supervisor exit timestamp and is charged conservatively
through that timestamp. Recovery preserves every original journal and receipt.
"""
import json
import os
from pathlib import Path
import time


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())


class IntervalJournal:
    def __init__(self, path, attempt, step, spent_ns, now_ns=time.perf_counter_ns):
        self.path, self.attempt, self.step, self.spent_ns = Path(path), attempt, step, spent_ns
        self.now_ns = now_ns
        self.started = None
        self.finished = False

    def tick(self):
        stamp = self.now_ns()
        if self.started is None:
            self.started = stamp
            append(self.path, dict(event='start', attempt=self.attempt, step=self.step,
                                  spent_before_ns=self.spent_ns, stamp_ns=stamp))
        else:
            assert not self.finished and stamp > self.started
            append(self.path, dict(event='end', attempt=self.attempt, step=self.step,
                                  stamp_ns=stamp, duration_ns=stamp-self.started))
            self.finished = True
        return stamp


def accounted_intervals(rows, *, exit_ns=None):
    """Validate one attempt journal. Caller binds it to its boot/exit receipt."""
    output=[]; pending=None
    for row in rows:
        if row['event']=='start':
            assert pending is None
            assert type(row['stamp_ns']) is int and type(row['spent_before_ns']) is int
            assert row['spent_before_ns'] >= 0
            if output:
                assert row['attempt']==output[-1]['attempt']
                assert row['step']==output[-1]['step']+1
                assert row['spent_before_ns']==output[-1]['spent_after_ns']
                assert row['stamp_ns']>=output[-1]['end_stamp_ns']
            pending=row
        elif row['event']=='end':
            assert pending is not None
            assert (row['attempt'],row['step'])==(pending['attempt'],pending['step'])
            duration=row['stamp_ns']-pending['stamp_ns']
            assert duration>0 and duration==row['duration_ns']
            output.append(dict(attempt=row['attempt'],step=row['step'],duration_ns=duration,
                spent_before_ns=pending['spent_before_ns'],spent_after_ns=pending['spent_before_ns']+duration,
                end_stamp_ns=row['stamp_ns'],conservative=False))
            pending=None
        else: raise ValueError('Unknown journal event')
    if pending is not None:
        if type(exit_ns) is not int or exit_ns <= pending['stamp_ns']:
            raise ValueError('Unfinished interval requires audited same-boot supervisor exit timestamp')
        duration=exit_ns-pending['stamp_ns']
        output.append(dict(attempt=pending['attempt'],step=pending['step'],duration_ns=duration,
            spent_before_ns=pending['spent_before_ns'],spent_after_ns=pending['spent_before_ns']+duration,
            end_stamp_ns=exit_ns,conservative=True))
    return output


def recovery_charge(rows, restored_step, restored_spent_ns, *, exit_ns=None):
    intervals=accounted_intervals(rows,exit_ns=exit_ns)
    relevant=[r for r in intervals if r['step']>restored_step]
    if relevant:
        assert relevant[0]['step']==restored_step+1
        assert relevant[0]['spent_before_ns']==restored_spent_ns
    return dict(duration_ns=sum(r['duration_ns'] for r in relevant),
                intervals=relevant, conservative=any(r['conservative'] for r in relevant))
