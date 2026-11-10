"""Prospective equal-time worker primitive. Not used by frozen exposure runs.

The caller puts sampling, transfer, forward/backward, optimizer and averaging
inside one interval. Device synchronization bounds the measured work. Evaluation
and serialization stay outside; the worker must report those costs separately.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import time


@dataclass
class TrainingClock:
    budget_ns: int
    spent_ns: int = 0

    def __post_init__(self):
        if type(self.budget_ns) is not int or self.budget_ns <= 0:
            raise ValueError('budget_ns must be a positive integer')
        if type(self.spent_ns) is not int or self.spent_ns < 0:
            raise ValueError('spent_ns must be a nonnegative integer')

    @property
    def done(self):
        return self.spent_ns >= self.budget_ns

    def lr_ratio(self):
        # LR for the next update, based only on already charged time.
        fraction = self.spent_ns / self.budget_ns
        return 1.0 if fraction <= 2/3 else max(0.1, 1.0-2.7*(fraction-2/3))

    def state_dict(self):
        return dict(budget_ns=self.budget_ns, spent_ns=self.spent_ns)

    @classmethod
    def restore(cls, state, *, uncheckpointed_work_ns=0):
        if type(uncheckpointed_work_ns) is not int or uncheckpointed_work_ns < 0:
            raise ValueError('lost work must be explicitly charged, never subtracted')
        return cls(state['budget_ns'], state['spent_ns']+uncheckpointed_work_ns)

    @contextmanager
    def update(self, synchronize, now_ns=time.perf_counter_ns):
        if self.done:
            raise RuntimeError('training-time budget exhausted')
        synchronize()
        start = now_ns()
        try:
            yield
        finally:
            try:
                synchronize()
            finally:
                duration = now_ns()-start
                if duration < 0:
                    raise RuntimeError('monotonic clock moved backwards')
                self.spent_ns += duration
