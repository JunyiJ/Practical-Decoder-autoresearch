"""Draft v2 quality-worker resource recording, not a dispatcher or launch gate.

Use inside a separately gated/locked worker under the serial parent and existing
memory watchdog. Samples are not instantaneous limits. An earlier hard kill by
another guard can leave an incomplete trace, which MUST block stage acceptance.
No CLI and no retry path. Tests replace observation with synthetic values only.
"""
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import threading
import time

from .protocol import file_digest
from .tokenizer_wide_hybrid_retention_audit import RESOURCE_LIMITS
from .tokenizer_wide_hybrid_retention_receipts import _sha
from .tokenizer_wide_hybrid_simulation import _publish

EVENT_LIMIT = 2*1024**2
REPORT_LIMIT = 2*1024**2
MAX_SAMPLES = 7203
POLL_SECONDS = 1.


def observation(root):
    import torch
    if (os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1'
            or not torch.backends.mps.is_available()):
        raise RuntimeError('Strict MPS resource observation required')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(peak_rss_bytes=rss if sys.platform == 'darwin' else rss*1024,
                mps_driver_bytes=torch.mps.driver_allocated_memory(),
                free_bytes=shutil.disk_usage(root).free)


def reason(sample):
    if set(sample) != {'elapsed_seconds','peak_rss_bytes','mps_driver_bytes','free_bytes'}:
        raise ValueError('Resource sample schema differs')
    elapsed = sample['elapsed_seconds']
    if type(elapsed) not in (float,int) or not 0 <= elapsed < float('inf'):
        raise ValueError('Invalid elapsed observation')
    for key in ('peak_rss_bytes','mps_driver_bytes','free_bytes'):
        if type(sample[key]) is not int or not 0 <= sample[key] < 2**63:
            raise ValueError('Invalid resource byte observation')
    if sample['peak_rss_bytes'] == 0: raise ValueError('Missing RSS observation')
    if sample['peak_rss_bytes'] >= RESOURCE_LIMITS['rss_limit_bytes']: return 'rss_limit'
    if sample['mps_driver_bytes'] >= RESOURCE_LIMITS['driver_limit_bytes']: return 'driver_limit'
    if sample['free_bytes'] < RESOURCE_LIMITS['reserve_bytes']: return 'disk_reserve'
    if elapsed >= RESOURCE_LIMITS['timeout_seconds']: return 'wall_time_limit'
    return None


def header(plan_sha, request_sha):
    return dict(event='header',kind='wide_hybrid_trial_resources_v2',
                plan_sha256=plan_sha,request_sha256=request_sha,limits=RESOURCE_LIMITS)


class QualityResourceRecorder:
    """Fresh quality attempt only; mark_success binds the completed result.

    Opening this recorder is not permission to train. Caller must already have
    the full v2 frozen-plan, numerical, pilot, storage and worker-lock gates.
    """
    def __init__(self, trial, plan_sha256, request_sha256):
        self.trial = Path(trial).resolve()
        self.plan_sha = _sha(plan_sha256); self.request_sha = _sha(request_sha256)
        self.trace = self.trial/'resource-events.jsonl'
        self.output = self.trial/'resource-audit.json'
        self.samples = []; self.result_sha = None; self.started = None
        self.stream = None; self.thread = None
        self.stop = threading.Event(); self.lock = threading.RLock()

    def _emit(self, event, emergency=False):
        data = (json.dumps(event,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()
        cap = EVENT_LIMIT if emergency else EVENT_LIMIT-4096
        if self.stream.tell()+len(data) > cap:
            raise RuntimeError('Resource event log cap exceeded')
        self.stream.write(data); self.stream.flush(); os.fsync(self.stream.fileno())

    def _sample(self):
        sample = dict(observation(self.trial),elapsed_seconds=round(time.monotonic()-self.started,6))
        violation = reason(sample)
        with self.lock:
            if len(self.samples) >= MAX_SAMPLES: raise RuntimeError('Resource sample cap exceeded')
            if self.samples and sample['elapsed_seconds'] < self.samples[-1]['elapsed_seconds']:
                raise RuntimeError('Resource clock moved backwards')
            self._emit(dict(event='sample',**sample))
            self.samples.append(sample)
        return violation

    def _terminal(self, status, problem):
        with self.lock:
            self._emit(dict(event='terminal',status=status,result_sha256=self.result_sha,
                            reason=problem),emergency=True)
            self.stream.close()
            report = dict(kind='wide_hybrid_trial_resources_v2',status=status,
                plan_sha256=self.plan_sha,request_sha256=self.request_sha,result_sha256=self.result_sha,
                limits=RESOURCE_LIMITS.copy(),violations=[] if problem is None else [problem],
                samples=self.samples,events=dict(path='resource-events.jsonl',sha256=file_digest(self.trace)))
            if len((json.dumps(report,sort_keys=True,allow_nan=False,indent=2)+'\n').encode()) > REPORT_LIMIT:
                raise RuntimeError('Resource report cap exceeded')
            _publish(self.output,report,exclusive=True)

    def _fatal(self, status, problem, code):
        # Own independent FD survives pytest/stdout/stderr redirection. If even
        # diagnostic writing fails, no completed report can be published.
        try:
            self._terminal(status,str(problem)[:1000])
        finally:
            os._exit(code)

    def _watch(self):
        try:
            while not self.stop.wait(POLL_SECONDS):
                violation = self._sample()
                if violation: self._fatal('failed',violation,90)
        except BaseException as exc:
            self._fatal('guard_error',repr(exc),91)

    def __enter__(self):
        if self.started is not None: raise RuntimeError('Resource recorder cannot be reused')
        if self.output.exists() or self.output.is_symlink() or self.trace.is_symlink():
            raise FileExistsError('Existing resource attempt requires review, never retry')
        request = self.trial/'request.json'
        if request.is_symlink() or file_digest(request) != self.request_sha:
            raise ValueError('Resource request binding changed')
        self.stream = self.trace.open('xb')
        self.started = time.monotonic()
        try:
            self._emit(header(self.plan_sha,self.request_sha))
            violation = self._sample()
            if violation: self._fatal('failed',violation,90)
        except BaseException as exc:
            self._fatal('guard_error',repr(exc),91)
        self.thread = threading.Thread(target=self._watch,daemon=True,name='wide-hybrid-resource-recorder')
        self.thread.start()
        return self

    def mark_success(self, result_sha256):
        if self.started is None or self.stop.is_set() or self.result_sha is not None:
            raise RuntimeError('Result binding must occur once inside the active recorder')
        self.result_sha = _sha(result_sha256)

    def __exit__(self, exc_type, exc, tb):
        problem = repr(exc)[:1000] if exc is not None else None
        try:
            if problem is None:
                result = self.trial/'result.json'
                if self.result_sha is None: raise ValueError('Explicit completed result binding missing')
                if (result.is_symlink() or (self.trial/'request.json').is_symlink()
                        or file_digest(result) != self.result_sha
                        or file_digest(self.trial/'request.json') != self.request_sha):
                    raise ValueError('Request/result changed before resource completion')
                if result.stat().st_size > 4*1024**2: raise ValueError('Result metadata cap exceeded')
                saved = json.loads(result.read_text())
                if (saved.get('status') != 'completed' or saved.get('holdout_consumed') is not False
                        or saved.get('request_sha256') != self.request_sha):
                    raise ValueError('Result incomplete or mismatched')
        except BaseException as error:
            problem = repr(error)[:1000]
        self.stop.set(); self.thread.join(timeout=2)
        if self.thread.is_alive(): self._fatal('guard_error','Resource observer did not stop',91)
        try:
            violation = self._sample()
            if violation: self._fatal('failed',violation,90)
            if problem is None and not any(s['mps_driver_bytes'] > 0 for s in self.samples):
                problem = 'No MPS driver observation'
            self._terminal('completed' if problem is None else 'failed',problem)
        except BaseException as error:
            self._fatal('guard_error',repr(error),91)
        if problem is not None and exc is None: raise ValueError(problem)
        return False


def verify_events(report, events):
    """Exact trace-to-summary agreement, used after the trace hash is checked."""
    expected = [header(report['plan_sha256'],report['request_sha256'])]
    expected += [dict(event='sample',**sample) for sample in report['samples']]
    expected += [dict(event='terminal',status='completed',result_sha256=report['result_sha256'],reason=None)]
    if (report['status'] != 'completed' or len(report['samples']) > MAX_SAMPLES
            or report.get('events',{}).get('path') != 'resource-events.jsonl' or events != expected):
        raise ValueError('Resource trace incomplete or differs from summary')
