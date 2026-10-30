"""Recorder checks only; never runs the actual-scale diagnostic."""
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def trace():
    path=Path(__file__).resolve().parents[1]/'research/tokenizer-evidence/wide-hybrid-preparation-001/run_metadata_contention_trace_v3.py'
    spec=importlib.util.spec_from_file_location('contention_trace_v3',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    writer=SimpleNamespace(_serialized=contextlib.nullcontext)
    return module.Trace(writer)


def test_json_piece_volume_is_aggregated_without_spending_detail_budget(trace):
    for i in range(20000):
        trace.record(dict(kind='operation',operation='metadata._write',duration=i/1000000,
                          bytes=4,error=None))
    report=trace.report();stat=report['operations']['metadata._write']
    assert report['records']==[] and report['complete']
    assert stat['count']==20000 and stat['total_bytes']==80000
    assert stat['max_seconds']==.019999 and stat['slowest']['duration']==.019999


def test_overflow_is_bounded_and_does_not_replace_original_failure(trace):
    for _ in range(16385):
        trace.record(dict(kind='lock',wait=.01,hold=.02,error=None))
    @contextlib.contextmanager
    def fails():
        raise TimeoutError('original lock timeout')
        yield
    trace.original=fails
    with pytest.raises(TimeoutError,match='original lock timeout'):
        with trace.context():pass
    report=trace.report()
    assert len(report['records'])==16384 and report['dropped_records']==2
    assert not report['complete']
    assert 'original lock timeout' in report['failures'][0]['error']
    assert report['failures'][0]['hold'] is None


def test_failure_archive_has_an_explicit_finite_bound(trace):
    for _ in range(65):
        trace.record(dict(kind='operation',operation='os.fsync',duration=.1,error='failed'))
    report=trace.report()
    assert len(report['failures'])==64 and report['dropped_failures']==1
    assert report['operations']['os.fsync']['errors']==65 and not report['complete']
