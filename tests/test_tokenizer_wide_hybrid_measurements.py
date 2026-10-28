"""Tiny deterministic two-logit CPU recurrence; never a campaign model/retry."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from src.research import tokenizer_wide_hybrid_measurements as m


class Cache(nn.Module):
    def __init__(self, batch, bad_reset=False):
        super().__init__();self.register_buffer('state',torch.zeros(batch,1));self.position=0
        self.bad_reset=bad_reset
    def reset(self):
        self.position=0
        if not self.bad_reset:self.state.zero_()


class Toy(nn.Module):
    def __init__(self, bad=None, events=None):
        super().__init__();self.bad=bad;self.calls=[];self.events=events
    def forward(self, ids, cache=None, start_pos=0):
        assert not torch.is_grad_enabled()
        self.calls.append((cache is not None,start_pos,ids.shape[1]))
        if self.events is not None:self.events.append(('forward',start_pos,ids.shape[1]))
        x=ids.float().cumsum(dim=1)
        if cache is not None:
            c=cache[0];assert c.position==start_pos
            x=x+c.state;c.state.copy_(x[:,-1:]);c.position+=ids.shape[1]
            if self.bad=='drift' and start_pos==95:x=x+1
        logits=torch.stack((x,-x),dim=-1)
        if self.bad=='nan':logits[:]=float('nan')
        if self.bad=='inf':logits[:]=float('inf')
        return logits,None


def ids(batch):return torch.ones(batch,96,dtype=torch.int64)


@pytest.mark.parametrize('batch',[1,8])
def test_full_cached_reset_replay_all96(batch):
    model=Toy().eval();caches=[]
    def factory():
        c=Cache(batch);caches.append(c);return [c]
    result=m.check_kernel(model,ids(batch),factory)
    assert result==dict(passed=True,max_absolute_error=0.)
    sequence=[(True,lo,hi-lo) for lo,hi in m.CHUNKS]
    assert model.calls==[(False,0,96)]+sequence*2
    assert len(caches)==1 and caches[0].position==96
    assert torch.equal(caches[0].state,torch.full((batch,1),96.))


@pytest.mark.parametrize('bad',['drift','nan','inf','reset'])
def test_wrong_cache_or_nonfinite_logits_rejected(bad):
    with pytest.raises((ValueError,AssertionError)):
        m.check_kernel(Toy(bad).eval(),ids(1),lambda:[Cache(1,bad_reset=bad=='reset')])


@pytest.mark.parametrize('batch',[1,8])
def test_timing_boundaries_positions_and_fresh_cache(batch):
    events=[];model=Toy(events=events).eval();caches=[]
    def factory():
        events.append('allocate');c=Cache(batch);caches.append(c);return [c]
    times=iter([10.,12.,20.,25.,30.,32.,40.,45.])
    def clock():events.append('clock');return next(times)
    def sync():events.append('sync')
    def memory():events.append('memory');return (1000,2000)
    for _ in range(2):
        result=m.measure_kernel(model,ids(batch),factory,synchronize=sync,clock=clock,memory=memory)
        assert result==dict(prefill_seconds=2.,decode_seconds=5.,state_bytes=batch*4,
            observed_mps_current_bytes=1000,observed_mps_driver_bytes=2000)
    expected=['allocate','sync','clock',('forward',0,32),'sync','clock','clock']
    expected += [('forward',p,1) for p in range(32,96)]
    expected += ['sync','clock','memory']
    assert events==expected*2 and caches[0] is not caches[1]
    assert all(c.position==96 for c in caches)


@pytest.mark.parametrize('kind',['batch','length','dtype'])
def test_bad_input_rejected_before_model(kind):
    data=ids(1)
    if kind=='batch':data=data.repeat(2,1)
    elif kind=='length':data=data[:,:95]
    else:data=data.float()
    model=Toy()
    with pytest.raises(ValueError):m.check_kernel(model,data,lambda:[Cache(1)])
    assert not model.calls


def test_timing_exception_stops_without_retry_or_memory_read():
    events=[]
    class Broken(Toy):
        def forward(self,ids,cache=None,start_pos=0):
            if start_pos==35:raise RuntimeError('synthetic decode failure')
            return super().forward(ids,cache,start_pos)
    model=Broken(events=events)
    with pytest.raises(RuntimeError,match='synthetic'):
        m.measure_kernel(model,ids(1),lambda:[Cache(1)],synchronize=lambda:None,
                         clock=lambda:0.,memory=lambda:events.append('memory'))
    assert events[-1]==('forward',34,1) and 'memory' not in events


@pytest.mark.parametrize('fallback,available',[('1',True),('0',False)])
def test_public_adapter_rejects_unavailable_or_fallback_mps(monkeypatch,fallback,available):
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK',fallback)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:available)
    with pytest.raises(RuntimeError,match='Strict MPS'):m.mps_callbacks({}, {}, [], '')


def test_public_adapter_rejects_missing_families(monkeypatch):
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK',raising=False)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    with pytest.raises(ValueError,match='three'):m.mps_callbacks({}, {}, [], '')


def test_public_adapter_rejects_changed_token_hash(monkeypatch):
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK',raising=False)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    objects=dict.fromkeys(m.FAMILIES,None)
    with pytest.raises(ValueError,match='token identity'):m.mps_callbacks(objects,objects,list(range(96)),'0'*64)
