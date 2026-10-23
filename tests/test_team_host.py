import pytest
import torch

from src.research.team.host import inspect_host


def test_mps_unavailable_is_not_certified(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    with pytest.raises(ValueError, match="device is unavailable"):
        inspect_host("mps")


def test_mps_cpu_fallback_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(ValueError, match="Disable MPS CPU fallback"):
        inspect_host("mps")


def test_host_check_claims_only_availability(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    facts = inspect_host("mps")
    assert facts["device_available"] is True
    assert facts["memory_limit_enforced"] is False
    assert "no model operations" in facts["verification_scope"]
