"""Continuous control tests use mock roles/suites, never training or model calls."""
from copy import deepcopy
import signal
from types import SimpleNamespace

import pytest
import yaml

from src.research.team import coordinator
from src.research.team.control import request_stop, run_control
from test_team_coordinator import (campaign, install_fake_handoffs, read, save,
                                   advance_to)


def enable_continuous(campaign):
    state = read(campaign)
    state["config"]["continuous"] = {"target_hours": 48, "min_free_disk_mb": 1}
    for key in ("max_iterations", "max_agent_calls", "total_agent_seconds", "total_training_seconds"):
        state["config"]["limits"][key] = None
    save(campaign, state)
    return state


def test_continuous_passes_old_caps_and_48_hours_with_knowledge(campaign):
    state = enable_continuous(campaign)
    state.update(agent_seconds=4000, training_seconds=9000,
                 run_started_at="2000-01-01T00:00:00Z")
    save(campaign, state)
    install_fake_handoffs(campaign)
    for _ in range(20):  # Four complete iterations, exceeding the old cap of three.
        state = coordinator.advance(campaign["output"], run_agents=True, allow_training=True)
        assert state["status"] != "needs_attention", state["iterations"][-1].get("error")
    assert len(state["iterations"]) == 4
    assert all(item["stage"] == "completed" for item in state["iterations"])
    assert len(state["agent_calls"]) == 16
    assert all(event["timeout"] == 10 for event in campaign["events"]["roles"])
    suites = [event["spec"] for event in campaign["events"]["suites"]]
    assert all(spec["limits"]["session_seconds"] == 30 for spec in suites)
    assert all(spec["audit_test"] is False for spec in suites)
    assert state["iterations"][0]["learning_ids"][0] in campaign["events"]["roles"][4]["prompt"]


def test_stop_finishes_current_agent_and_resumes_next_phase(campaign):
    enable_continuous(campaign)
    install_fake_handoffs(campaign)
    fake = coordinator.run_call

    def stop_during_call(*args, **kwargs):
        request_stop(campaign["output"])
        return fake(*args, **kwargs)

    campaign["monkeypatch"].setattr(coordinator, "run_call", stop_during_call)
    state = coordinator.run_campaign(campaign["output"], run_agents=True, allow_training=True)
    assert state["status"] == "paused"
    assert state["iterations"][0]["stage"] == "coder"
    assert state["agent_calls"][0]["status"] == "completed"
    assert len(campaign["events"]["roles"]) == 1
    # A plain run respects STOP; only explicit resume clears it.
    assert coordinator.run_campaign(campaign["output"], run_agents=True)["status"] == "paused"
    campaign["monkeypatch"].setattr(coordinator, "run_call", fake)
    state = coordinator.run_campaign(campaign["output"], run_agents=True, resume=True)
    assert state["status"] == "awaiting_training_authorization"
    assert [event["role"] for event in campaign["events"]["roles"]] == ["idea", "coder", "executor"]


def test_stop_during_training_preserves_results_and_next_evaluation(campaign):
    enable_continuous(campaign)
    install_fake_handoffs(campaign)
    advance_to(campaign, "train")
    fake = coordinator.run_suite

    def stop_during_suite(*args, **kwargs):
        request_stop(campaign["output"])
        return fake(*args, **kwargs)

    campaign["monkeypatch"].setattr(coordinator, "run_suite", stop_during_suite)
    state = coordinator.run_campaign(campaign["output"], run_agents=True, allow_training=True)
    assert state["status"] == "paused"
    assert state["iterations"][0]["stage"] == "evaluator"
    assert (campaign["output"] / "iteration-001/evidence.json").is_file()
    assert len(campaign["events"]["suites"]) == 1


def test_low_disk_prevents_new_work_without_discarding_phase(campaign):
    enable_continuous(campaign)
    campaign["monkeypatch"].setattr(coordinator.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    state = coordinator.run_campaign(campaign["output"], run_agents=True, allow_training=True)
    assert state["status"] == "insufficient_disk_space"
    assert state["agent_calls"] == []


def test_failure_still_stops_instead_of_spinning(campaign):
    enable_continuous(campaign)
    def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")
    campaign["monkeypatch"].setattr(coordinator, "run_call", fail)
    state = coordinator.run_campaign(campaign["output"], run_agents=True, allow_training=True)
    assert state["status"] == "needs_attention"
    assert state["iterations"][0]["failure_learning_id"]
    assert len(state["agent_calls"]) == 1


def test_signals_request_stop_and_restore_handlers(campaign):
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    with run_control(campaign["output"]):
        for signum in previous:
            signal.getsignal(signum)(signum, None)
            assert (campaign["output"] / "STOP").exists()
    assert all(signal.getsignal(s) == handler for s, handler in previous.items())


def test_duplicate_run_cannot_clear_stop(campaign):
    request_stop(campaign["output"])
    with run_control(campaign["output"]):
        with pytest.raises(ValueError, match="already active"):
            with run_control(campaign["output"], resume=True):
                pytest.fail("Duplicate run acquired the lock")
    assert (campaign["output"] / "STOP").exists()


def test_resume_requires_explicit_execution(campaign):
    request_stop(campaign["output"])
    with pytest.raises(ValueError, match="requires --run-agents"):
        coordinator.run_campaign(campaign["output"], resume=True)
    assert (campaign["output"] / "STOP").exists()


def test_stop_unknown_campaign_does_not_create_it(tmp_path):
    with pytest.raises(ValueError, match="No initialized"):
        request_stop(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("change", [
    lambda cfg: cfg.pop("continuous"),
    lambda cfg: cfg["limits"].update(agent_seconds=None),
    lambda cfg: cfg["limits"].update(max_iterations=3),
    lambda cfg: cfg["continuous"].update(target_hours=0),
    lambda cfg: cfg["continuous"].update(min_free_disk_mb=float("inf")),
])
def test_invalid_continuous_config_rejected(campaign, tmp_path, change):
    config = deepcopy(enable_continuous(campaign)["config"])
    change(config)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        coordinator.load_config(path)


def test_shipped_continuous_config_valid():
    cfg = coordinator.load_config(coordinator.ROOT / "research/team_mac_continuous.yaml")
    assert cfg["continuous"]["target_hours"] == 48
    assert cfg["limits"]["max_iterations"] is None


def test_host_failure_is_recorded_before_agent_dispatch(campaign):
    def unavailable(device):
        raise ValueError("Requested training device is unavailable: mps")
    campaign["monkeypatch"].setattr(coordinator, "inspect_host", unavailable)
    state = coordinator.advance(campaign["output"], run_agents=True, allow_training=True)
    assert state["status"] == "needs_attention"
    assert state["agent_calls"] == []
    assert "device is unavailable" in state["iterations"][0]["error"]


def test_authorization_and_host_checks_reach_executor(campaign):
    install_fake_handoffs(campaign)
    advance_to(campaign, "train")
    prompt = campaign["events"]["roles"][-1]["prompt"]
    assert "has explicit training authorization (--allow-training)" in prompt
    assert '"device_available": true' in prompt
    assert '"checked_at"' in prompt
