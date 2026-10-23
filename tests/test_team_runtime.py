"""Pure process-boundary tests: no Codex process or model is started."""
import json
from pathlib import Path

import pytest

from src.research.team import runtime
from src.research.team.roles import schema_for


@pytest.fixture(autouse=True)
def no_real_process(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must never launch a real subprocess")
    monkeypatch.setattr(runtime.subprocess, "Popen", forbidden)


@pytest.mark.parametrize("role,sandbox,search", [
    ("idea", "read-only", "live"), ("coder", "workspace-write", "disabled"),
    ("executor", "read-only", "disabled"), ("evaluator", "read-only", "disabled"),
])
def test_commands_use_role_sandbox_inherited_model_and_no_bypass_flags(tmp_path, role, sandbox, search):
    command = runtime.command_for(role, tmp_path, tmp_path / "handoff", {})
    assert command[:2] == ["codex", "exec"]
    assert command[command.index("--sandbox") + 1] == sandbox
    assert f'web_search="{search}"' in command
    assert 'approval_policy="never"' in command
    assert "--ephemeral" in command and "--json" in command
    assert "--model" not in command
    assert command[-1] == "-"
    assert not {"--dangerously-bypass-approvals-and-sandbox", "--yolo", "--full-auto", "--add-dir"} & set(command)


def test_explicit_runtime_choices_are_argv_values(tmp_path):
    command = runtime.command_for("idea", tmp_path, tmp_path, {"executable": "/tool/codex", "model": "chosen-model", "web_search": False})
    assert command[0] == "/tool/codex"
    assert command[command.index("--model") + 1] == "chosen-model"
    assert 'web_search="disabled"' in command


def test_prepare_call_writes_reviewable_files_without_starting_agent(tmp_path):
    output = tmp_path / "handoff"
    receipt = runtime.prepare_call("executor", "Review the frozen plan.", tmp_path, output, {})
    assert receipt["state"] == "prepared"
    assert (output / "prompt.md").read_text() == "Review the frozen plan."
    assert json.loads((output / "schema.json").read_text()) == schema_for("executor")
    assert json.loads((output / "request.json").read_text()) == receipt
    assert not (output / "response.json").exists()
    assert not (output / "receipt.json").exists()
    with pytest.raises(FileExistsError):
        runtime.prepare_call("executor", "Do not overwrite.", tmp_path, output, {})


def test_execution_requires_explicit_gate_before_artifact_creation(tmp_path):
    output = tmp_path / "handoff"
    with pytest.raises(runtime.AgentRunError, match="disabled"):
        runtime.run_call("executor", "prompt", tmp_path, output, {})
    assert not output.exists()


@pytest.mark.parametrize("argument,value", [("timeout", 0), ("timeout", -1), ("timeout", float("nan")),
                                          ("timeout", float("inf")), ("max_output_mb", 0),
                                          ("max_output_mb", float("nan")), ("max_output_mb", float("inf"))])
def test_nonfinite_or_nonpositive_resource_limits_fail_before_process(tmp_path, argument, value):
    with pytest.raises(ValueError):
        runtime.run_call("executor", "prompt", tmp_path, tmp_path / "handoff", {}, allow_agents=True, **{argument: value})


def fake_process(monkeypatch, *, response=None, raw=None, events=(), returncode=0, running=False):
    calls = []

    class Process:
        def __init__(self, command, **kwargs):
            self.returncode = None if running else returncode
            calls.append((command, kwargs, self))
            result = Path(command[command.index("--output-last-message") + 1])
            if response is not None:
                result.write_text(json.dumps(response))
            if raw is not None:
                result.write_text(raw)
            for event in events:
                kwargs["stdout"].write(event + "\n")
            kwargs["stdout"].flush()

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runtime.shutil, "which", lambda executable: "/fake/codex")
    monkeypatch.setattr(runtime.subprocess, "Popen", Process)
    return calls


def test_structured_response_receipt_records_usage_and_ignores_nonobject_events(tmp_path, monkeypatch):
    response = {"ready": True, "reasons": ["Preflight only."]}
    calls = fake_process(monkeypatch, response=response, events=["not JSON", "null", "[]", "42",
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 4}})])
    output = tmp_path / "handoff"
    receipt = runtime.run_call("executor", "prompt", tmp_path, output, {}, allow_agents=True, lock_fd=17)
    assert receipt["state"] == "completed" and receipt["response"] == response
    assert receipt["usage"] == [{"input_tokens": 10, "output_tokens": 4}]
    assert receipt["response_sha256"] == runtime.file_digest(output / "response.json")
    assert json.loads((output / "receipt.json").read_text()) == receipt
    assert calls[0][1]["pass_fds"] == (17,)
    assert calls[0][1]["start_new_session"] is True
    assert "shell" not in calls[0][1]


@pytest.mark.parametrize("response,raw,match", [
    (None, None, "no structured response"), (None, "not-json", "Invalid executor response"),
    ({"ready": "true", "reasons": []}, None, "Invalid executor response"),
    ({"ready": True, "reasons": [], "outcome": "supported"}, None, "Invalid executor response"),
])
def test_missing_or_invalid_agent_response_is_not_accepted(tmp_path, monkeypatch, response, raw, match):
    fake_process(monkeypatch, response=response, raw=raw)
    output = tmp_path / "handoff"
    with pytest.raises(runtime.AgentRunError, match=match):
        runtime.run_call("executor", "prompt", tmp_path, output, {}, allow_agents=True)
    assert not (output / "receipt.json").exists()


def test_nonzero_exit_does_not_accept_response(tmp_path, monkeypatch):
    fake_process(monkeypatch, response={"ready": True, "reasons": []}, returncode=2)
    with pytest.raises(runtime.AgentRunError, match="exited 2"):
        runtime.run_call("executor", "prompt", tmp_path, tmp_path / "handoff", {}, allow_agents=True)


@pytest.mark.parametrize("resource", ["deadline", "output"])
def test_running_agent_is_terminated_when_capped(tmp_path, monkeypatch, resource):
    calls = fake_process(monkeypatch, running=True)
    killed = []
    monkeypatch.setattr(runtime, "terminate", lambda process: killed.append(process))
    if resource == "deadline":
        ticks = iter([0, 2])
        monkeypatch.setattr(runtime.time, "monotonic", lambda: next(ticks))
        kwargs, message = {"timeout": 1}, "deadline"
    else:
        kwargs, message = {"max_output_mb": .00001}, "output-size"
    with pytest.raises(runtime.AgentRunError, match=message):
        runtime.run_call("executor", "prompt", tmp_path, tmp_path / "handoff", {}, allow_agents=True, **kwargs)
    assert killed == [calls[0][2]]


def test_late_completion_is_rejected_even_after_clean_exit(tmp_path, monkeypatch):
    fake_process(monkeypatch, response={"ready": True, "reasons": []})
    ticks = iter([0, 2])
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(ticks))
    with pytest.raises(runtime.AgentRunError, match="after its deadline"):
        runtime.run_call("executor", "prompt", tmp_path, tmp_path / "handoff", {}, allow_agents=True, timeout=1)
