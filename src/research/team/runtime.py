"""Bounded Codex CLI calls. No API credentials or model defaults are embedded."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from ..protocol import file_digest
from ..runner import atomic_json, terminate
from .roles import schema_for, validate_response


class AgentRunError(RuntimeError):
    pass


def command_for(role: str, cwd: Path, output: Path, settings: dict) -> list[str]:
    schema_for(role)  # Reject unknown roles before constructing a command.
    command = [settings.get("executable", "codex"), "exec", "--cd", str(cwd),
               "--sandbox", "workspace-write" if role == "coder" else "read-only",
               "-c", 'approval_policy="never"',
               "-c", 'web_search="live"' if role == "idea" and settings.get("web_search", True) else 'web_search="disabled"',
               "--ephemeral", "--skip-git-repo-check", "--json", "--color", "never",
               "--output-schema", str(output / "schema.json"),
               "--output-last-message", str(output / "response.json")]
    if settings.get("model"):
        command.extend(["--model", settings["model"]])
    return command + ["-"]


def prepare_call(role: str, prompt: str, cwd: Path, output: Path, settings: dict) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    (output / "prompt.md").write_text(prompt, encoding="utf-8")
    atomic_json(output / "schema.json", schema_for(role))
    command = command_for(role, cwd.resolve(), output.resolve(), settings)
    request = {"role": role, "cwd": str(cwd.resolve()), "command": command,
               "prompt_sha256": file_digest(output / "prompt.md"), "state": "prepared"}
    atomic_json(output / "request.json", request)
    return request


def run_call(role: str, prompt: str, cwd: Path, output: Path, settings: dict,
             *, allow_agents=False, timeout=600.0, max_output_mb=64.0,
             lock_fd: int | None = None) -> dict:
    if not allow_agents:
        raise AgentRunError("Agent execution is disabled; prepare a handoff or explicitly request --run-agents")
    if (not math.isfinite(timeout) or not math.isfinite(max_output_mb)
            or timeout <= 0 or max_output_mb <= 0):
        raise ValueError("Agent time and log budgets must be positive")
    request = prepare_call(role, prompt, cwd, output, settings)
    command = request["command"]
    if shutil.which(command[0]) is None:
        raise AgentRunError(f"Codex CLI not found: {command[0]}")
    started = time.monotonic()
    result_path = output / "response.json"
    with (output / "prompt.md").open("r") as stdin, (output / "events.jsonl").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr,
                                   start_new_session=True, pass_fds=() if lock_fd is None else (lock_fd,),
                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        try:
            while process.poll() is None:
                if time.monotonic() - started >= timeout:
                    raise AgentRunError(f"{role} exceeded its {timeout:g}s agent deadline")
                size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
                if size >= max_output_mb * 1024**2:
                    raise AgentRunError(f"{role} exceeded its output-size budget")
                time.sleep(.2)
        except BaseException:
            terminate(process)
            raise
    if process.returncode:
        raise AgentRunError(f"{role} exited {process.returncode}; inspect {output / 'stderr.log'}")
    if time.monotonic() - started > timeout:
        raise AgentRunError(f"{role} completed after its deadline; response not accepted")
    if not result_path.is_file():
        raise AgentRunError(f"{role} returned no structured response")
    if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > max_output_mb * 1024**2:
        raise AgentRunError(f"{role} response exceeded output-size budget")
    try:
        response = validate_response(role, json.loads(result_path.read_text()))
    except (ValueError, TypeError, KeyError) as exc:
        raise AgentRunError(f"Invalid {role} response: {exc}") from exc
    usage = []
    for line in (output / "events.jsonl").read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usage.append(event["usage"])
    receipt = {**request, "state": "completed", "response": response,
               "response_sha256": file_digest(result_path),
               "duration_seconds": time.monotonic() - started, "usage": usage}
    atomic_json(output / "receipt.json", receipt)
    return receipt
