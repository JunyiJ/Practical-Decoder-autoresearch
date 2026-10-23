"""Human stop requests take effect at phase boundaries, never inside a trial."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
import signal
import threading


def request_stop(output: Path) -> dict:
    output = output.resolve()
    if not (output / "campaign.json").is_file():
        raise ValueError("No initialized campaign at this path")
    (output / "STOP").touch(exist_ok=True)
    return {"status": "stop_requested", "campaign": str(output),
            "message": "The active phase will finish before the campaign pauses."}


@contextmanager
def run_control(output: Path, *, resume=False):
    """Exclude duplicate run loops and turn SIGINT/SIGTERM into durable stops."""
    output = output.resolve()
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("A campaign run loop is already active") from exc
        previous = {}
        try:
            if resume:
                (output / "STOP").unlink(missing_ok=True)
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGINT, signal.SIGTERM):
                    previous[signum] = signal.signal(signum, lambda *_: request_stop(output))
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            fcntl.flock(lock, fcntl.LOCK_UN)
