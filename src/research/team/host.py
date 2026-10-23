"""Read-only execution prerequisites; this does not allocate a model or train."""
import os
import platform
import sys


def inspect_host(device: str) -> dict:
    import torch

    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    if device == "mps":
        available = torch.backends.mps.is_available()
        if fallback == "1":
            raise ValueError("Disable MPS CPU fallback for strict-device experiments")
    elif device.startswith("cuda"):
        index = int(device.split(":")[1]) if ":" in device else 0
        available = torch.cuda.is_available() and index < torch.cuda.device_count()
    else:
        available = device == "cpu"
    if not available:
        raise ValueError(f"Requested training device is unavailable: {device}")
    return {"platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version, "torch": torch.__version__, "device": device,
            "device_available": available, "mps_cpu_fallback": fallback,
            "memory_limit_enforced": False,
            "memory_note": "No hard RAM quota or peak-memory guarantee. Trial time, disk, and failure limits are enforced; actual resource use is measured during trials.",
            "verification_scope": "Availability only; no model operations, tests, benchmarks, or optimizer steps were run."}
