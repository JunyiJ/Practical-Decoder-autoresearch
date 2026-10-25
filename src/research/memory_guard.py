"""Opt-in worker memory watchdog for bounded MoE research only."""
import math
import os
import resource
import sys
import threading
import time


def exceeded(rss_bytes, driver_bytes, rss_mb=4096, driver_mb=6144):
    for value in (rss_bytes, driver_bytes, rss_mb, driver_mb):
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid memory observation or limit')
    return rss_bytes >= rss_mb*1024**2 or driver_bytes >= driver_mb*1024**2


def start_memory_guard():
    # Polling is not a total-system or instantaneous peak-memory guarantee.
    # os._exit stops this worker only; its controller records failure/no retry.
    def watch():
        import torch
        try:
            while True:
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform != 'darwin':rss*=1024
                driver=torch.mps.driver_allocated_memory() if torch.backends.mps.is_available() else 0
                if exceeded(rss,driver):
                    print(f'Memory guard: rss_bytes={rss} mps_driver_bytes={driver}',file=sys.stderr,flush=True)
                    os._exit(86)
                time.sleep(.5)
        except BaseException as error:
            print(f'Memory guard unavailable: {error}',file=sys.stderr,flush=True)
            os._exit(87)
    thread=threading.Thread(target=watch,daemon=True,name='research-memory-guard')
    thread.start()
