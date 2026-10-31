"""Strict sampled MPS prerequisite guard; no fallback or training dispatch."""
import os
import time
import torch
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard, observation, violation
from src.research.memory_guard import exceeded


class StrictMpsGuard(CpuPrerequisiteGuard):
    def __init__(self,journal,root):
        super().__init__(journal,root,timeout_seconds=600,poll_seconds=.25)
        self.limits['mps_driver_limit_bytes']=6*1024**3
        self.peak_driver=0

    def __enter__(self):
        if (os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0'
                or os.environ.get('PYTORCH_MPS_FAST_MATH')!='0'
                or not torch.backends.mps.is_available()):
            raise RuntimeError('Available strict MPS, fallback=0 and fast_math=0 required')
        return super().__enter__()

    def _sample(self):
        sample=observation(self.root)
        driver=torch.mps.driver_allocated_memory()
        self.peak_driver=max(self.peak_driver,driver)
        sample.update(mps_driver_bytes=driver,mps_current_bytes=torch.mps.current_allocated_memory(),
                      peak_observed_mps_driver_bytes=self.peak_driver)
        limits={k:v for k,v in self.limits.items() if k!='mps_driver_limit_bytes'}
        reason=violation(sample,time.monotonic()-self.started,**limits)
        if reason is None and exceeded(sample['peak_rss_bytes'],driver):reason='rss_or_mps_driver_limit'
        return sample,reason
