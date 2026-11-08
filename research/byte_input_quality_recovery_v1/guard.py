"""Retain original resource guards while tolerating atomic publication races."""
import os
from pathlib import Path
import stat

from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.byte_input_quality_v1.contract import OUTPUT_CAP


def _raise(error):
    raise error


def retained_bytes(roots):
    """Count every regular path, including temporaries and hard-link overlap.

    A disappearing path invalidates the whole sample: rescan at most twice.
    Never substitute zero for a missing file or reuse a stale lower count.
    Persistent churn and all other errors still fail the guard.
    """
    for attempt in range(3):
        try:
            total = 0
            for root in roots:
                root = Path(root)
                if not stat.S_ISDIR(os.lstat(root).st_mode):
                    raise ValueError(f'Unsafe output root: {root}')
                for directory, folders, files in os.walk(root, followlinks=False, onerror=_raise):
                    for name in folders + files:
                        path = Path(directory) / name
                        info = os.lstat(path)
                        if stat.S_ISLNK(info.st_mode):
                            raise ValueError(f'Unsafe output symlink: {path}')
                        if stat.S_ISREG(info.st_mode):
                            total += info.st_size
                        elif not stat.S_ISDIR(info.st_mode):
                            raise ValueError(f'Unsafe output type: {path}')
            return total
        except FileNotFoundError:
            if attempt == 2:
                raise


class Guard(CpuPrerequisiteGuard):
    def __init__(self, *args, weight_roots, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_roots = tuple(Path(p) for p in weight_roots)
        self.torch = None

    def _sample(self):
        sample, reason = super()._sample()
        sample['retained_weight_bytes'] = retained_bytes(self.weight_roots)
        sample['metadata_allowance_bytes'] = 512 * 1024**2
        sample['charged_output_bytes'] = sample['retained_weight_bytes'] + sample['metadata_allowance_bytes']
        if sample['charged_output_bytes'] > OUTPUT_CAP:
            reason = 'output_limit'
        if self.torch is not None:
            sample['mps_driver_bytes'] = self.torch.mps.driver_allocated_memory()
            if sample['mps_driver_bytes'] >= 6 * 1024**3:
                reason = 'mps_limit'
        return sample, reason
