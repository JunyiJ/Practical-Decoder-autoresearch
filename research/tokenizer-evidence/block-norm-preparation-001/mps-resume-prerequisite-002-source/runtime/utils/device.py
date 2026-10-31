import torch


def synchronize_device(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def memory_measurement(device):
    """Return a truthful measurement: MPS exposes current, not peak, allocation."""
    import resource
    import sys
    device = torch.device(device)
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2), 'CUDA peak allocated MiB'
    if device.type == 'mps':
        return torch.mps.current_allocated_memory() / (1024 ** 2), 'MPS current allocated MiB (not peak)'
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 2 if sys.platform == 'darwin' else 1024), 'Process lifetime peak RSS MiB'
