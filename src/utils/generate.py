from __future__ import annotations

import argparse
import time
import resource
import sys
from typing import Optional

import torch
from omegaconf import OmegaConf

from ..data.loaders import TinyDataLoader
from ..models.gpt import GPT
from .checkpoint import read_checkpoint, load_model_weights
from .device import synchronize_device, reset_peak_memory, memory_measurement
from ..attention.cache import KVCache

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency for memory stats
    psutil = None

def _select_device(requested: Optional[str]) -> str:
    if requested is None:
        return "cpu"
    device = requested
    if device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _encode(prompt: str, stoi: dict[str, int]) -> torch.Tensor:
    try:
        ids = [stoi[ch] for ch in prompt]
    except KeyError as exc:
        raise ValueError(f"Unknown character in prompt: {exc.args[0]!r}") from exc
    return torch.tensor(ids, dtype=torch.long)

def _device_type(device) -> str:
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":")[0]


def get_mem_usage(device):
    """Returns memory usage in MB."""
    device_type = _device_type(device)
    if device_type == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    if device_type == "mps" and hasattr(torch, "mps"):
        # MPS memory is part of unified memory; we track the process RSS 
        # or use torch.mps.current_allocated_memory() in newer versions
        current = getattr(torch.mps, "current_allocated_memory", None)
        if callable(current):
            return current() / 1e6
        return 0.0
    # CPU
    if psutil is not None:
        return psutil.Process().memory_info().rss / 1e6
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    return rss_bytes / 1e6

@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    cache: Optional[list[KVCache]] = None,
    greedy: bool = False,
) -> torch.Tensor:
    if max_new_tokens <= 0:
        return idx
    if idx.size(1) == 0:
        raise ValueError("Prompt must be non-empty.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")
    if cache is not None and idx.size(1) + max_new_tokens - 1 > model.cfg.block_size:
        raise ValueError("Requested generation exceeds KV cache capacity; reduce tokens or disable cache.")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    def _sample_next(logits: torch.Tensor) -> torch.Tensor:
        if greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            cutoff = values[:, [-1]]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    if cache is not None:
        if idx.size(1) == 0:
            raise ValueError("Prompt must be non-empty when using KV cache.")
        logits, _ = model(idx, cache=cache, start_pos=0)
        next_id = _sample_next(logits[:, -1, :])
        idx = torch.cat((idx, next_id), dim=1)
        for _ in range(max_new_tokens - 1):
            start_pos = idx.size(1) - 1
            logits, _ = model(idx[:, -1:], cache=cache, start_pos=start_pos)
            next_id = _sample_next(logits[:, -1, :])
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond)
        next_id = _sample_next(logits[:, -1, :])
        idx = torch.cat((idx, next_id), dim=1)
    return idx

@torch.no_grad()
def benchmark_generate(model, idx, max_new_tokens, **kwargs):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    device = idx.device
    
    reset_peak_memory(device)
    # --- Warmup ---
    _ = generate(model, idx, max_new_tokens=min(5, max_new_tokens), **kwargs)
    if kwargs.get("cache"):
        for c in kwargs["cache"]: c.reset()

    # --- Benchmark Start ---
    synchronize_device(idx.device)
    start_time = time.perf_counter()
    
    output = generate(model, idx, max_new_tokens=max_new_tokens, **kwargs)
    
    synchronize_device(idx.device)
    end_time = time.perf_counter()
    memory_mb, memory_label = memory_measurement(device)
    
    # Metrics calculation
    total_time = end_time - start_time
    tokens_sec = idx.size(0) * max_new_tokens / total_time
    ms_per_token = (total_time / max_new_tokens) * 1000
    
    print(f"\n{'='*30}")
    print(f"BENCHMARK RESULTS ({model.cfg.attn_type.upper()})")
    print(f"{'='*30}")
    print(f"Throughput:      {tokens_sec:.2f} tokens/sec")
    print(f"Latency:         {ms_per_token:.2f} ms/token")
    print(f"{memory_label}: {memory_mb:.2f}")
    print(f"{'='*30}\n")
    
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tokens from a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument("--data-path", required=True, help="Path to training text (for vocab).")
    parser.add_argument("--prompt", default="", help="Prompt string.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu, mps, or cuda")
    parser.add_argument("--enable-cache", action="store_true", help="Enable KV cache for decoding")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (argmax)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    args = parser.parse_args()

    device = _select_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    ckpt = read_checkpoint(args.checkpoint, map_location=device)
    if "cfg" not in ckpt:
        raise ValueError("Checkpoint missing cfg; cannot reconstruct model.")
    cfg = OmegaConf.create(ckpt["cfg"])

    loader = TinyDataLoader(
        data_path=args.data_path,
        batch_size=1,
        block_size=cfg.model.block_size,
        device=device,
    )
    cfg.model.vocab_size = loader.vocab_size

    model = GPT(cfg.model).to(device)
    load_model_weights(model, ckpt)
    model.eval()

    idx = _encode(args.prompt, loader.stoi).unsqueeze(0).to(device)
    caches = None
    if args.enable_cache and cfg.model.attn_type in ("gqa", "mha", "mla", "gated_mla", "kda"):
        caches = [
            KVCache(
                cfg,
                device,
                batch_size=idx.size(0),
                dtype=next(model.parameters()).dtype,
            )
            for _ in range(cfg.model.n_layers)
        ]
    out = generate(
        model,
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        cache=caches,
        greedy=args.greedy,
    )
    text = loader.decode(out[0].tolist())
    if caches is not None:
        for cache in caches:
            cache.reset()
    print(text)


if __name__ == "__main__":
    main()
