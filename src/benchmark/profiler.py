from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class ProfileResult:
    """Results from a single profiling run."""

    latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    peak_gpu_memory_bytes: int = 0
    peak_gpu_memory_mb: float = 0.0
    peak_cpu_memory_bytes: int = 0
    peak_cpu_memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "peak_cpu_memory_bytes": self.peak_cpu_memory_bytes,
            "peak_cpu_memory_mb": self.peak_cpu_memory_mb,
            "gpu_utilization": self.gpu_utilization,
            "metadata": self.metadata,
        }


@dataclass
class AggregatedProfileResult:
    """Aggregated results from multiple profiling runs."""

    runs: list[ProfileResult] = field(default_factory=list)
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_mean_tokens_per_sec: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    peak_cpu_memory_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_runs": len(self.runs),
            "latency_mean_ms": self.latency_mean_ms,
            "latency_std_ms": self.latency_std_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "throughput_mean_tokens_per_sec": self.throughput_mean_tokens_per_sec,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "peak_cpu_memory_mb": self.peak_cpu_memory_mb,
            "metadata": self.metadata,
        }


def _get_gpu_memory_usage() -> int:
    """Get current GPU memory usage in bytes (0 if CUDA unavailable)."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.memory_allocated()


def _get_gpu_peak_memory() -> int:
    """Get peak GPU memory usage in bytes."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.max_memory_allocated()


def _reset_gpu_peak_memory() -> None:
    """Reset GPU peak memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _get_cpu_memory_usage() -> int:
    """Get current CPU memory usage in bytes (approximate)."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KB on Linux, bytes on macOS
        if hasattr(resource, "RUSAGE_SELF"):
            return usage.ru_maxrss * 1024
    except ImportError:
        pass
    return 0


@contextmanager
def track_gpu_memory() -> Generator[dict[str, int]]:
    """Context manager that tracks GPU memory usage.

    Yields a dict that is populated with peak_memory_bytes on exit.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
    result: dict[str, int] = {"peak_memory_bytes": 0, "start_memory_bytes": 0}
    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            result["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
            result["start_memory_bytes"] = start_mem


@contextmanager
def timer() -> Generator[dict[str, float]]:
    """Context manager that measures wall-clock time.

    Yields a dict populated with elapsed_ms on exit.
    """
    result: dict[str, float] = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        end = time.perf_counter()
        result["elapsed_ms"] = (end - start) * 1000.0


@contextmanager
def profile_forward(
    model: nn.Module,
    input_tensor: torch.Tensor,
    warmup_runs: int = 3,
    num_runs: int = 10,
    sync: bool = True,
    device: torch.device | None = None,
) -> Generator[AggregatedProfileResult]:
    """Profile a model's forward pass over multiple runs.

    Performs warmup, then timed runs with GPU memory tracking.
    Yields the AggregatedProfileResult which is populated on exit.

    Args:
        model: The nn.Module to profile.
        input_tensor: Input tensor (or tuple of args).
        warmup_runs: Number of untimed warmup iterations.
        num_runs: Number of timed iterations.
        sync: Whether to call torch.cuda.synchronize() before timing.
        device: Device to move model/inputs to. If None, uses model's device.
    """
    model.eval()
    if device is not None:
        model = model.to(device)
        input_tensor = input_tensor.to(device)

    results: list[ProfileResult] = []

    with torch.no_grad():
        # Warmup
        for _ in range(warmup_runs):
            _ = model(input_tensor)
            if sync and torch.cuda.is_available():
                torch.cuda.synchronize()

        # Timed runs
        for _ in range(num_runs):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            start_time = time.perf_counter()
            _ = model(input_tensor)
            if sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000.0

            # Count tokens (assuming batch x seq_len input)
            if input_tensor.dim() >= 2:
                num_tokens = input_tensor.numel()
            else:
                num_tokens = input_tensor.shape[0]

            throughput = num_tokens / (latency_ms / 1000.0) if latency_ms > 0 else 0.0

            peak_gpu = _get_gpu_peak_memory()
            peak_cpu = _get_cpu_memory_usage()

            results.append(ProfileResult(
                latency_ms=latency_ms,
                throughput_tokens_per_sec=throughput,
                peak_gpu_memory_bytes=peak_gpu,
                peak_gpu_memory_mb=peak_gpu / (1024 * 1024),
                peak_cpu_memory_bytes=peak_cpu,
                peak_cpu_memory_mb=peak_cpu / (1024 * 1024),
            ))

    # Aggregate
    latencies = [r.latency_ms for r in results]
    throughputs = [r.throughput_tokens_per_sec for r in results]

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    aggregated = AggregatedProfileResult(
        runs=results,
        latency_mean_ms=statistics.mean(latencies) if latencies else 0.0,
        latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        latency_p50_ms=sorted_latencies[n // 2] if n > 0 else 0.0,
        latency_p95_ms=sorted_latencies[int(n * 0.95)] if n > 0 else 0.0,
        latency_p99_ms=sorted_latencies[int(n * 0.99)] if n > 0 else 0.0,
        throughput_mean_tokens_per_sec=statistics.mean(throughputs) if throughputs else 0.0,
        peak_gpu_memory_mb=max(r.peak_gpu_memory_mb for r in results) if results else 0.0,
        peak_cpu_memory_mb=max(r.peak_cpu_memory_mb for r in results) if results else 0.0,
    )

    yield aggregated


def profile_model_forward(
    model: nn.Module,
    batch_size: int = 1,
    seq_len: int = 512,
    vocab_size: int = 50257,
    warmup_runs: int = 3,
    num_runs: int = 10,
    device: torch.device | None = None,
    causal: bool = True,
) -> AggregatedProfileResult:
    """Convenience function to profile a transformer model.

    Creates random input, runs the model, and returns aggregated results.
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    model = model.to(device)

    with profile_forward(
        model,
        input_ids,
        warmup_runs=warmup_runs,
        num_runs=num_runs,
        sync=True,
        device=device,
    ) as result:
        pass

    return result


def profile_generation(
    model: nn.Module,
    batch_size: int = 1,
    prompt_len: int = 32,
    max_new_tokens: int = 128,
    vocab_size: int = 50257,
    warmup_runs: int = 1,
    num_runs: int = 5,
    device: torch.device | None = None,
) -> AggregatedProfileResult:
    """Profile autoregressive generation (token-by-token decoding).

    Measures time-per-token and memory for the generate() method.
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)
    model = model.to(device)
    model.eval()

    results: list[ProfileResult] = []

    with torch.no_grad():
        # Warmup
        for _ in range(warmup_runs):
            _ = model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        for _ in range(num_runs):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            start_time = time.perf_counter()
            output = model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000.0
            # Per-sequence tokens generated * batch size = total tokens
            # produced across the batch in this call, matching
            # profile_forward's num_tokens convention (which counts all
            # elements, not just one sequence's).
            total_tokens = (output.shape[-1] - prompt_len) * output.shape[0]
            tokens_per_sec = total_tokens / (latency_ms / 1000.0) if latency_ms > 0 else 0.0

            peak_gpu = _get_gpu_peak_memory()
            peak_cpu = _get_cpu_memory_usage()

            results.append(ProfileResult(
                latency_ms=latency_ms,
                throughput_tokens_per_sec=tokens_per_sec,
                peak_gpu_memory_bytes=peak_gpu,
                peak_gpu_memory_mb=peak_gpu / (1024 * 1024),
                peak_cpu_memory_bytes=peak_cpu,
                peak_cpu_memory_mb=peak_cpu / (1024 * 1024),
                metadata={"total_tokens_generated": total_tokens},
            ))

    latencies = [r.latency_ms for r in results]
    throughputs = [r.throughput_tokens_per_sec for r in results]
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    return AggregatedProfileResult(
        runs=results,
        latency_mean_ms=statistics.mean(latencies) if latencies else 0.0,
        latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        latency_p50_ms=sorted_latencies[n // 2] if n > 0 else 0.0,
        latency_p95_ms=sorted_latencies[int(n * 0.95)] if n > 0 else 0.0,
        latency_p99_ms=sorted_latencies[int(n * 0.99)] if n > 0 else 0.0,
        throughput_mean_tokens_per_sec=statistics.mean(throughputs) if throughputs else 0.0,
        peak_gpu_memory_mb=max(r.peak_gpu_memory_mb for r in results) if results else 0.0,
        peak_cpu_memory_mb=max(r.peak_cpu_memory_mb for r in results) if results else 0.0,
        metadata={"profiling_mode": "generation"},
    )


def profile_sequence_lengths(
    model: nn.Module,
    seq_lengths: list[int],
    batch_size: int = 1,
    vocab_size: int = 50257,
    warmup_runs: int = 3,
    num_runs: int = 10,
    device: torch.device | None = None,
) -> dict[int, AggregatedProfileResult]:
    """Profile a model across multiple sequence lengths.

    Returns a dict mapping seq_len -> AggregatedProfileResult.
    """
    results: dict[int, AggregatedProfileResult] = {}
    for sl in seq_lengths:
        results[sl] = profile_model_forward(
            model,
            batch_size=batch_size,
            seq_len=sl,
            vocab_size=vocab_size,
            warmup_runs=warmup_runs,
            num_runs=num_runs,
            device=device,
        )
    return results


def profile_batch_sizes(
    model: nn.Module,
    batch_sizes: list[int],
    seq_len: int = 512,
    vocab_size: int = 50257,
    warmup_runs: int = 3,
    num_runs: int = 10,
    device: torch.device | None = None,
) -> dict[int, AggregatedProfileResult]:
    """Profile a model across multiple batch sizes.

    Returns a dict mapping batch_size -> AggregatedProfileResult.
    """
    results: dict[int, AggregatedProfileResult] = {}
    for bs in batch_sizes:
        results[bs] = profile_model_forward(
            model,
            batch_size=bs,
            seq_len=seq_len,
            vocab_size=vocab_size,
            warmup_runs=warmup_runs,
            num_runs=num_runs,
            device=device,
        )
    return results
